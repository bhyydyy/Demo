import argparse
import yaml
import torch
import torch.optim as optim
import warnings
from sklearn.cluster import KMeans

# ================== 导入自定义模块 ==================
# 1. 根目录下的核心模块
from dataset import get_dataset
from model import DeepGraphClusteringModel

# 2. utils 文件夹下的工具模块 (注意这里加上了 utils. 前缀)
from utils.util import drop_feature, drop_edge, get_logger, setup_seed
from utils.high_order_graph import get_motif_adjacency, get_khop_adjacency
from utils.graph_adjacency import get_similarity_matrix, get_masked_adjacency_matrix
from utils.loss import ContrastiveLoss, MaskedReconLoss, DECLoss

# 3. utils 文件夹下的评估与可视化模块
from utils.eval import evaluate_clustering
from utils.std_utils import cal_std
from utils.visualization import plot_tsne

warnings.filterwarnings('ignore')

# ================== 稠密矩阵转稀疏边索引 ==================
def dense_to_edge_index(dense_adj):
    """将稠密邻接矩阵转换为 PyG 的 edge_index"""
    row, col = torch.nonzero(dense_adj, as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index

# ================== 主训练流程 ==================
def main():
    # 1. 基础配置与 YAML 参数读取
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 使用 argparse 允许在命令行切换数据集，例如: python train.py --dataset CiteSeer
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name (Cora, CiteSeer, PubMed, DBLP)')
    args = parser.parse_args()
    
    # 读取 config.yaml 文件
    with open('config.yaml', 'r', encoding='utf-8') as f:
        all_configs = yaml.safe_load(f)
        
    config = all_configs[args.dataset]
    config['dataset'] = args.dataset  # 将数据集名称写入 config，方便 logger 记录

    # 初始化顶会级 Logger
    logger = get_logger(config)
    logger.info("=" * 50)
    logger.info("   Starting Deep Graph Clustering Experiment   ")
    logger.info("=" * 50)
    for k, v in config.items():
        logger.info(f"{k:<20}: {v}")
    logger.info("=" * 50)

    # 2. 获取数据
    dataset = get_dataset('./data', config['dataset'])
    data = dataset[0].to(device)
    num_nodes = data.x.size(0)
    config['input_dim'] = dataset.num_features
    config['n_clusters'] = dataset.num_classes

    # 3. 提前计算全局高阶图 (掩码重构的目标)
    logger.info("Generating high-order target graph (Motif)...")
    # 使用强大的三角形模体 (Motif) 作为高阶重构目标
    target_high_order_adj = get_motif_adjacency(data.edge_index, num_nodes, use_norm=True).to(device)
    
    # 用于收集每次 run 的最终测试指标
    fold_acc, fold_nmi, fold_ari, fold_f1 = [], [], [], []

    # ================== 多次独立实验循环 ==================
    for run in range(config['runs']):
        logger.info(f"\n" + "-" * 40)
        logger.info(f"          Starting Run {run + 1}/{config['runs']}          ")
        logger.info("-" * 40)
        
        # 设置随机种子以保证可复现性
        setup_seed(run * 42)

        # 实例化模型、优化器和损失函数
        model = DeepGraphClusteringModel(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
        criterion_con = ContrastiveLoss(temperature=config['tau'])
        criterion_rec = MaskedReconLoss(k=config['k_neighbors'], t=2.0)
        criterion_dec = DECLoss(alpha=1.0)

        # 定义单轮训练逻辑
        def train_epoch(epoch, is_finetune=False):
            model.train()
            optimizer.zero_grad()

            # Step A: 数据增强产生两个视图 (这里完美对应了你的单视图增强Idea)
            edge_index_1 = drop_edge(data.edge_index, drop_prob=config['drop_edge_1'])
            x_1 = drop_feature(data.x, drop_prob=config['drop_feat_1'])
            
            edge_index_2 = drop_edge(data.edge_index, drop_prob=config['drop_edge_2'])
            x_2 = drop_feature(data.x, drop_prob=config['drop_feat_2'])

            # Step B: 共享参数的 GCN + 投影头提取基础嵌入
            z1, z_proj1 = model.forward_gcn_proj(x_1, edge_index_1)
            z2, z_proj2 = model.forward_gcn_proj(x_2, edge_index_2)

            # Step C: 动态构建新图 (根据嵌入的余弦相似度取 Top-K)
            sim1 = get_similarity_matrix(z1, method='cos')
            sim2 = get_similarity_matrix(z2, method='cos')
            
            dense_adj_1 = get_masked_adjacency_matrix(sim1, k=config['k_neighbors'])
            dense_adj_2 = get_masked_adjacency_matrix(sim2, k=config['k_neighbors'])
            
            new_edge_index_1 = dense_to_edge_index(dense_adj_1)
            new_edge_index_2 = dense_to_edge_index(dense_adj_2)

            # Step D: 共享参数的 GAT 提取高级特征供重构使用
            h1 = model.forward_gat(z1, new_edge_index_1)
            h2 = model.forward_gat(z2, new_edge_index_2)

            # Step E: 计算损失 (权重暂定全部为 1)
            loss_con = criterion_con(z_proj1, z_proj2)
            loss_rec = criterion_rec(h1, h2, target_high_order_adj)
            
            loss = loss_con + loss_rec
            loss_kl_val = 0.0

            # 聚类 KL 损失 (仅在微调阶段加入，使用两路特征的均值)
            if is_finetune:
                z_mean = (z1 + z2) / 2.0  
                loss_kl = criterion_dec(z_mean, model.cluster_layer.cluster_centers)
                loss += loss_kl
                loss_kl_val = loss_kl.item()

            loss.backward()
            optimizer.step()
            return loss.item(), loss_con.item(), loss_rec.item(), loss_kl_val

        # ================= 阶段 1: 预训练 (Warm-up) =================
        logger.info("=== Pre-training (Contrastive + Recon) ===")
        for epoch in range(1, config['epochs_pretrain'] + 1):
            loss, l_con, l_rec, _ = train_epoch(epoch, is_finetune=False)
            if epoch % 50 == 0:
                logger.info(f"Pre-train Epoch {epoch:03d} | Total Loss: {loss:.4f} | Con: {l_con:.4f} | Rec: {l_rec:.4f}")

        # [可视化] 预训练结束时的特征分布 (只在第一次运行生成)
        if run == 0:
            model.eval()
            with torch.no_grad():
                z_pre, _ = model.forward_gcn_proj(data.x, data.edge_index)
                plot_tsne(z_pre, data.y, save_path=f'./logs/{config["dataset"]}_tsne_01_pretrain.pdf', title="After Pre-training")

        # ================= 阶段 2: K-Means 冷启动 =================
        logger.info("=== K-Means Initialization ===")
        model.eval()
        with torch.no_grad():
            z_orig, _ = model.forward_gcn_proj(data.x, data.edge_index)
            kmeans = KMeans(n_clusters=config['n_clusters'], n_init=20, random_state=42)
            y_pred_kmeans = kmeans.fit_predict(z_orig.cpu().numpy())
            
            # 使用 K-Means 中心初始化聚类层
            cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
            model.cluster_layer.cluster_centers.data = cluster_centers
            
            y_true = data.y.cpu().numpy()
            acc, nmi, ari, f1 = evaluate_clustering(y_true, y_pred_kmeans)
            logger.info(f"K-Means Init -> ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, F1: {f1:.4f}")

        # ================= 阶段 3: 微调 (加入 KL 损失) =================
        logger.info("=== Fine-tuning (Adding KL Loss) ===")
        best_acc = 0.0
        best_metrics = (0, 0, 0, 0) # 存储 acc, nmi, ari, f1
        
        for epoch in range(1, config['epochs_finetune'] + 1):
            loss, l_con, l_rec, l_kl = train_epoch(epoch, is_finetune=True)
            
            # 每 10 轮评估一次聚类效果
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    z_orig, _ = model.forward_gcn_proj(data.x, data.edge_index)
                    q = model.cluster_layer(z_orig)
                    y_pred = q.argmax(dim=1).cpu().numpy()
                    
                    acc, nmi, ari, f1 = evaluate_clustering(y_true, y_pred)
                    
                    # 记录最佳结果
                    if acc > best_acc:
                        best_acc = acc
                        best_metrics = (acc, nmi, ari, f1)
                        
                    if epoch % 50 == 0:
                        logger.info(f"Fine-tune Epoch {epoch:03d} | Loss: {loss:.4f} (KL:{l_kl:.4f}) | ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f}")

        # [可视化] 微调结束后的特征分布 (只在第一次运行生成)
        if run == 0:
            model.eval()
            with torch.no_grad():
                z_final, _ = model.forward_gcn_proj(data.x, data.edge_index)
                plot_tsne(z_final, data.y, save_path=f'./logs/{config["dataset"]}_tsne_02_finetune.pdf', title="After Fine-tuning")

        logger.info(f"--- Run {run + 1} Best Metrics -> ACC: {best_metrics[0]:.4f}, NMI: {best_metrics[1]:.4f}, ARI: {best_metrics[2]:.4f}, F1: {best_metrics[3]:.4f}")
        
        # 记录本次实验的最佳结果，供最后计算标准差
        fold_acc.append(best_metrics[0])
        fold_nmi.append(best_metrics[1])
        fold_ari.append(best_metrics[2])
        fold_f1.append(best_metrics[3])

    # ================== 最终统计与论文表格输出 ==================
    cal_std(logger, fold_acc, fold_nmi, fold_ari, fold_f1)

if __name__ == '__main__':
    main()