import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class SharedGCNEncoder(nn.Module):
    """
    参数共享的 GCN 编码器
    用于从图拓扑和节点属性中提取基础特征嵌入 Z
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2):
        super(SharedGCNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        
        # 最后一层不加激活函数，以保留完整的特征空间分布
        z = self.convs[-1](x, edge_index)
        return z

class ProjectionHead(nn.Module):
    """
    非线性投影头 (MLP)
    用于将 GCN 提取的 Z 映射到对比学习的隐空间，保护原始 Z 不被对比损失过度改变
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

class SharedGATEncoder(nn.Module):
    """
    参数共享的 GAT 编码器
    接收原始嵌入 Z 和 动态生成的新邻接矩阵 (new_edge_index)，提取高阶特征 H
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 1):
        super(SharedGATEncoder, self).__init__()
        # concat=False 表示多头注意力结果取平均而不是拼接，保持维度一致
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.gat2 = GATConv(hidden_channels, out_channels, heads=1, concat=False)

    def forward(self, z: torch.Tensor, new_edge_index: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.gat1(z, new_edge_index))
        h = self.gat2(h, new_edge_index)
        return h

class ClusterLayer(nn.Module):
    """
    DEC 聚类层
    维护可学习的聚类中心，并计算 Student-t 分布下的软分配概率 Q
    """
    def __init__(self, n_clusters: int, embedding_dim: int, alpha: float = 1.0):
        super(ClusterLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # 聚类中心作为模型的可学习参数
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, embedding_dim))
        torch.nn.init.xavier_normal_(self.cluster_centers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        输入节点嵌入 Z，输出每个节点属于各个簇的概率矩阵 Q
        """
        # 计算每个样本到每个聚类中心的距离平方
        # z: [N, D], centers: [K, D] -> dist: [N, K]
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, dim=2)
        
        # 使用 Student-t 分布计算软分配概率
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()  # 按行归一化
        return q

class DeepGraphClusteringModel(nn.Module):
    """
    总体模型封装，将 GCN, 投影头, GAT 和聚类层组装在一起
    """
    def __init__(self, config: dict):
        """
        config 需要包含以下键值:
        'input_dim', 'gcn_hidden', 'z_dim', 
        'proj_hidden', 'proj_out',
        'gat_hidden', 'h_dim', 'n_clusters'
        """
        super(DeepGraphClusteringModel, self).__init__()
        
        # 1. GCN 编码器 (输出基础嵌入 Z)
        self.gcn = SharedGCNEncoder(
            in_channels=config['input_dim'], 
            hidden_channels=config['gcn_hidden'], 
            out_channels=config['z_dim']
        )
        
        # 2. 投影头 (输出投影嵌入 Z_proj 供对比学习)
        self.projector = ProjectionHead(
            in_channels=config['z_dim'], 
            hidden_channels=config['proj_hidden'], 
            out_channels=config['proj_out']
        )
        
        # 3. GAT 编码器 (输出高级特征 H 供掩码重构)
        self.gat = SharedGATEncoder(
            in_channels=config['z_dim'], 
            hidden_channels=config['gat_hidden'], 
            out_channels=config['h_dim']
        )
        
        # 4. 聚类层 (供 DEC 损失计算)
        self.cluster_layer = ClusterLayer(
            n_clusters=config['n_clusters'], 
            embedding_dim=config['z_dim']
        )

    def forward_gcn_proj(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        阶段 1：通过 GCN 提取特征，并经过投影头。
        返回原始嵌入 (用于构图和进 GAT) 和投影嵌入 (用于对比损失)。
        """
        z = self.gcn(x, edge_index)
        z_proj = self.projector(z)
        return z, z_proj

    def forward_gat(self, z: torch.Tensor, new_edge_index: torch.Tensor) -> torch.Tensor:
        """
        阶段 2：基于原始嵌入 Z 和新的动态图结构，通过 GAT 提取高阶特征 H。
        返回高阶特征 (用于掩码重构损失)。
        """
        h = self.gat(z, new_edge_index)
        return h