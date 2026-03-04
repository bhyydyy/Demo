import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull

def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP'], "不支持的数据集名称！"
    name = 'dblp' if name == 'DBLP' else name

    # 显式使用 kwargs (root, name, transform) 来避免 PyG 版本间的参数顺序冲突
    if name == 'dblp':
        dataset = CitationFull(
            root=path, 
            name=name, 
            transform=T.NormalizeFeatures()
        )
    else:
        dataset = Planetoid(
            root=path, 
            name=name, 
            transform=T.NormalizeFeatures()
        )
        
    return dataset
    