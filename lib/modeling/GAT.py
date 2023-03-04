# 导入需要的包，遇到安装问题可在官方文档或其他文章查找解决方案
import torch
import torch.nn.functional as F
# 导入GCN层、GraphSAGE层和GAT层
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GAE
from torch_geometric.datasets import Planetoid

class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=8, concat=False)
        self.conv2 = GATConv(h_feats, out_feats, heads=8, concat=False)

    def forward(self, x,edge_index):
        # x = x.cuda()
        edge_index = edge_index.cuda()
        # print(x.is_cuda,edge_index.is_cuda)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        z = x
        new_edge = GAE(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = F.softmax(x, dim=1)

        return x, z, new_edge



# model = GAE(4096,4096//4,20)
# print(model.state_dict())

