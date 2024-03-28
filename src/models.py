import copy
import math
import torch
import torch.nn as nn
from torch.nn import Embedding
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv,GCNConv, global_max_pool,GATConv
from torch_geometric.utils import dropout_edge, get_laplacian


def MLP(dimensions, dropout=False, batch_norm=False, batch_norm_momentum=1e-3):
    return nn.Sequential(*[
        nn.Sequential(
            nn.Dropout(p=0.5) if dropout else nn.Identity(),
            nn.Linear(dimensions[i - 1], dimensions[i]),
            nn.PReLU(dimensions[i]),
            nn.BatchNorm1d(dimensions[i], affine=True, momentum=batch_norm_momentum) if batch_norm else nn.Identity())
        for i in range(1, len(dimensions))])

def MLP_adj(dimension):
    return nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Tanh())

class MLPModel(nn.Module):
    def __init__(self, dim):
        super(MLPModel, self).__init__()
        self.dim = dim
        self.layers = nn.Sequential(MLP(self.dim[0:-1], batch_norm=True),
                                    nn.Linear(self.dim[-2], self.dim[-1]))

    def forward(self, x):
        return self.layers(x)

class GRU_Model(nn.Module):
    def __init__(self, input_num, hidden_num, output_num, shortcut=False):
        super(GRU_Model, self).__init__()
        self.hidden_num = hidden_num
        self.cell = nn.GRUCell(input_num, hidden_num)
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        self.out_linear = nn.Sequential(MLP(self.output_num[0:-1], batch_norm=True, dropout=True),
                                        nn.Dropout(p=0),
                                        nn.Linear(self.output_num[-2], self.output_num[-1]))
        self.shortcut = shortcut

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)
        next_hidden = self.cell(x, hidden)
        y = self.out_linear(next_hidden)
        if self.shortcut:
            y = y + x[:, :-3]
        return y, next_hidden

class MyGRU_GCN_Model(nn.Module):
    def __init__(self, input_num, hidden_num, output_num, gru_out_dim, gcn_dim, edge_index=None, batch_norm=True):
        
        super(MyGRU_GCN_Model, self).__init__()
        self.hidden_num = hidden_num
        self.cell = nn.GRUCell(input_num, hidden_num)
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        self.gcn_dim = gcn_dim
        self.gru_out_linear = nn.Sequential(MLP(self.output_num, batch_norm=batch_norm, dropout=True))
        self.gru_out_dim = gru_out_dim

        self.out_linear = nn.Sequential(nn.Linear(self.gcn_dim[-1] + self.gru_out_dim, 1))
        self.edge_index = edge_index

        self.motion_net.append(nn.GRUCell(self.motion_input_dim,self.gru_dim))
        self.motion_net.append(nn.Sequential(MLP([self.gru_dim,self.pos_dim[-1]**2],batch_dimensions=self.pos_dim[-1]**2,batch_norm=batch_norm,dropout=True)))
        

        self.dim_conv = nn.ModuleList()
        self.edge_conv_layers = nn.ModuleList()

        for i in range(len(self.gcn_dim) - 1):
            self.edge_conv_layers.append(GCNConv(self.gcn_dim[i],self.gcn_dim[i+1]))

        for i in range(1):
            temp_net = nn.ModuleList()
            temp_net.append(self.edge_conv_layers)
            self.dim_conv.append(temp_net)
        

    def dropout_edges(self, input_edge_index, p=0.9, force_undirected=True):
        if self.training:
            edge_index, edge_mask = dropout_edge(input_edge_index, p=p, force_undirected=force_undirected)
        else:
            edge_index = input_edge_index
        return edge_index,edge_mask

    def forward(self, x, smoothed_vert_pos, hidden=None):


        batch_size = x.shape[0]
        smoothed_vert_pos = smoothed_vert_pos.view((batch_size, -1, 3)) 

        res = torch.zeros((1,batch_size, smoothed_vert_pos.shape[1], self.gcn_dim[-1])).to(x.device)

        for i in range(batch_size):
            for j in range(res.shape[0]):
                tmp = self.dim_conv[j][0][0](smoothed_vert_pos[i,:,j+2].reshape(-1,1),self.edge_index)
                res[j,i,:,:] = self.dim_conv[j][0][1](tmp,self.edge_index)

        if not self.training:
            res = res.detach()

        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)

        next_hidden = self.cell(x, hidden)
        gru_out = self.gru_out_linear(next_hidden).view((batch_size, -1, self.gru_out_dim))

        y = self.out_linear(torch.cat([gru_out,res[0]], axis=2)).view((batch_size, -1))
        return y, next_hidden
    
    def motion_weight(self,motion_data,hidden=None):

        batchsize = motion_data.shape[0]
        motion_data = motion_data.to(self.device)
        if hidden is None:
            hidden = torch.zeros(batchsize, self.gru_dim).to(self.motion_net[0].bias_hh.device)
        else:
            hidden = hidden.to(self.device)
        next_hidden = self.motion_net[0](motion_data,hidden).to(self.device)
        gru_out = self.motion_net[1](next_hidden).view((batchsize,self.pos_dim[-1],self.pos_dim[-1])) 

        return gru_out,next_hidden

class GRU_GAT_Model(nn.Module):

    def __init__(self, input_num, hidden_num, output_num, gru_out_dim, gat_dim, edge_index, batch_norm=True):
        super(GRU_GAT_Model, self).__init__()
        self.hidden_num = hidden_num
        self.cell = nn.GRUCell(input_num, hidden_num)
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        self.gat_dim = gat_dim
        self.gru_out_linear = nn.Sequential(MLP(self.output_num, batch_norm=batch_norm, dropout=True))
        self.gru_out_dim = gru_out_dim
        self.out_linear = nn.Sequential(nn.Linear(self.gat_dim[-1] + self.gru_out_dim, 3))
        self.edge_index = edge_index
        self.GAT_net = nn.ModuleList()

        for i in range(len(self.gat_dim) - 1):
            self.GAT_net.append(GATConv(self.gat_dim[i],self.gat_dim[i+1],heads=1,dropout=0.6))
            
    def dropout_edge(self, input_edge_index, p=0.9, force_undirected=True):
        if self.training:
            edge_index, _ = dropout_edge(input_edge_index, p=p, force_undirected=force_undirected)
        else:
            edge_index = input_edge_index
        return edge_index

    def forward(self, x, smoothed_vert_pos, hidden=None):
        batch_size = x.shape[0]
        smoothed_vert_pos = smoothed_vert_pos.view((batch_size, -1, 3))
        smoothed_x0 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gat_dim[1])).to(x.device)
        smoothed_x1 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gat_dim[2])).to(x.device)

        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x0[i] = self.GAT_net[0](smoothed_vert_pos[i], edge_index)

        if not self.training:
            smoothed_x0 = smoothed_x0.detach()

        for i in range(batch_size):
            edge_index = self.dropout_edge(self.edge_index)
            smoothed_x1[i] = self.GAT_net[1](smoothed_x0[i], edge_index)

        if not self.training:
            smoothed_x1 = smoothed_x1.detach()

        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)
        next_hidden = self.cell(x, hidden)
        gru_out = self.gru_out_linear(next_hidden).view((batch_size, -1, self.gru_out_dim))
        y = self.out_linear(torch.cat([gru_out, smoothed_x1], axis=2)).view((batch_size, -1))
        return y, next_hidden

class MyGRU_GCN_Model_motion(nn.Module):

    def __init__(self, input_num, hidden_num, output_num, gru_out_dim, gnn_dim, edge_index,p=0.9,batch_norm=True):
        super(MyGRU_GCN_Model_motion, self).__init__()
        self.hidden_num = hidden_num
        self.cell = nn.GRUCell(input_num, hidden_num)
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        self.gnn_dim = gnn_dim
        self.gru_out_linear = nn.Sequential(MLP(self.output_num, batch_norm=batch_norm, dropout=True))
        
        self.gru_out_dim = gru_out_dim
        self.out_linear = nn.Sequential(nn.Linear(self.gnn_dim[-1] + self.gru_out_dim, 3))
        self.edge_index = edge_index
        self.edge_conv_layers = nn.ModuleList()
        self.motion_net = nn.ModuleList()

        self.motion_input_dim = 63
        self.gru_dim = 128
        self.edge_num_motion = 18*17
        self.motion_net.append(nn.GRUCell(self.motion_input_dim,self.gru_dim))
        self.motion_net.append(nn.Sequential(
            nn.Dropout(p=0.5) ,
            nn.Linear(self.gru_dim,self.edge_num_motion),
            nn.Sigmoid(),
            ))
        
        
        self.p = p
        for i in range(len(self.gnn_dim) - 1):
            self.edge_conv_layers.append(GCNConv(self.gnn_dim[i], self.gnn_dim[i + 1]))

    def dropout_edges(self, input_edge_index, p=0.9, force_undirected=True):
        if self.training:
            edge_index, _ = dropout_edge(input_edge_index, p=p, force_undirected=force_undirected)
        else:
            edge_index = input_edge_index
        return edge_index

    def forward(self, x,motion_data,smoothed_vert_pos, hidden=None,motion_hidden=None):
        self.device = x.device
        batch_size = x.shape[0]
        edge_weights,next_motion_hidden = self.motion_weight(motion_data,motion_hidden)
        smoothed_vert_pos = smoothed_vert_pos.view((batch_size, -1, 3))
        smoothed_x0 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[1])).to(x.device)

        smoothed_x1 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[2])).to(x.device)

        for i in range(batch_size):
            if self.p > 0:
                edge_index = self.dropout_edges(self.edge_index,p=self.p)
                smoothed_x0[i] = self.edge_conv_layers[0](smoothed_vert_pos[i], edge_index)
            else:
                smoothed_x0[i] = self.edge_conv_layers[0](smoothed_vert_pos[i], self.edge_index,edge_weights[i])


        if not self.training:
            smoothed_x0 = smoothed_x0.detach()

        for i in range(batch_size):
            if self.p > 0:
                edge_index = self.dropout_edges(self.edge_index,p=self.p)
                smoothed_x1[i] = self.edge_conv_layers[1](smoothed_x0[i], edge_index)
            else:
                smoothed_x1[i] = self.edge_conv_layers[1](smoothed_x0[i], self.edge_index,edge_weights[i])   

        if not self.training:
            smoothed_x1 = smoothed_x1.detach()

        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)

        next_hidden = self.cell(x, hidden)
        gru_out = self.gru_out_linear(next_hidden).view((batch_size, -1, self.gru_out_dim))
        y = self.out_linear(torch.cat([gru_out, smoothed_x1], axis=2)).view((batch_size, -1))
        return y, next_hidden,next_motion_hidden

    
    def motion_weight(self,motion_data,hidden=None):
        batchsize = motion_data.shape[0]
        motion_data = motion_data.to(self.device)
        if hidden is None:
            hidden = torch.zeros(batchsize, self.gru_dim).to(self.motion_net[0].bias_hh.device)
        else:
            hidden = hidden.to(self.device)

    
        next_hidden = self.motion_net[0](motion_data,hidden).to(self.device)
        gru_out = self.motion_net[1](next_hidden).view((batchsize,self.edge_num_motion)) 

        return gru_out,next_hidden
    
class GRU_GNN_Model(nn.Module):
    def __init__(self, input_num, hidden_num, output_num, gru_out_dim, gnn_dim, edge_index,p=0.9,batch_norm=True):
        super(GRU_GNN_Model, self).__init__()
        self.hidden_num = hidden_num
        self.cell = nn.GRUCell(input_num, hidden_num)
        self.output_num = copy.deepcopy(output_num)
        self.output_num.insert(0, hidden_num)
        self.gnn_dim = gnn_dim
        self.gru_out_linear = nn.Sequential(MLP(self.output_num, batch_norm=batch_norm, dropout=True))

        self.gru_out_dim = gru_out_dim
        self.out_linear = nn.Sequential(nn.Linear(self.gnn_dim[-1] + self.gru_out_dim, 3))
        self.edge_index = edge_index
        self.edge_conv_layers = nn.ModuleList()
        self.p = p
        for i in range(len(self.gnn_dim) - 1):
            self.edge_conv_layers.append(GCNConv(self.gnn_dim[i], self.gnn_dim[i + 1]))

    def dropout_edges(self, input_edge_index, p=0.9, force_undirected=True):
        if self.training:
            edge_index, _ = dropout_edge(input_edge_index, p=p, force_undirected=force_undirected)
        else:
            edge_index = input_edge_index
        return edge_index

    def forward(self, x, smoothed_vert_pos, hidden=None):
        batch_size = x.shape[0]
        smoothed_vert_pos = smoothed_vert_pos.view((batch_size, -1, 3))
        smoothed_x0 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[1])).to(x.device)

        smoothed_x1 = torch.zeros((batch_size, smoothed_vert_pos.shape[1], self.gnn_dim[2])).to(x.device)

        for i in range(batch_size):
            if self.p > 0:
                edge_index = self.dropout_edges(self.edge_index,p=self.p)
                smoothed_x0[i] = self.edge_conv_layers[0](smoothed_vert_pos[i], edge_index)
            else:
                smoothed_x0[i] = self.edge_conv_layers[0](smoothed_vert_pos[i], self.edge_index)


        if not self.training:
            smoothed_x0 = smoothed_x0.detach()

        for i in range(batch_size):
            if self.p > 0:
                edge_index = self.dropout_edges(self.edge_index,p=self.p)
                smoothed_x1[i] = self.edge_conv_layers[1](smoothed_x0[i], edge_index)
            else:
                smoothed_x1[i] = self.edge_conv_layers[1](smoothed_x0[i], self.edge_index)   

        if not self.training:
            smoothed_x1 = smoothed_x1.detach()


        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_num).to(self.cell.bias_hh.device)

        next_hidden = self.cell(x, hidden)
        gru_out = self.gru_out_linear(next_hidden).view((batch_size, -1, self.gru_out_dim))
        y = self.out_linear(torch.cat([gru_out, smoothed_x1], axis=2)).view((batch_size, -1))
        return y, next_hidden


if __name__ == "__main__":
    pass
