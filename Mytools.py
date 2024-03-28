# 工具函数定义: Loss,collision,laplace
import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from torch_geometric.utils import get_laplacian,to_dense_adj,add_self_loops
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm,gcn_norm_my
from torch_geometric.nn.conv import MessagePassing
from src.obj_parser import Mesh_obj
import fcl
from sklearn.decomposition import PCA,KernelPCA,TruncatedSVD,FactorAnalysis,DictionaryLearning,NMF
from sklearn.manifold import Isomap,MDS,LocallyLinearEmbedding,TSNE
import scipy 
from scipy.spatial.transform import Rotation
import copy


def collision_detect(body_pose,body_trans,pred_pos):
    """基于fcl对执行布料和人体的碰撞检测&布料自碰撞检测
    input:
    布料人体碰撞:返回二者的最短距离,保证这个距离始终>0
    布料自碰撞:再研究研究,与上一帧布料对比？？也不对==
    """
    # body_model是常量,mesh_body只需要输入trans和rotations修改即可
    body = Mesh_obj("./assets/dress02/garment.obj")  
    mesh_body = fcl.BVHModel()
    mesh_body.beginModel(body.v.shape[0], body.f.shape[0])
    mesh_body.addSubModel(body.v,body.f)
    mesh_body.endModel()

    # mesh_cloth的faces是常量,需要修改cloth.v,旋转和平移量来源于动作数据pose&trans
    cloth = Mesh_obj("./assets/dress03/garment.obj")
    mesh_cloth = fcl.BVHModel()
    mesh_cloth.beginModel(cloth.v.shape[0], cloth.f.shape[0])
    mesh_cloth.addSubModel(pred_pos,cloth.f)
    mesh_cloth.endModel()

    # 创建人体碰撞体
    # ！！！这里有点问题:输入的pose是[52,3]带关节的,是否得根据关节旋转重新计算mesh_body顶点？

    tb = fcl.Transform(body_pose,body_trans)
    ob = fcl.CollisionObject(mesh_body, tb)

    # 创建布料碰撞体
    tc = fcl.Transform(Rotation.from_rotvec(body_pose[0]).as_matrix(),body_trans+np.array(np.array([0,-2.1519510746002397,90.4766845703125]) / 100.0))
    oc = fcl.CollisionObject(mesh_cloth,tc)
    
    request = fcl.DistanceRequest()
    result = fcl.DistanceResult()

    # 返回两个碰撞体之间的距离作为Loss惩罚项
    ret = fcl.distance(ob,oc, request, result)
    
    return ret 
# endregion

class Laplacian_Smooth(MessagePassing):

    """对原始数据Laplacian平滑:继承MessagePassing基类;
    可实现: 1)求解laplacian矩阵;2)lap局部坐标;3)平滑后的lap全局坐标
    L = I - D(-1)*A
    input:edge_index & Vertex_positions
    output:laplacian变化后的局部坐标 - 还需要求解一个线性方程组还原到全局坐标 - 也就是平滑后的全局网格
    这里的结果只是局部坐标,不能直接作为平滑后的结果(可以通过可视化obj文件验证)
    """
    def __init__(self):
        super(Laplacian_Smooth, self).__init__(aggr='add')  # "Add" aggregation.
        
    def forward(self,x,edge_index,edge_weight=None):
        # node_dim (int, optional): The axis along which to propagate.(default:`-2`)
        # gcn_norm(self_loop=False) => : D(-0.5)*A*D(-0.5) => norm
        edge_index, norm = gcn_norm_my( 
                        edge_index=edge_index, edge_weight=edge_weight, num_nodes=x.size(self.node_dim),improved=False,
                        add_self_loops=False, flow="source_to_target", dtype=x.dtype)
        
        # print(edge_index)
        # print(norm)
        edge_index, norm = add_self_loops(edge_index=edge_index, edge_attr = -norm,num_nodes=x.size(self.node_dim))
        self.lap_edge = edge_index
        self.lap_weight = norm
        return self.propagate(edge_index,x=x,norm=norm)
    def message(self,x_i,x_j,norm):

        # x_j has shape [E, features]: x_j对应的是edge_index[0]索引对应的坐标取值,不是edge_index[1]
        # I - norm
        #return norm.view(-1, 1) * (x_i  - x_j)
        return (norm.view(-1, 1)) * (x_j)
 
    def update(self, aggr_out):
        return aggr_out
    
    def laplacian_smooth(self,x_global,edge_index,iter_num=10,is_train=True):
        # 对原始网格实现laplacian平滑: 剔除高频信息; -- 不是执行laplacian变换
        # x_global:torch_tensor:[numverts,3] -> 初始全局坐标
        # output:final_res -> 拉普拉斯平滑后的全局坐标 
        # 平滑iter_num次,权重 = 0.1
        weight = 0.2
        if is_train:
            for i in range(iter_num):
                # print("平滑中...")
                x_local = self(x_global,edge_index = edge_index)
                x_global = x_global - weight * x_local
            return x_global
        else:
            pos_list = []
            for i in range(iter_num):
                # print("平滑中...")
                x_local = self(x_global,edge_index = edge_index)
                x_global = x_global - weight * x_local
                pos_list.append(x_global)
            
            return pos_list
    
    def laplacian_with_ancher(self,x=None,x_init=None,edge_index=None,Ancher=None):
        # 执行laplacian变换,需要指定锚点来对解进行定位！# 根据拉普拉斯变换后的局部坐标来还原全局坐标
        # Ancher:[indexlist,np.array(shape=(len(indexlist),3))] -> 表示ancher的索引和目标位置,第一个维度存储锚点索引列表;第二个表示目标位置
        # x:torch_tensor:[numverts,3] -> 拉普拉斯变换后的局部坐标
        # x_init:torch_tensor:[numverts,3] -> 原始模型的全局坐标
        # edge_index: 模型的初始邻接信息
        # output:final_res -> 拉普拉斯变换后的全局坐标
        import copy
        from numpy.linalg import lstsq # 解超定方程
           
        if x == None:
            # 如果输入的是x_init即初始的全局坐标,需要结合邻接矩阵先计算对应的laplacian局部坐标
            x = self(x_init,edge_index)
   
        weight = 0.5
        # 获得laplacian矩阵的矩阵表示
        adj = to_dense_adj(edge_index = self.lap_edge,edge_attr = self.lap_weight)[0].T
        # print(adj)
        adj_temp = copy.deepcopy(adj.numpy())
        x_temp = copy.deepcopy(x.numpy())
        # print(x.shape) [numverts,3]

        # 变换需要指定anchor,但平滑不需要
        for index in Ancher[0]:
            tmp = np.zeros((1,x.shape[0]))
            tmp[0][index] = weight
            # print(adj_temp.shape) #[numvert,numvert]
            adj_temp = np.concatenate([adj_temp,tmp],axis = 0)
        Ancher[1] = Ancher[1].reshape(len(Ancher[0]),3)
        # print(x_temp.shape,Ancher[1].shape)
        x_temp = np.concatenate([x_temp,weight*Ancher[1]],axis = 0)

        print('开始求解超定齐次线性方程....')

        final_res = lstsq(adj_temp,x_temp,rcond=None)
        # print(final_res[0])
        return final_res[0]

class reduction_tool_plus():
    # 对原始布料模型的顶点降维&还原工具升级版 -- 顶点层次的降维:非三维坐标&时间长度
    # 支持PCA/KPCA/...等多种降维方式

    def __init__(self,init_pos,method="PCA",n_componment=18):
        self.choice = ["PCA","KPCA","FA","TSVD","DictLearn","NMF"]
        self.numverts = init_pos.shape[0]
        self.method = method
        self.init_pos = init_pos.T #[500*3,numverts] -> 初始顶点坐标数据:取转置为了保证对顶点数目降维
        self.n_componment = n_componment # 任意一套动作降维之后得到降维dim=18,取18作为基准

    def init_process(self):
        # 首次降维操作: 对初始布料模型(不包含任何动作信息)降维,并保存fit后的模型作为预训练模型
        # 由于初始布料模型相当于只有一个样本 - 无法降维,只能使用三维坐标作为三个样本降维 -- 主成分 <= 3

        if self.method == "PCA":
            self.percent = 0.999
            self.tool = PCA(n_components=self.percent) 
            self.tool.fit(self.init_pos)
            new = self.tool.transform(self.init_pos)
            self.n_componment = new.shape[1]
            print("PCA_compoments:{},dim = {}.".format(np.sum(self.tool.explained_variance_ratio_),self.n_componment))

        elif self.method == "KPCA":
            self.tool = KernelPCA(n_components=self.n_componment,kernel="rbf",gamma=20,degree=3,fit_inverse_transform=True) 
            self.tool.fit(self.init_pos)
            print("KPCA_compoments:{},dim = {}.".format(1.0,self.n_componment))
        
        elif self.method == "TSVD":
            """截断SVD分解"""
            self.tool = TruncatedSVD(n_components = self.n_componment,n_iter = 7,random_state = 3439)
            self.tool.fit(self.init_pos)
            print("TSVD_compoments:{},dim = {}.".format(1.0,self.n_componment))
        
        elif self.method == "dict":
            self.tool = DictionaryLearning(n_components=self.n_componment,transform_algorithm='lasso_lars',transform_alpha=0.1,random_state=3439)
            self.tool.fit(self.init_pos)
            print("DictLearning: ...,dim = {}.".format(self.n_componment))
        
        elif self.method == "FA":
            self.tool = FactorAnalysis(n_components=self.n_componment,random_state=3439) 
            self.tool.fit(self.init_pos)
            Ih = np.eye(len(self.tool.components_))
            self.tool.Wpsi = self.tool.components_ / self.tool.noise_variance_
            self.tool.cov_z_inv = Ih + np.dot(self.tool.Wpsi, self.tool.components_.T)
            print("FA_process: ...,dim = {}.".format(self.n_componment))

        elif self.method == "NMF":
            self.tool = NMF(n_components=self.n_componment,init="random",random_state=3439,solver="mu",max_iter=100) 
            self.min_temp = np.min(self.init_pos,axis=0)
            input_data = self.init_pos - self.min_temp
            self.tool.fit(input_data)
            print("NMF process :...,dim = {}.".format(self.n_componment))

        return self.n_componment
    
    def dim_reduction(self,x,is_fit=False):

        batch_size = x.shape[0]
        x = torch.transpose(x,2,1).reshape(batch_size*3,-1)
        x_ = x.detach().clone() 
        if not is_fit:
            if self.method == "KPCA":                
                x_ = self.tool._validate_data(x_.cpu().numpy(), accept_sparse="csr", reset=False)
                K = self.tool._centerer.transform(self.tool._get_kernel(x_, self.tool.X_fit_)) # [3,12273]
                non_zeros = np.flatnonzero(self.tool.eigenvalues_)
                scaled_alphas = np.zeros_like(self.tool.eigenvectors_)
                scaled_alphas[:, non_zeros] = self.tool.eigenvectors_[:, non_zeros] / np.sqrt(self.tool.eigenvalues_[non_zeros])
                x_reduce = torch.from_numpy(np.dot(K, scaled_alphas)).to(torch.float32).to(x.device)

            elif self.method == "PCA":
                x_reduce = torch.matmul(x - torch.from_numpy(self.tool.mean_).to(x.device), torch.from_numpy(self.tool.components_.T).to(x.device)).to(torch.float32) 

            elif self.method == "TSVD":
                x_reduce = torch.matmul(x, torch.from_numpy(self.tool.components_.T).to(x.device).to(torch.float32))
            
            elif self.method == "dict":
                x_reduce = torch.matmul(x, torch.from_numpy(self.tool.components_.T).to(x.device)).to(torch.float32)
                mask = ~(x_reduce == (torch.max(x_reduce,dim = -2)[0].repeat(x_reduce.shape[0],1))).
                x_reduce[mask] = 0  
            
            elif self.method == "FA":        
                Ih = np.eye(len(self.tool.components_))
                x_transformed = x - torch.from_numpy(self.tool.mean_).to(x.device)

                self.tool.Wpsi = self.tool.components_ / self.tool.noise_variance_
                self.tool.cov_z_inv = Ih + np.dot(self.tool.Wpsi, self.tool.components_.T)
                cov_z = torch.from_numpy(scipy.linalg.inv(self.tool.cov_z_inv)).to(x.device)
                tmp = torch.matmul(x_transformed,torch.from_numpy(self.tool.Wpsi.T).to(x.device))
                x_reduce = torch.matmul(tmp,cov_z).float()   
            
            elif self.method == "NMF":
                input_x = x_.cpu().numpy()
                self.min_temp = np.min(input_x,axis=0)
                x_reduce = torch.matmul(x-torch.min(x,dim=0), torch.from_numpy(scipy.linalg.pinv(self.tool.components_)).to(torch.float32).to(x.device))
                
        else:
            if self.method == "PCA":
                self.tool = PCA(n_components=self.n_componment) 
                self.tool.fit(x_.cpu().numpy()) 
                x_reduce = torch.matmul(x - torch.from_numpy(self.tool.mean_).to(x.device), torch.from_numpy(self.tool.components_.T).to(x.device)).to(torch.float32) 

                
            elif self.method == "KPCA":
                self.tool = KernelPCA(n_components=self.n_componment,kernel="rbf",gamma=20,degree=3,fit_inverse_transform=True) 
                self.tool.fit(x_.cpu().numpy())
                x_ = self._validate_data(x_.cpu().numpy(), accept_sparse="csr", reset=False)
                K = self.tool._centerer.transform(self.tool._get_kernel(x_, self.tool.X_fit_))
                non_zeros = np.flatnonzero(self.tool.eigenvalues_)
                scaled_alphas = np.zeros_like(self.tool.eigenvectors_)
                scaled_alphas[:, non_zeros] = self.tool.eigenvectors_[:, non_zeros] / np.sqrt(self.tool.eigenvalues_[non_zeros])
                x_reduce = torch.from_numpy(np.dot(K, scaled_alphas)).to(torch.float32).to(x.device)
            
            elif self.method == "TSVD":
                self.tool = TruncatedSVD(n_components = 3,n_iter = 7,random_state = 3439)
                self.tool.fit(x_.cpu().numpy())
                x_reduce = torch.matmul(x, torch.from_numpy(self.tool.components_.T).to(x.device).to(torch.float32))
                
        
            elif self.method == "dict":
                self.tool = DictionaryLearning(n_components=self.n_componment,transform_algorithm='lasso_lars',transform_alpha=0.1,random_state=3439)
                self.tool.fit(x_.cpu().numpy())
                x_reduce = torch.matmul(x, torch.from_numpy(self.tool.components_.T).to(x.device)).to(torch.float32)
                mask = ~(x_reduce == (torch.max(x_reduce,dim = -2)[0].repeat(x_reduce.shape[0],1)))
                x_reduce[mask] = 0  
                
            elif self.method == "FA":
                self.tool = FactorAnalysis(n_components=self.n_componment,random_state=3439) 
                self.tool.fit(x_.cpu().numpy())
                
                Ih = np.eye(len(self.tool.components_))
                x_transformed = x - torch.from_numpy(self.tool.mean_).to(x.device)
                self.tool.Wpsi = self.tool.components_ / self.tool.noise_variance_
                self.tool.cov_z_inv = Ih + np.dot(self.tool.Wpsi, self.tool.components_.T)
                cov_z = torch.from_numpy(scipy.linalg.inv(self.tool.cov_z_inv)).to(x.device)
                tmp = torch.matmul(x_transformed,torch.from_numpy(self.tool.Wpsi.T).to(x.device))
                x_reduce = torch.matmul(tmp,cov_z).to(torch.float32) 
 
 
            elif self.method == "NMF":
                input_x = x_.cpu().numpy()
                self.min_temp = np.min(input_x,axis=0)
                input_x = input_x - self.min_temp
                self.tool = NMF(n_components=self.n_componment,init="random",random_state=3439) 
                self.tool.fit(input_x)  
                x_reduce = torch.matmul(x-torch.min(x,dim=0), torch.from_numpy(scipy.linalg.pinv(self.tool.components_)).to(torch.float32).to(x.device))
                
        return x_reduce.reshape(batch_size,1,-1,3)
     
    def recover_matrix(self,matrix):

        if self.method == "PCA":
            x_recover = torch.matmul(matrix.float(),torch.from_numpy(self.tool.components_).float().to(matrix.device)) # pca.components_ :[pca_dim,12273]pca转换矩阵
            x_recover = torch.matmul(torch.from_numpy(self.tool.components_.T),x_recover) + 0.5 * torch.from_numpy(self.tool.mean_).float().to(matrix.device)
            x_recover = x_recover.T + + 0.5*torch.from_numpy(self.tool.mean_).float().to(matrix.device)

        elif self.method == "TSVD":
            x_recover = (torch.matmul(matrix.float(),torch.from_numpy(self.tool.components_).float().to(matrix.device))) #[12273,reduce_dim]
            x_recover = torch.matmul(torch.from_numpy(self.tool.components_.T).float().to(matrix.device),x_recover)
        
        elif self.method == "dict":
            x_recover = (torch.matmul(matrix.float(),torch.from_numpy(self.tool.components_).float().to(matrix.device)))
            x_recover = torch.matmul(torch.from_numpy(self.tool.components_.T).to(matrix.device),x_recover)

        elif self.method == "FA":
            tmp = torch.matmul(matrix.float(),torch.from_numpy(self.tool.cov_z_inv).float().to(matrix.device))  
            inv_w = torch.from_numpy(scipy.linalg.pinv(self.tool.Wpsi.T)).float().to(matrix.device)
            x_recover = torch.matmul(tmp,inv_w).T

            x_recover = torch.matmul(x_recover,torch.from_numpy(self.tool.cov_z_inv).float().to(matrix.device))
            x_recover = torch.matmul(x_recover,inv_w) + 0.5 * torch.from_numpy(self.tool.mean_).float().to(matrix.device)
            x_recover = x_recover.T + 0.5 * torch.from_numpy(self.tool.mean_).float().to(matrix.device)

        elif self.method == "NMF":        
            x_recover = torch.matmul(matrix.float(),torch.from_numpy(self.tool.components_).float().to(matrix.device)) 
            x_recover = torch.matmul(torch.from_numpy(self.tool.components_.T).float().to(matrix.device),x_recover) + 0.5 * torch.from_numpy(self.min_temp).float()
            x_recover = x_recover.T + 0.5 * torch.from_numpy(self.min_temp).float()

        return x_recover.reshape(self.numverts,self.numverts)
    
    def dim_recover(self,x):
   
        batch_size = x.shape[0]
        x = x.reshape(x.shape[0]*3,-1)
        x_ = x.detach().clone() 

        if self.method == "PCA":
            x_recover = (torch.matmul(x.float(),torch.from_numpy(self.tool.components_).float().to(x.device)) + torch.from_numpy(self.tool.mean_).float().to(x.device)).T
        
        elif self.method == "KPCA":
            K = self.tool._get_kernel(x_.cpu().numpy(), self.tool.X_transformed_fit_)
            x_recover = 2.0 * torch.from_numpy(np.dot(K, self.tool.dual_coef_)).to(torch.float32).to(x.device).T

        elif self.method == "TSVD":
            x_recover = (torch.matmul(x.float(),torch.from_numpy(self.tool.components_).float().to(x.device))).T
        
        elif self.method == "dict":
            x_recover = (torch.matmul(x.float(),torch.from_numpy(self.tool.components_).float().to(x.device))).T

        elif self.method == "FA":
            tmp = torch.matmul(x.float(),torch.from_numpy(self.tool.cov_z_inv).float().to(x.device))             
            x_recover = torch.matmul(tmp,torch.from_numpy(scipy.linalg.pinv(self.tool.Wpsi.T)).to(torch.float32).to(x.device))
            x_recover = (x_recover + torch.from_numpy(self.tool.mean_).float().to(x.device)).T
            
        elif self.method == "NMF":      
            x_recover = (torch.matmul(x.float(),torch.from_numpy(self.tool.components_).float().to(x.device)) + torch.from_numpy(self.min_temp).float()).T            
        return x_recover.reshape(batch_size,-1,3)

class reduction_tool():
    def __init__(self,init_pos):
        self.method = "PCA"
        self.init_pos = init_pos.T

    def PCA_process(self):
        self.pca_tool = PCA(n_components=3) 
        self.pca_tool.fit(self.init_pos)
        print("PCA_compoments:{},dim = {}.".format(np.sum(self.pca_tool.explained_variance_ratio_),3))

    def dim_reduction(self,x):
        
        x = x.reshape(x.shape[0]*3,-1) 
        x_reduce = torch.matmul(x - torch.from_numpy(self.pca_tool.mean_).to(x.device), torch.from_numpy(self.pca_tool.components_.T).to(x.device)).to(torch.float32) 
        return x_reduce.reshape(x.shape[0],1,-1,3)
    
    def dim_recover(self,x):
        x_recover = torch.matmul(x.reshape(x.shape[0]*3,-1).float(),torch.from_numpy(self.pca_tool.components_).float().to(x.device)) + torch.from_numpy(self.pca_tool.mean_).float().to(x.device) 
        return x_recover.reshape(x.shape[0],-1,3)
    
    def adj_trans(self,adj_matrix):
        adj_reduce = self.pca_tool.transform(self.pca_tool.transform(adj_matrix).T)
        return adj_reduce

class cluster_data(Dataset):
    def __init__(self,state,id_list,length=429,path="VirtualBoneDataset/dress02/HFdata",cluster_id=0):

        self.pose = []
        self.trans = []
        self.cluster_id = cluster_id
        self.length = length 
        self.id_list = id_list 
        self.cluster_res = []

        for i in os.listdir(path)[:]:

            if "npz" in i :
                tmp = np.load(os.path.join(path,i),allow_pickle=True)
                if (tmp["pose"].shape != (500,52,3))or(tmp["trans"].shape != (500,3))or(tmp["sim_res"].shape!=(500,12273,3)) :
                    continue
                final = torch.zeros(500,self.length,3)
                tmp2 = np.load(os.path.join("VirtualBoneDataset/dress02/spectral_cosine",i),allow_pickle=True)

                self.pose.append(torch.from_numpy(tmp["pose"]))
                self.trans.append(torch.from_numpy(tmp["trans"]))

                final[:,:(self.id_list[cluster_id+1]-self.id_list[cluster_id]),:] = torch.from_numpy(tmp2["spectral_res"][:,self.id_list[cluster_id]:self.id_list[cluster_id+1],:])
                self.cluster_res.append(final)

        self.len = len(self.pose)

        for i in range(self.len):
            self.pose[i] = (self.pose[i] - state["pose_mean"]) / state["pose_std"]
            self.trans[i] = (self.trans[i] - self.trans[0] - state["trans_mean"]) / state["trans_std"]   

    def __getitem__(self,idx):

        return self.pose[idx],self.trans[idx],self.cluster_res[idx]

    def __len__(self):
        return self.len

class Mydata(Dataset):
    def __init__(self,path,state,mode="LF"):
        self.pose = []
        self.trans = []
        self.mode = mode
        if mode == "LF":

            self.sim_res = []
            self.dpos = []
            for i in os.listdir(path)[:]:
                if "npz" in i:
                    tmp = np.load(os.path.join(path,i),allow_pickle=True)
                    if (tmp["pose"].shape != (500,52,3))or(tmp["trans"].shape != (500,3)) :
                        continue
                    tmp2 = np.load(os.path.join("VirtualBoneDataset/dress02/LF_res",i),allow_pickle=True)

                    self.pose.append(torch.from_numpy(tmp["pose"]))
                    self.trans.append(torch.from_numpy(tmp["trans"]))
                    self.sim_res.append(torch.from_numpy(tmp2["lap_pos"]))
                    self.dpos.append(torch.from_numpy(tmp2["dpos"]))
                    
        elif mode == "HF":
            self.sim_res = []
            for i in os.listdir(path)[:]:
                if "npz" in i:
                    tmp = np.load(os.path.join(path,i),allow_pickle=True)
                    if (tmp["pose"].shape != (500,52,3))or(tmp["trans"].shape != (500,3))or(tmp["sim_res"].shape!=(500,12273,3)) :
                        continue
                    tmp2 = np.load(os.path.join("VirtualBoneDataset/dress02/HF_res",i),allow_pickle=True)

                    self.pose.append(torch.from_numpy(tmp["pose"]))
                    self.trans.append(torch.from_numpy(tmp["trans"]))
                    self.sim_res.append(torch.from_numpy(tmp2["final_ground"]))

        self.len = len(self.pose)
        for i in range(self.len):
            self.pose[i] = (self.pose[i] - state["pose_mean"]) / state["pose_std"]
            self.trans[i] = (self.trans[i] - self.trans[i][0] - state["trans_mean"]) / state["trans_std"]   

    def __getitem__(self,idx):
        if self.mode == "HF": 
            return self.pose[idx],self.trans[idx],self.sim_res[idx]
        if self.mode == "LF": 
            return self.pose[idx],self.trans[idx],self.sim_res[idx],self.dpos[idx]
            
    def __len__(self):
        return self.len


def get_full_edge_index(numverts=1,mode="empty"):
    # 根据顶点数量创建全孤立&全连通的邻接矩阵  -- 返回稀疏格式[2,edge_num]
    from torch_geometric.utils import dense_to_sparse    
    if mode == "empty":
        adj = torch.zeros((numverts,numverts))  
    elif mode == "full":
        adj = torch.ones((numverts,numverts)) - torch.eye(numverts)
    return dense_to_sparse(adj)[0]

def Loss_func(pos,pred_pos,mode="L2",bool_col = 0):
    """
    # https://blog.csdn.net/WANGWUSHAN/article/details/105903765
    定义反向传播的loss函数：
    ## 只接受tensor张量输入
    1. 均方损失
    2. 绝对值损失
    3. 混合损失
    4. BCEloss: 需要将数据处理到[0,1]之间 -- sigmoid
    """
    if mode == "L2":
        # RMSE
        loss_fn = torch.nn.MSELoss(reduction='mean')
        return torch.sqrt(loss_fn(pos,pred_pos))

    elif mode == "L1":
        loss_fn = torch.nn.L1Loss(reduction='mean')
        return loss_fn(pos,pred_pos)
      
    elif mode == "Mix":
        loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        return loss_fn(pos,pred_pos)
    
    elif mode == "BCE":
        sigmoid = torch.nn.Sigmoid()
        loss_fn = torch.nn.BCELoss()
        sigmoid_pos=sigmoid(pos)
        sigmoid_pos_pred=sigmoid(pred_pos)
        return loss_fn(sigmoid_pos,sigmoid_pos_pred)
    
    elif mode == "BCE_L":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        return loss_fn(pos,pred_pos)  

    
    elif mode == "CrossE":
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(pos,pred_pos)
    
    elif mode == "NL":
        log_ = torch.nn.LogSoftmax(dim=1)
        loss_fn = torch.nn.NLLLoss()
        log_pos = log_(pos)
        return loss_fn(log_pos,pred_pos)
    
    elif mode == "KL":
        logp_y = F.log_softmax(pred_pos, dim = -1)
        p_x = F.softmax(pos, dim =- 1)

        dist = F.kl_div(logp_y, p_x, reduction='batchmean')
        return dist   
    
def KL_distance(x,y):

    logp_y = F.log_softmax(y, dim = -1)
    p_x = F.softmax(x, dim =- 1)

    dist = F.kl_div(logp_y, p_x, reduction='batchmean')
    return dist     

class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    

if __name__ == "__main__":

    x1 = torch.ones((9,12273,3))
    x2 = torch.ones((9,12273,3))

    res = KL_distance(x1,x2)
    print(res.item())
