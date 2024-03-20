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

# region Defined-self about laplaican_smooth -- too slow    
# def Smoothing_pos(edge_index,pos_x,normalization='rw'):
#      """
#      网格的Laplacian平滑,执行效率还是很低,pyg里貌似有可以直接执行的模块-GCN_norm?
#      L_rw = I - D(-1)A
#      L_sym = I - D(-0.5)A(-0.5)
#      """
#      # pos_x:[numvert,3]
#      # 1.计算modified laplacian平滑矩阵L
#      indexs,weights = get_laplacian(edge_index,normalization=normalization)
#      print(indexs,weights)
#      # 2.根据L对原始网格坐标进行平滑
#      dpos = torch.zeros_like(pos_x)
#      res = torch.mul(weights.reshape(-1,1),pos_x[indexs[1]])
#      #for i in range(res.shape[1]):
#      #    dpos[:,indexs[0,i]] = dpos[:,indexs[0,i]] + res[:,i]
#      for i in range(pos_x.shape[1]):
#          # 上一个循环遍历所有边,效率很低,修改为遍历顶点. [1,0,2,1]==[0,0,0,0]=[FTFF],然后sum(res[F,T,F,F])->dpos[0]
#          # 经验证,两种方法返回结果一致
#          tmpp = res[indexs[0]==(torch.zeros_like(indexs[0])+i)]
#          dpos[i] = torch.sum(tmpp,dim=0)
#      return dpos

# def Laplacian_matrix(adj_matrix):
#     """根据邻接矩阵(不需要self-loop)计算Laplacian变换矩阵
#     PYG库中自带计算L矩阵的函数:get_laplacian"""
#     # 1. 顶点规模:adj_matrix[1,num,num]
#     num = adj_matrix.shape[1]
#     # 2. L = (I-D^(-1)A)
#     # 这里注意维度一定要对应:依赖输入matrix的shape
#     L = torch.transpose(torch.sum(adj_matrix,dim=2).expand(1,num,num),1,2).to("cuda")
#     L = torch.eye(num).reshape(1,num,num).to("cuda") - torch.div(adj_matrix,L)
#     return L 


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
            # input_data = np.exp(self.init_pos) # 将原始数据转换为非负
            self.min_temp = np.min(self.init_pos,axis=0)
            input_data = self.init_pos - self.min_temp
            self.tool.fit(input_data)
            print("NMF process :...,dim = {}.".format(self.n_componment))

        return self.n_componment
    
    def dim_reduction(self,x,is_fit=False):

        batch_size = x.shape[0]
        x = torch.transpose(x,2,1).reshape(batch_size*3,-1)
        x_ = x.detach().clone() # 从计算图剥离并拷贝
        if not is_fit:
            if self.method == "KPCA":                
                x_ = self.tool._validate_data(x_.cpu().numpy(), accept_sparse="csr", reset=False)
                K = self.tool._centerer.transform(self.tool._get_kernel(x_, self.tool.X_fit_)) # [3,12273]
                non_zeros = np.flatnonzero(self.tool.eigenvalues_)
                scaled_alphas = np.zeros_like(self.tool.eigenvectors_)
                scaled_alphas[:, non_zeros] = self.tool.eigenvectors_[:, non_zeros] / np.sqrt(self.tool.eigenvalues_[non_zeros])
                x_reduce = torch.from_numpy(np.dot(K, scaled_alphas)).to(torch.float32).to(x.device)
                #x_reduce = torch.from_numpy(self.tool.transform(x.cpu().numpy())).to(x.device).to(torch.float32) 

            elif self.method == "PCA":
                x_reduce = torch.matmul(x - torch.from_numpy(self.tool.mean_).to(x.device), torch.from_numpy(self.tool.components_.T).to(x.device)).to(torch.float32) 

            elif self.method == "TSVD":
                x_reduce = torch.matmul(x, torch.from_numpy(self.tool.components_.T).to(x.device).to(torch.float32))
            
            elif self.method == "dict":
                x_reduce = torch.matmul(x, torch.from_numpy(self.tool.components_.T).to(x.device)).to(torch.float32) # shape:[3,3];[batchsize*3,reduce_dim]
                mask = ~(x_reduce == (torch.max(x_reduce,dim = -2)[0].repeat(x_reduce.shape[0],1)))
                #mask = not(x_reduce == (torch.max(x_reduce,dim = -2)[0].repeat(x_reduce.shape[0],1))) # 不能用not,会报错 -- not应该只能作用在单个逻辑变量..
                x_reduce[mask] = 0  # 保留第一个维度的最大值,其余=0,按照源码transfrom的结果执行此操作 -- 原理暂不清楚
            
            elif self.method == "FA":        
                Ih = np.eye(len(self.tool.components_))
                x_transformed = x - torch.from_numpy(self.tool.mean_).to(x.device)
                # FA中的Wpsi*Wpsi.T不等于单位阵,所以后续recover阶段需要使用pinv求解Wpsi.T矩阵的伪逆
                self.tool.Wpsi = self.tool.components_ / self.tool.noise_variance_
                self.tool.cov_z_inv = Ih + np.dot(self.tool.Wpsi, self.tool.components_.T)
                cov_z = torch.from_numpy(scipy.linalg.inv(self.tool.cov_z_inv)).to(x.device)
                tmp = torch.matmul(x_transformed,torch.from_numpy(self.tool.Wpsi.T).to(x.device))
                x_reduce = torch.matmul(tmp,cov_z).float()   
            
            elif self.method == "NMF":
                input_x = x_.cpu().numpy()
                self.min_temp = np.min(input_x,axis=0)
                # input_x = input_x - self.min_temp  # input_x - self.min_temp <==> x - torch.min(x,dim=0)
                x_reduce = torch.matmul(x-torch.min(x,dim=0), torch.from_numpy(scipy.linalg.pinv(self.tool.components_)).to(torch.float32).to(x.device))
                #x_reduce = torch.from_numpy(self.tool.transform(input_x.astype(np.float64))).to(torch.float32).to(x.device)
                
        else:
            if self.method == "PCA":
                self.tool = PCA(n_components=self.n_componment) 
                self.tool.fit(x_.cpu().numpy()) # 刷新当前类的pca_tool
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
                self.tool.fit(input_x)  # 这里构造input_x主要是为了fit模型,只支持numpy数组类型
                x_reduce = torch.matmul(x-torch.min(x,dim=0), torch.from_numpy(scipy.linalg.pinv(self.tool.components_)).to(torch.float32).to(x.device))
                #x_reduce = torch.from_numpy(self.tool.transform(input_x.astype(np.float64))).to(torch.float32).to(x.device) 
                
        return x_reduce.reshape(batch_size,1,-1,3)
     
    def recover_matrix(self,matrix):
        # input: matrix:[reduce_dim,reduce_dim]
        # output: recover_adj:[init_dim,init_dim]


        if self.method == "PCA":
            x_recover = torch.matmul(matrix.float(),torch.from_numpy(self.tool.components_).float().to(matrix.device)) # pca.components_ :[pca_dim,12273]pca转换矩阵
            x_recover = torch.matmul(torch.from_numpy(self.tool.components_.T),x_recover) + 0.5 * torch.from_numpy(self.tool.mean_).float().to(matrix.device)
            x_recover = x_recover.T + + 0.5*torch.from_numpy(self.tool.mean_).float().to(matrix.device)

        
        # elif self.method == "KPCA":
        #     K = self.tool._get_kernel(matrix.cpu().numpy(), self.tool.X_transformed_fit_)
        #     x_recover = 2.0 * torch.from_numpy(np.dot(K, self.tool.dual_coef_)).to(torch.float32).to(matrix.device).T

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
            #x_recover = (x_recover + 0.5 * torch.from_numpy(self.tool.mean_).float().to(matrix.device)).T

            x_recover = torch.matmul(x_recover,torch.from_numpy(self.tool.cov_z_inv).float().to(matrix.device))
            x_recover = torch.matmul(x_recover,inv_w) + 0.5 * torch.from_numpy(self.tool.mean_).float().to(matrix.device)
            x_recover = x_recover.T + 0.5 * torch.from_numpy(self.tool.mean_).float().to(matrix.device)

        elif self.method == "NMF":
            #x_recover = torch.log(torch.matmul(x.float(),torch.from_numpy(self.tool.components_).float().to(x.device))).T            
            x_recover = torch.matmul(matrix.float(),torch.from_numpy(self.tool.components_).float().to(matrix.device)) 
            x_recover = torch.matmul(torch.from_numpy(self.tool.components_.T).float().to(matrix.device),x_recover) + 0.5 * torch.from_numpy(self.min_temp).float()
            x_recover = x_recover.T + 0.5 * torch.from_numpy(self.min_temp).float()

        return x_recover.reshape(self.numverts,self.numverts)
    
    def dim_recover(self,x):
        # x:[batchsize,3*3]
   
        batch_size = x.shape[0]
        x = x.reshape(x.shape[0]*3,-1)
        x_ = x.detach().clone() # 从计算图剥离并拷贝

        if self.method == "PCA":
            x_recover = (torch.matmul(x.float(),torch.from_numpy(self.tool.components_).float().to(x.device)) + torch.from_numpy(self.tool.mean_).float().to(x.device)).T # pca.components_ :[pca_dim,12273]pca转换矩阵
            #x_recover = self.tool.inverse_transform(x).T
        
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
            #x_recover = torch.log(torch.matmul(x.float(),torch.from_numpy(self.tool.components_).float().to(x.device))).T            
            x_recover = (torch.matmul(x.float(),torch.from_numpy(self.tool.components_).float().to(x.device)) + torch.from_numpy(self.min_temp).float()).T            
        return x_recover.reshape(batch_size,-1,3)

class reduction_tool():
    # 0414_reduction_plus1模型基于这个代码
    # 对原始布料模型的顶点降维&还原 -- 针对顶点的降维
    # 两种思路: 每个动作帧都执行一次PCA降维；2)只对初始模型进行PCA降维,后续降维直接使用初始的PCA变换矩阵计算

    def __init__(self,init_pos):
        self.method = "PCA"
        self.init_pos = init_pos.T #[3,numverts] -> 对顶点降维

    def PCA_process(self):
        # 首次PCA降维操作: 
        self.pca_tool = PCA(n_components=3) 
        self.pca_tool.fit(self.init_pos)
        #new = self.pca_tool.fit_transform(self.init_pos)
        print("PCA_compoments:{},dim = {}.".format(np.sum(self.pca_tool.explained_variance_ratio_),3))

    def dim_reduction(self,x):
        
        x = x.reshape(x.shape[0]*3,-1) #[batchsize*3,numverts]
        x_reduce = torch.matmul(x - torch.from_numpy(self.pca_tool.mean_).to(x.device), torch.from_numpy(self.pca_tool.components_.T).to(x.device)).to(torch.float32) 
        return x_reduce.reshape(x.shape[0],1,-1,3)
    
    def dim_recover(self,x):
        # x:[batchsize,3*3]
        x_recover = torch.matmul(x.reshape(x.shape[0]*3,-1).float(),torch.from_numpy(self.pca_tool.components_).float().to(x.device)) + torch.from_numpy(self.pca_tool.mean_).float().to(x.device) # pca.components_ :[pca_dim,12273]pca转换矩阵  
        # print(x_recover.shape)  # [15,12273]
        return x_recover.reshape(x.shape[0],-1,3)
    
    def adj_trans(self,adj_matrix):
        # 原始邻接矩阵降维
        # 如果是3*3的话直接指定全连通即可
        adj_reduce = self.pca_tool.transform(self.pca_tool.transform(adj_matrix).T)
        return adj_reduce

class cluster_data(Dataset):
    def __init__(self,state,id_list,length=429,path="VirtualBoneDataset/dress02/HFdata",cluster_id=0):

        self.pose = []
        self.trans = []
        self.cluster_id = cluster_id
        self.length = length # 类内最大成员数
        self.id_list = id_list  # 重排之后前i-1类成员的个数
        self.cluster_res = []

        for i in os.listdir(path)[:]:
            # path指动作序列对应的path
            if "npz" in i :
                tmp = np.load(os.path.join(path,i),allow_pickle=True)
                if (tmp["pose"].shape != (500,52,3))or(tmp["trans"].shape != (500,3))or(tmp["sim_res"].shape!=(500,12273,3)) :
                    continue
                final = torch.zeros(500,self.length,3)
                #tmp2 = np.load(os.path.join("VirtualBoneDataset/dress02/affinity_cosine",i),allow_pickle=True)
                tmp2 = np.load(os.path.join("VirtualBoneDataset/dress02/spectral_cosine",i),allow_pickle=True)

                self.pose.append(torch.from_numpy(tmp["pose"]))
                self.trans.append(torch.from_numpy(tmp["trans"]))

                # 按聚类个数分批次的加载数据集: id表示第i个类别
                # print("{}-th cluster loading...with {}.".format(cluster_id,self.id_list[cluster_id+1]-self.id_list[cluster_id]))
                # 统一不同类的输入数据规模: 使用最大类内成员数作为统一维度,不足用0填充
                
                final[:,:(self.id_list[cluster_id+1]-self.id_list[cluster_id]),:] = torch.from_numpy(tmp2["spectral_res"][:,self.id_list[cluster_id]:self.id_list[cluster_id+1],:])
                #final[:,:(self.id_list[cluster_id+1]-self.id_list[cluster_id]),:] = torch.from_numpy(tmp2["affinity_res"][:,self.id_list[cluster_id]:self.id_list[cluster_id+1],:])
                self.cluster_res.append(final)

        self.len = len(self.pose)

        for i in range(self.len):
            self.pose[i] = (self.pose[i] - state["pose_mean"]) / state["pose_std"]
            self.trans[i] = (self.trans[i] - self.trans[0] - state["trans_mean"]) / state["trans_std"]   
        # print(self.len) # 80个有效数据: 测试集5;训练集75

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

        # 标准化处理
        self.len = len(self.pose)
        for i in range(self.len):
            self.pose[i] = (self.pose[i] - state["pose_mean"]) / state["pose_std"]
            # self.trans[i] = (self.trans[i] - self.trans[0] - state["trans_mean"]) / state["trans_std"]   
            self.trans[i] = (self.trans[i] - self.trans[i][0] - state["trans_mean"]) / state["trans_std"]   
        # print(self.len) # 80个有效数据: 测试集5;训练集75

    def __getitem__(self,idx):
        if self.mode == "HF": 
            return self.pose[idx],self.trans[idx],self.sim_res[idx]
        if self.mode == "LF": 
            #return self.pose[idx],self.trans[idx],self.sim_res[idx]
            return self.pose[idx],self.trans[idx],self.sim_res[idx],self.dpos[idx]

    def __len__(self):
        return self.len

# def find_nearest(pred_pos):
#     """
#     输入预测布料顶点坐标
#     返回与每个顶点距离最近的人体顶点坐标和法线 - 注意是人体坐标非虚拟骨骼坐标
#     """
#     body_pos = np.load() # 加载人体坐标数据

#     return bpos,bnormal
def get_full_edge_index(numverts=1,mode="empty"):
    # 根据顶点数量创建全孤立&全连通的邻接矩阵  -- 返回稀疏格式[2,edge_num]
    from torch_geometric.utils import dense_to_sparse    
    if mode == "empty":
        adj = torch.zeros((numverts,numverts))  # 使用全连接导致最终输出无法区分不同维度的差别，所以由全连通改为全不连通
    elif mode == "full":
        # 全连通邻接矩阵
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
        # 这个loss直接算回归问题有点勉强 -- 后续看怎么处理
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
    # 支持torch_cuda
    # x,y : []
    # KL散度支持高维分布计算, logp_y是被指导的,需要取对数
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

    # 测试下KL_distance
    x1 = torch.ones((9,12273,3))
    x2 = torch.ones((9,12273,3))

    res = KL_distance(x1,x2)
    print(res.item())