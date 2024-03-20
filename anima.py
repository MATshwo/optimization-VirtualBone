# Total: 本模块用于结果数据可视化
#第三方库导入 
#region
# from ursina import *
from src.obj_parser import Mesh_obj
import numpy as np
from tqdm import tqdm
import copy

# import meshio
from torch_geometric.utils import get_laplacian,to_dense_adj
import torch
from torch.utils.data import Dataset,DataLoader
from torch_geometric.transforms import FaceToEdge
from torch_geometric.data import Data
import os
import torch_geometric
from scipy.spatial.transform import Rotation

import matplotlib
# 添加这个,同样是为了防止内存爆炸
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#endregion


# region 0314Loss修改
# region 顶点坐标随动画帧变动可视化:500帧动作下,12273个顶点的三个坐标分量变化折线图,每张图表示一个顶点。
x1 = np.load("./VirtualBoneDataset/dress02/HFdata/0.npz",allow_pickle=True)
x2 = np.load("./VirtualBoneDataset/dress02/HFdata/1.npz",allow_pickle=True)

# 绘制曲线图并保存到本地
path = "./photo/"
if not os.path.exists(path):
    os.makedirs(path)
for i in range(12273)[:]:
    plt.figure()
    # plt.xlim(x1, x2)
    plt.ylim(-1.0, 1.5)
    plt.plot(x1["sim_res"][:,i,0],label="X")
    plt.plot(x1["sim_res"][:,i,1],label="Y")
    plt.plot(x1["sim_res"][:,i,2],label="Z")
    plt.legend()
    name = "point_index = " + str(i)
    plt.title(name)
    figname = path+'pos_motion0_'+str(i)+'.png'
    plt.savefig(figname,bbox_inches='tight')   
    # 及时清除,不然内存会爆掉
    plt.clf()
    plt.close()

# 加载本地png生成动画文件
path = r"./photo/pos_motion0_"
fps = 24  # 可以随意调整视频的帧速率

#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('TestVideo.mp4',fourcc,fps,(559,433),True)#最后一个是保存图片的尺寸

for i in range(12273):
    frame = cv2.imread(path+str(i)+'.png')
    #print(frame.shape)
    cv2.imshow('frame',frame)
    videoWriter.write(frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
            break
videoWriter.release()
cv2.destroyAllWindows()

# endregion




# def Loss_func(pos,pred_pos,mode="L1",bool_col = 0):
#     """
#     定义反向传播的loss函数：
#     1. Loss_LF() = 
#     2. Loss_HF() = 
#     3. Loss_collision() =
#     """
#     # bool_col:是否使用碰撞loss,暂不使用,没有找到合适的人体模型和检测算法
    
#     laplace_loss = 1.0
#     lambda_col = 0.1
#     de = torch.tensor(0.0001) # barrier between cloth and body,原文取值=0,但感觉太极限了,可以留一段距离！
#     loss_lf,loss_hf,losscol = 0.0,0.0,0.0

#     if mode == "Lf":
#         # 1. 低频loss
#         # torch.mean(torch.norm(y-y_pred,p=2,dim=1))
#         loss_lf = torch.mean(torch.norm(pos-pred_pos,p=2,dim=2))

#     if mode == "Hf":
#         # 2. 高频loss
#         loss_hf = torch.mean(torch.norm(pos-pred_pos,p=2,dim=2))
#         #loss_hf = loss_hf + lambda_laplace*torch.mean(torch.norm(Laplace(pos)-Laplace(pred_pos),p=2,dim=1))

#     # 3. 碰撞loss
#     if bool_col:
        
#         # bpos,bnormal = find_nearest(pred_pos)
#         # tmp = torch.matmul(b_normal,torch.transpose(pred_pos-bpos,0,1))
#         # tmp = torch.diag(tmp).reshape(1,-1) # [N,1]
#         # losscol = lambda_col * torch.mean(de-torch.min(tmp,de))
#         losscol = 0.0
#     return loss_lf+loss_hf+losscol
    
#endregion


# 数据加载
#region 
pred_obj_path = "./out/"
dataset_path = "./VirtualBoneDataset/dress02/HFdata/"
picture_path = "./out/"
# anima_path = "./"
obj_template_path = "assets/dress02/garment.obj"
garment_template = Mesh_obj(obj_template_path)
out_path = "./True_obj_anima0"
state_path =  "assets/dress02/state.npz"
#endregion

# 0308：解决数据集与网络输出的对齐问题以及obj结果初步可视化操作
#region
"""1. 先验证下真实数据集的布料位置"""
#region
#if not os.path.exists(out_path):
#    os.makedirs(out_path)
#data_anima0 = np.load(os.path.join(dataset_path,"0.npz"))
#for frame in tqdm(range(500)):
#    out_obj = copy.deepcopy(garment_template)
#    # 模型其他信息不变，但需要更改顶点坐标，得到预测后的模型
#    out_obj.v = data_anima0["sim_res"][frame]
#    out_obj.write(os.path.join(out_path, "{}.obj".format(frame)))
"""经测试:仿真数据的pos是结合动作信息的全局坐标,即需要添加run_motion.py:149-161行,145行得到的布料还原动画足够,149-161是根据人体移动
另,146行*ssdr_res_std + ssdr_res_mean必须要有,否则无法还原布料模型"""
#endregion

"""2. obj如何导出截图,生成动画更困难点,先从截图开始,尝试使用pytorch3D库"""
#region
# import pytorch3d
#endregion

"""3. 真实数据与高频输出对齐处理,得到HF网络输出的标签用于计算loss"""
#region
#注: 使用自定义数据集必须有
#   1.K组动作序列-[帧数,poses:{52,3},trans:{3}];
#   2.布料原始模型;
#   3.布料仿真结果K组:[帧数,顶点数,3];
#   4.K组动作序列的pose均值方差,trans均值方差,以及对应布料仿真数据的均值方差;
#   5.SSDR蒙皮分解得到虚拟骨骼的pose和trans均值方差;
#   6.预训练LF后得到的低频输出"K_train"组布料仿真数据的均值和方差
#   7. ...
#state = np.load(state_path)
#for i in os.listdir(dataset_path)[:]:
#    if "npz" in i:
#        tmp = np.load(os.path.join(dataset_path,i),allow_pickle=True)
#        if (tmp["pose"].shape != (500,52,3))or(tmp["trans"].shape != (500,3))or(tmp["sim_res"].shape!=(500,12273,3)) :
#            continue
#        pose_arr = (tmp["pose"] - state["pose_mean"]) / state["pose_std"]
#        trans_arr = (tmp["trans"] - tmp["trans"][0] - state["trans_mean"]) / state["trans_std"]
#        result = np.zeros_like(tmp["sim_res"])
#        for frame in range(500)[:]:
#            pose = pose_arr[frame] * state["pose_std"] + state["pose_mean"]
#            trans = trans_arr[frame] * state["trans_std"] + state["trans_mean"]
#            trans_off = np.array([0,-2.1519510746002397,90.4766845703125]) / 100.0
#            trans += trans_off
#            final_res_ori = (tmp["sim_res"][frame] - trans).transpose() # 先转置
#            t = (Rotation.from_rotvec(pose[0]).as_matrix()).transpose() # 旋转矩阵转置即为逆矩阵
#            # final_res_ori = torch.from_numpy(np.matmul(t,final_res_ori).transpose())
#            result[frame,:,:] = np.matmul(t,final_res_ori).transpose()
#            # 经测试,final_res_ori与HF+LF网络输出对齐,处理无误.
#            #out_obj = copy.deepcopy(garment_template) 
#            #out_obj.v = result[frame,:,:] 
#            #out_obj.write(os.path.join(out_path, "{}.obj".format("Testing")))
#        np.savez(r"D:\ProgramTotal\ProgramDemo\VS2019\VirtualBone0302\VirtualBone0302\VirtualBoneDataset\dress02\HF_res/"+i,final_ground=result)
#endregion
#endregion





# 0307 Laplacian平滑计算
#region 
def Smoothing_pos(edge_index,pos_x,normalization='sym'):
    """网格的Laplacian平滑"""
    # pos_x:[batchsize,numvert,3]
    # 1.计算modified laplacian平滑矩阵L
    indexs,weights = get_laplacian(edge_index,normalization=normalization)
    # 2.根据L对原始网格坐标进行平滑
    dpos = torch.zeros_like(pos_x)
    res = torch.mul(weights.reshape(1,-1,1),pos_x[:,indexs[0]])
    #for i in range(res.shape[1]):
    #    dpos[:,indexs[1,i]] = dpos[:,indexs[1,i]] + res[:,i]
    for i in range(pos_x.shape[1]):
        # 上一个循环遍历所有边,效率很低,修改为遍历顶点. [1,0,2,1]==[0,0,0,0]=[FTFF],然后sum(res[F,T,F,F])->dpos[0]
        # 经验证,两种方法返回结果一致
        tmpp = res[:,indexs[1]==(torch.zeros_like(indexs[1])+i)]
        dpos[:,i] = torch.sum(tmpp,dim=1)
    return dpos
#endregion
# 0307 为了解决速度问题:需要重新生成一个低频网络训练集：LR_i.npz文件["pose","trans","dpos","ddpos"]
#region 还是未解决:涉及计算laplacian时间成本太高了,需要考虑替代方案！
#garment_template = Mesh_obj("assets/dress02/garment.obj")
#data = FaceToEdge()(Data(num_nodes=garment_template.v.shape[0],face=torch.from_numpy(garment_template.f.astype(int).transpose() - 1).long()))
#edge_index = data.edge_index
#for i in os.listdir("./VirtualBoneDataset/dress02/HFdata/")[:]:
#    if "npz" in i:
#        tmp = np.load(os.path.join("./VirtualBoneDataset/dress02/HFdata/",i),allow_pickle=True)
#        name = "LF_"+ i
#        dpos = []
#        ddpos = []
#        if (tmp["pose"].shape != (500,52,3))or(tmp["trans"].shape != (500,3))or(tmp["sim_res"].shape!=(500,12273,3)) :
#            continue
#        for j in range(tmp["sim_res"].shape[0]):
#            dpos.append(Smoothing_pos(edge_index,torch.from_numpy(tmp["sim_res"][j]).reshape(1,12273,3)).numpy())
#            ddpos.append(Smoothing_pos(edge_index,torch.from_numpy(dpos[j])).numpy())
#        np.savez(os.path.join("./VirtualBoneDataset/dress02/LFdata/",name),pose=tmp["pose"],trans=tmp["trans"],dpos=np.array(dpos),ddpos=np.array(ddpos))
#endregion

# 尝试用ursina库播放obj文件:通过刷新顶点坐标;结果失败,ursina导入模型后不支持逐顶点更新模型动作,只能把模型按照顶点列表的形式导入,之后尝试下！
#region
# 使用ursina制作obj动画文件
# 根据预训练模型的结果提取坐标
#positions = []
#garment_template = Mesh_obj("assets/dress02/garment.obj")
#path = "./out/"
#for i in os.listdir(path)[:]:
#    if "obj" in i:
#        positions.append(Mesh_obj(os.path.join(path,i)).v)
#print(positions[0].shape)
#np.save("./00.npy",np.array(positions))

#positions = np.load("./00.npy")

#class world(Entity):#Entity # 定义构造方法
#    def __init__(self):
#        super().__init__()
#        s =100
#        grid = Entity(model=Grid(s, s), scale=s, color=color.color(0, 0, .8, 1), rotation_x=90,
#                      position=(-20, -0, 0))
#        #vertsx = ((0, 0, 0), (10, 0, 0))
#        #Entity(model=Mesh(vertices=vertsx, mode='line', thickness=3), color=color.cyan)
#        #vertsyz = [(0, 0, 0), (0, 5, 0), (0, 0, 0), (0, 0, 10)]
#        #self.linesyz =Entity(model=Mesh(vertices=vertsyz, mode='line', thickness=3), color=color.yellow)

##a = [1,2,3,4,5,6,7,8,9]
#n = 0
#def update():
#    global positions
#    global n
#    an1.position += positions[int(n/100)]
#    n += 1
#    #an1.y += 0.1
#    #print(int(time.dt*40))
#app = Ursina()
#world()
#l1 = PointLight(x=3,y=2,color=color.red)
#an1 = Entity(model="./out/0.obj",texture ='brick',rotation_y=0,rotation_x=-90,scale=5)
#EditorCamera()

#app.run()
#endregion
