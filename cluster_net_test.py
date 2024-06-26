import copy
import os
import random
from tqdm import tqdm
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import json

import torch
# from torch.utils.cpp_extension import *
from torch.utils.data import DataLoader, Dataset
from torch_geometric.transforms import FaceToEdge
from torch_geometric.utils import get_laplacian,to_dense_adj,dense_to_sparse

from Mytools import *
from src.obj_parser import Mesh_obj
from src.SSDRLBS import SSDRLBS
from src.models import *


def get_results(config_path, anim_path, out_path, device="cpu"):
    with open(config_path, "r") as f:
        config = json.load(f)

    gru_dim = config["gru_dim"]
    gru_out_dim = config["gru_out_dim"]
    joint_list = config["joint_list"]

    ssdrlbs_bone_num = config["ssdrlbs_bone_num"]
    ssdrlbs_root_dir = config["ssdrlbs_root_dir"]

    ssdrlbs_net_path = config["ssdrlbs_net_path1"]
    detail_net_path = []

    state_path = config["state_path"]

    obj_template_path = config["obj_template_path"]
    garment_template = Mesh_obj(obj_template_path)

    joint_num = len(joint_list)

    cluster_affinity_path = config["affinity_message_path"]
    cluster_adj_path = config["affinity_adj_path"]
    cluster_affinity = np.load(cluster_affinity_path,allow_pickle=True)
    cluster_adj = torch.from_numpy(np.load(cluster_adj_path))
    affinity_max = cluster_affinity["dims"][0]
    affinity_id_list = torch.from_numpy(cluster_affinity["index"])
    affinity_order = torch.from_numpy(cluster_affinity["order"])

    ssdr_model = GRU_Model((joint_num + 1) * 3, gru_dim, [ssdrlbs_bone_num * 6]) 
    ssdr_model.load_state_dict(torch.load(ssdrlbs_net_path,map_location=torch.device('cuda:0')))
    ssdr_model = ssdr_model.to(device)
    ssdr_model.eval()

    data = FaceToEdge()(Data(num_nodes=garment_template.v.shape[0],
                             face=torch.from_numpy(garment_template.f.astype(int).transpose() - 1).long()))
    detail_model = GRU_GNN_Model(6 * ssdrlbs_bone_num, gru_dim, [gru_out_dim * affinity_max],
                                 gru_out_dim, [3,8,16],edge_index=None).to(device)
    

    ssdrlbs = SSDRLBS(os.path.join(ssdrlbs_root_dir, "u.obj"),
                      os.path.join(ssdrlbs_root_dir, "skin_weights.npy"),
                      os.path.join(ssdrlbs_root_dir, "u_trans.npy"),
                      device)

    state = np.load(state_path)
    cloth_pose_mean = torch.from_numpy(state["cloth_pose_mean"]).to(device)
    cloth_pose_std = torch.from_numpy(state["cloth_pose_std"]).to(device)
    cloth_trans_mean = torch.from_numpy(state["cloth_trans_mean"]).to(device)
    cloth_trans_std = torch.from_numpy(state["cloth_trans_std"]).to(device)

    detail_net_path_save = config["detail_net_path_save"]
    ssdr_res_mean = torch.from_numpy(state["ssdr_res_mean"])
    ssdr_res_std = torch.from_numpy(state["ssdr_res_std"])

    vert_std = state["sim_res_std"]
    vert_mean = state["sim_res_mean"]

    anim = np.load(anim_path)
    pose_arr = (anim["pose"] - state["pose_mean"]) / state["pose_std"]
    trans_arr = (anim["trans"] - anim["trans"][0] - state["trans_mean"]) / state["trans_std"]
    item_length = pose_arr.shape[0]

    ssdr_hidden = None
    hidden_list = [None,None,None,None,None,None,None,None,None]
    
    res_list = []

    with torch.no_grad():
        for frame in tqdm(range(item_length)[:100]):

            motion_signature = np.zeros(((len(joint_list) + 1) * 3), dtype=np.float32)
            for j in range(len(joint_list)):
                motion_signature[j * 3: j * 3 + 3] = pose_arr[frame, joint_list[j]]
            motion_signature[len(joint_list) * 3:] = trans_arr[frame]

            motion_signature = torch.from_numpy(motion_signature)
            motion_signature = motion_signature.view((1, -1)).to(device)

            pred_rot_trans, new_ssdr_hidden = ssdr_model(motion_signature, ssdr_hidden)
            ssdr_hidden = new_ssdr_hidden

            pred_pose = pred_rot_trans.view((-1, 6))[:, 0:3] * cloth_pose_std + \
                        cloth_pose_mean
            pred_trans = pred_rot_trans.view((-1, 6))[:, 3:6] * cloth_trans_std + \
                         cloth_trans_mean

            ssdr_res = ssdrlbs.batch_pose(pred_trans.reshape((1, 1, pred_trans.shape[0], pred_trans.shape[1])),
                                          torch.deg2rad(pred_pose).reshape(
                                              (1, 1, pred_pose.shape[0], pred_pose.shape[1])))

            
            ssdr_cluster = torch.zeros(1,affinity_max,3).to(device)
            ssdr_res = ssdr_res.reshape(1,-1,3)

            for cluster_id in range(9):
                path = os.path.join(detail_net_path_save,"spectral/Cluster_cos_{}_0413.pth.tar".format(cluster_id))
                cluster_edge_index = dense_to_sparse(cluster_adj[cluster_id])[0].to(device)   
                detail_model.load_state_dict(torch.load(path,map_location=device))
                detail_model.edge_index = cluster_edge_index
                detail_model.eval()
      
                ssdr_cluster[:,:affinity_id_list[cluster_id+1]-affinity_id_list[cluster_id],:] = ssdr_res[:,affinity_order,:][:,affinity_id_list[cluster_id]:affinity_id_list[cluster_id+1],:]
                
                detail_res, new_detail_hidden = detail_model(pred_rot_trans, ssdr_cluster, hidden_list[cluster_id])
                
                hidden_list[cluster_id] = new_detail_hidden
                cluster_std = torch.zeros((affinity_max,3))
                cluster_mean = torch.zeros((affinity_max,3))
                cluster_std[:affinity_id_list[cluster_id+1]-affinity_id_list[cluster_id],:] = ssdr_res_std[affinity_order,:][affinity_id_list[cluster_id]:affinity_id_list[cluster_id+1],:]
                cluster_mean[:affinity_id_list[cluster_id+1]-affinity_id_list[cluster_id],:] = ssdr_res_mean[affinity_order,:][affinity_id_list[cluster_id]:affinity_id_list[cluster_id+1],:]
              

                final_res = ssdr_cluster.detach().cpu().numpy().reshape((-1, 3)) + \
                        (detail_res.detach().cpu().numpy().reshape((-1, 3)) * cluster_std.cpu().numpy() + cluster_mean.cpu().numpy())
                res_list.append(final_res[:(affinity_id_list[cluster_id+1]-affinity_id_list[cluster_id]),:])
            out_obj = copy.deepcopy(garment_template)

            out_obj.v[affinity_order,:] = np.concatenate([res_list[0],res_list[1],res_list[2],res_list[3],res_list[4],res_list[5],\
                                        res_list[6],res_list[7],res_list[8]],axis=0)
            out_obj.write(os.path.join(out_path, "cluster_{}.obj".format(frame)))


if __name__ == "__main__":
    config_path = "assets/dress02/config.json"
    anim_path = "./VirtualBoneDataset/dress02/HFdata/84.npz"
    out_path = "out_LF"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    device = "cuda:0"  # 
    get_results(config_path, anim_path, out_path, device)
