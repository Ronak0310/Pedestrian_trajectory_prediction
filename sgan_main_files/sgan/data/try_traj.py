import sys
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from trajectories import TrajectoryDataset, seq_collate
import matplotlib.pyplot as plt
import numpy as np
import math
from attrdict import AttrDict

# sys.path.append('../data')
from loader import data_loader
sys.path.append('../')
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path
from collections import defaultdict




def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    # generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

data_dir = r'E:\thesis_files_IVI\trajectory_prediction\Git_repos\sgan\datasets\zara1\test'
obs_len = 8
pred_len = 12
skip = 1
num_peds_in_seq = []
seq_list = []
seq_list_rel = []
loss_mask_list = []
non_linear_ped = []
seq_len = obs_len + pred_len
data = read_file(data_dir+'\\'+'crowds_zara01.txt')
# data = read_file(data_dir+'\\'+'new_dataset.txt')
 
frames = np.unique(data[:, 0]).tolist()
frame_data = []
counter=0
for frame in frames:
    frame_data.append(data[frame == data[:, 0], :])
    num_sequences = int(
    math.ceil((len(frames) - seq_len + 1) / skip))
    

for idx in range(0, num_sequences * skip + 1, skip):
    curr_seq_data = np.concatenate(
        frame_data[idx:idx + seq_len], axis=0)
    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
    curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                seq_len))
    curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len))
    curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                seq_len))
    num_peds_considered = 0
    _non_linear_ped = []
    for _, ped_id in enumerate(peds_in_curr_seq):
        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                        ped_id, :]
        
        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
        if pad_end - pad_front != seq_len:
            continue
        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
        curr_ped_seq = curr_ped_seq
        
        # Make coordinates relative
        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
        rel_curr_ped_seq[:, 1:] = \
            curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
        _idx = num_peds_considered
        try:
            curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
        except:
            continue
        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
        num_peds_considered += 1
    
    if num_peds_considered > 1:
        non_linear_ped += _non_linear_ped
        num_peds_in_seq.append(num_peds_considered)
        loss_mask_list.append(curr_loss_mask[:num_peds_considered])
        seq_list.append(curr_seq[:num_peds_considered])
        seq_list_rel.append(curr_seq_rel[:num_peds_considered])

num_seq = len(seq_list)
seq_list = np.concatenate(seq_list, axis=0)
seq_list_rel = np.concatenate(seq_list_rel, axis=0)

# Convert numpy -> Torch Tensor
obs_traj = torch.from_numpy(
    seq_list[:, :, :obs_len]).type(torch.float)
pred_traj = torch.from_numpy(
    seq_list[:, :, obs_len:]).type(torch.float)
obs_traj_rel = torch.from_numpy(
    seq_list_rel[:, :, :obs_len]).type(torch.float)
pred_traj_rel = torch.from_numpy(
    seq_list_rel[:, :, obs_len:]).type(torch.float)
    
cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
seq_start_end = [
    (start, end)
    for start, end in zip(cum_start_idx, cum_start_idx[1:])
]

index = 0
start, end = seq_start_end[index]
seq_start_end = torch.LongTensor(seq_start_end)

# out = [
#         obs_traj[start:end, :], pred_traj[start:end, :],
#         obs_traj_rel[start:end, :], pred_traj_rel[start:end, :]
#         ]

# traj_obs, traj_pred_gt, traj_obs_rel, traj_pred_gt_rel = out

# traj_obs = traj_obs.permute(2, 0, 1).cuda()
# traj_obs_rel = traj_obs_rel.permute(2,0,1).cuda()
# traj_pred_gt = traj_pred_gt.permute(2,0,1).cuda()
# traj_pred_gt_rel = traj_pred_gt_rel.permute(2,0,1).cuda()

# ****************************** trial code from dictionary values (below) ***********************************************************
# ********************************* custom dataset- processing ***********************************************
dc = defaultdict(list)
with open(r'E:\thesis_files_IVI\trajectory_prediction\Git_repos\sgan\new_dataset.txt', 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        line = [float(i) for i in line]
        # print(line)
        dc[line[1]].append([line[2], line[3]])
# ********************************* end of custom data processing ********************************************
import random
tracker = random.choice(list(dc.keys()))
if len(dc[tracker])>20:
    tracker = tracker
else:
    tracker = 1
p1 = np.array(dc[tracker][0:20]).T
# p2 = np.array(dc[2][0:20]).T
# p3 = np.array(dc[3][0:20]).T

# Code for obs trajectory
obs1 = np.array([p1[0][0:8], p1[1][0:8]]).astype(np.float32)
# obs2 = np.array([p2[0][0:8], p2[1][0:8]]).astype(np.float32)
# obs3 = np.array([p3[0][0:8], p3[1][0:8]]).astype(np.float32)

p1_ = torch.tensor(obs1).view((1,2,8))
# p2_ = torch.tensor(obs2).view((1,2,8))
# p3_ = torch.tensor(obs3).view((1,2,8))
# ls = torch.cat((p1_,p2_,p3_),0)
# obs_traj_try = ls.permute(2,0,1).cuda()

## only one object
obs_traj_try = p1_.permute(2,0,1).cuda()

# Code for relative obs trajectory
rel_1 = np.zeros(p1.shape)
# rel_2 = np.zeros(p1.shape)
# rel_3 = np.zeros(p1.shape)

rel_1[:, 1:] = p1[:, 1:] - p1[:, :-1]
# rel_2[:, 1:] = p2[:, 1:] - p2[:, :-1]
# rel_3[:, 1:] = p3[:, 1:] - p3[:, :-1]

obs_rel_1 = np.array([rel_1[0][0:8], rel_1[1][0:8]]).astype(np.float32)
# obs_rel_2 = np.array([rel_2[0][0:8], rel_2[1][0:8]]).astype(np.float32)
# obs_rel_3 = np.array([rel_3[0][0:8], rel_3[1][0:8]]).astype(np.float32)

obs_rel_1_ = torch.tensor(obs_rel_1).view((1,2,8))
# obs_rel_2_ = torch.tensor(obs_rel_2).view((1,2,8))
# obs_rel_3_ = torch.tensor(obs_rel_3).view((1,2,8))

# ls_rel = torch.cat((obs_rel_1_,obs_rel_2_,obs_rel_3_),0)
# obs_rel_traj_try = ls_rel.permute(2,0,1).cuda()

## only one object
obs_rel_traj_try = obs_rel_1_.permute(2,0,1).cuda()

# code for ground truth trajectory
gt1 = np.array([p1[0][8:], p1[1][8:]]).astype(np.float32)
gt1_ = torch.tensor(gt1).view((1,2,12))
gt1_try = gt1_.permute(2,0,1).cuda()

gt_rel_1 = np.array([rel_1[0][8:], rel_1[1][8:]]).astype(np.float32)
gt_rel_1_ = torch.tensor(gt_rel_1).view((1,2,12))
gt_rel_1_try = gt_rel_1_.permute(2,0,1).cuda()

# *********************************************************** end trial code *********************************************************

checkpoint = torch.load(r"E:\thesis_files_IVI\trajectory_prediction\Git_repos\sgan\models\sgan-models\eth_12_model.pt")
generator = get_generator(checkpoint)
_args = AttrDict(checkpoint['args'])
# print(traj_obs.shape)
# pred_traj_fake_rel = generator(
#                     traj_obs.cuda(), traj_obs_rel.cuda(), (start,end)
#                 )
# pred_traj_fake = relative_to_abs(
#                     pred_traj_fake_rel, traj_obs[-1]
#                 )


# gt = traj_pred_gt[:,2,:].data
# input_trj = traj_obs[:,2,:].data
# out_trj = pred_traj_fake[:,2,:].data


# gt = gt.cpu().detach().numpy()
# input_trj = input_trj.cpu().detach().numpy()
# out_trj = out_trj.cpu().detach().numpy()
# print("Ground truth",gt)
# print("Predicted", out_trj)

# # plot trajectories
# plt.figure(figsize=(10,8))
# plt.scatter(gt[:,0], gt[:,1], c='green')e:
# plt.scatter(input_trj[:,0], input_trj[:,1], c='gray')
# plt.scatter(out_trj[:,0], out_trj[:,1], c='red')
# plt.legend(['ground_truth', 'obs_trj', 'predicted_trj'])
# plt.show()

# ****************************** trial code from dictionary for visualization (below) ********************************

print(obs_traj_try)
pred_traj_fake_rel = generator(
                    obs_traj_try.cuda(), obs_rel_traj_try.cuda(), (start,end)
                )

pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj_try[-1]
                )

print(pred_traj_fake)

input_trj = obs_traj_try[:,0,:].data
out_trj = pred_traj_fake[:,0,:].data
gt_traj = gt1_try[:,0,:].data
input_trj = input_trj.cpu().detach().numpy()
out_trj = out_trj.cpu().detach().numpy()
gt_traj = gt_traj.cpu().detach().numpy()
plt.figure(figsize=(10,8))
plt.scatter(input_trj[:,0], input_trj[:,1], c='gray')
plt.scatter(out_trj[:,0], out_trj[:,1], c='red')
plt.scatter(gt_traj[:,0], gt_traj[:,1], c='green')
plt.legend(['Observed','Predicted','Ground-truth'])
plt.show()

# ****************************** end trial code ******************************















# dset = TrajectoryDataset(
#     data_dir=data_dir,
#     obs_len=8,
#     pred_len=12,
#     skip=1,
#     delim='\t')
