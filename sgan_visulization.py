import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
from attrdict import AttrDict
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append('./sgan/sgan/data')
from trajectories import TrajectoryDataset, seq_collate
from loader import data_loader

sys.path.append('./sgan/')
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path
from collections import defaultdict



class Predict_trajectory():

    def __init__(self):
        self.checkpoint = torch.load(r"E:\thesis_files_IVI\Pedestrian_trajectory_prediction\sgan\models\sgan-models\eth_12_model.pt")

    def get_generator(self, checkpoint):
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


    def predict(self, dc, trk_id):
        generator = self.get_generator(self.checkpoint)
        _args = AttrDict(self.checkpoint['args'])

        width = 512
        height = 640
        # Code for obs trajectory
        p1 = np.array(dc[trk_id][:]).T
        # obs1 = np.array([p1[0][-9:-1], p1[1][-9:-1]]).astype(np.float32)
        obs1 = np.array([p1[0][:], p1[1][:]]).astype(np.float32)
        p1_ = torch.tensor(obs1).view((1,2,8))
        ## only one object
        obs_traj_try = p1_.permute(2,0,1).cuda()

        # Code for relative obs trajectory
        rel_1 = np.zeros(p1.shape)
        rel_1[:, 1:] = p1[:, 1:] - p1[:, :-1]
        obs_rel_1 = np.array([rel_1[0][0:8], rel_1[1][0:8]]).astype(np.float32)
        obs_rel_1_ = torch.tensor(obs_rel_1).view((1,2,8))
        ## only one object
        obs_rel_traj_try = obs_rel_1_.permute(2,0,1).cuda()



        num_peds_in_seq = []
        num_peds_considered = 1 #len(dc.keys())
        num_peds_in_seq.append(num_peds_considered)
        num_peds_in_seq = 1
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        index = 0
        start, end = seq_start_end[index]
        seq_start_end = torch.LongTensor(seq_start_end)



        # predict the future trajectory points
        pred_traj_fake_rel = generator(
                            obs_traj_try.cuda(), obs_rel_traj_try.cuda(), (start,end)
                        )

        pred_traj_fake = relative_to_abs(
                            pred_traj_fake_rel, obs_traj_try[-1]
                        )

        out_trj = pred_traj_fake[:,0,:].data
        out_trj = out_trj.cpu().detach().numpy()

        return out_trj