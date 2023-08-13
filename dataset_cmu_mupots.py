import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from data_utils import *


class CMUMocap(Dataset):

    def __init__(self, config, file="./data/cmu_mupots/test_3_120_mocap.npy"):

        self.config = config

        data = np.load(file, allow_pickle=True)[:, :, ::2] # from 30fps to 15fps
        
        data = data.reshape(*data.shape[:-1], 15, 3)
        
        self.data = torch.from_numpy(data).to(config.device)

        print("CMUMocap shape:", self.data.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        INPUT_LEN = self.config.input_len

        if idx < self.data.shape[0]:
            x0, y0 = self.data[idx, 0, :INPUT_LEN], self.data[idx, 0, INPUT_LEN:]
            x1, y1 = self.data[idx, 1, :INPUT_LEN], self.data[idx, 1, INPUT_LEN:]
            x2, y2 = self.data[idx, 2, :INPUT_LEN], self.data[idx, 2, INPUT_LEN:]

            sctx01 = 1
            sctx02 = 1
            sctx12 = 0 if idx < 400 else 1

        sample = {
            'keypoints0': x0.requires_grad_(False).to(self.config.device),
            'keypoints1': x1.requires_grad_(False).to(self.config.device),
            'keypoints2': x2.requires_grad_(False).to(self.config.device),
            'out_keypoints0': y0.requires_grad_(False).to(self.config.device),
            'out_keypoints1': y1.requires_grad_(False).to(self.config.device),
            'out_keypoints2': y2.requires_grad_(False).to(self.config.device),

            'sctx01': torch.from_numpy(np.array([sctx01])).to(self.config.device), # 0 - socially dependent, 1 - independent, 2 - same person
            'sctx02': torch.from_numpy(np.array([sctx02])).to(self.config.device),
            'sctx12': torch.from_numpy(np.array([sctx12])).to(self.config.device)
        }

        return sample


class Mupots(Dataset):

    def __init__(self, config, file="./data/cmu_mupots/mupots_120_3persons.npy"):

        self.config = config

        data = np.load(file, allow_pickle=True)[:, :, ::2] # from 30fps to 15fps
        
        data = data.reshape(*data.shape[:-1], 15, 3)

        self.data = torch.from_numpy(data).to(config.device)

        print("Mupots shape:", self.data.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        INPUT_LEN = self.config.input_len

        if idx < self.data.shape[0]:
            x0, y0 = self.data[idx, 0, :INPUT_LEN], self.data[idx, 0, INPUT_LEN:]
            x1, y1 = self.data[idx, 1, :INPUT_LEN], self.data[idx, 1, INPUT_LEN:]
            x2, y2 = self.data[idx, 2, :INPUT_LEN], self.data[idx, 2, INPUT_LEN:]

            sctx01 = 0
            sctx02 = 0
            sctx12 = 0

        sample = {
            'keypoints0': x0.requires_grad_(False).to(self.config.device),
            'keypoints1': x1.requires_grad_(False).to(self.config.device),
            'keypoints2': x2.requires_grad_(False).to(self.config.device),
            'out_keypoints0': y0.requires_grad_(False).to(self.config.device),
            'out_keypoints1': y1.requires_grad_(False).to(self.config.device),
            'out_keypoints2': y2.requires_grad_(False).to(self.config.device),

            'sctx01': torch.from_numpy(np.array([sctx01])).to(self.config.device), # 0 - socially dependent, 1 - independent, 2 - same person
            'sctx02': torch.from_numpy(np.array([sctx02])).to(self.config.device),
            'sctx12': torch.from_numpy(np.array([sctx12])).to(self.config.device)
        }

        return sample


class TrainCMUMocap(Dataset):

    def __init__(self, config, file="./data/cmu_mupots/train_3_120_mocap.npy"):
        self.config = config
        self.use_augmentation = True

        data = np.load(file, allow_pickle=True)[:, :, ::2] # 30 to 15 fps
        
        data = data.reshape(*data.shape[:-1], 15, 3)
        
        self.data = torch.from_numpy(data).to(config.device)
        print("TrainCMUMocap shape:", self.data.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        INPUT_LEN = self.config.input_len

        if idx < self.data.shape[0]:
            x0, y0 = self.data[idx, 0, :INPUT_LEN], self.data[idx, 0, INPUT_LEN:]
            x1, y1 = self.data[idx, 1, :INPUT_LEN], self.data[idx, 1, INPUT_LEN:]
            x2, y2 = self.data[idx, 2, :INPUT_LEN], self.data[idx, 2, INPUT_LEN:]

            sctx01 = 1
            sctx12 = 0 if idx < 3000 else 1 # based on mix_mocap.py
            sctx02 = 1

        if self.use_augmentation:
            x0, y0, x1, y1, x2, y2, sctx01, sctx02, sctx12 = self._augment(x0, y0, x1, y1, x2, y2, sctx01, sctx02, sctx12)

        sample = {
            'keypoints0': x0,
            'keypoints1': x1,
            'keypoints2': x2,
            'out_keypoints0': y0,
            'out_keypoints1': y1,
            'out_keypoints2': y2,
            # 0 - socially dependent, 1 - independent, 2 - same person
            'sctx01': torch.from_numpy(np.array([sctx01])).to(self.config.device),
            'sctx02': torch.from_numpy(np.array([sctx02])).to(self.config.device),
            'sctx12': torch.from_numpy(np.array([sctx12])).to(self.config.device),

        }

        return sample

    def _augment(self, x0, y0, x1, y1, x2, y2, sctx01, sctx02, sctx12):
        INPUT_LEN = self.config.input_len

        if np.random.rand() > 0.5:
            return x0, y0, x1, y1, x2, y2, sctx01, sctx02, sctx12

        seq0 = torch.cat((x0, y0), dim=0)
        seq1 = torch.cat((x1, y1), dim=0)
        seq2 = torch.cat((x2, y2), dim=0)

        if self.config.augment.backward_movement and np.random.rand() > 0.5: # backward movement, is this flip?? torch.flip(seq0, dim=0)
            seq0 = seq0[np.arange(-seq0.shape[0]+1, 1)]
            seq1 = seq1[np.arange(-seq1.shape[0]+1, 1)]
            seq2 = seq2[np.arange(-seq2.shape[0]+1, 1)]

        if self.config.augment.reversed_order and np.random.rand() > 0.5: # random order of sctx of people
            if np.random.rand() > 0.5:
                seq0, seq1, seq2 = seq1, seq2, seq0
                sctx01, sctx02, sctx12 = sctx12, sctx01, sctx02
            else:
                seq0, seq1, seq2 = seq1, seq0, seq2
                sctx01, sctx02, sctx12 = sctx01, sctx12, sctx02

        if self.config.augment.random_scale and np.random.rand() > 0.5: # random scale
            r1=0.1#0.8
            r2=5#1.2
            def _rand_scale(_x):
                if np.random.rand() > 0.5:
                    rnd = ((r1 - r2) * np.random.rand() + r2)

                    scld = _x * rnd
                    scld += (_x[:, 7] - scld[:, 7]).reshape(-1, 1, 3) # restore global position, TODO: scaled-motion
                    return scld
                return _x
            seq0 = _rand_scale(seq0)
            seq1 = _rand_scale(seq1)
            seq2 = _rand_scale(seq2)

        if self.config.augment.random_rotate_y and np.random.rand() > 0.75:
            seq0, seq1, seq2 = random_rotate_sequences(seq0, seq1, seq2, rotate_around="y", device=self.config.device)
        if self.config.augment.random_rotate_x and np.random.rand() > 0.75:
            seq0, seq1, seq2 = random_rotate_sequences(seq0, seq1, seq2, rotate_around="x", device=self.config.device)
        if self.config.augment.random_rotate_z and np.random.rand() > 0.75:
            seq0, seq1, seq2 = random_rotate_sequences(seq0, seq1, seq2, rotate_around="z", device=self.config.device)

        if self.config.augment.random_reposition and np.random.rand() > 0.5:
            seq0, seq1, seq2 = random_reposition_sequences(seq0, seq1, seq2, device=self.config.device)

        return seq0[:INPUT_LEN], seq0[INPUT_LEN:], seq1[:INPUT_LEN], seq1[INPUT_LEN:], seq2[:INPUT_LEN], seq2[INPUT_LEN:], sctx01, sctx02, sctx12


def create_datasets(config, train_name="mocaptrain", valid_name="mocaptest", test_name="mupots"):
    num_workers = 0 if config.device == "cuda" else 10

    if train_name == "mocaptrain":
        print("Loading *{0}* dataset for training...".format(train_name))
        train_loader = DataLoader(TrainCMUMocap(config), batch_size=256, shuffle=True, num_workers=num_workers)

    if valid_name == "mocaptest":
        print("Loading *{0}* dataset for validation...".format(valid_name))
        cmu_test_loader = DataLoader(CMUMocap(config), batch_size=1, shuffle=False, num_workers=num_workers)

    if test_name == "mupots":
        print("Loading *{0}* dataset for testing...".format(test_name))
        mupots_test_loader = DataLoader(Mupots(config), batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, cmu_test_loader, mupots_test_loader
