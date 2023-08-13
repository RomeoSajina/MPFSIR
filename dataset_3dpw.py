import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os

from data_utils import *
from amass import AMASSDataset

def create_3dpw_test_out_json_if_needed():
    if os.path.exists("./data/somof_data_3dpw/3dpw_test_out.json"):
        return
    
    ds, _ = load_3dpw(split="test", out_type="dict")
    
    with open("./data/somof_data_3dpw/3dpw_test_in.json") as f:
        X = np.array(json.load(f))
    with open("./data/somof_data_3dpw/3dpw_test_frames_in.json") as f:
        X_f = np.array(json.load(f))
    
    i = X_f[0]

    y_test = []
    for i in X_f:
        key, last_idx = i[-1].split("/")[0], int( i[-1].split("/")[-1].split(".jpg")[0].split("_")[1] )
        key, last_idx

        indicies = np.arange(last_idx+2, last_idx+2+14*2, 2)

        y_i_test = ds[key][:, indicies]
        y_test.append(y_i_test)
    
    print("Creating '3dpw_test_out.json' file...")
    with open("./data/somof_data_3dpw/3dpw_test_out.json", "w") as outfile:
        json.dump(np.array(y_test).tolist(), outfile)

    
create_3dpw_test_out_json_if_needed()


class SoMof3DPW(Dataset):

    def __init__(self, config, dir_path="./data/somof_data_3dpw/", name="test"):
        self.config = config

        with open(dir_path + "3dpw_{0}_in.json".format(name)) as f:
            X = np.array(json.load(f))

        with open(dir_path + "3dpw_{0}_out.json".format(name)) as f:
            Y = np.array(json.load(f))

        X = X if X.shape[-1] == 3 else X.reshape(*X.shape[:-1], 13, 3)
        Y = Y if Y.shape[-1] == 3 else Y.reshape(*Y.shape[:-1], 13, 3)

        data = np.concatenate((X, Y), axis=2)

        self.data = torch.from_numpy(data).to(config.device)

        print("Stats for SoMoF3DPW {0} ctx-num-of-examples:".format(name), {0: self.data.shape[0]})

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        INPUT_LEN = self.config.input_len

        x0, y0 = self.data[idx, 0, :INPUT_LEN], self.data[idx, 0, INPUT_LEN:]
        x1, y1 = self.data[idx, 1, :INPUT_LEN], self.data[idx, 1, INPUT_LEN:]
        sctx = 0

        sample = {
            'keypoints0': x0.requires_grad_(False).to(self.config.device),
            'keypoints1': x1.requires_grad_(False).to(self.config.device),
            'out_keypoints0': y0.requires_grad_(False).to(self.config.device),
            'out_keypoints1': y1.requires_grad_(False).to(self.config.device),
            'sctx': torch.from_numpy(np.array([sctx])).to(self.config.device), # 0 - socially dependent, 1 - independent, 2 - same person
        }

        return sample


class Train3DPWAmass(Dataset):

    def __init__(self, config, x3dpw, x3dpw_single, xamass, x3dpw_val, x3dpw_val_single):
        self.config = config
        self.use_augmentation = True
        self.use_amass = True
        self.xamass = torch.from_numpy(xamass).to(config.device)
        self.x3dpw = torch.from_numpy(x3dpw).to(config.device)
        self.x3dpw_single = torch.from_numpy(x3dpw_single).to(config.device)
        
        self.x3dpw_val = torch.from_numpy(x3dpw_val).to(config.device)
        self.x3dpw_val_single = torch.from_numpy(x3dpw_val_single).to(config.device)

        print("Stats for 3DPWAmass ctx-num-of-examples:", {  0: self.x3dpw.shape[0],
                                                             1: self.x3dpw_single.shape[0]+(self.xamass.shape[0] if self.use_amass else 0),
                                                             2: self.x3dpw_single.shape[0]+(self.xamass.shape[0] if self.use_amass else 0)
                                                          })

    def __len__(self):
        return self.x3dpw.shape[0] + self.x3dpw_single.shape[0] * 2 + (self.xamass.shape[0] * 2 if self.use_amass else 0)

    def __getitem__(self, idx):
        INPUT_LEN = self.config.input_len

        if idx < self.x3dpw.shape[0]:
            x0, y0 = self.x3dpw[idx, 0, :INPUT_LEN], self.x3dpw[idx, 0, INPUT_LEN:]
            x1, y1 = self.x3dpw[idx, 1, :INPUT_LEN], self.x3dpw[idx, 1, INPUT_LEN:]
            sctx = 0

        elif idx < self.x3dpw.shape[0] + self.x3dpw_single.shape[0]:
            idx -= self.x3dpw.shape[0]
            partner_idx = np.random.randint(0, self.x3dpw_single.shape[0])
            x0, y0 = self.x3dpw_single[idx, 0, :INPUT_LEN], self.x3dpw_single[idx, 0, INPUT_LEN:]
            x1, y1 = self.x3dpw_single[partner_idx, 0, :INPUT_LEN], self.x3dpw_single[partner_idx, 0, INPUT_LEN:]
            sctx = 1 if idx != partner_idx else 2

        elif idx < self.x3dpw.shape[0] + self.x3dpw_single.shape[0] * 2:
            idx -= (self.x3dpw.shape[0] + self.x3dpw_single.shape[0])
            x0, y0 = self.x3dpw_single[idx, 0, :INPUT_LEN], self.x3dpw_single[idx, 0, INPUT_LEN:]
            x1, y1 = x0, y0
            sctx = 2

        elif self.use_amass and idx < self.x3dpw.shape[0] + self.x3dpw_single.shape[0] * 2 + self.xamass.shape[0]:
            idx -= self.x3dpw.shape[0] + self.x3dpw_single.shape[0] * 2
            partner_idx = np.random.randint(0, self.xamass.shape[0])
            x0, y0 = self.xamass[idx, 0, :INPUT_LEN], self.xamass[idx, 0, INPUT_LEN:]
            x1, y1 = self.xamass[partner_idx, 0, :INPUT_LEN], self.xamass[partner_idx, 0, INPUT_LEN:]
            sctx = 1 if idx != partner_idx else 2

        elif self.use_amass and idx < self.x3dpw.shape[0] + self.x3dpw_single.shape[0] * 2 + self.xamass.shape[0] * 2:
            idx -= self.x3dpw.shape[0] + self.x3dpw_single.shape[0] * 2 + self.xamass.shape[0]
            x0, y0 = self.xamass[idx, 0, :INPUT_LEN], self.xamass[idx, 0, INPUT_LEN:]
            x1, y1 = x0, y0
            sctx = 2

        if self.use_augmentation:
            x0, y0, x1, y1, sctx = self._augment(x0, y0, x1, y1, sctx)

        sample = {
            'keypoints0': x0.requires_grad_(False).to(self.config.device),
            'keypoints1': x1.requires_grad_(False).to(self.config.device),
            'out_keypoints0': y0.requires_grad_(False).to(self.config.device),
            'out_keypoints1': y1.requires_grad_(False).to(self.config.device),
            'sctx': torch.from_numpy(np.array([sctx])).to(self.config.device), # 0 - socially dependent, 1 - independent, 2 - same person
        }

        return sample

    def _augment(self, x0, y0, x1, y1, sctx):

        if np.random.rand() > 0.5:
            return x0, y0, x1, y1, sctx

        seq0 = torch.cat((x0, y0), dim=0)
        seq1 = torch.cat((x1, y1), dim=0)

        if self.config.augment.backward_movement and np.random.rand() > 0.5: # backward movement, is this flip?? torch.flip(seq0, dim=0)
            seq0 = seq0[np.arange(-seq0.shape[0]+1, 1)]
            seq1 = seq1[np.arange(-seq1.shape[0]+1, 1)]

        if self.config.augment.reversed_order and np.random.rand() > 0.5: # reversed order of people
            seq0, seq1 = seq1, seq0

        if self.config.augment.random_scale and np.random.rand() > 0.5: # random scale
            r1=0.1#0.8
            r2=5.#1.2
            def _rand_scale(_x):
                if np.random.rand() > 0.5:
                    rnd = ((r1 - r2) * np.random.rand() + r2)

                    scld = _x * rnd
                    scld += (_x[:, 7] - scld[:, 7]).reshape(-1, 1, 3) # restore global position, TODO: scaled-motion
                    return scld
                return _x
            seq0 = _rand_scale(seq0)
            seq1 = _rand_scale(seq1)

        if self.config.augment.random_rotate_y and np.random.rand() > 0.75:
            seq0, seq1 = random_rotate_sequences(seq0, seq1, rotate_around="y", device=self.config.device)
        if self.config.augment.random_rotate_x and np.random.rand() > 0.75:
            seq0, seq1 = random_rotate_sequences(seq0, seq1, rotate_around="x", device=self.config.device)
        if self.config.augment.random_rotate_z and np.random.rand() > 0.75:
            seq0, seq1 = random_rotate_sequences(seq0, seq1, rotate_around="z", device=self.config.device)

        if self.config.augment.random_reposition and np.random.rand() > 0.5:
            seq0, seq1 = random_reposition_sequences(seq0, seq1, device=self.config.device)
        
        return seq0[:self.config.input_len], seq0[self.config.input_len:], seq1[:self.config.input_len], seq1[self.config.input_len:], sctx


class Original3DPW(Dataset):

    def __init__(self, config, name="test"):
        INPUT_LEN, OUTPUT_LEN = config.input_len, config.output_len

        x_3dpw_orig, y_3dpw_orig, x_3dpw_orig_single, y_3dpw_orig_single = load_original_3dw(input_window=INPUT_LEN, output_window=OUTPUT_LEN, split=name)
        x3dpw = np.concatenate((x_3dpw_orig, y_3dpw_orig), axis=2)
        x3dpw_single = np.concatenate((x_3dpw_orig_single, y_3dpw_orig_single), axis=2)

        self.config = config

        self.x3dpw = torch.from_numpy(x3dpw).to(config.device)
        self.x3dpw_single = torch.from_numpy(x3dpw_single).to(config.device)

        print("Stats for Original3DPW {0} ctx-num-of-examples:".format(name), {0: self.x3dpw.shape[0],
                                                                               1: self.x3dpw_single.shape[0],
                                                                               2: 0
                                                                              })

    def __len__(self):
        return self.x3dpw.shape[0] + self.x3dpw_single.shape[0]

    def __getitem__(self, idx):
        INPUT_LEN = self.config.input_len

        if idx < self.x3dpw.shape[0]:
            x0, y0 = self.x3dpw[idx, 0, :INPUT_LEN], self.x3dpw[idx, 0, INPUT_LEN:]
            x1, y1 = self.x3dpw[idx, 1, :INPUT_LEN], self.x3dpw[idx, 1, INPUT_LEN:]
            sctx = 0

        elif idx < self.x3dpw.shape[0] + self.x3dpw_single.shape[0]:
            idx -= self.x3dpw.shape[0]
            x0, y0 = self.x3dpw_single[idx, 0, :INPUT_LEN], self.x3dpw_single[idx, 0, INPUT_LEN:]
            x1, y1 = x0, y0
            sctx = 2

        sample = {
            'keypoints0': x0.requires_grad_(False).to(self.config.device),
            'keypoints1': x1.requires_grad_(False).to(self.config.device),
            'out_keypoints0': y0.requires_grad_(False).to(self.config.device),
            'out_keypoints1': y1.requires_grad_(False).to(self.config.device),
            'sctx': torch.from_numpy(np.array([sctx])).to(self.config.device), # 0 - socially dependent, 1 - independent, 2 - same person
        }

        return sample


class TestCTXDS(Dataset):

    def __init__(self, config, x3dpw, x3dpw_single):
        self.config = config

        # split single 3dpw into one that contains repeated sequences, other two independent sequences
        indicies = np.random.choice(x3dpw_single.shape[0], x3dpw.shape[0]*2, replace=False)
        assert np.unique(indicies).shape == indicies.shape

        x3dpw_single_independent = x3dpw_single[indicies[:x3dpw.shape[0]]]
        x3dpw_single_twice = x3dpw_single[indicies[x3dpw.shape[0]:]]

        self.x3dpw = torch.from_numpy(x3dpw).to(config.device)
        self.x3dpw_single_independent = torch.from_numpy(x3dpw_single_independent).to(config.device)
        self.x3dpw_single_twice = torch.from_numpy(x3dpw_single_twice).to(config.device)

        print("Stats for ctx-test ctx-num-of-examples:", {0: self.x3dpw.shape[0],
                                                         1: self.x3dpw_single_independent.shape[0],
                                                         2: self.x3dpw_single_twice.shape[0]
                                                        })

    def __len__(self):
        return self.x3dpw.shape[0] + self.x3dpw_single_independent.shape[0] + self.x3dpw_single_twice.shape[0]

    def __getitem__(self, idx):
        INPUT_LEN = self.config.input_len

        if idx < self.x3dpw.shape[0]:
            x0, y0 = self.x3dpw[idx, 0, :INPUT_LEN], self.x3dpw[idx, 0, INPUT_LEN:]
            x1, y1 = self.x3dpw[idx, 1, :INPUT_LEN], self.x3dpw[idx, 1, INPUT_LEN:]
            sctx = 0

        elif idx < self.x3dpw.shape[0] + self.x3dpw_single_independent.shape[0]:
            idx -= self.x3dpw.shape[0]
            partner_idx = idx
            while partner_idx == idx:
                partner_idx = np.random.randint(0, self.x3dpw_single_independent.shape[0])

            x0, y0 = self.x3dpw_single_independent[idx, 0, :INPUT_LEN], self.x3dpw_single_independent[idx, 0, INPUT_LEN:]
            x1, y1 = self.x3dpw_single_independent[partner_idx, 0, :INPUT_LEN], self.x3dpw_single_independent[partner_idx, 0, INPUT_LEN:]
            sctx = 1

        elif idx < self.x3dpw.shape[0] + self.x3dpw_single_independent.shape[0] + self.x3dpw_single_twice.shape[0]:
            idx -= (self.x3dpw.shape[0] + self.x3dpw_single_independent.shape[0])
            x0, y0 = self.x3dpw_single_twice[idx, 0, :INPUT_LEN], self.x3dpw_single_twice[idx, 0, INPUT_LEN:]
            x1, y1 = x0, y0
            sctx = 2

        sample = {
            'keypoints0': x0.requires_grad_(False).to(self.config.device),
            'keypoints1': x1.requires_grad_(False).to(self.config.device),
            'out_keypoints0': y0.requires_grad_(False).to(self.config.device),
            'out_keypoints1': y1.requires_grad_(False).to(self.config.device),
            'sctx': torch.from_numpy(np.array([sctx])).to(self.config.device), # 0 - socially dependent, 1 - independent, 2 - same person
        }

        return sample


def create_datasets(config, train_name="3dpw+amass", valid_name="somofvalid", test_name="somoftest"):
    INPUT_LEN, OUTPUT_LEN = config.input_len, config.output_len
    num_workers = 0 if config.device == "cuda" else 10

    if train_name == "3dpw+amass":
        print("Loading *{0}* dataset for training...".format(train_name))
        x_amass = AMASSDataset(split_name="train", data_aug=False, input_length=INPUT_LEN, output_length=OUTPUT_LEN).load()

        x_3dpw_orig, y_3dpw_orig, x_3dpw_orig_single, y_3dpw_orig_single = load_original_3dw(input_window=INPUT_LEN, output_window=OUTPUT_LEN, frequency=2) # Take every other frame
        tmp_3dpw_orig = np.concatenate((x_3dpw_orig, y_3dpw_orig), axis=2)
        tmp_3dpw_orig_single = np.concatenate((x_3dpw_orig_single, y_3dpw_orig_single), axis=2)
        
        x_val, y_val, x_val_single, y_val_single = \
        load_original_3dw(input_window=INPUT_LEN, output_window=OUTPUT_LEN, frequency=2, split="valid") # Take every other frame
        tmp_val_orig = np.concatenate((x_val, y_val), axis=2)
        tmp_val_single = np.concatenate((x_val_single, y_val_single), axis=2)

        pfds = Train3DPWAmass(config=config, x3dpw=tmp_3dpw_orig, x3dpw_single=tmp_3dpw_orig_single, xamass=x_amass.reshape(-1, 1, INPUT_LEN+OUTPUT_LEN, 13, 3).numpy(), x3dpw_val=tmp_val_orig, x3dpw_val_single=tmp_val_single)

        train_loader = DataLoader(pfds, batch_size=256, shuffle=True, num_workers=num_workers)

    if valid_name == "somofvalid":
        print("Loading *{0}* dataset for validation...".format(valid_name))
        valid_loader = DataLoader(SoMof3DPW(config=config, name="valid"), batch_size=256, shuffle=False, num_workers=num_workers)

    if test_name == "somoftest":
        print("Loading *{0}* dataset for testing...".format(test_name))
        test_loader = DataLoader(SoMof3DPW(config=config, name="test"), batch_size=256, shuffle=False, num_workers=num_workers)
    elif test_name == "somofvalid":
        print("Loading *{0}* dataset for testing...".format(valid_name))
        test_loader = DataLoader(SoMof3DPW(config=config, name="valid"), batch_size=256, shuffle=False, num_workers=num_workers)
    elif test_name == "3dpwtest":
        print("Loading *{0}* dataset for testing...".format(test_name))
        test_loader = DataLoader(Original3DPW(config=config, name="test"), batch_size=256, shuffle=False, num_workers=num_workers)

    if True:
        print("Loading *{0}* dataset for ctx testing...".format("3dpwtestctx"))
        x_3dpw_orig, y_3dpw_orig, x_3dpw_orig_single, y_3dpw_orig_single = load_original_3dw(input_window=INPUT_LEN, output_window=OUTPUT_LEN, split="test")
        tmp_3dpw_orig = np.concatenate((x_3dpw_orig, y_3dpw_orig), axis=2)
        tmp_3dpw_orig_single = np.concatenate((x_3dpw_orig_single, y_3dpw_orig_single), axis=2)

        ctxtestds = TestCTXDS(config=config, x3dpw=tmp_3dpw_orig, x3dpw_single=tmp_3dpw_orig_single)
        ctx_test_loader = DataLoader(ctxtestds, batch_size=256, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, ctx_test_loader
