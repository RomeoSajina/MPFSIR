import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import torch
import torch.utils.data as data

def ang2joint(p3d0, pose,
              parent={0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9,
                      15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}):
    """
    :param p3d0:[batch_size, joint_num, 3]
    :param pose:[batch_size, joint_num, 3]
    :param parent:
    :return:
    """
    batch_num = p3d0.shape[0]
    jnum = len(parent.keys())
    J = p3d0
    R_cube_big = rodrigues(pose.contiguous().view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)
    results = []
    results.append(
        with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
    )

    for i in range(1, jnum):

        results.append(
            torch.matmul(
                results[parent[i]],
                with_zeros(
                    torch.cat(
                        (R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:, parent[i], :], (-1, 3, 1))),
                        dim=2
                    )
                )
            )
        )

    stacked = torch.stack(results, dim=1)
    J_transformed = stacked[:, :, :3, 3]
    return J_transformed

def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.
    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].
    Return:
    -------
    Rotation matrix of shape [batch_size * angle_num, 3, 3].
    """
    eps = r.clone().normal_(std=1e-8)
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
    # theta = torch.norm(r, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float).to(r.device)
    m = torch.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
         -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float).unsqueeze(dim=0) \
              + torch.zeros((theta_dim, 3, 3), dtype=torch.float)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R

def with_zeros(x):
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.
    Parameter:
    ---------
    x: Tensor to be appended.
    Return:
    ------
    Tensor after appending of shape [4,4]
    """
    ones = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float
    ).expand(x.shape[0], -1, -1).to(x.device)
    ret = torch.cat((x, ones), dim=1)
    return ret


class AMASSDataset(data.Dataset):
    def __init__(self, split_name="train", data_aug=False, input_length=16, output_length=14):
        super(AMASSDataset, self).__init__()
        self._split_name = split_name
        self._data_aug = data_aug
        self._root_dir = "."

        self._amass_anno_dir = "./data/amass"
        self.amass_motion_input_length = input_length
        self.amass_motion_target_length = output_length

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._all_amass_motion_poses)

    def load(self, path="./data/amass/amass_somof_16-14.npy"):

        if os.path.exists(path):
            return torch.from_numpy(np.load(path, allow_pickle=True))

        print("Creating amass dataset....")
        
        self._amass_file_names = self._get_amass_names()

        self._load_skeleton()
        self._all_amass_motion_poses = self._load_all()
        self._file_length = len(self._all_amass_motion_poses)

        AMASS_KPS = [1, 2, 4, 5, 7, 8, 12,  16, 17, 18, 19, 20, 21]

        x_amass = torch.cat( [torch.cat(self[i], dim=0).reshape(1, 30, -1) for i in range(len(self))], dim=0)

        x_amass = x_amass.reshape(-1, 30, 22, 3)[:, :, AMASS_KPS]
        print("amass dataset created:", x_amass.shape)
        np.save(path.split(".npy")[0], x_amass.numpy())
        return self.load(path)

    def _get_amass_names(self):

        if self._split_name == 'train':
            seq_names = ["CMU", "BMLmovi", "BioMotionLab_NTroje"]
        else:
            seq_names = ["KIT"]

        file_list = []
        for dataset in seq_names:
            subjects = glob.glob(self._amass_anno_dir + '/' + dataset + '/*')
            for subject in subjects:
                if os.path.isdir(subject):
                    files = glob.glob(subject + '/*poses.npz')
                    file_list.extend(files)
        return file_list

    def _preprocess(self, amass_motion_feats):
        if amass_motion_feats is None:
            return None
        amass_seq_len = amass_motion_feats.shape[0]

        if self.amass_motion_input_length + self.amass_motion_target_length < amass_seq_len:
            start = np.random.randint(amass_seq_len - self.amass_motion_input_length  - self.amass_motion_target_length + 1)
            end = start + self.amass_motion_input_length
        else:
            return None
        amass_motion_input = torch.zeros((self.amass_motion_input_length, amass_motion_feats.shape[1]))
        amass_motion_input[:end-start] = amass_motion_feats[start:end]

        amass_motion_target = torch.zeros((self.amass_motion_target_length, amass_motion_feats.shape[1]))
        amass_motion_target[:self.amass_motion_target_length] = amass_motion_feats[end:end+self.amass_motion_target_length]

        amass_motion = torch.cat([amass_motion_input, amass_motion_target], axis=0)

        return amass_motion

    def _load_skeleton(self):

        #skeleton_info = np.load(os.path.join(self._root_dir, 'body_models', 'smpl_skeleton.npz'))
        skeleton_info = np.load(os.path.join(self._root_dir, 'data/amass', 'smpl_skeleton.npz'))
        
        self.p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()
        parents = skeleton_info['parents']
        self.parent = {}
        for i in range(len(parents)):
            self.parent[i] = parents[i]

    def _load_all(self):
        all_amass_motion_poses = []
        for amass_motion_name in tqdm(self._amass_file_names):
            try:
                amass_info = np.load(amass_motion_name)
                amass_motion_poses = amass_info['poses'] # 156 joints(all joints of SMPL)
                N = len(amass_motion_poses)
                if N < self.amass_motion_target_length + self.amass_motion_input_length:
                    continue

                frame_rate = amass_info['mocap_framerate']
                sample_rate = int(frame_rate // 25)
                sampled_index = np.arange(0, N, sample_rate)
                amass_motion_poses = amass_motion_poses[sampled_index]

                T = amass_motion_poses.shape[0]
                amass_motion_poses = R.from_rotvec(amass_motion_poses.reshape(-1, 3)).as_rotvec()
                amass_motion_poses = amass_motion_poses.reshape(T, 52, 3)
                amass_motion_poses[:, 0] = 0

                p3d0_tmp = self.p3d0.repeat([amass_motion_poses.shape[0], 1, 1])
                amass_motion_poses = ang2joint(p3d0_tmp, torch.tensor(amass_motion_poses).float(), self.parent).reshape(-1, 52, 3)[:, :22].reshape(T, -1)

                all_amass_motion_poses.append(amass_motion_poses)
            except Exception as e:
                print("Unable to load", amass_motion_name, e)
        return all_amass_motion_poses

    def __getitem__(self, index):
        amass_motion_poses = self._all_amass_motion_poses[index]
        amass_motion = self._preprocess(amass_motion_poses)
        if amass_motion is None:
            while amass_motion is None:
                index = np.random.randint(self._file_length)
                amass_motion_poses = self._all_amass_motion_poses[index]
                amass_motion = self._preprocess(amass_motion_poses)

        if self._data_aug:
            if np.random.rand() > .5:
                idx = [i for i in range(amass_motion.size(0)-1, -1, -1)]
                idx = torch.LongTensor(idx)
                amass_motion = amass_motion[idx]

        amass_motion_input = amass_motion[:self.amass_motion_input_length].float()
        amass_motion_target = amass_motion[-self.amass_motion_target_length:].float()
        return amass_motion_input, amass_motion_target
