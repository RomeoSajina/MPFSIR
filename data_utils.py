import numpy as np
import torch
import os
import pickle
from scipy.spatial.transform import Rotation as R


def build_windowed_sequences(seq_list, input_window, output_window, frequency=1):

    x_out, y_out = [], []

    for seq in seq_list:

        freq_seqs = [seq[i::frequency] for i in range(frequency)]

        _x_out, _y_out = [], []

        for fs in freq_seqs:

            for i in range( len(fs) - (input_window+output_window) ):
                _x_out.append(fs[i:i+input_window])
                _y_out.append(fs[i+input_window:i+input_window+output_window])

        if len(_x_out) > 0:
            x_out.append(_x_out)
            y_out.append(_y_out)

    return np.array(x_out), np.array(y_out)


def load_3dpw(dataset_dir="./data/3dpw/", split="train", out_type="array"):
    # TRAIN AND TEST SETS ARE REVERSED FOR SOMOF Benchmark
    SPLIT_3DPW = {
        "train": "test",
        "val": "validation",
        "valid": "validation",
        "test": "train"
    }

    out = {} if out_type == "dict" else []
    
    out_single_person_poses = []
    path_to_data = os.path.join(dataset_dir, "sequenceFiles", SPLIT_3DPW[split])

    for pkl in os.listdir(path_to_data):
        with open(os.path.join(path_to_data, pkl), 'rb') as reader:
            annotations = pickle.load(reader, encoding='latin1')

        seq_poses = [[], []]

        for actor_index in range(len(annotations['genders'])):

            joints_2D = annotations['poses2d'][actor_index].transpose(0, 2, 1)
            joints_3D = annotations['jointPositions'][actor_index]

            track_joints = []

            for image_index in range(len(joints_2D)):
                J_3D_real = joints_3D[image_index].reshape(-1, 3)
                J_3D_mask = np.ones(J_3D_real.shape[:-1])
                track_joints.append(J_3D_real)

            track_joints = np.asarray(track_joints)

            SOMOF_JOINTS = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]
            poses = []
            for i in range(len(track_joints)):
                poses.append(track_joints[i][SOMOF_JOINTS])

            if len(annotations['genders']) > 1: # only if 2 people in the scene
                seq_poses[actor_index] = poses
            elif len(poses) > 0:
                out_single_person_poses.append(poses)

        if len(seq_poses[0]) > 0:
            if out_type == "dict":
                out[pkl.split(".")[0]] = np.array(seq_poses.copy())
            else:
                out.append(seq_poses.copy())

    return out, out_single_person_poses


def load_original_3dw(input_window=16, output_window=14, split="train", frequency=2):
    sequences, single_person_sequences = load_3dpw(split=split)

    x_out, y_out = None, None
    for seq in sequences:
        _x, _y = build_windowed_sequences(seq, input_window=input_window, output_window=output_window, frequency=frequency)
        _x = np.transpose(_x, (1, 0, 2, 3, 4))
        _y = np.transpose(_y, (1, 0, 2, 3, 4))
        if x_out is None:
            x_out = _x
            y_out = _y
        else:
            x_out = np.concatenate((x_out, _x), axis=0)
            y_out = np.concatenate((y_out, _y), axis=0)

    x_single_out, y_single_out = None, None
    for seq in single_person_sequences:
        seq = np.array(seq)
        seq = seq.reshape(1, *seq.shape)

        _x, _y = build_windowed_sequences(seq, input_window=input_window, output_window=output_window, frequency=2)
        _x = np.transpose(_x, (1, 0, 2, 3, 4))
        _y = np.transpose(_y, (1, 0, 2, 3, 4))
        if x_single_out is None:
            x_single_out = _x
            y_single_out = _y
        else:
            x_single_out = np.concatenate((x_single_out, _x), axis=0)
            y_single_out = np.concatenate((y_single_out, _y), axis=0)

    return x_out, y_out, x_single_out, y_single_out


def rotate_around_axis(seq, x=0, y=0, z=0):

    r = R.from_euler('ZXY', [z, x, y], degrees=True)

    centroid = np.sum(seq[0], axis=0) / seq.shape[1]

    seq_rotated = np.array([r.apply(x) for x in (seq - centroid)]) # Position on coordinate system (0,0)

    seq_rotated += centroid # Return the sequence to initial position

    return seq_rotated


def random_rotate_sequences(x, y, z=None, rotate_around="y", device="cpu"):
    x = x.detach().cpu().numpy().astype(np.float64)
    y = y.detach().cpu().numpy().astype(np.float64)
    z = None if z is None else z.detach().cpu().numpy().astype(np.float64)

    rotatedx, rotatedy, rotatedz = None, None, None

    rnd = np.random.rand()*360

    if rotate_around == "x":
        rotatedx, rotatedy = rotate_around_axis(x, x=rnd), rotate_around_axis(y, x=rnd)
        rotatedz = None if z is None else rotate_around_axis(z, x=rnd)
    if rotate_around == "y":
        rotatedx, rotatedy = rotate_around_axis(x, y=rnd), rotate_around_axis(y, y=rnd)
        rotatedz = None if z is None else rotate_around_axis(z, y=rnd)
    if rotate_around == "z":
        rotatedx, rotatedy = rotate_around_axis(x, z=rnd), rotate_around_axis(y, z=rnd)
        rotatedz = None if z is None else rotate_around_axis(z, z=rnd)
    
    if z is None:
        return torch.from_numpy(rotatedx).to(device), torch.from_numpy(rotatedy).to(device)
    else:
        return torch.from_numpy(rotatedx).to(device), torch.from_numpy(rotatedy).to(device), torch.from_numpy(rotatedz).to(device)


def random_reposition_sequences(x, y, z=None, rs=3, device="cpu"):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    z = None if z is None else z.detach().cpu().numpy()

    rdist = lambda x: np.linalg.norm(x[0, 0] - x[0, 7]) * rs * np.random.rand() # Lhip - Lshoulder

    offset = np.array([rdist(x), rdist(x), rdist(x)]).reshape(-1, 1, 3)
    repositionedx = x + offset
    repositionedy = y + offset
    repositionedz = None if z is None else z + offset
                                                                                                                           
    if z is None:
        return torch.from_numpy(repositionedx).to(device), torch.from_numpy(repositionedy).to(device)
    else:
        return torch.from_numpy(repositionedx).to(device), torch.from_numpy(repositionedy).to(device), torch.from_numpy(repositionedz).to(device)
