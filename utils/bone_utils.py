import numpy as np
import torch


bones = [[0,1],[1,2],[2,3], [0,4],[4,5],[5,6], [0,7],[7,8],[8,9],[9,10], [8,11],[11,12],[12,13], [8,14],[14,15],[15,16]]
bone_child = [1,2,3, 4,5,6, 7,8,9,10, 11,12,13, 14,15,16]
bone_parent = [0,1,2, 0,4,5, 0,7,8,9, 8,11,12, 8,14,15]
bone_symmetry = [[0,3],[1,4],[2,5], [13,10],[14,11],[15,12]]
bone_left = [3,4,5, 10,11,12]
bone_right = [0,1,2, 13,14,15]
bone_partial = [12,15]

bone_name = [
    'r_hip', 'r_thigh', 'r_calf',
    'l_hip', 'l_thigh', 'l_calf',
    'spine_1', 'spine_2', 'neck', 'head',
    'l_shoulder', 'l_upper_arm', 'l_lower_arm',
    'r_shoulder', 'r_upper_arm', 'r_lower_arm'
]

def poses2bone_numpy(seq):
    # seq.shape = (N, 17, 3)
    global bone_child, bone_parent
    
    bone_directions = seq[:, bone_child] - seq[:, bone_parent]
    bone_lengths = np.linalg.norm(bone_directions, axis=2)
    bone_directions = bone_directions / np.expand_dims(bone_lengths, axis=-1)
    bone_lengths = np.mean(bone_lengths, axis=0)
    return bone_directions, bone_lengths

def poses2bone_torch(seq):
    # seq.shape = (N, 17, 3)
    global bone_child, bone_parent
    
    bone_directions = seq[:, bone_child] - seq[:, bone_parent]
    bone_lengths = torch.norm(bone_directions, dim=2)
    bone_directions = bone_directions / bone_lengths.unsqueeze(-1)
    return bone_directions, bone_lengths

def poses2direction_torch(seq):
    # seq.shape = (N, 1, 17, 3)
    global bone_child, bone_parent
    
    bone_directions = seq[:, :, bone_child] - seq[:, :, bone_parent]
    bone_directions = bone_directions / torch.norm(bone_directions, dim=-1).unsqueeze(-1)
    return bone_directions

def poses2direction_training(seq):
    # seq.shape = (N, 17, 2)
    global bone_child, bone_parent
    
    bone_directions = seq[:, bone_child] - seq[:, bone_parent]
    return bone_directions

def bone_length_aug(bone_lengths, mean, std):
    # bone_lengths.shape = (16,)
    bone_lengths = np.random.normal(bone_lengths, std)
    # bone_lengths = np.random.normal(mean, std)
    for p in bone_symmetry:
        bone_lengths[p] = bone_lengths[p[0]]
    return bone_lengths

def bone_length_aug_batch(bone_lengths):
    # bone_lengths.shape = (b, 16)
    bone_lengths_mean = np.mean(bone_lengths, axis=0)
    bone_lengths = bone_lengths + (np.random.rand(16)-0.5) * bone_lengths * 0.6
    
    for p in bone_symmetry:
        bone_lengths[p] = bone_lengths[p[0]]
    return bone_lengths

def bone_length_aug_partial(bone_lengths):
    bone_lengths[bone_partial] = bone_lengths[bone_partial] + (np.random.rand(2)-0.5) * bone_lengths[bone_partial] * 0.2
    bone_lengths[bone_partial] = np.mean(bone_lengths[bone_partial])
    return bone_lengths

def bones2poses_numpy(bone_directions, bone_lengths):
    # bone_directions.shape = (N, 16, 3)
    # bone_lengths.shape = (16,)
    
    seq_with_len = bone_directions * np.expand_dims(bone_lengths, axis=-1)
    res = np.empty((seq_with_len.shape[0],17,3))
    
    # hip
    res[:,0] = 0
    # legs
    res[:,1] = seq_with_len[:,0]
    res[:,2] = res[:,1] + seq_with_len[:,1]
    res[:,3] = res[:,2] + seq_with_len[:,2]
    res[:,4] = seq_with_len[:,3]
    res[:,5] = res[:,4] + seq_with_len[:,4]
    res[:,6] = res[:,5] + seq_with_len[:,5]
    # spine
    res[:,7] = seq_with_len[:,6]
    res[:,8] = res[:,7] + seq_with_len[:,7]
    res[:,9] = res[:,8] + seq_with_len[:,8]
    res[:,10] = res[:,9] + seq_with_len[:,9]
    # arms
    res[:,11] = res[:,8] + seq_with_len[:,10]
    res[:,12] = res[:,11] + seq_with_len[:,11]
    res[:,13] = res[:,12] + seq_with_len[:,12]
    res[:,14] = res[:,8] + seq_with_len[:,13]
    res[:,15] = res[:,14] + seq_with_len[:,14]
    res[:,16] = res[:,15] + seq_with_len[:,15]
    
    return res

def bones2poses_torch(bone_directions, bone_lengths):
    # bone_directions.shape = (B, N, 16, 3)
    # bone_lengths.shape = (B, 16)
    
    bone_lengths = bone_lengths.unsqueeze(1).unsqueeze(-1)
    seq_with_len = bone_directions * bone_lengths
    
    res = torch.empty((seq_with_len.shape[0],seq_with_len.shape[1],17,3), dtype=torch.float32).to(bone_lengths.device)
    
    # hip
    res[:,:,0] = 0
    # legs
    res[:,:,1] = seq_with_len[:,:,0]
    res[:,:,2] = res[:,:,1] + seq_with_len[:,:,1]
    res[:,:,3] = res[:,:,2] + seq_with_len[:,:,2]
    res[:,:,4] = seq_with_len[:,:,3]
    res[:,:,5] = res[:,:,4] + seq_with_len[:,:,4]
    res[:,:,6] = res[:,:,5] + seq_with_len[:,:,5]
    # spine
    res[:,:,7] = seq_with_len[:,:,6]
    res[:,:,8] = res[:,:,7] + seq_with_len[:,:,7]
    res[:,:,9] = res[:,:,8] + seq_with_len[:,:,8]
    res[:,:,10] = res[:,:,9] + seq_with_len[:,:,9]
    # arms
    res[:,:,11] = res[:,:,8] + seq_with_len[:,:,10]
    res[:,:,12] = res[:,:,11] + seq_with_len[:,:,11]
    res[:,:,13] = res[:,:,12] + seq_with_len[:,:,12]
    res[:,:,14] = res[:,:,8] + seq_with_len[:,:,13]
    res[:,:,15] = res[:,:,14] + seq_with_len[:,:,14]
    res[:,:,16] = res[:,:,15] + seq_with_len[:,:,15]
    
    return res