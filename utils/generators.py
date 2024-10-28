# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np

from utils.bone_utils import *

class ChunkedGenerator:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_2d, lengths,
                 chunk_length=1, pad=0, shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, aug_chunk_length=512):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)
    
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_2d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        # preprocess bone lengths
        bone_directions = []
        poses_3d_traj = []
        for i in range(len(poses_3d)):
            poses_3d_traj.append(poses_3d[i][:,:1])
            tmp = poses_3d[i].copy()
            tmp[:, 0] = 0
            directions, _ = poses2bone_numpy(tmp)
            bone_directions.append(directions)
        
        # Initialize buffers
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length + 2*pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))
        self.batch_len = np.empty((batch_size, lengths.shape[-1]))
        self.batch_2d_chunk_aug = np.empty((batch_size, aug_chunk_length, poses_2d[0].shape[-2], 2))
        self.batch_3d_chunk_aug = np.empty((batch_size, aug_chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.endless = endless
        self.state = None
        
        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_3d_traj = poses_3d_traj
        self.poses_2d = poses_2d
        self.bone_directions = bone_directions
        self.lengths = lengths
        
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.aug_chunk_length = aug_chunk_length
        
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state
    
    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    start_2d = start_3d - self.pad
                    end_2d = end_3d + self.pad
                    
                    # lengths
                    self.batch_len[i] = self.lengths[seq_i].copy()
                    
                    # 2D poses
                    seq_2d = self.poses_2d[seq_i]
                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_2d.shape[0])
                    pad_left_2d = low_2d - start_2d
                    pad_right_2d = end_2d - high_2d
                    if pad_left_2d != 0 or pad_right_2d != 0:
                        self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
                    else:
                        self.batch_2d[i] = seq_2d[low_2d:high_2d]

                    if flip:
                        # Flip 2D keypoints
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]
                    
                    # augmented 3d pose
                    idx = np.random.randint(low=0, high=len(seq_2d)-self.aug_chunk_length)
                    seq_2d_chunk = seq_2d[idx:idx+self.aug_chunk_length,:,:2].copy()
                    self.batch_2d_chunk_aug[i] = seq_2d_chunk
                    if flip:
                        self.batch_2d_chunk_aug[i, :, :, 0] *= -1
                        self.batch_2d_chunk_aug[i, :, self.joints_left + self.joints_right] = \
                                self.batch_2d_chunk_aug[i, :, self.joints_right + self.joints_left]
                    
                    # 3D poses
                    if self.poses_3d is not None:
                        seq_3d = self.poses_3d[seq_i]
                        low_3d = max(start_3d, 0)
                        high_3d = min(end_3d, seq_3d.shape[0])
                        pad_left_3d = low_3d - start_3d
                        pad_right_3d = end_3d - high_3d
                        if pad_left_3d != 0 or pad_right_3d != 0:
                            self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_3d[i] = seq_3d[low_3d:high_3d]

                        if flip:
                            # Flip 3D joints
                            self.batch_3d[i, :, :, 0] *= -1
                            self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                    self.batch_3d[i, :, self.joints_right + self.joints_left]

                    # Cameras
                    if self.cameras is not None:
                        self.batch_cam[i] = self.cameras[seq_i]
                        if flip:
                            # Flip horizontal distortion coefficients
                            self.batch_cam[i, 2] *= -1
                            self.batch_cam[i, 7] *= -1

                if self.endless:
                    self.state = (b_i + 1, pairs)
                
                yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)], \
                    self.batch_len[:len(chunks)], self.batch_2d_chunk_aug[:len(chunks)]
            
            if self.endless:
                self.state = None
            else:
                enabled = False

class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, cameras, poses_3d, poses_2d, bone_lengths, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = cameras
        self.poses_2d = poses_2d
        
        # preprocess bone lengths
        bone_lengths_gt = []
        bone_directions = []
        for i in range(len(poses_3d)):
            tmp = poses_3d[i].copy()
            tmp[:, 0] = 0
            directions, lengths = poses2bone_numpy(tmp)
            bone_directions.append(directions)
            bone_lengths_gt.append(lengths)
        
        self.poses_3d = poses_3d
        self.bone_lengths = bone_lengths
        self.bone_lengths_gt = bone_lengths_gt
        self.bone_directions = bone_directions
        
        self.num_batches = len(poses_2d)
        
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count
    
    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d, seq_len, seq_len_gt, seq_dir in \
                zip_longest(self.cameras, self.poses_3d, self.poses_2d, self.bone_lengths, self.bone_lengths_gt, self.bone_directions):
            batch_bone_len = np.expand_dims(seq_len, axis=0)
            batch_bone_len_gt = np.expand_dims(seq_len_gt, axis=0)
            batch_bone_directions = np.expand_dims(seq_dir, axis=0)
            batch_3d = np.expand_dims(seq_3d, axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            batch_cam = None if seq_cam is None else np.full((batch_3d.shape[1],len(seq_cam)), seq_cam)
            
            if self.augment:
                # Append flipped version
                batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                batch_cam[1, 2] *= -1
                batch_cam[1, 7] *= -1
                
                batch_bone_len = np.concatenate((batch_bone_len, batch_bone_len), axis=0)
                
                batch_bone_len_gt = np.concatenate((batch_bone_len_gt, batch_bone_len_gt), axis=0)
                
                batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                batch_3d[1, :, :, 0] *= -1
                batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]
            
            yield batch_cam, batch_3d, batch_2d, batch_bone_len, batch_bone_len_gt, batch_bone_directions


class ChunkedGeneratorBoneLengths:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_2d, bone_lengths,
                 chunk_length=256, segment_shift=32, shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, length_aug='smpl'):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is not None and len(cameras) == len(poses_2d)
    
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_2d[i].shape[0] == poses_3d[i].shape[0]
            
            bounds = []
            for j in range(0, poses_2d[i].shape[0]-chunk_length, segment_shift):
                bounds.append([j,j+chunk_length])
            bounds = np.array(bounds)
            
            augment_vector = np.full(len(bounds), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds)), bounds[:,0], bounds[:,1], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds)), bounds[:,0], bounds[:,1], ~augment_vector)

        # preprocess bone lengths
        bone_directions = []
        poses_3d_traj = []
        for i in range(len(poses_3d)):
            poses_3d_traj.append(poses_3d[i][:,:1])
            tmp = poses_3d[i].copy()
            tmp[:, 0] = 0
            directions, lengths = poses2bone_numpy(tmp)
            bone_directions.append(directions)
        
        # Initialize buffers
        self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        self.batch_bone_len = np.empty((batch_size, bone_lengths[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.endless = endless
        self.state = None
        
        self.cameras = cameras
        self.poses_3d_traj = poses_3d_traj
        self.bone_directions = bone_directions
        self.bone_lengths = bone_lengths
        self.bone_lengths_size = len(bone_lengths)
        self.poses_2d = poses_2d
        
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.length_aug = length_aug
        
        if self.length_aug == 'random_norm_all':
            self.lengths_mean = np.mean(bone_lengths, axis=0)
            self.lengths_std = np.std(bone_lengths, axis=0)
        elif self.length_aug == 'random_norm_train':
            self.lengths_mean = np.mean(bone_lengths[:-2], axis=0)
            self.lengths_std = np.std(bone_lengths[:-2], axis=0)
        
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state
    
    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                if self.length_aug == 'random_unif':
                    for i, (seq_i, start_2d, end_2d, flip) in enumerate(chunks):
                        no = int(seq_i / 120)
                        self.batch_bone_len[i] = self.bone_lengths[no].copy()
                    self.batch_bone_len[i] = bone_length_aug_batch(self.batch_bone_len[i])
                
                for i, (seq_i, start_2d, end_2d, flip) in enumerate(chunks):
                    # Cameras
                    self.batch_cam[i] = self.cameras[seq_i]
                    if flip:
                        # Flip horizontal distortion coefficients
                        self.batch_cam[i, 2] *= -1
                        self.batch_cam[i, 7] *= -1
                    
                    # Bone
                    if self.length_aug == 'smpl':
                        no = np.random.randint(low=0, high=self.bone_lengths_size)
                        self.batch_bone_len[i] = self.bone_lengths[no].copy()
                    elif 'norm' in self.length_aug:
                        no = int(seq_i / 120)
                        self.batch_bone_len[i] = bone_length_aug(self.bone_lengths[no].copy(), self.lengths_mean, self.lengths_std)
                    seq_3d = bones2poses_numpy(self.bone_directions[seq_i][start_2d:end_2d], self.batch_bone_len[i])
                    rand_traj = np.random.normal(loc=0.0, scale=0.5, size=3)
                    self.batch_3d[i] = seq_3d + self.poses_3d_traj[seq_i][start_2d:end_2d] + rand_traj
                    
                    if flip:
                        # Flip 3D joints
                        self.batch_3d[i, :, :, 0] *= -1
                        self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                self.batch_3d[i, :, self.joints_right + self.joints_left]
                    
                if self.endless:
                    self.state = (b_i + 1, pairs)
                
                yield self.batch_cam[:len(chunks)], self.batch_bone_len[:len(chunks)], self.batch_3d[:len(chunks)]
            
            if self.endless:
                self.state = None
            else:
                enabled = False

class UnchunkedGeneratorBoneLengths:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 bone_lengths_aug=None, length_aug='smpl'):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is not None and len(cameras) == len(poses_2d)

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = cameras
        self.poses_2d = poses_2d
        self.length_aug = length_aug
        
        # preprocess bone lengths
        bone_lengths = []
        bone_directions = []
        poses_3d_traj = []
        for i in range(len(poses_3d)):
            poses_3d_traj.append(poses_3d[i][:,:1])
            tmp = poses_3d[i].copy()
            tmp[:, 0] = 0
            directions, lengths = poses2bone_numpy(tmp)
            bone_directions.append(directions)
            bone_lengths.append(lengths)
        
        self.poses_3d = poses_3d
        self.poses_3d_traj = poses_3d_traj
        self.bone_lengths = bone_lengths
        self.bone_lengths_size = len(bone_lengths)
        self.bone_lengths_aug = bone_lengths_aug
        self.bone_directions = bone_directions
        
        self.num_batches = len(poses_2d)
        
        if self.length_aug == 'random_norm_all':
            self.lengths_mean = np.mean(bone_lengths, axis=0)
            self.lengths_std = np.std(bone_lengths, axis=0)
        elif self.length_aug == 'random_norm_train':
            self.lengths_mean = np.mean(bone_lengths[:-2], axis=0)
            self.lengths_std = np.std(bone_lengths[:-2], axis=0)
        
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count
    
    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for seq_i in range(len(self.poses_2d)):
            batch_bone_len = np.expand_dims(self.bone_lengths[seq_i], axis=0)
            batch_2d = np.expand_dims(self.poses_2d[seq_i], axis=0)
            batch_cam = np.expand_dims(self.cameras[seq_i], axis=0)
            
            # augmnentation
            if self.bone_lengths_aug is not None:
                if self.length_aug == 'smpl':
                    no = np.random.randint(low=0, high=self.bone_lengths_size)
                    batch_bone_len_aug = self.bone_lengths[no].copy()
                elif self.length_aug == 'random_unif':
                    no = int(seq_i / 120)
                    batch_bone_len_aug = bone_length_aug_batch(self.bone_lengths_aug[no].copy())
                elif 'norm' in self.length_aug:
                    no = int(seq_i / 120)
                    batch_bone_len_aug = bone_length_aug(self.bone_lengths_aug[no].copy(), self.lengths_mean, self.lengths_std)
                seq_3d = bones2poses_numpy(self.bone_directions[seq_i], batch_bone_len_aug)
                rand_traj = np.random.normal(loc=0.0, scale=0.5, size=3)
                batch_3d_aug = seq_3d + self.poses_3d_traj[seq_i] + rand_traj
                
                batch_bone_len_aug = np.expand_dims(batch_bone_len_aug, axis=0)
                batch_3d_aug = np.expand_dims(batch_3d_aug, axis=0)
            else:
                batch_bone_len_aug = None
                batch_3d_aug = None
            
            if self.augment:
                # Append flipped version
                batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                batch_cam[1, 2] *= -1
                batch_cam[1, 7] *= -1
                
                batch_bone_len = np.concatenate((batch_bone_len, batch_bone_len), axis=0)
                batch_bone_len_aug = np.concatenate((batch_bone_len_aug, batch_bone_len_aug), axis=0)

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]
                
                batch_3d_aug = np.concatenate((batch_3d_aug, batch_3d_aug), axis=0)
                batch_3d_aug[1, :, :, 0] *= -1
                batch_3d_aug[1, :, self.joints_left + self.joints_right] = batch_3d_aug[1, :, self.joints_right + self.joints_left]
            
            yield batch_cam, batch_bone_len, batch_2d, batch_bone_len_aug, batch_3d_aug
