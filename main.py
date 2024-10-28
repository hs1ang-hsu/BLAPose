import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os

from utils.camera import *
from utils.model_length import BoneLengthModel
from utils.model_mix import MixModel
from utils.generators import *
from utils.loss import *
from utils.bone_utils import *

def load_dataset(args):
    # Loading 3D dataset
    print('Loading 3D dataset...')
    dataset_path = 'data/data_3d_' + args.dataset + '.npz'
    if args.dataset == 'h36m':
        from utils.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
    else:
        raise KeyError('Invalid dataset')
    
    # Processing 3D dataset
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(
                        anim['positions'],
                        R=cam['orientation'],
                        t=cam['translation']
                    )
                    
                    # Remove global offset, but keep trajectory in first position
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d
    
    # Loading 2D dataset
    print('Loading 2D dataset...')
    keypoints = np.load(
        'data/data_2d_'+args.dataset+'_'+args.keypoints+'.npz',
        allow_pickle=True
    )
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()
    
    # Correcting data lengths in 2D dataset
    for subject in dataset.subjects():
        assert subject in keypoints, \
            'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], \
                'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue
                
            for cam_idx in range(len(keypoints[subject][action])):
                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length, f'{keypoints[subject][action][cam_idx].shape[0]}, {mocap_length}'
                
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = \
                        keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])
    
    # Processing 2D dataset
    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])

    return keypoints, dataset, keypoints_metadata, \
        kps_left, kps_right, joints_left, joints_right

def fetch(subjects, dataset, keypoints, action_filter=None,
        subset=1, downsample=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
                
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                
            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])
                
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
    
    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    
    stride = downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    
    return out_camera_params, out_poses_3d, out_poses_2d

def load_model(args, dataset, num_joints):
    # Creating model
    filter_widths = [int(x) for x in args.architecture.split(',')]
    
    model_params = 0
    if args.task == 'length':
        model_train = BoneLengthModel(num_joints, args.in_chans, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional)
        model_valid = BoneLengthModel(num_joints, args.in_chans, num_layers=args.num_layers, dropout=0, bidirectional=args.bidirectional)
        for parameter in model_valid.parameters():
            model_params += parameter.numel()
    elif args.task == 'mix':
        num_frame = filter_widths[0] ** len(filter_widths)
        
        model_train = MixModel(num_joints, args.in_chans, filter_widths=filter_widths, num_layers=args.num_layers,
                dropout=args.dropout, channels=args.channels, bidirectional=args.bidirectional)
        model_valid = MixModel(num_joints, args.in_chans, filter_widths=filter_widths, num_layers=args.num_layers,
                dropout=0, channels=args.channels, bidirectional=args.bidirectional)
        for parameter in model_valid.parameters():
            model_params += parameter.numel()
    else:
        print('Wrong task name')
        exit()
    
    receptive_field = None
    pad = None
    causal_shift = 0
    if args.task != 'length':
        receptive_field = model_train.receptive_field()
        pad = (receptive_field - 1) // 2 # Padding on each side
        print('INFO: Receptive field: {} frames'.format(receptive_field))
    
    print('INFO: Trainable parameter count:', model_params)
    
    if torch.cuda.is_available():
        model_train = model_train.cuda()
        model_valid = model_valid.cuda()
    
    # Loading model
    if args.task == 'length':
        assert args.length_model == '', 'Direction model is not used in length predicting task'
    if args.task == 'mix':
        assert args.mix_model or (args.length_model and args.direction_model), \
            'Both length and direction pretrained model are needed in mixed task'
    if args.evaluate:
        assert args.task == 'mix', 'Only mixed model can be evaluated'
    
    checkpoint = {}
    if args.length_model and args.task == 'length':
        chk_filename = os.path.join(args.checkpoint, args.length_model)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model_train.load_state_dict(checkpoint['model_pos'])
        model_valid.load_state_dict(checkpoint['model_pos'])
    if args.task == 'mix':
        # load two pretrained models
        if args.mix_model:
            chk_filename = os.path.join(args.checkpoint, args.mix_model)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_train.load_state_dict(checkpoint['model_pos'], strict=True)
            model_valid.load_state_dict(checkpoint['model_pos'], strict=True)
        else:
            length_model_chk = os.path.join(args.checkpoint, args.length_model)
            direction_model_chk = os.path.join(args.checkpoint, args.direction_model)
            model_train._init_weights(length_model_chk, direction_model_chk)
            model_valid._init_weights(length_model_chk, direction_model_chk)
    
    return model_train, model_valid, checkpoint, pad, causal_shift, receptive_field

def train_mix(args, model_train, train_generator, optimizer):
    h_aug = model_train.init_hidden(args.batch_size)
    epoch_loss_3d_train = 0
    N = 0
    model_train.train()
    progress = tqdm(total=train_generator.num_batches)
    for cam, batch_3d, batch_2d, batch_len, batch_3d_chunk_aug in train_generator.next_epoch():
        cam = torch.from_numpy(cam.astype('float32'))
        inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
        inputs_len = torch.from_numpy(batch_len.astype('float32'))
        inputs_3d_chunk_aug = torch.from_numpy(batch_3d_chunk_aug.astype('float32'))
        if torch.cuda.is_available():
            cam = cam.cuda()
            inputs_3d = inputs_3d.cuda()
            inputs_2d = inputs_2d.cuda()
            inputs_len = inputs_len.cuda()
            inputs_3d_chunk_aug = inputs_3d_chunk_aug.cuda()
        
        inputs_3d[:, :, 0] = 0
        inputs_2d_chunk_aug = inputs_3d_chunk_aug
        
        optimizer.zero_grad()
        
        # Predict 3D poses
        if inputs_2d.shape[0] != args.batch_size:
            h_aug = model_train.init_hidden(inputs_2d.shape[0])
        predicted_3d_pos, predicted_dir = model_train(inputs_2d, inputs_2d_chunk_aug, h_aug, L=inputs_len)
        
        D, L = poses2bone_torch(inputs_3d.squeeze())
        
        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
        loss_3d_pos_d = mpjpe(predicted_dir, D)
        total_loss = loss_3d_pos + loss_3d_pos_d

        epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
        N += inputs_3d.shape[0]*inputs_3d.shape[1]
        progress.update(1)

        total_loss.backward()
        optimizer.step()
    return epoch_loss_3d_train / N

def train_bone_length(args, model_train, train_generator, optimizer):
    # h = model_train.init_hidden(args.batch_size)
    h_aug = model_train.init_hidden(args.batch_size)
    epoch_loss_3d_train = 0
    N = 0
    model_train.train()
    progress = tqdm(total=train_generator.num_batches)
    for cam, batch_bone_len, batch_3d in train_generator.next_epoch():
        cam = torch.from_numpy(cam.astype('float32'))
        inputs_bone_len = torch.from_numpy(batch_bone_len.astype('float32'))
        inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
        if torch.cuda.is_available():
            cam = cam.cuda()
            inputs_bone_len = inputs_bone_len.cuda()
            inputs_3d = inputs_3d.cuda()
        
        # evaluate inputs_2d_aug using projection
        inputs_2d = project_to_2d(inputs_3d, cam)
        
        optimizer.zero_grad()

        # Predict 3D poses
        if inputs_2d.shape[0] != args.batch_size:
            h_aug = model_train.init_hidden(inputs_2d.shape[0])
        predicted_3d_pos = model_train(inputs_2d, h_aug)
        
        del inputs_2d, cam, inputs_3d
        loss_3d_pos = boneLoss(predicted_3d_pos, inputs_bone_len)
        
        loss_bone_len = torch.mean(torch.abs(predicted_3d_pos - inputs_bone_len))
        
        total_loss = loss_3d_pos
        epoch_loss_3d_train += inputs_bone_len.shape[0]*inputs_bone_len.shape[1] * loss_bone_len.item()
        
        N += inputs_bone_len.shape[0]*inputs_bone_len.shape[1]
        progress.update(1)

        total_loss.backward()
        optimizer.step()
    return epoch_loss_3d_train / N

def eval_mix(args, model_train_dict, model_valid, test_generator, receptive_field):
    N = 0
    epoch_loss_3d_valid = 0
    epoch_loss_len_valid = 0
    epoch_loss_dir_valid = 0
    boneindex = [[16,15],[15,14],[13,12],[12,11], [10,9],[9,8],[8,7], [8,11],[8,14], [7,0], [3,2],[2,1],[6,5],[5,4], [1,0],[4,0]]
    
    progress = tqdm(total=test_generator.num_batches)
    with torch.no_grad():
        model_valid.load_state_dict(model_train_dict)
        model_valid.eval()

        # Evaluate on test set
        for _, batch_3d, batch_2d, batch_len, batch_len_gt, batch_dir in test_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_len = torch.from_numpy(batch_len.astype('float32'))
            inputs_len_gt = torch.from_numpy(batch_len_gt.astype('float32'))
            inputs_dir = torch.from_numpy(batch_dir.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                inputs_len = inputs_len.cuda()
                inputs_len_gt = inputs_len_gt.cuda()
                inputs_dir = inputs_dir.cuda()
            inputs_3d[:, :, 0] = 0

            # Predict 3D poses
            predicted_3d_pos, predicted_length, predicted_dir = model_valid(inputs_2d, L=inputs_len, eval=True)
            
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            loss_bone_len = torch.mean(torch.abs(predicted_length - inputs_len_gt))
            loss_dir = mpjpe(predicted_dir, inputs_dir)
            
            epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            epoch_loss_len_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_bone_len.item()
            epoch_loss_dir_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_dir.item()
            
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            progress.update(1)
            
        losses_3d_valid_ave = epoch_loss_3d_valid / N
        losses_len_valid_ave = epoch_loss_len_valid / N
        losses_dir_valid_ave = epoch_loss_dir_valid / N
    
    return losses_3d_valid_ave, losses_len_valid_ave, losses_dir_valid_ave

def eval_bone_length(args, model_train_dict, model_valid, test_generator, receptive_field, kps_left=None, kps_right=None):
    N = 0
    epoch_loss_3d_valid = 0
    epoch_loss_3d_valid_aug = 0

    progress = tqdm(total=test_generator.num_batches)
    with torch.no_grad():
        model_valid.load_state_dict(model_train_dict)
        model_valid.eval()

        # Evaluate on test set
        for cam, batch_bone_len, batch_2d, batch_bone_len_aug, batch_3d_aug in test_generator.next_epoch():
            cam = torch.from_numpy(cam.astype('float32'))
            inputs_bone_len = torch.from_numpy(batch_bone_len.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_bone_len_aug = torch.from_numpy(batch_bone_len_aug.astype('float32'))
            inputs_3d_aug = torch.from_numpy(batch_3d_aug.astype('float32'))
            
            if torch.cuda.is_available():
                cam = cam.cuda()
                inputs_bone_len = inputs_bone_len.cuda()
                inputs_2d = inputs_2d.cuda()
                inputs_bone_len_aug = inputs_bone_len_aug.cuda()
                inputs_3d_aug = inputs_3d_aug.cuda()
            
            # evaluate inputs_2d_aug using projection
            inputs_2d_aug = project_to_2d(inputs_3d_aug, cam)

            # Predict lengths
            predicted_bone_len = model_valid(inputs_2d)
            
            predicted_bone_len_aug = model_valid(inputs_2d_aug)
            for p in bone_symmetry:
                predicted_bone_len[:,p] = predicted_bone_len[:,p].mean(dim=1)[:,None]
            
            # loss
            loss_bone_len = torch.mean(torch.abs(predicted_bone_len - inputs_bone_len))
            loss_bone_len_aug = torch.mean(torch.abs(predicted_bone_len_aug - inputs_bone_len_aug))
            
            epoch_loss_3d_valid += inputs_bone_len.shape[0] * inputs_bone_len.shape[1] * loss_bone_len.item()
            epoch_loss_3d_valid_aug += inputs_bone_len.shape[0] * inputs_bone_len.shape[1] * loss_bone_len_aug.item()
            N += inputs_bone_len.shape[0] * inputs_bone_len.shape[1]
            progress.update(1)
            
        losses_3d_valid_ave = epoch_loss_3d_valid / N
        losses_3d_valid_aug_ave = epoch_loss_3d_valid_aug / N
    
    return losses_3d_valid_ave, losses_3d_valid_aug_ave

def evaluate_mix(model_valid, test_generator, receptive_field,
        action=None, return_predictions=False, use_trajectory_model=False,
        kps_left=None, kps_right=None, joints_left=None, joints_right=None):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_endjoint = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    
    with torch.no_grad():
        model_valid.eval()
        N = 0
        progress = tqdm(total=test_generator.num_batches)
        for cameras, batch, batch_2d, batch_len, _, batch_dir in test_generator.next_epoch():
            cameras = torch.from_numpy(cameras.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            inputs_len = torch.from_numpy(batch_len.astype('float32'))
            
            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip [:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]
            inputs_len_flip = inputs_len.clone()

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                inputs_len = inputs_len.cuda()
                inputs_len_flip = inputs_len_flip.cuda()
                inputs_3d = inputs_3d.cuda()
                # cameras = cameras.cuda()
            inputs_traj = inputs_3d[:, :, :1].clone()
            inputs_3d[:, :, 0] = 0

            predicted_3d_pos, predicted_rnn_lengths, _ = model_valid(inputs_2d, eval=True)
            predicted_3d_pos_flip, predicted_rnn_lengths_flip, _ = model_valid(inputs_2d_flip, eval=True)
            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]
            predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=0), dim=0,
                                          keepdim=True)
            # predicted_rnn_lengths = torch.mean(torch.cat((predicted_rnn_lengths, predicted_rnn_lengths_flip), dim=0), dim=0,
                                          # keepdim=True)
            
            torch.cuda.empty_cache()
            error = mpjpe(predicted_3d_pos, inputs_3d)
            
            predicted_dir, predicted_lengths = poses2bone_torch(predicted_3d_pos.squeeze())
            gt_dir, gt_lengths = poses2bone_torch(inputs_3d.squeeze())
            error_rnn_lengths = torch.mean(torch.abs(predicted_rnn_lengths - gt_lengths)).item()
            error_lengths = torch.mean(torch.abs(predicted_lengths - gt_lengths)).item()
            error_dirs = mpjpe(predicted_dir, gt_dir).item()
            
            # error_endjoint = mpjpe(predicted_3d_pos[:,:,[12, 13, 15, 16, 5, 6, 2, 3]], inputs_3d[:,:,[12, 13, 15, 16, 5, 6, 2, 3]])
            # epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * error_rnn_lengths

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            epoch_loss_3d_pos_endjoint += inputs_3d.shape[0]*inputs_3d.shape[1] * error_lengths
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            progress.update(1)

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            # epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)
            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * error_dirs

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e11 = (epoch_loss_3d_pos_endjoint / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #11 Error (MPJPE endjoint):', e11, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e11, e2, e3, ev
 
def evaluate_mix_aug(model_valid, test_generator, receptive_field,
        action=None, return_predictions=False, use_trajectory_model=False,
        kps_left=None, kps_right=None, joints_left=None, joints_right=None):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_endjoint = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    
    middle_frame = receptive_field//2
    with torch.no_grad():
        model_valid.eval()
        N = 0
        progress = tqdm(total=test_generator.num_batches)
        for cam, batch_bone_len, batch_3d in test_generator.next_epoch():
            cam = torch.from_numpy(cam.astype('float32'))
            inputs_len = torch.from_numpy(batch_bone_len.astype('float32'))
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            
            if torch.cuda.is_available():
                cam = cam.cuda()
                inputs_len = inputs_len.cuda()
                inputs_3d = inputs_3d.cuda()
            
            inputs_2d = project_to_2d(inputs_3d, cam)
            
            
            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip [:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]
            inputs_len_flip = inputs_len.clone()

            ##### convert size
            inputs_3d = inputs_3d - inputs_3d[:, :, :1]

            predicted_3d_pos, _ = model_valid(inputs_2d, inputs_len)
            predicted_3d_pos_flip, _ = model_valid(inputs_2d_flip, inputs_len_flip)
            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]
            predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                          keepdim=True)
            
            
            torch.cuda.empty_cache()
            inputs_3d = inputs_3d[:,middle_frame:middle_frame+1]
            error = mpjpe(predicted_3d_pos, inputs_3d)
            error_endjoint = mpjpe(predicted_3d_pos[:,:,[12, 13, 15, 16, 5, 6, 2, 3]], inputs_3d[:,:,[12, 13, 15, 16, 5, 6, 2, 3]])
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            epoch_loss_3d_pos_endjoint += inputs_3d.shape[0]*inputs_3d.shape[1] * error_endjoint.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            progress.update(1)

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e11 = (epoch_loss_3d_pos_endjoint / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #11 Error (MPJPE endjoint):', e11, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e11, e2, e3, ev

def fetch_actions(args, keypoints, dataset, actions, lengths):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    out_lengths = []

    for subject, action in actions:
        poses_2d = keypoints[subject][action]
        for i in range(len(poses_2d)): # Iterate across cameras
            out_poses_2d.append(poses_2d[i])
        
        cams = dataset.cameras()[subject]
        assert len(cams) == len(poses_2d), 'Camera count mismatch'
        for cam in cams:
            out_camera_params.append(cam['intrinsic'])

        poses_3d = dataset[subject][action]['positions_3d']
        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
        for i in range(len(poses_3d)): # Iterate across cameras
            out_poses_3d.append(poses_3d[i])
        
        lengths_tmp = lengths[subject][action]
        assert len(lengths_tmp) == len(poses_2d), 'Camera count mismatch'
        for i in range(len(lengths_tmp)):
            out_lengths.append(lengths_tmp[i])
            

    stride = args.downsample
    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    
    return out_camera_params, out_poses_3d, out_poses_2d, np.array(out_lengths)

def run_evaluation(
        args, model_valid, keypoints, dataset, lengths, receptive_field,
        actions, pad, action_filter=None, kps_left=None, kps_right=None,
        joints_left=None,joints_right=None):
    errors_p1 = []
    errors_p2 = []
    errors_p3 = []
    errors_vel = []
    
    # bone_lengths = np.load('data/bone_lengths.npz')['data']
    for action_key in actions.keys():
        if action_filter is not None:
            found = False
            for a in action_filter:
                if action_key.startswith(a):
                    found = True
                    break
            if not found:
                continue

        cam_act, poses_act, poses_2d_act, lengths_act = fetch_actions(
            args, keypoints, dataset, actions[action_key], lengths
        )
        
        gen = UnchunkedGenerator(
            cam_act, poses_act, poses_2d_act, lengths_act, pad=pad, augment=False,
            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right
        )
        e1, e11, e2, e3, ev = evaluate_mix(
            model_valid, gen, receptive_field, action_key,
            kps_left=kps_left, kps_right=kps_right,
            joints_left=joints_left, joints_right=joints_right
        )
        errors_p1.append(e1)
        errors_p2.append(e2)
        errors_p3.append(e3)
        errors_vel.append(ev)

    print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 2), 'mm')
    print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 2), 'mm')
    print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 2), 'mm')
    print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')
