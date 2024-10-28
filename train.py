# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import os
import sys
from time import time
import errno
import pickle

from utils.arguments import parse_args
from utils.generators import ChunkedGenerator, UnchunkedGenerator
from main import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = parse_args()
try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # Loading dataset
    keypoints, dataset, keypoints_metadata, kps_left, \
        kps_right, joints_left, joints_right = load_dataset(args)
    
    # Preprocessing dataset
    subjects_train = args.subjects_train.split(',')
    subjects_test = args.subjects_test.split(',')
    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)
    
    cameras_valid, poses_valid, poses_valid_2d = fetch(
            subjects_test, dataset, keypoints, action_filter,
            downsample=args.downsample
        )
    if not args.evaluate:
        cameras_train, poses_train, poses_train_2d = fetch(
            subjects_train, dataset, keypoints, action_filter,
            downsample=args.downsample
        )
    
    # Loading model
    num_joints = keypoints_metadata['num_joints']
    model_train, model_valid, checkpoint, pad, causal_shift, receptive_field = \
        load_model(args, dataset, num_joints)

    # Creating dataset generator
    if args.task == 'length':
        bone_lengths = np.load(args.bone_length_aug)['data']
        test_generator = UnchunkedGeneratorBoneLengths(
            cameras_valid, poses_valid, poses_valid_2d, augment=False,
            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right,
            bone_lengths_aug=bone_lengths, length_aug=args.length_aug_type
        )
        if not args.evaluate:
            train_generator = ChunkedGeneratorBoneLengths(
                args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, bone_lengths,
                args.chunk_len, segment_shift=args.segment_shift, shuffle=True, augment=args.data_augmentation,
                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right,
                length_aug=args.length_aug_type
            )
            print('INFO: Training on {} frames'.format(train_generator.num_frames()))
    elif args.task == 'mix':
        lengths = np.load(args.bone_length_list)['lengths']
        
        test_generator = UnchunkedGenerator(
            cameras_valid, poses_valid, poses_valid_2d, lengths[600:], pad=pad, augment=False, 
            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right
        )
        print('INFO: Testing on {} frames'.format(test_generator.num_frames()))
        if not args.evaluate:
            train_generator = ChunkedGenerator(
                args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, lengths[:600],
                args.stride, pad=pad, shuffle=True, augment=args.data_augmentation,
                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right
            )
            print('INFO: Training on {} frames'.format(train_generator.num_frames()))

    if not args.evaluate:
        # Initializing parameters
        lr = args.learning_rate
        optimizer = optim.Adam(model_train.parameters(), lr=lr, amsgrad=True)
        lr_decay = args.lr_decay
        scheduler = ExponentialLR(optimizer, lr_decay)

        losses_3d_train = []
        losses_3d_valid = []
        losses_3d_valid_aug = []
        epoch = 0
        initial_momentum = 0.001
        final_momentum = 0.001
    
        resume = False
        if args.task != 'mix' and (args.length_model or args.direction_model):
            resume = True
        elif args.task == 'mix' and args.mix_model:
            resume = True
        
        if resume:
            epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
                train_generator.set_random_state(checkpoint['random_state'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']

        # Training
        loss_min = 200
        while epoch < args.epochs:
            start_time = time()
            
            # train
            if args.task == 'length':
                epoch_loss_3d = train_bone_length(args, model_train, train_generator, optimizer)
            elif args.task == 'mix':
                epoch_loss_3d = train_mix(args, model_train, train_generator, optimizer)
            losses_3d_train.append(epoch_loss_3d)

            # validate
            if not args.no_eval:
                if args.task == 'length':
                    losses_3d_valid_ave, losses_3d_valid_aug_ave = eval_bone_length(
                        args, model_train.state_dict(), model_valid, test_generator, receptive_field, kps_left, kps_right
                    )
                    losses_3d_valid_aug.append(losses_3d_valid_aug_ave)
                elif args.task == 'mix':
                    losses_3d_valid_ave, losses_3d_valid_len, losses_3d_valid_dir = eval_mix(
                        args, model_train.state_dict(), model_valid, test_generator, receptive_field
                    )
                    losses_3d_valid_aug.append(np.array([np.round(losses_3d_valid_len,6), np.round(losses_3d_valid_dir,4)]))
                losses_3d_valid.append(losses_3d_valid_ave)
            
            elapsed = (time() - start_time)/60
            print(f'[{epoch + 1}] time {elapsed:.2f} lr {lr:.6f} 3d_train {losses_3d_train[-1] * 1000 if len(losses_3d_train)>0 else 0:.4f} ' + \
                    f'3d_valid {losses_3d_valid[-1] * 1000 if len(losses_3d_valid)>0 else 0:.4f} ' + \
                    f'3d_valid_aug {losses_3d_valid_aug[-1] * 1000 if len(losses_3d_valid_aug)>0 else 0}')
            
            # Decay learning rate exponentially
            epoch += 1
            if optimizer.param_groups[0]['lr'] >= 1e-6:
                scheduler.step() 
            lr = optimizer.param_groups[0]['lr']
        
            # Decay BatchNorm momentum
            momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
            model_train.set_bn_momentum(momentum)
            
            # Saving the best result
            if losses_3d_valid[-1]*1000 < loss_min:
                chk_path = os.path.join(args.checkpoint, 'epoch_best.bin')
                print('Saving checkpoint to', chk_path)
                torch.save({
                    'epoch': epoch,
                    'lr': lr,
                    'random_state': train_generator.random_state(),
                    'optimizer': optimizer.state_dict(),
                    'model_pos': model_train.state_dict()
                }, chk_path)
                loss_min = losses_3d_valid[-1]*1000
        
            # Save checkpoint if necessary
            if epoch % args.checkpoint_frequency == 0:
                chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
                print('Saving checkpoint to', chk_path)
                
                torch.save({
                    'epoch': epoch,
                    'lr': lr,
                    'random_state': train_generator.random_state(),
                    'optimizer': optimizer.state_dict(),
                    'model_pos': model_train.state_dict()
                }, chk_path)
            
            chk_path = os.path.join(args.checkpoint, 'final.bin')
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_train.state_dict()
            }, chk_path)
            
            pickle.dump(
                {'train':losses_3d_train, 'valid':losses_3d_valid, 'valid_aug':losses_3d_valid_aug},
                open('checkpoint/result.pkl', 'wb')
            )
            
            # Save training curves after every epoch, as .png images (if requested)
            if args.export_training_curves:
                if 'matplotlib' not in sys.modules:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                
                plt.figure()
                epoch_x = np.arange(0, len(losses_3d_train)) + 1
                plt.plot(epoch_x, np.array(losses_3d_train)*1000, '--', color='C0')
                plt.plot(epoch_x, np.array(losses_3d_valid)*1000, color='C1')
                plt.legend(['3d train', '3d valid (eval)'])
                plt.ylabel('Bone Length Error (mm)')
                plt.xlabel('Epoch')
                plt.xlim((1, epoch+1))
                plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

                plt.close('all')
    
    # Evaluate
    if args.evaluate:
        print('Evaluating...')
        all_actions = {}
        all_actions_by_subject = {}
        for subject in subjects_test:
            if subject not in all_actions_by_subject:
                all_actions_by_subject[subject] = {}

            for action in dataset[subject].keys():
                action_name = action.split(' ')[0]
                if action_name not in all_actions:
                    all_actions[action_name] = []
                if action_name not in all_actions_by_subject[subject]:
                    all_actions_by_subject[subject][action_name] = []
                all_actions[action_name].append((subject, action))
                all_actions_by_subject[subject][action_name].append((subject, action))
        
        lengths = np.load(args.bone_length_dict, allow_pickle=True)['lengths'].item()
        run_evaluation(
            args, model_valid, keypoints, dataset, lengths, receptive_field,
            all_actions, pad, action_filter,
            kps_left, kps_right, joints_left, joints_right
        )