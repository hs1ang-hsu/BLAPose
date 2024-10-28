# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str, metavar='LIST',
                        help='training subjects separated by comma')
    parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str, metavar='LIST', help='test subjects separated by comma')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--checkpoint-frequency', default=40, type=int, metavar='N',
                        help='create a checkpoint every N epochs')
    ''' model and bone length data '''
    parser.add_argument('--length_model', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--length_aug_type', default='smpl', type=str, metavar='NAME',
                        choices=['smpl','random_unif', 'random_norm_train', 'random_norm_all'],
                        help='type of augmentation on bone lengths')
    parser.add_argument('-bla', '--bone_length_aug', default='data/bone_lengths_smpl_neutral_all.npz', type=str, metavar='NAME',
                        help='Bone length data for augmentation')
    parser.add_argument('-bll', '--bone_length_list', default='data/lengths_list.npz', type=str, metavar='NAME',
                        help='predicted bone lengths in list format')
    parser.add_argument('-bld', '--bone_length_dict', default='data/lengths_dict.npz', type=str, metavar='NAME',
                        help='predicted bone lengths in dictionary format')
    parser.add_argument('--direction_model', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--mix_model', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    ''' settings '''
    parser.add_argument('-t', '--task', default='', type=str, metavar='NAME', choices=['length','mix'],
                        help='task to train (length or direction or mix)')
    parser.add_argument('--evaluate', action='store_true', help='checkpoint to evaluate (file name)')
    parser.add_argument('--render', action='store_true', help='visualize a particular video')
    parser.add_argument('--by-subject', action='store_true', help='break down error by subject (on evaluation)')
    parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')
    parser.add_argument('--seed', default=5731, type=int, metavar='N', help='random seed')


    # Model arguments
    parser.add_argument('-s', '--stride', default=1, type=int, metavar='N', help='chunk size to use during training')
    parser.add_argument('-e', '--epochs', default=60, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=1024, type=int, metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('-l', '--chunk_len', default=512, type=int, metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('--segment_shift', default=64, type=int, metavar='N', help='segment_shift of data sequences.')
    parser.add_argument('-drop', '--dropout', default=0.25, type=float, metavar='P', help='dropout probability')
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.95, type=float, metavar='LR', help='learning rate decay per epoch')
    parser.add_argument('-no-da', '--no-data-augmentation', dest='data_augmentation', action='store_false',
                        help='disable train-time flipping')
    parser.add_argument('-no-tta', '--no-test-time-augmentation', dest='test_time_augmentation', action='store_false',
                        help='disable test-time flipping')
    parser.add_argument('-arc', '--architecture', default='3,3,3,3', type=str, metavar='LAYERS', help='filter widths separated by comma')
    parser.add_argument('-ich', '--in-chans', default=2, type=int, help='number of hypotheses')
    parser.add_argument('-layer', '--num-layers', default=1, type=int, help='number of layers')
    parser.add_argument('-ch', '--channels', default=1024, type=int, metavar='N', help='number of channels in convolution layers')
    parser.add_argument('-bi', '--bidirectional', action='store_true', help='bidirectional GRU or not')

    # Experimental
    parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor (semi-supervised)')
    parser.add_argument('--no-eval', action='store_true', help='disable epoch evaluation while training (small speed-up)')
    
    
    parser.set_defaults(data_augmentation=True)
    parser.set_defaults(test_time_augmentation=True)

    args = parser.parse_args()
    # Check invalid configuration
    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()
    
    
    return args