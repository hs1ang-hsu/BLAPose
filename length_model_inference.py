import argparse
import torch
import os
import numpy as np
from tqdm import tqdm

from main import load_dataset, fetch
from utils.generators import UnchunkedGeneratorBoneLengths
from utils.bone_utils import bone_symmetry
from utils.model_length import BoneLengthModel 

ckpt_root = 'checkpoint'
dataset_root = 'data'


def arg_parse():
    parser = argparse.ArgumentParser('Generating skeleton demo.')
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('--ckpt', type=str, default='epoch_best.bin', help='checkpoint')
    parser.add_argument('--subjects', type=str, default='S9,S11', help='subjects')
    
    parser.add_argument('-layer', '--num_layers', default=1, type=int, help='number of layers')
    parser.add_argument('-bi', '--bidirectional', action='store_true', help='bidirectional GRU or not')
    
    parser.add_argument('--causal', action='store_true', help='Causal inference')
    args = parser.parse_args()
    
    if args.bidirectional and args.causal:
        print('The causal mode only allows GRU-models')
        exit()

    return args

def h36m_dataset(args):
    # Loading dataset
    keypoints, dataset, keypoints_metadata, \
        kps_left, kps_right, joints_left, joints_right = load_dataset(args)
    
    # Preprocessing dataset
    subjects_test = args.subjects.split(',')
    action_filter = None 
    
    cameras_valid, poses_valid, poses_valid_2d = fetch(
            subjects_test, dataset, keypoints, action_filter, downsample=1
        )
    
    test_generator = UnchunkedGeneratorBoneLengths(
        cameras_valid, poses_valid, poses_valid_2d, augment=False,
        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right
    )
    return test_generator

def load_model(chk_path, args):
    model = BoneLengthModel(17, 2, num_layers=args.num_layers, dropout=0, bidirectional=args.bidirectional)
    checkpoint = torch.load(chk_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_pos'])
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def inference(model_valid, test_generator):
    N = 0
    epoch_loss_3d_valid = 0
    pred_lengths = []

    progress = tqdm(total=test_generator.num_batches)
    with torch.no_grad():
        model_valid.eval()
        # Evaluate on test set
        for cam, batch_bone_len, batch_2d, _, _ in test_generator.next_epoch():
            cam = torch.from_numpy(cam.astype('float32'))
            inputs_bone_len = torch.from_numpy(batch_bone_len.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            
            if torch.cuda.is_available():
                cam = cam.cuda()
                inputs_bone_len = inputs_bone_len.cuda()
                inputs_2d = inputs_2d.cuda()
            
            # Predict lengths
            if args.causal:
                bone_lengths = []
                h_0 = [model_valid.init_hidden(inputs_2d.shape[0]), model_valid.init_hidden(inputs_2d.shape[0])]
                for i in range(inputs_2d.shape[1]):
                    predicted_bone_len, h_0 = model_valid(inputs_2d[:,i:i+1], h_0, causal=True)
                    for p in bone_symmetry:
                        predicted_bone_len[:,p] = predicted_bone_len[:,p].mean(dim=1)[:,None]
                    bone_lengths.append(predicted_bone_len[0].cpu().numpy())
                    
                    h_0[0] = h_0[0].unsqueeze(1)
                    h_0[1] = h_0[1].unsqueeze(1)
                pred_lengths.append(np.array(bone_lengths))
            else:
                predicted_bone_len = model_valid(inputs_2d)
                for p in bone_symmetry:
                    predicted_bone_len[:,p] = predicted_bone_len[:,p].mean(dim=1)[:,None]
                pred_lengths.append(predicted_bone_len.squeeze().cpu().numpy())
            
            
            # loss
            loss_bone_len = torch.mean(torch.abs(predicted_bone_len - inputs_bone_len))
            
            epoch_loss_3d_valid += inputs_bone_len.shape[0] * inputs_bone_len.shape[1] * loss_bone_len.item()
            N += inputs_bone_len.shape[0] * inputs_bone_len.shape[1]
            progress.update(1)
    
    # save lengths
    dataset = np.load('data/data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)['positions_2d'].item()
    lengths = {}
    cnt = 0
    subjects_test = args.subjects.split(',')
    for subject in subjects_test:
        lengths[subject] = {}
        for action in dataset[subject].keys():
            tmp = []
            act = action.split(' ')[0]
            for i in range(4):
                tmp.append(pred_lengths[cnt])
                cnt += 1
            lengths[subject][action] = tmp.copy()
    if args.causal:
        np.savez_compressed('data/lengths_dict_causal.npz', lengths=lengths)
        np.savez_compressed('data/lengths_list_causal.npz', lengths=np.array(pred_lengths, dtype=object))
    else:
        np.savez_compressed('data/lengths_dict.npz', lengths=lengths)
        np.savez_compressed('data/lengths_list.npz', lengths=pred_lengths)


if __name__ == '__main__':
    args = arg_parse()
    
    model = load_model(os.path.join(ckpt_root, args.ckpt), args)
    test_generator = h36m_dataset(args)
    
    inference(model, test_generator)