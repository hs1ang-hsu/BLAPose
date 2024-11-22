import argparse
import torch
import os
import numpy as np
import cv2
from tqdm import tqdm

from utils.camera import normalize_screen_coordinates, camera_to_world
from utils.bone_utils import *
from utils.model_mix import MixModel

import length_model_inference
from vis_utils.detector import detect2d
import vis_utils.visualization as visualization

rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)

ckpt_root = 'checkpoint'
output_root = 'output'

def arg_parse():
    parser = argparse.ArgumentParser('Inference.')
    # model
    parser.add_argument('--length_ckpt', type=str, default='biGRU_all.bin', help='checkpoint')
    parser.add_argument('-layer', '--num_layers', default=1, type=int, help='number of layers')
    parser.add_argument('-bi', '--bidirectional', action='store_true', help='bidirectional GRU or not')
    parser.add_argument('--causal', action='store_true', help='Causal inference')
    
    parser.add_argument('--finetuned_ckpt', type=str, default='videopose_finetuned.bin', help='checkpoint')
    parser.add_argument('-arc', '--architecture', default='3,3,3,3,3', type=str, help='filter widths separated by comma')
    parser.add_argument('-ch', '--channels', default=1024, type=int, help='number of channels in convolution layers')
    
    # inference
    parser.add_argument('--video', type=str, required=True, help='path to the video')
    parser.add_argument('--poses', type=str, default="", help='path to predicted poses')
    
    args = parser.parse_args()
    
    if args.bidirectional and args.causal:
        print('The causal mode only allows GRU-models')
        exit()

    return args

def load_finetuned_model(args):
    filter_widths = [int(x) for x in args.architecture.split(',')]
    model = MixModel(17, 2, filter_widths=filter_widths, num_layers=args.num_layers,
            dropout=0, channels=args.channels, bidirectional=args.bidirectional)
    chk_filename = os.path.join(ckpt_root, args.finetuned_ckpt)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_pos'], strict=True)
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def inference_finetuned_model(model, inputs_2d):
    receptive_field = model.receptive_field()
    pad = (receptive_field - 1) // 2
    inputs_2d = np.expand_dims(np.pad(inputs_2d, ((pad, pad), (0, 0), (0, 0)), 'edge'), axis=0)
    inputs_2d = torch.from_numpy(inputs_2d).float()
    
    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()
        pred_pose_camera, pred_len, pred_dir = model(inputs_2d, eval=True)
    pred_pose_world = camera_to_world(pred_pose_camera.cpu().numpy(), R=rot, t=0)
    return pred_pose_world

def inference(args):
    video = args.video
    cap = cv2.VideoCapture(video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # evaluate 2d keypoints
    re_kpts = detect2d(video, save_video=False)
    norm_kpts = normalize_screen_coordinates(re_kpts, w=width, h=height)
    
    if args.poses != "":
        ''' inference with pre-evaluated 3D poses '''
        # evaluate bone length
        length_model = length_model_inference.load_model(os.path.join(ckpt_root, args.length_ckpt), args)
        norm_kpts = torch.from_numpy(norm_kpts).unsqueeze(0).float()
        bone_length = length_model_inference.inference_wild(length_model, norm_kpts, args.causal)
        bone_length = torch.from_numpy(bone_length).view(1,16)
        
        # load 3D poses
        # pred_pose.shape = (# of frames, 17, 3)
        pred_pose = np.load(args.poses)['poses']
        pred_pose = torch.from_numpy(pred_pose)
    
        # adjust
        predicted_dir, predicted_lengths = poses2bone_torch(pred_pose)
        pred_pose_adjusted = bones2poses_torch(predicted_dir.unsqueeze(0), bone_length)
        pred_pose_adjusted = pred_pose_adjusted.numpy()
        pred_pose_adjusted[0][:, :, 2] -= np.amin(pred_pose_adjusted[0][:, :, 2])
    else:
        ''' inference with fine-tuned models '''
        # load fine-tuned model
        finetuned_model = load_finetuned_model(args)
        
        # inference
        pred_pose_adjusted = inference_finetuned_model(finetuned_model, norm_kpts)
        pred_pose_adjusted[0][:, :, 2] -= np.amin(pred_pose_adjusted[0][:, :, 2])
    
    # visualize
    out_name = os.path.basename(args.video).split('.')[0] + '_3D_pose.mp4'
    visualization.draw(pred_pose_adjusted, args.video, os.path.join(output_root, out_name), re_kpts=re_kpts)


if __name__ == '__main__':
    args = arg_parse()
    os.makedirs(output_root, exist_ok=True)
    inference(args)