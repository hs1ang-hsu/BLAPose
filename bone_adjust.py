import os
import argparse
import pickle
import numpy as np
import pandas as pd

from utils.camera import *
from utils.bone_utils import *
from utils.h36m_dataset import Human36mDataset
from utils.loss import *


def arg_parse():
    parser = argparse.ArgumentParser('Adjusting bone lengths of predicted poses.')
    parser.add_argument('--act', type=str, default='3D_prediction/action_dict.pkl', help='target action dict')
    parser.add_argument('--pred', type=str, default='3D_prediction/prediction.pkl', help='target predicted poses')
    parser.add_argument('--gt', type=str, default='data/data_3d_h36m.npz', help='groundtruth poses')
    parser.add_argument('--bone', type=str, default='data/lengths_dict.npz', help='bone dict for adjustment')
    parser.add_argument('--print_action', action='store_true', help='print action-wise error or not')
    
    parser.add_argument('--causal', action='store_true', help='lengths predicted with causal model')
    parser.add_argument('--plot_frame_error', action='store_true', help='draw error versus input frames')
    
    args = parser.parse_args()
    
    if args.plot_frame_error:
        args.causal = True

    return args

def length_inconsistency(pred):
    import matplotlib.pyplot as plt
    plt.rc('axes', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rcParams["font.family"] = "Times New Roman"
    
    pred = np.linalg.norm(pred[:,16] - pred[:,15], axis=1) * 1000
    ours = pickle.load(open('test.pkl', 'rb'))[:,-1] * 1000
    x = np.arange(len(ours))
    
    colors = np.array(plt.cm.tab20(range(20)))
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(x, pred, label='Chen et al.', color=colors[2])
    ax.plot(x, ours, label='Ours', color=colors[4])
    
    ax.set_ylabel('Bone length (mm)')
    ax.set_xlabel('Frame')
    ax.set_ylim(200,270)
    ax.legend()
    
    plt.show()

def load_gt(gt_path):
    dataset = Human36mDataset(gt_path)
    # preprocess dataset
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(
                    anim['positions'],
                    R=cam['orientation'],
                    t=cam['translation']
                )
                
                # Remove global offset
                pos_3d -= pos_3d[:, :1]
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d
    return dataset

def evalaute(actions_dict, pred_pose_list, bone_length_dict, dataset, f, args):
    # evaluation
    idx = 0
    result = []
    result_sub = {'S9': {'error': 0, 'N': 0}, 'S11': {'error': 0, 'N': 0}}
    
    error_length_avg = []
    error_direction_avg = []
    error_pose_avg = []
    error_pmpjpe_avg = []
    error_mpjve_avg = []
    
    error_length_adjusted_avg = []
    error_joint_adjusted_avg = []
    error_pose_adjusted_avg = []
    error_adjusted_pmpjpe_avg = []
    error_adjusted_mpjve_avg = []
    
    data_reformat = []
    for actions in actions_dict:
        error_length_action = 0
        error_direction_action = 0
        error_pose_action = 0
        error_pmpjpe_action = 0
        error_mpjve_action = 0
        
        error_length_adjusted_action = np.zeros(16)
        error_joint_adjusted_action = np.zeros(17)
        error_pose_adjusted_action = 0
        error_adjusted_pmpjpe_action = 0
        error_adjusted_mpjve_action = 0
        
        N = 0
        for subject, action in actions_dict[actions]:
            for i in range(4):
                # prepare data
                gt_pose = torch.from_numpy(dataset[subject][action]['positions_3d'][i]) # (N, 17, 3)
                
                pred_pose = torch.from_numpy(pred_pose_list[idx]) # (N, 17, 3)
                # length_inconsistency(pred_pose)
                # exit()
                
                if args.causal:
                    bone_length = torch.from_numpy(bone_length_dict[subject][action][i]) # (N, 16)
                    assert len(bone_length.shape) == 2, "Please run length_model_inference.py with --causal flag."
                    if args.plot_frame_error:
                        bone_length = bone_length[f-1:f]
                else:
                    bone_length = torch.from_numpy(bone_length_dict[subject][action][i]).view(1,16) # (1, 16)
                
                
                # non-adjusted error
                error = mpjpe(pred_pose, gt_pose).item()
                error_procrustes = p_mpjpe(pred_pose.numpy(), gt_pose.numpy())
                error_velocity = mean_velocity_error(pred_pose.numpy(), gt_pose.numpy())
                predicted_dir, predicted_lengths = poses2bone_torch(pred_pose)
                gt_dir, gt_lengths = poses2bone_torch(gt_pose)
                error_lengths = torch.mean(torch.abs(gt_lengths - predicted_lengths)).item()
                error_dirs = mpjpe(predicted_dir, gt_dir).item()
                
                # adjusted error
                if args.causal and not args.plot_frame_error:
                    pred_pose_adjusted = bones2poses_torch(predicted_dir.unsqueeze(1), bone_length).squeeze()
                    error_lengths_adjusted = torch.mean(torch.abs(gt_lengths - bone_length), dim=0).numpy()
                else:
                    pred_pose_adjusted = bones2poses_torch(predicted_dir.unsqueeze(0), bone_length).squeeze()
                    error_lengths_adjusted = torch.mean(torch.abs(gt_lengths[:1] - bone_length), dim=0).numpy()
                error_adjusted = mpjpe(pred_pose_adjusted, gt_pose).item()
                error_adjusted_procrustes = p_mpjpe(pred_pose_adjusted.numpy(), gt_pose.numpy())
                error_adjusted_velocity = mean_velocity_error(pred_pose_adjusted.numpy(), gt_pose.numpy())
                
                # statistics
                error_length_action += pred_pose.shape[0] * error_lengths
                error_direction_action += pred_pose.shape[0] * error_dirs
                error_pose_action += pred_pose.shape[0] * error
                error_pmpjpe_action += pred_pose.shape[0] * error_procrustes
                error_mpjve_action += pred_pose.shape[0] * error_velocity
                
                error_length_adjusted_action += pred_pose.shape[0] * error_lengths_adjusted
                error_joint_adjusted_action += pred_pose.shape[0] * torch.mean(torch.norm(pred_pose - gt_pose, dim=-1), dim=0).numpy()
                error_pose_adjusted_action += pred_pose.shape[0] * error_adjusted
                error_adjusted_pmpjpe_action += pred_pose.shape[0] * error_adjusted_procrustes
                error_adjusted_mpjve_action += pred_pose.shape[0] * error_adjusted_velocity
                
                result_sub[subject]['error'] += pred_pose.shape[0] * np.mean(error_lengths_adjusted) * 1000
                result_sub[subject]['N'] += pred_pose.shape[0]
                N += pred_pose.shape[0]
                idx += 1
        
        error_length_avg.append(error_length_action / N * 1000)
        error_direction_avg.append(error_direction_action / N * 1000)
        error_pose_avg.append(error_pose_action / N * 1000)
        error_pmpjpe_avg.append(error_pmpjpe_action / N * 1000)
        error_mpjve_avg.append(error_mpjve_action / N * 1000)
        
        error_length_adjusted_avg.append(error_length_adjusted_action / N * 1000)
        error_joint_adjusted_avg.append(error_joint_adjusted_action / N * 1000)
        error_pose_adjusted_avg.append(error_pose_adjusted_action / N * 1000)
        error_adjusted_pmpjpe_avg.append(error_adjusted_pmpjpe_action / N * 1000)
        error_adjusted_mpjve_avg.append(error_adjusted_mpjve_action / N * 1000)
        
        result.append([
            actions, round(error_pose_avg[-1],1), round(error_pmpjpe_avg[-1],1), round(error_mpjve_avg[-1],1),
            round(error_length_avg[-1],1), round(error_direction_avg[-1],1),
            round(error_pose_adjusted_avg[-1],1), round(error_adjusted_pmpjpe_avg[-1],1), round(error_adjusted_mpjve_avg[-1],1),
            round(np.mean(error_length_adjusted_avg[-1]),1)
        ])
        if args.print_action and not args.plot_frame_error:
            print('----'+actions+'----')
            print('Protocol #1 Error (MPJPE):', error_pose_avg[-1], 'mm')
            print('Protocol #2 Error (P-MPJPE):', error_pmpjpe_avg[-1], 'mm')
            print('Velocity Error (MPJVE):', error_mpjve_avg[-1], 'mm')
            print('Length Error:', error_length_avg[-1], 'mm')
            print('Direction Error:', error_direction_avg[-1])
            print('Adjusted Protocol #1 Error (MPJPE):', error_pose_adjusted_avg[-1], 'mm')
            print('Adjusted Protocol #2 Error (P-MPJPE):', error_adjusted_pmpjpe_avg[-1], 'mm')
            print('Adjusted Velocity Error (MPJVE):', error_adjusted_mpjve_avg[-1], 'mm')
            print('Adjusted Length Error:', np.mean(error_length_adjusted_avg[-1]), 'mm')
            print('----------')
    
    # pickle.dump(data_reformat, open('prediction.pkl', 'wb'))
    
    if not args.plot_frame_error:
        print('')
        print('Protocol #1 Error (MPJPE) action-wise average:', round(np.mean(error_pose_avg), 1), 'mm')
        print('Protocol #2 Error (P-MPJPE) action-wise average:', round(np.mean(error_pmpjpe_avg), 1), 'mm')
        print('Velocity Error (MPJVE):', round(np.mean(error_mpjve_avg), 1), 'mm')
        print('Length Error action-wise average:', round(np.mean(error_length_avg), 1), 'mm')
        print('Direction Error action-wise average:', round(np.mean(error_direction_avg), 1))
        
        print('Adjusted Protocol #1 Error (MPJPE) action-wise average:', round(np.mean(error_pose_adjusted_avg), 1), 'mm')
        print('Adjusted Protocol #2 Error (P-MPJPE) action-wise average:', round(np.mean(error_adjusted_pmpjpe_avg), 1), 'mm')
        print('Velocity Error (MPJVE):', round(np.mean(error_adjusted_mpjve_avg), 1), 'mm')
        print('Adjusted Length Error action-wise average:', round(np.mean(error_length_adjusted_avg), 1), 'mm')
        
        result = np.array(result)
        result = result[result[:, 0].argsort()]
        result[[-2,-1]] = result[[-1,-2]] # swap Walk and WalkTogether
        result = np.array(result).T
        df = pd.DataFrame(result)
        df.to_csv("result.csv")
    else:
        return round(np.mean(error_pose_adjusted_avg), 1), round(np.mean(error_adjusted_pmpjpe_avg), 1), \
            round(np.mean(error_adjusted_mpjve_avg), 1), round(np.mean(error_length_adjusted_avg), 1)


if __name__ == '__main__':
    args = arg_parse()
    
    # load dataset
    actions_dict = pickle.load(open(args.act, 'rb'))
    pred_pose_list = pickle.load(open(args.pred, 'rb'))
    bone_length_dict = np.load(args.bone, allow_pickle=True)['lengths'].item()
    dataset = load_gt(args.gt)
    
    if args.plot_frame_error:
        x = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500]
        result = {'x': x, 'mpjpe': [], 'pmpjpe': [], 'mpjve': [], 'le': []}
        for i in x:
            print("-----", i, "-----")
            frame_mpjpe, frame_pmpjpe, frame_mpjve, frame_le = evalaute(actions_dict, pred_pose_list, bone_length_dict, dataset, i, args)
            result['mpjpe'].append(frame_mpjpe)
            result['pmpjpe'].append(frame_pmpjpe)
            result['mpjve'].append(frame_mpjve)
            result['le'].append(frame_le)
        
        pickle.dump(result, open('frame_error.pkl', 'wb'))
    else:
        evalaute(actions_dict, pred_pose_list, bone_length_dict, dataset, -1, args)
