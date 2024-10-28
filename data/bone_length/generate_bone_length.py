# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os.path as osp
import argparse
import numpy as np
import torch
from tqdm import tqdm
from glob import glob

import smplx

# seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


bones = [[0,1],[1,2],[2,3], [0,4],[4,5],[5,6], [0,7],[7,8],[8,9],[9,10], [8,11],[11,12],[12,13], [8,14],[14,15],[15,16]]
bone_child = [0,1,2, 0,4,5, 0,7,8,9, 8,11,12, 8,14,15]
bone_parent = [1,2,3, 4,5,6, 7,8,9,10, 11,12,13, 14,15,16]
bone_symmetry = [[0,3],[1,4],[2,5], [10,13],[11,14],[12,15]]
mapping = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]

def bone_length_tune(bone_lengths):
    for p in bone_symmetry:
        bone_lengths[p] = np.mean(bone_lengths[p])
    return bone_lengths

def visualize(vertices, joints, model):
    import pyrender
    import trimesh
    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(vertices, model.faces,
                               vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    scene = pyrender.Scene()
    scene.add(mesh)

    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    tfs[:, :3, 3] = joints
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)

    pyrender.Viewer(scene, use_raymond_lighting=True)

def generate_smpl_lengths(gender, num):
    model = smplx.create('body_models', model_type='smpl',
                         gender=gender, use_face_contour=False,
                         num_betas=10,
                         num_expression_coeffs=10,
                         ext='pkl')
    print(model)


    res = []
    J_regressor = torch.from_numpy(np.load('J_regressor_h36m_correct.npy')).float()
    for i in tqdm(range(num)):
        betas = torch.randn([1, model.num_betas], dtype=torch.float32)
        
        # run model
        output = model(betas=betas, return_verts=True)
        
        # evaluate joints
        h36m_joints = J_regressor @ output.vertices.squeeze()
        joints = h36m_joints.detach().cpu().numpy().squeeze()
        bone_lengths = np.linalg.norm(joints[bone_parent] - joints[bone_child], axis=1)
        res.append(bone_length_tune(bone_lengths))
        
        # visualize
        # vertices = output.vertices.detach().cpu().numpy().squeeze()
        # joints = output.joints.detach().cpu().numpy().squeeze()
        # joints = np.concatenate((pose3d, joints))
        # visualize(vertices, joints, model)
        # exit()
    
    return res
    

def generate_smpl(args):
    # generate bone lengths
    gender = args.gender
    if gender == 'mix':
        male = generate_smpl_lengths('male', int(args.amount/2))
        female = generate_smpl_lengths('female', int(args.amount/2))
        res = male + female
    else:
        res = generate_smpl_lengths('neutral', args.amount)
    res = np.array(res)
    
    # save data if not applying data alignment
    if not args.align:
        np.savez_compressed(f'../bone_lengths_smpl_{gender}.npz', data=res)
        return
    
    # load h36m data
    h36m_data = np.load('../data_3d_h36m.npz', allow_pickle=True)['positions_3d'].item()
    h36m = []
    for sub in h36m_data.keys():
        bone_lengths = []
        for act in h36m_data[sub].keys():
            data_3d = h36m_data[sub][act][:,mapping]
            bone_lengths.append(
                np.mean(
                    np.linalg.norm(data_3d[:,bone_parent] - data_3d[:,bone_child], axis=2),
                axis=0)
            )
        h36m.append(np.mean(bone_lengths, axis=0))
    h36m = np.array(h36m)
    np.savez_compressed(f'../bone_lengths_h36m.npz', data=h36m)
    
    # alignment
        # Align with training set only
    shift = np.mean(h36m[:-2], axis=0) - np.mean(res, axis=0)
    smpl_train = res + shift
    np.savez_compressed(f'../bone_lengths_smpl_{gender}_train.npz', data=smpl_train)
        # Align with the entire dataset
    shift = np.mean(h36m, axis=0) - np.mean(res, axis=0)
    smpl_all = res + shift
    np.savez_compressed(f'../bone_lengths_smpl_{gender}_all.npz', data=smpl_all)


def arg_parse():
    parser = argparse.ArgumentParser('Generate synthetic bone lengths.')
    parser.add_argument('-g', '--gender', type=str, default='neutral', choices=['mix','neutral'], help='specify generated genders')
    parser.add_argument('-amt', '--amount', type=int, default=200000, help='number of generated bone lengths')
    parser.add_argument('--align', action="store_true", help='whether to apply the data alignment.')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = arg_parse()
    generate_smpl(args)
