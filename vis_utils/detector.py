import os
import sys
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
from .mpii_coco_h36m import coco_h36m
from .sort import Sort


CURSOR_UP_ONE = '\x1b[1A' 
ERASE_LINE = '\x1b[2K'

def search_bbox_id(bbox, bboxes):
    bbox_id = 0
    minimum = 100000
    for i in range(len(bboxes)):
        tmp = np.linalg.norm(bboxes[i]-bbox)
        if tmp < minimum:
            minimum = tmp
            bbox_id = i
    return bbox_id

def detect2d(video_path, save_video=False):
    pose_model = MMPoseInferencer(
        pose2d='td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192',
        det_model='yolox_l_8x8_300e_coco',
        device='cuda:0',
        det_cat_ids=[0]
    )
    
    cap = cv2.VideoCapture(video_path)
    if save_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter('output/result_2d.mp4', fourcc, fps, size)

    N = int(cap.get(7))
    people_sort = Sort()
    ret = []
    for i in range(N):
        success, frame = cap.read()
        if not success:
            break
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_generator = pose_model(img, return_vis=True)
        result = next(result_generator)
        
        kpts = []
        bboxes = []
        for j in range(min(5, len(result['predictions'][0]))): # Choose first 5 poses
            kpts.append(result['predictions'][0][j]['keypoints'])
            bboxes.append(result['predictions'][0][j]['bbox'][0])
        bboxes = np.array(bboxes, dtype=np.float32)
        people_track = people_sort.update(bboxes)
        ret_bbox = people_track[-1, :-1]
        bbox_id = search_bbox_id(ret_bbox, bboxes)
        ret.append(kpts[bbox_id])
        
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)
        
        if save_video:
            videoWriter.write(result['visualization'][0].astype(np.uint8))
    print('')
    
    cap.release()
    if save_video:
        videoWriter.release()
        cv2.destroyAllWindows()
    return coco_h36m(np.array(ret))

def detect2d_image(image_path, save_image=False):
    pose_model = MMPoseInferencer(
        pose2d='td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192',
        det_model='yolox_l_8x8_300e_coco',
        device='cuda:0',
        det_cat_ids=[0]
    )
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_generator = pose_model(img, return_vis=True)
    result = next(result_generator)
    
    # np.savez_compressed('test.npz', data=result)
    
    if save_image:
        cv2.imwrite('output/result_img.png', result['visualization'][0].astype(np.uint8))
    print(len(result['predictions']))
    print(len(result['predictions'][0]))
    print(result['predictions'][0][0].keys())
    return coco_h36m(np.array([result['predictions'][0][0]['keypoints']]))[0]
    