import subprocess as sp
from tqdm import tqdm
import cv2
import numpy as np

import matplotlib.pyplot as plt

colors = ['k','k','r','r','r','b','b','b','k','k','b','b','b','r','r','r']
bones = [
    [8,9],[9,10], [8,14],[14,15],[15,16], [8,11],[11,12],[12,13],
    [8,7],[7,0], [0,4],[4,5],[5,6], [0,1],[1,2],[2,3]
]

fig_ske = plt.figure()
ax_ske = fig_ske.add_subplot(111, projection='3d')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
ax_ske.view_init(elev=15, azim=66)
ax_ske.axis('off')

def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)

def plotSke(pts):
    ax_ske.clear()
    radius = 1.7
    lims = [[-radius,radius], [-radius,radius], [0,radius]]
    ax_ske.set_xlim(lims[0])
    ax_ske.set_ylim(lims[1])
    ax_ske.set_zlim(lims[2])
    
    for i, p in enumerate(bones):
        xs = [pts[p[0]][0], pts[p[1]][0]]
        ys = [pts[p[0]][1], pts[p[1]][1]]
        zs = [pts[p[0]][2], pts[p[1]][2]]
        ax_ske.plot(xs, ys, zs, marker='o', linewidth=3, markersize=2, color=colors[i])
    
    ax_ske.set_proj_type('persp')
    fig_ske.canvas.draw()
    image = np.frombuffer(fig_ske.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig_ske.canvas.get_width_height()[::-1] + (3,))
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def draw(result, video_path, name, re_kpts=None):
    # result.shape = (# of results, # of frames, 17, 3)
    # re_kpts.shape = (# of frames, 17, 2)
    fps = get_fps(video_path)
    k = len(result)
    n = len(result[0])
    
    print('Drawing...')
    frames = []
    cap = cv2.VideoCapture(video_path)
    for i in tqdm(range(n)):
        success, frame = cap.read()
        
        fig, axes = plt.subplots(1, k+1, figsize=(10+8*k,10), constrained_layout=True)
        if re_kpts is not None:
            for kpt in re_kpts[i]:
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), radius=5, color=(0, 0, 255), thickness=-1)
        axes[0].imshow(frame)
        for p in range(k):
            img = plotSke(result[p][i])
            axes[p+1].imshow(img)
        
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
                    
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)

    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))
    print('Generating video...')
    for i in range(len(frames)):
        video.write(frames[i])

    cv2.destroyAllWindows()
    video.release()