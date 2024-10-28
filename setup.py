import gdown
import os

## Dataset
bone_length_data = 'https://drive.google.com/drive/folders/10CLfvL5izh-V_IXPlbVYhDSwklXuMkQh'
gdown.download_folder(bone_length_data)

os.makedirs('data/', exist_ok=True)
os.rename('./synthetic_lengths/bone_lengths_h36m.npz', './data/bone_lengths_h36m.npz')
os.rename('./synthetic_lengths/bone_lengths_smpl_neutral_all.npz', './data/bone_lengths_smpl_neutral_all.npz')
os.rename('./synthetic_lengths/bone_lengths_smpl_neutral_train.npz', './data/bone_lengths_smpl_neutral_train.npz')
os.rmdir('./synthetic_lengths')

## Checkpoint
checkpoint = 'https://drive.google.com/drive/folders/1dwCqZs8HCqlijyHXD0KYOnKqGsGuoi8E'
gdown.download_folder(checkpoint)


## 3D lifting model prediction
prediction = 'https://drive.google.com/drive/folders/16N9hnOAkmmnfZGix_kTFKyjoQbfYBF4v'
gdown.download_folder(prediction)
