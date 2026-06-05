import os
import re
import json
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import imageio.v2 as imageio
import argparse
from skimage.color import rgb2gray

dense_volume = {
                    'OCTV1L_0001_Mode3D' : [289, 546],
                    'OCTV7L_04' : [196, 294],
                    'OCTV9L_03' : [316, 404],
                    'OCTV10L_02' : [212, 336],
                    'OCTV11L_01' : [346, 418]
               }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PNG slices to nnUNet format"
    )
    
    parser.add_argument("--train_img_dir", type=str, default="Cochlear_Fibrosis_OCT_Dataset/ALL_raw_volumes",
                        help="Path to training directory")
    
    parser.add_argument("--output_base", type=str, default="3d_fullres_exp/dense",
                        help="Base output directory")
    parser.add_argument("--task_name", type=str, default="Task001_Cochlea",
                        help="nnUNet task name")
    
    parser.add_argument("--description", type=str,  default="OCT Cochlea Segmentation",
                        help="Dataset description")
    
    return parser.parse_args()

def extract_case_id(filename):
    """
    Extract volume ID by removing slice-specific parts.
    Works for:
    - OCTV1L_0001_Mode3D_page_289.png
    - OCTV7L_04_0196.png
    """
    name = filename.replace(".png", "")
    
    # Case 1: Mode3D pattern
    if "page_" in name:
        return name.split("_Mode3D")[0]
    
    # Case 2: simple numeric slice index at end
    parts = name.split("_")
    return "_".join(parts[:-1])


def extract_slice_index(filename):
    name = filename.replace(".png", "")
    
    # Case 1: page_XXX
    match = re.search(r'page_(\d+)', name)
    if match:
        return int(match.group(1))
    
    # Case 2: last numeric part
    parts = name.split("_")
    return int(parts[-1])


def load_image(path):
    img = imageio.imread(path)

    if img.ndim == 3:
        img = rgb2gray(img)

    return img.astype(np.float32)


def load_mask(path):
    mask = imageio.imread(path)

    if mask.ndim == 3:
        mask = mask[..., 0]

    return mask.astype(np.uint8)


def save_nifti(volume, path):
    affine = np.eye(4)
    nifti = nib.Nifti1Image(volume, affine)
    nib.save(nifti, path)


def extract_volume_number(case_id):
    match = re.search(r'OCT(?:V)?(\d+)L', case_id)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Cannot extract number from {case_id}")


def group_files(image_dir, image_path):
    files = glob(os.path.join(image_dir, "*.jpg"))

    files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

    files = files[dense_volume[image_path][0] : dense_volume[image_path][1]+1]
    
        
    images = [load_image(f) for f in files]
    volume = np.stack(images, axis=0)
    
    return volume

def convert(args):
    OUTPUT_BASE = args.output_base
    DATASET_NAME = args.dataset_name
    DATASET_DIR = os.path.join(OUTPUT_BASE, DATASET_NAME)

    os.makedirs(DATASET_DIR, exist_ok=True)
    
    imagesTr = os.path.join(DATASET_DIR, "imagesTr")
    
    os.makedirs(imagesTr, exist_ok=True)
    
    print("Processing training data...")

    for vid_dir in dense_volume.keys():

        vid_path = os.path.join(args.train_img_dir,vid_dir)

        train_volume = group_files(vid_path,vid_dir)
        
        training_entries = []

        img_path = os.path.join(imagesTr, f"{vid_dir}_0000.nii.gz")
        
        print(img_path)
        print(train_volume.shape)

        train_volume = np.transpose(train_volume, (1, 2, 0))

        save_nifti(train_volume, img_path)
        
    print("Conversion complete!")

def main():
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()