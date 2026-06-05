import os
import re
import json
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import imageio
import argparse
from skimage.color import rgb2gray

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PNG slices to nnUNet format"
    )
    
    parser.add_argument("--train_img_dir", type=str, default="/home/ayushrai/Cochlear_Segmentation_3D/Cochlear_Fibrosis_OCT_Dataset/split/train_images",
                        help="Path to training images directory")
    parser.add_argument("--train_mask_dir", type=str, default="/home/ayushrai/Cochlear_Segmentation_3D/Cochlear_Fibrosis_OCT_Dataset/split/train_masks",
                        help="Path to training masks directory")
    parser.add_argument("--test_img_dir", type=str, default="/home/ayushrai/Cochlear_Segmentation_3D/Cochlear_Fibrosis_OCT_Dataset/split/test_images",
                        help="Path to test images directory")
    parser.add_argument("--test_mask_dir", type=str, default="/home/ayushrai/Cochlear_Segmentation_3D/Cochlear_Fibrosis_OCT_Dataset/split/test_masks",
                        help="Path to test masks directory")
    
    parser.add_argument("--output_base", type=str, default="nnUNet_raw_data_base/nnUNet_raw_data",
                        help="Base output directory")
    parser.add_argument("--task_name", type=str, default="Task001_Cochlea",
                        help="nnUNet task name")
    
    parser.add_argument("--dataset_name", type=str, default="CochleaFibrosis",
                        help="Dataset name in dataset.json")
    parser.add_argument("--description", type=str,
                        default="OCT Cochlea Segmentation",
                        help="Dataset description")
    parser.add_argument("--modality", type=str, default="OCT",
                        help="Imaging modality name")
    
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


def group_files(image_dir, mask_dir=None):
    files = glob(os.path.join(image_dir, "*.png"))
    
    groups = {}
    
    for f in files:
        name = os.path.basename(f)
        case_id = extract_case_id(name)
        groups.setdefault(case_id, []).append(f)
    
    volumes = []
    
    #for case_id in :
    for case_id in sorted(groups.keys(), key=extract_volume_number):
        file_list = groups[case_id]
        
        file_list_sorted = sorted(
            file_list,
            key=lambda x: extract_slice_index(os.path.basename(x))
        )
        
        images = [load_image(f) for f in file_list_sorted]
        volume = np.stack(images, axis=0)

        if mask_dir:
            masks = []
            for f in file_list_sorted:
                mask_path = os.path.join(mask_dir, os.path.basename(f))
        
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Missing mask for {f}")
        
                masks.append(load_mask(mask_path))
    
            mask_volume = np.stack(masks, axis=0)
        else:
            mask_volume = None
        
        volumes.append((case_id, volume, mask_volume))
    
    return volumes

def convert(args):
    OUTPUT_BASE = args.output_base
    TASK_NAME = args.task_name
    TASK_DIR = os.path.join(OUTPUT_BASE, TASK_NAME)

    os.makedirs(TASK_DIR, exist_ok=True)
    
    imagesTr = os.path.join(TASK_DIR, "imagesTr")
    labelsTr = os.path.join(TASK_DIR, "labelsTr")
    imagesTs = os.path.join(TASK_DIR, "imagesTs")
    labelsTs = os.path.join(TASK_DIR, "labelsTs")
    
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    os.makedirs(labelsTs, exist_ok=True)
    
    print("Processing training data...")
    train_volumes = group_files(args.train_img_dir, args.train_mask_dir)
    
    print("Processing test data...")
    test_volumes = group_files(args.test_img_dir, args.test_mask_dir)
    
    training_entries = []
    testing_entries = []
    
    # Save training
    for i, (case_id, vol, mask) in enumerate(tqdm(train_volumes)):
        volume_id = case_id.split('_')[0]
        #case_name = f"la_{i:03d}"

        case_name = f"{volume_id}_{i:03d}"
        
        img_path = os.path.join(imagesTr, f"{case_name}_0000.nii.gz")
        lbl_path = os.path.join(labelsTr, f"{case_name}.nii.gz")

        print(vol.shape)
        print(mask.shape)

        vol = np.transpose(vol, (1, 2, 0))
        mask = np.transpose(mask, (1, 2, 0))
        
        save_nifti(vol, img_path)
        save_nifti(mask, lbl_path)
        
        training_entries.append({
            "image": f"./imagesTr/{case_name}.nii.gz",
            "label": f"./labelsTr/{case_name}.nii.gz"
        })
    
    # Save test
    for i, (case_id, vol, mask) in enumerate(tqdm(test_volumes)):

        volume_id = case_id.split('_')[0]
        #case_name = f"la_{i:03d}"
        case_name = f"{volume_id}_{i:03d}"
        #case_name = f"la_{i+len(train_volumes):03d}"

        img_path = os.path.join(imagesTs, f"{case_name}_0000.nii.gz")
        lbl_path = os.path.join(labelsTs, f"{case_name}.nii.gz")

        vol = np.transpose(vol, (1, 2, 0))
        mask = np.transpose(mask, (1, 2, 0))
        
        save_nifti(vol, img_path)
        save_nifti(mask, lbl_path)

        #save_nifti(vol, img_path)

        testing_entries.append(f"./imagesTs/{case_name}.nii.gz")
    
    # dataset.json
    dataset = {
        "name": args.dataset_name,
        "description": args.description,
        "tensorImageSize": "3D",
        "modality": {"0": args.modality},
        "labels": {
            "0": "background",
            "1": "class1",
            "2": "class2",
            "3": "class3"
        },
        "numTraining": len(training_entries),
        "numTest": len(test_volumes),
        "training": training_entries,
        # "test": [
        #     f"./imagesTs/la_{i+len(train_volumes):03d}_0000.nii.gz"
        #     for i in range(len(test_volumes))
        # ]
        "test" : testing_entries
    }
    
    with open(os.path.join(TASK_DIR, "dataset.json"), "w") as f:
        json.dump(dataset, f, indent=4)
    
    print("Conversion complete!")


def main():
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()