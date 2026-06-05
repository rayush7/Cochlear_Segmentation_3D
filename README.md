# 🦻 3D Segmentation of Cochlear Implants using nnUNet

Go through documentation of nnUNetv1 : https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1/documentation 

---

# Directions
Install nnUNetv1 : https://github.com/MIC-DKFZ/nnUNet 

---

# Export Paths

Create 3 folders `nnUNet_raw_data_base`, `nnUNet_preprocessed`, `nnUNet_results` and export following paths.

Ensure the `Task001_Cochlea` is defined for cochlear and directory structure follows nnUNet conventions.

```bash
export nnUNet_raw_data_base="/home/ayushrai/Image_Segmentation_3D/nnUNet_raw_data_base"
export nnUNet_preprocessed="/home/ayushrai/Image_Segmentation_3D/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/ayushrai/Image_Segmentation_3D/nnUNet_results"
```

---

# To Create NIfTI Files

```bash
python convert_to_nnunet.py \
  --train_img_dir Cochlear_Fibrosis_OCT_Dataset/split/train_images \
  --train_mask_dir Cochlear_Fibrosis_OCT_Dataset/split/train_masks \
  --test_img_dir Cochlear_Fibrosis_OCT_Dataset/split/test_images \
  --test_mask_dir Cochlear_Fibrosis_OCT_Dataset/split/test_masks
```

---

# To Check NIfTI Files

```python
import nibabel as nib

nii = nib.load(<PATH TO NIFTY FILE>)
data = nii.get_fdata()

print(data.shape)
```

---

# Verify Dataset

```bash
nnUNet_plan_and_preprocess -t 2 --verify_dataset_integrity
```

---

# Training

## Train Fold 1

```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 1 1 --npz
```

---

# Inference

## Standard Prediction

```bash
nnUNet_predict \
-i /home/ayushrai/Image_Segmentation_3D/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Cochlea/imagesTs/ \
-o /home/ayushrai/Image_Segmentation_3D/nnUNet_results/ \
-t 1 \
-m 3d_fullres \
-chk model_best \
-f 1
```

## Fold 5 Prediction

```bash
nnUNet_predict \
-i /home/ayushrai/Image_Segmentation_3D/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Cochlea/imagesTs/ \
-o /home/ayushrai/Image_Segmentation_3D/nnUNet_results/nnUNet/3d_fullres/Task001_Cochlea/nnUNetTrainerV2__nnUNetPlansv2.1/fold_5/ \
-t 1 \
-m 3d_fullres \
-chk model_best \
-f 5
```

## Fold 1 Prediction

```bash
nnUNet_predict \
-i /home/ayushrai/Image_Segmentation_3D/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Cochlea/imagesTs/ \
-o /home/ayushrai/Image_Segmentation_3D/nnUNet_results/nnUNet/3d_fullres/Task001_Cochlea/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/ \
-t 1 \
-m 3d_fullres \
-chk model_best \
-f 1
```

## Dense Volume Prediction

```bash
nnUNet_predict -i dense/ -o dense_pred/ -t 1 -m 3d_fullres -chk model_best -f 1
```

### Reference

https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/inference_example_Prostate.md

---

# Evaluate Folder

```bash
nnUNet_evaluate_folder \
-ref nnUNet_raw_data_base/nnUNet_raw_data/Task001_Cochlea/labelsTs/ \
-pred nnUNet_results/nnUNet/3d_fullres/Task001_Cochlea/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/ \
-l 1 2 3
```

---

# Create Dense Volume
```bash
python create_dense_volume.py
```

---

# Visualization
Ensure the prediction are stored in ```3d_fullres_exp/dense_pred``` folder.

```bash
python visualise.py
```
