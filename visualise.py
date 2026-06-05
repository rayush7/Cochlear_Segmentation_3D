# Generate Overlay Visualizations for Nifty Volumes

import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


DENSE_DIR = "3d_fullres_exp/dense"
PRED_DIR = "3d_fullres_exp/dense_pred"
VIS_DIR = "3d_fullres_exp/vis"


CLASS_COLORS = {
    1: "tab:red",
    2: "tab:blue",
    3: "tab:green"
}


os.makedirs(VIS_DIR, exist_ok=True)


nifti_files = sorted(glob.glob(os.path.join(DENSE_DIR, "*.nii.gz")))


for image_path in nifti_files:

    basename = os.path.basename(image_path)

    # Prediction filename should match image filename
    pred_path = os.path.join(PRED_DIR, basename)

    if not os.path.exists(pred_path):
        print(f"Prediction missing for: {basename}")
        continue

    print(f"Processing: {basename}")

    # Create output folder inside vis/
    volume_name = basename.replace(".nii.gz", "")

    output_folder = os.path.join(VIS_DIR, volume_name)
    os.makedirs(output_folder, exist_ok=True)

    # Slice range dictionary
    dense_volume = {
        'OCTV1L_0001_Mode3D_0000': [289, 546],
        'OCTV7L_04_0000': [196, 294],
        'OCTV9L_03_0000': [316, 404],
        'OCTV10L_02_0000': [212, 336],
        'OCTV11L_01_0000': [346, 418]
    }

    if volume_name not in dense_volume:
        print(f"Slice range not found for: {volume_name}")
        continue

    start_slice, end_slice = dense_volume[volume_name]

    # Read images
    image = sitk.ReadImage(image_path)
    pred = sitk.ReadImage(pred_path)

    image_np = sitk.GetArrayFromImage(image)
    pred_np = sitk.GetArrayFromImage(pred)

    # Normalize image volume
    image_np = image_np.astype(np.float32)
    image_np = (image_np - image_np.min()) / (
        image_np.max() - image_np.min() + 1e-8
    )

    num_slices = image_np.shape[0]

    # Clamp slice range safely
    start_slice = max(0, start_slice)
    end_slice = min(end_slice, num_slices - 1)

    for slice_idx in range(0, num_slices):

        img = image_np[slice_idx]
        pred_slice = pred_np[slice_idx]

        classes = np.unique(pred_slice)
        classes = classes[classes != 0]

        fig, ax = plt.subplots(figsize=(7, 7))

        ax.imshow(img, cmap="gray")

        # Overlay masks
        for cls in classes:

            mask = (pred_slice == cls)

            if np.any(mask):

                color = CLASS_COLORS.get(cls, "yellow")

                ax.imshow(
                    np.ma.masked_where(~mask, mask),
                    cmap=plt.cm.colors.ListedColormap([color]),
                    alpha=0.6
                )

        ax.axis("off")

        # Add legend only if classes exist
        if len(classes) > 0:
            legend_patches = [
                mpatches.Patch(
                    color=CLASS_COLORS.get(c, "yellow"),
                    label=f"Class {c}"
                )
                for c in classes
            ]

            ax.legend(
                handles=legend_patches,
                loc="lower left",
                frameon=False,
                labelcolor="white",
                handlelength=2,
                fontsize=12
            )

        save_path = os.path.join(
            output_folder,
            f"slice_{start_slice + slice_idx:03d}.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close(fig)

    print(f"Saved overlays to: {output_folder}")


print("Done.")