# %%
import pandas as pd
import os
from pathlib import Path
import dicom2nifti
import nibabel as nib
import glob
import imageio
import numpy as np

output_dir = Path(os.getenv("DATA_DIR"))
meta_dir = Path(os.getenv("METADATA"))
metadata = pd.read_csv(meta_dir)
subjectid_to_volidx = {}

# %%
# Filter out Whole Body PET scans
pet = metadata[metadata.apply(lambda x: x['Modality'] == "PT", axis=1)]
correct_path = lambda x: '/'.join(x.split("\\"))
file_paths = [correct_path(i) for i in (pet["File Location"])]


def create_nifti(row):
    corrected_file_path = lambda x: '/'.join(x.split("\\"))
    dir_path = corrected_file_path(row["File Location"])
    subjectid_to_volidx[row["Subject ID"]] = len(subjectid_to_volidx)
    vol_dir = output_dir / "nifti" / f"volume_{str(len(subjectid_to_volidx)).zfill(6)}"
    vol_dir.mkdir(parents=True, exist_ok=True)
    dicom2nifti.convert_directory(meta_dir.parent / dir_path, vol_dir)


# pet.apply(lambda row: create_nifti(row), axis=1)

# %%
#
# save slices from nifti
def save_slices(img, out_dir, slice_axis=1):

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = img.swapaxes(2, slice_axis)

    for i in range(img.shape[slice_axis]):
        slice = np.squeeze(np.expand_dims(tmp[:, :, 1], 2).swapaxes(2, 1), 1)
        slice = slice.astype(np.uint8)
        imageio.imwrite(f"{out_dir}/slice_{str(i).zfill(6)}.jpeg", slice)


i = 0
for vol in (output_dir/"nifti").glob("*"):
    print(vol.name)
    if os.path.isdir(vol):
        file = next(vol.glob("*.nii.gz"))
        img = nib.load(file).get_fdata()
        print(img.shape)
        save_slices(img=img,
                    out_dir=(output_dir / "train" / vol.name),
                    slice_axis=1)


    if i == 3:
        break

    i += 1
