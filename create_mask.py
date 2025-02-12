import os
import nibabel as nib
import numpy as np
import glob
from typing import Optional


# ------------------------------Directory-Structure------------------------------------------
"""
-Proj
    -derivatives 
        -dual regression
        -fmriprep
            -sub-001
            -sub-002
            -sub-003
            -sub-004
                -anat
                -figures
                -log
                -func
                    -sub-004_task-rest_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz
                    ...
            ...
        -group_ICA
            -group_mask_sum.nii.gz (be created by SumMaskCreator)
    -scripts
"""
# -------------------------------------------------------------------------------------------


class SumMask:
    def __init__(self, proj_path, opath=None):

        self.proj_path = proj_path
        self.fmriprep_path = os.path.join(self.proj_path, 'derivatives', 'fmriprep')

        # Set up 'opath' input conditions.
        self.opath = opath if opath else os.path.join(self.proj_path, 'derivatives', 'group_ICA')
        os.makedirs(self.opath, exist_ok=True)
        self.opath = os.path.join(self.opath, "group_mask_sum.nii.gz")

        # BIDS-ids, (bids-id1, bids-id2, bids-id3, ...)
        self.bids_ids = self.extract_bids_ids()
        # The data format of mask_paths is [(bids_id1, path1), (bids_id2, path2), ...]
        self.mask_ids_paths = sorted(self.combine_mask_paths())

        # Parameters for saving Nifti file
        self.affine = None
        self.header = None

    def extract_bids_ids(self) -> tuple:
        # Get all bids-ids which meet bids format pattern
        contents = glob.glob(os.path.join(self.fmriprep_path, 'sub-*'))
        print(contents)

        # Ensure that only directory level name will be record
        bids_dir_paths = [dir_name for dir_name in contents if os.path.isdir(dir_name)]

        # Extract bids-ids from directory path
        bids_ids = sorted([os.path.basename(p) for p in bids_dir_paths])

        return tuple(bids_ids)

    def combine_mask_paths(self) -> Optional[tuple]:
        """Based on BIDS format, using bids_ids to combine mask file's paths"""
        # Get all mask files' path
        # The data format of mask_paths is [(bids_id1, path1), (bids_id2, path2)...]
        mask_ids_paths = [(
            bids_id,
            os.path.join(
                self.fmriprep_path,
                f'{bids_id}',
                'func',
                f'{bids_id}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz'))
            for bids_id in self.bids_ids]

        # Check whether all mask files' path exist.
        absent_mask_paths = []

        for bids_id, path in mask_ids_paths:
            if not os.path.exists(path):
                absent_mask_paths.append(bids_id)

        if not absent_mask_paths:
            return tuple(mask_ids_paths)
        else:
            raise FileNotFoundError('The mask file of following IDs could not be found:\n'
                                    + '\n'.join(absent_mask_paths))

    def check_mask_shapes_consistency(self) -> bool:
        """To check whether all mask matrix are same shape."""
        shapes = set()

        for _, mask_path in self.mask_ids_paths:
            image = nib.load(mask_path)
            mask_data = image.get_fdata()
            shapes.add(mask_data.shape)

        if len(shapes) == 1:
            print(f'All mask have same shape:{shapes}')
            return True
        else:
            return False

    def get_sum_mask(self) -> np.ndarray:
        """Get the sum mask by add all mask matrix together."""
        # Using one individual mask to get fundamental information (shape, affine, header)
        sample_mask_path = self.mask_ids_paths[0][1]

        img = nib.load(sample_mask_path)
        sample_mask = img.get_fdata()
        shape = sample_mask.shape
        print(f'shape:{shape}')

        self.affine = img.affine
        self.header = img.header

        # Create a base mask based on mask shape.
        mask_shape = list(shape)
        sum_mask = np.zeros(mask_shape)
        # Add all mask to base mask
        for i, (_, mask_path) in enumerate(self.mask_ids_paths):

            image = nib.load(mask_path)
            mask_data = image.get_fdata()
            sum_mask += mask_data

        return sum_mask

    def save_sum_mask(self, sum_mask: np.ndarray) -> None:
        if self.affine is not None and self.header is not None:
            sum_mask_nifti = nib.Nifti1Image(sum_mask, affine=self.affine, header=self.header)
            nib.save(sum_mask_nifti, self.opath)
            print(f'Sum mask file saved to: {self.opath}')

    def count_ids(self) -> int:
        return len(self.bids_ids)

    def process(self) -> None:
        if self.check_mask_shapes_consistency():
            sum_mask = self.get_sum_mask()
            self.save_sum_mask(sum_mask)
        else:
            raise (ValueError(f'The mask shapes do not match each other, there are 0 or more than 1 types shape!'))


class LooseAndMask:
    def __init__(self, sum_mask_path: str, subj_n: int, bin_thresh_rate: float, opath=None):

        self.sum_mask_path = sum_mask_path
        self.subj_n = subj_n
        self.bin_thresh_rate = bin_thresh_rate

        self.opath = opath if opath else os.path.dirname(self.sum_mask_path)
        self.opath = os.path.join(self.opath, "binary_mask.nii.gz")

        self.thresh = round((self.subj_n * bin_thresh_rate), 2)

        self.affine = None
        self.header = None

    def create_binary_mask(self):
        img = nib.load(self.sum_mask_path)
        sum_mask = img.get_fdata()
        binary_mask = (sum_mask > self.thresh).astype(np.uint8)

        self.affine = img.affine
        self.header = img.header

        return binary_mask

    def save_binary_mask(self, binary_mask: np.ndarray) -> None:
        binary_mask_nifti = nib.Nifti1Image(binary_mask, affine=self.affine, header=self.header)
        nib.save(binary_mask_nifti, self.opath)


if __name__ == '__main__':
    # ---------------------------------Parameters------------------------------------
    project_path = '/Users/username/Proj/data_fMRIprep'

    # If output path is None, it will be set as default path as in directory structure
    # Make sure dirs listing in path are exist.
    sum_output_path = None
    binary_output_path = None

    # looseAND Mask bin thresh rate
    binarization_threshold_rate = 0.8
    # -------------------------------------------------------------------------------

    # ---------------------------------Executions------------------------------------
    # Execution 1 : Create Group Mask Sum
    summ_mask = SumMask(proj_path=project_path, opath=sum_output_path)
    summ_mask.process()
    print(type(summ_mask.get_sum_mask))

    # Execution 2 : Create looseAND mask
    # Create LooseAndMask base on different condition of output path.
    if sum_output_path:
        if os.path.exists(sum_output_path):
            loose_and_mask = LooseAndMask(sum_mask_path=sum_output_path,
                                          subj_n=summ_mask.count_ids(),
                                          bin_thresh_rate=binarization_threshold_rate,
                                          opath=binary_output_path)
        else:
            raise FileNotFoundError('The sum mask path does not exist! Make sure to create sum mask first!')
    elif not sum_output_path:
        sum_output_path = os.path.join(project_path, 'derivatives', 'group_ICA', "group_mask_sum.nii.gz")
        if os.path.exists(sum_output_path):
            loose_and_mask = LooseAndMask(sum_mask_path=sum_output_path,
                                          subj_n=summ_mask.count_ids(),
                                          bin_thresh_rate=binarization_threshold_rate,
                                          opath=binary_output_path)
        else:
            raise FileNotFoundError("'The sum mask path does not exist! Error in code please check 'execution 2'")

    if isinstance(loose_and_mask, LooseAndMask):
        bin_mask = loose_and_mask.create_binary_mask()
        loose_and_mask.save_binary_mask(bin_mask)
