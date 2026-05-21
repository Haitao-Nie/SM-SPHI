import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class HyperspectralDataset(Dataset):
    """
    Dataset that only provides hyperspectral datacubes X.
    Forward measurement is handled by the encoder (forward model).
    """

    def __init__(self, root_dir):
        """
        Parameters
        ----------
        root_dir : str
            Path to dataset split, e.g.
            ICVL_64/train
            ICVL_64/val
            ICVL_64/test
        """
        super().__init__()

        self.root_dir = root_dir
        self.files = sorted(
            [f for f in os.listdir(root_dir) if f.endswith(".mat")]
        )

        assert len(self.files) > 0, f"No .mat files found in {root_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fname = self.files[index]
        path = os.path.join(self.root_dir, fname)

        # --------------------------------------------------
        # Load hyperspectral cube
        # --------------------------------------------------
        with h5py.File(path, "r") as f:
            if "patch" in f:
                cube = f["patch"][:]     # (H, W, C)
            elif "cube" in f:
                cube = f["cube"][:]      # (H, W, C)
            else:
                raise KeyError(
                    f"{fname} does not contain 'patch' or 'cube'"
                )

        # --------------------------------------------------
        # HWC -> CHW
        # --------------------------------------------------
        cube = cube.astype(np.float32)
        cube = np.transpose(cube, (2, 0, 1))   # (C, H, W)

        # # --------------------------------------------------
        # # Normalization (per-cube)
        # # --------------------------------------------------
        # max_val = cube.max()
        # if max_val > 0:
        #     cube = cube / max_val

        X = torch.from_numpy(cube)

        return X
