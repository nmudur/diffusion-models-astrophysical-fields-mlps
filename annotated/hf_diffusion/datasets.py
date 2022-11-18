import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CustomTensorDataset(Dataset):
    r"""Dataset wrapping tensors.
    memmap is assumed to have shape N_trainxHxW

    Each sample will be retrieved by indexing tensors along the first dimension.
    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, memmap, transforms=None, labels_path=None) -> None:
        #self.tensor = tensor
        self.memmap = memmap
        self.transforms = transforms
        self.image_size = memmap[0].shape[-1]
        if labels_path is not None:
            self.conditional=True
            self.labels = np.loadtxt(labels_path, dtype=np.float32)
        else:
            self.conditional=False

    def __getitem__(self, index):
        #pre = self.tensor[index]
        pre = torch.from_numpy(np.array(self.memmap[index]).astype(np.float32)).view((1, self.image_size, self.image_size))
        if self.transforms:
           pre= self.transforms(pre)
        if self.conditional:
             return pre, self.labels[(index%9000)//15] #WARNING: only valid for the Nx=64 train cosmology dset
        else:
            return pre


    def __len__(self):
        return self.memmap.shape[0]

