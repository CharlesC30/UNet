import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize
from torchvision import transforms
import numpy as np
from skimage.filters import threshold_otsu


class DummyCTDataset(Dataset):
    """
    Dummy dataset that will just access the same image over and over again.
    Just using this for testing and trying things out.
    """
    def __init__(
            self, 
            sparse_view_path, 
            full_view_path,
            n_copies, 
            transform=transforms.Compose([ToTensor(), Resize((1008, 1008))]), 
            target_transform=transforms.Compose([ToTensor(), Resize((1008, 1008))]),
            # transform=ToTensor(), 
            # target_transform=ToTensor(),
            ):
        self.sparse_view_path = sparse_view_path
        self.full_view_path = full_view_path
        self.n_copies = n_copies
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.n_copies
    
    def __getitem__(self, idx):
        sparse_view = np.load(self.sparse_view_path)
        full_view = np.load(self.full_view_path)
        thresh = threshold_otsu(full_view)
        mask = full_view.copy()
        mask[full_view < thresh] = 1
        mask[full_view >= thresh] = 0
        masked_image = mask * sparse_view
        target = sparse_view.astype(np.float32) - full_view.astype(np.float32)
        return self.transform(masked_image), self.target_transform(target)
    
