import fnmatch
from torchvision.datasets import ImageFolder
import os
from torchvision.io import read_image
from typing import Any, Tuple, Optional, Callable
import torchvision.transforms as transforms


class SuperResolutionImageDataset(ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        sample_hr = self.loader(path)
        if self.transform is not None:
            sample_hr = self.transform(sample_hr)
        if self.target_transform is not None:
            sample_lr = self.target_transform(sample_hr)
        
        #Normalization
        hr_normalization = transforms.Compose([transforms.Normalize(
            mean = [0.5,0.5,0.5],
            std = [0.5,0.5,0.5]
        )])
        lr_normalization = transforms.Compose([transforms.Normalize(
            mean = [0,0,0],
            std = [1,1,1]
        ),])
        #ToTensor
        convert_to_tensor = transforms.Compose([transforms.ToTensor()])
        sample_hr = hr_normalization(convert_to_tensor(sample_hr))
        sample_lr = lr_normalization(convert_to_tensor(sample_lr))

        return {"hr_sample":sample_hr, "lr_sample":sample_lr}