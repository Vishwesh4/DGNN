"""
This script is basically a wrapper around extracted patches and coordinates
"""
import torch.utils.data as data

class TILData(data.Dataset):
    def __init__(self,
                 patches,
                 coordinates,
                 transform,
                 mode="til"
                 ) -> None:
        super().__init__()
        self.patches = patches
        self.coordinates = coordinates
        self.transform = transform
        self.mode = mode
    
    def __len__(self):
        return len(self.patches)

    def __getitem__(self,index):
        img = self.patches[index]
        if self.mode=="tumorbed":
            img = img[128:128+256,128:128+256,:]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.coordinates[index]
        