from pathlib import Path
import torch
from torch.utils.data import Dataset 
from torchvision.io import read_image

class SequenceImagesDataset(Dataset):
    def __init__(self, root, transform=None) -> None:
        super().__init__()
        
        self.image_paths = sorted(Path(root).glob('*'), key=lambda x: int(x.stem))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index:int) -> torch.Tensor:
        image_path = self.image_paths[index]
        
        image = read_image(str(image_path)).to(torch.float32)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image
    

