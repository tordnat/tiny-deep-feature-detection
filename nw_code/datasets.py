import torch
from PIL import Image
from io import BytesIO
import requests
import torch.utils.data
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import CocoDetection
from torchvision.transforms import v2

class SingleImageDataset(Dataset):
    """Dataset that returns the same image repeatedly for overfitting test"""
    
    def __init__(self, processor, num_samples=100):
        self.num_samples = num_samples
        
        # Load and resize image
        image = Image.open("assets/hovedbygg_left.jpg").convert('RGB')
        image.resize((240, 320))
        self.image = image  # Store for visualization
        
        # Pre-process once for consistent sizing
        inputs = processor(image, return_tensors="pt")
        self.pixel_values = inputs['pixel_values'].squeeze(0)
                
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.pixel_values

class CocoDataloader(CocoDetection):
    """Dataset that just returns Coco image without annotation """

    def __init__(self, dataset_root, annotation_file, processor, num_samples=100):
        super().__init__(dataset_root, annotation_file)
        self.processor = processor
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        image.resize((240, 320))
        inputs = self.processor(image, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0)
    
    def get_image(self, idx): # Mostly for visualization
        image, _ = super().__getitem__(idx)
        return image
