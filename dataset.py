import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np

class CUBDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of CUB_200_2011 dataset
            split (str): 'train' or 'test'
            transform: Optional transforms to apply
        """
        self.root_dir = root_dir
        self.transform = transform

        # Load image list
        images_file = os.path.join(root_dir, 'images.txt')
        image_data = pd.read_csv(images_file, sep=' ', names=['img_id', 'filepath'], header=None)

        # Load train/test split
        split_file = os.path.join(root_dir, 'train_test_split.txt')
        split_data = pd.read_csv(split_file, sep=' ', names=['img_id', 'is_training'], header=None)

        # Load class labels
        labels_file = os.path.join(root_dir, 'image_class_labels.txt')
        label_data = pd.read_csv(labels_file, sep=' ', names=['img_id', 'class_id'], header=None)

        # Merge all data
        self.data = pd.merge(image_data, split_data, on='img_id')
        self.data = pd.merge(self.data, label_data, on='img_id')

        # Filter based on split
        is_train = (split == 'train')
        self.data = self.data[self.data.is_training == is_train]

        # Load segmentation paths
        self.seg_dir = os.path.join(root_dir, 'segmentations')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.data.iloc[idx]['filepath'])
        img = Image.open(img_path).convert('RGB')

        # Load segmentation mask
        seg_path = os.path.join(self.seg_dir,
                              self.data.iloc[idx]['filepath'].replace('.jpg', '.png'))
        mask = Image.open(seg_path).convert('L')

        # Apply mask to image
        img_array = np.array(img)
        mask_array = np.array(mask)
        mask_array = np.stack([mask_array] * 3, axis=-1) / 255.0
        masked_img = Image.fromarray((img_array * mask_array).astype(np.uint8))

        if self.transform:
            masked_img = self.transform(masked_img)

        label = self.data.iloc[idx]['class_id'] - 1  # Convert to 0-based indexing
        return masked_img, label

def get_data_loaders(root_dir, batch_size=32):
    """Create train and validation data loaders."""
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CUBDataset(root_dir, split='train', transform=transform)
    test_dataset = CUBDataset(root_dir, split='test', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=4)

    return train_loader, test_loader
