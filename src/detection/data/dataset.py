import os
from PIL import Image

import torch
from torch.utils.data import Dataset


class USGDataset(Dataset):
    def __init__(self, data_file, root_dir, transform=None, mode='train'):

        self.transform = transform
        self.mode = mode
        self.root_dir = root_dir

        with open(data_file, 'r') as f:
            lines = f.read().splitlines()

        self.num_images = len(lines)

        self.images = []
        self.boxes = []
        self.labels = []

        for line in lines:
            line = line.strip().split(' ')
            self.images.append(line[0])
            num_boxes = (len(line)-1) // 5
            
            bbox, label = [], []
            for i in range(num_boxes):
                xmin, xmax, ymin, ymax, class_label = line[i*5+1], line[i*5+2], line[i*5+3], line[i*5+4], line[i*5+5]
                bbox.append(list(map(float, [xmin, xmax, ymin, ymax])))
                label.append(int(class_label))
            
            self.boxes.append(bbox)
            self.labels.append(label)
        
    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.images[index])
        bbox = torch.Tensor(self.boxes[index])
        label = torch.LongTensor(self.labels[index])

        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.mode=='test':
            if self.transform:
                img, _, _ = self.transform(img, bbox, label)
            return image_path, img, bbox, label
        
        if self.transform:
            img, bbox, label = self.transform(img, bbox, label)
        return img, bbox, label

    def __len__(self):
        return self.num_images