import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets
import os
from torchvision.datasets.folder import default_loader
class CelebADataset(datasets.VisionDataset):
    
    def __init__(self,root,transform=None):
        super(CelebADataset,self).__init__(root,transform=transform)
        self.imgs = [os.path.join(root,img) for img in os.listdir(root)]
        self.transform = transform
        self.loader = default_loader
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,index):
        img_path = self.imgs[index]
        img = self.loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img
    
    
def get_celeba_loader(batch_size,img_size,data_dir):
    transform = transforms.Compose([
        # transforms.Resize((img_size,img_size)),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset = CelebADataset(data_dir,transform=transform)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    
    return dataloader

