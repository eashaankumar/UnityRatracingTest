import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

class CNN_SD_Denoiser_Dataset(Dataset):
    def __init__(self, dataset_path, file_type):
        # import the modules
        import os
        from os import listdir
        trans = transforms.Compose([
            transforms.ToTensor(),
            ])
        self.data = []
        for images in os.listdir(dataset_path):
            if (images.endswith(file_type)):
                print(images)
                self.data.append(1)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
if __name__ == '__main__':
    import argparse
    print("Showing dataset")
    parser = argparse.ArgumentParser("trainer.data")
    parser.add_argument("--path", help="path to data directory", type=str)
    parser.add_argument("--type", help="image file extension (png, jpeg, etc...)", default='png')
    args = parser.parse_args()
    dataset = CNN_SD_Denoiser_Dataset(dataset_path= args.path, file_type=args.type)