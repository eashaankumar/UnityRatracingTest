import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import os
from tqdm import tqdm
import threading

def load_thread(iter, data, desiredKeys, file_type, trans, dataset_path, data_range):
    data.extend([0] * (len(iter[0])))
    assert (len(data) == len(iter[0]))
    with tqdm(iter[0], f"{iter[1]} {len(data)}") as tepoch:
        name = dataset_path.split(os.sep)[-2:]
        tepoch.set_description(f"{name[0]}-{name[1]}-{data_range}")
        for i, subdir in enumerate(tepoch):
            subdirPath = os.path.join(dataset_path, subdir)
            images = os.listdir(subdirPath)
            buffers = {}
            for image in images:
                tepoch.set_postfix(image=f"{image}")
                img_path = os.path.join(subdirPath, image)
                img = trans(Image.open(img_path))
                if (image.endswith(f'noisy-{subdir}.{file_type}')):
                    buffers['noisy'] = img
                elif (image.endswith(f'albedo-{subdir}.{file_type}')):
                    buffers['albedo'] = img
                elif (image.endswith(f'converged-{subdir}.{file_type}')):
                    buffers['converged'] = img
                elif (image.endswith(f'depth-{subdir}.{file_type}')):
                    buffers['depth'] = img[0:1]
                elif (image.endswith(f'emission-{subdir}.{file_type}')):
                    buffers['emission'] = img
                elif (image.endswith(f'normals-{subdir}.{file_type}')):
                    buffers['normals'] = img
                elif (image.endswith(f'shape-{subdir}.{file_type}')):
                    buffers['shape'] = img
                elif (image.endswith(f'specular-{subdir}.{file_type}')):
                    buffers['specular'] = img[0:1]
                else:
                    raise Exception(f"invalid buffer type: {image}")
                pass
            keys = buffers.keys()
            assert len(keys) == 8
            for key in keys:
                assert key in desiredKeys

            data[i] = buffers
            # data.append(buffers)
            pass


class CNN_SD_Denoiser_Dataset(Dataset):

   
    def __init__(self, dataset_path: str, file_type, num_dataset_threads=3, data_count=-1):
        # import the modules
        import os
        from os import listdir
        trans = transforms.Compose([
            transforms.ToTensor(),
            ])
        self.data = []
        self.desiredKeys = ['albedo', 'converged', 'depth', 'emission', 'noisy', 'normals', 'shape', 'specular']
        
        self.threads = []
        self.threadsData = []

        print(f"Loading {dataset_path} on {num_dataset_threads} threads")

        total_data = os.listdir(dataset_path)
        if (data_count > 0 and data_count < len(total_data)):
            total_data = total_data[:data_count]
        partition_size = int(len(total_data) / num_dataset_threads)
        start = 0
        for i in range(num_dataset_threads):
            tData = []
            end = start+partition_size
            if (i == num_dataset_threads-1):
                end = len(total_data)
            t = threading.Thread(target=load_thread, args=(
                (total_data[start:end], "datapoints"), tData, self.desiredKeys, file_type, trans, dataset_path, (start, end))
            )
            t.start()
            self.threads.append(t)
            self.threadsData.append(tData)
            start += partition_size
            print("Thread started")
        
        for i in range(num_dataset_threads):
            self.threads[i].join()

        for i in range(num_dataset_threads):
            self.data.extend(self.threadsData[i])
        
        assert (len(self.data) == len(total_data))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def load_data(dataset_path, num_workers=0, batch_size=128, num_dataset_threads=3, data_count=-1):
    dataset = CNN_SD_Denoiser_Dataset(dataset_path, file_type='jpg', num_dataset_threads=num_dataset_threads, data_count=data_count)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)
    
if __name__ == '__main__':
    import argparse
    print("Showing dataset")
    parser = argparse.ArgumentParser("trainer.data")
    parser.add_argument("--path", help="path to data directory", type=str)
    parser.add_argument("--type", help="image file extension (png, jpeg, etc...)", default='png')
    args = parser.parse_args()
    dataset = CNN_SD_Denoiser_Dataset(dataset_path= args.path, file_type=args.type)