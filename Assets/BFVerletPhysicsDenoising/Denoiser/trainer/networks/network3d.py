import torch
import os


class CNN_240p_Denoiser_3d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def DepthPoint(in_channel, out_channel, kernel_size):
            N = []
            N.append(torch.nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, groups=in_channel))
            N.append(torch.nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1))
            return torch.nn.Sequential(*N)

        def EncoderInitBlock():
            N = []
            N.append(DepthPoint(3, 32, (2,1,1)))
            N.append(torch.nn.ReLU())
            N.append(DepthPoint(32, 32, (2,1,1)))
            N.append(torch.nn.ReLU())
            N.append(DepthPoint(32, 32, (2,1,1)))
            N.append(torch.nn.ReLU())
            N.append(DepthPoint(32, 32, (2,1,1)))
            N.append(torch.nn.ReLU())
            N.append(DepthPoint(32, 32, (2,1,1)))
            N.append(torch.nn.ReLU())
            N.append(DepthPoint(32, 3, (2,1,1)))
            N.append(torch.nn.ReLU())
            return torch.nn.Sequential(*N)
        
        self.block1 = EncoderInitBlock()


    def forward(self, x):
        return self.block1(x)

    def make_input_tensor(self, noisy, normals, depth, albedo, shape, emission, specular):
        # convert grayscale to rgb
        depth = depth.repeat(1, 3, 1, 1)
        specular = specular.repeat(1, 3, 1, 1)
        # add depth dimension
        tensor = torch.cat([noisy[:, :, None, :, :], 
                            normals[:, :, None, :, :], 
                            depth[:, :, None, :, :], 
                            albedo[:, :, None, :, :], 
                            shape[:, :, None, :, :], 
                            emission[:, :, None, :, :], 
                            specular[:, :, None, :, :]], dim=2)
        return tensor 

if __name__=='__main__':
    from trainer.data import load_data
    dataset = load_data('C:\\Users\\seana\\Datasets\\CNN_240p_Denoiser\\val', batch_size=8, num_dataset_threads=3, data_count=10)
    model = CNN_240p_Denoiser_3d()
    sd_size = (240,426)
    for b in dataset:
        buffers = b
        break
    X = model.make_input_tensor(noisy=buffers['noisy'],
                                normals=buffers['normals'],
                                depth=buffers['depth'],
                                albedo=buffers['albedo'],
                                shape=buffers['shape'],
                                emission=buffers['emission'],
                                specular=buffers['specular'],
                                )
    print(X.shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    model = model.to(device)

    Y = model(X)
    print(Y.shape)
    pass