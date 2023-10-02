import torch
import os


class CNN_240p_Denoiser_3d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def EncoderInitBlock():
            N = []
            N.append(torch.nn.Conv3d(in_channels=3, out_channels=32, kernel_size=1))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.AvgPool3d(kernel_size=(1, 3, 3)))
            return torch.nn.Sequential(*N)
        
        def EncoderRepeatBlock():
            N = []
            N.append(torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3)))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.AvgPool3d(kernel_size=(1, 3, 3)))
            return torch.nn.Sequential(*N)
        
        def DecoderBlock():
            N = []
            # depth = 3
            N.append(torch.nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, output_padding=(0, 1, 1), stride=(1, 3,3)))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=1, output_padding=(0, 1, 1), stride=(1, 3,3)))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.AvgPool3d(kernel_size=(1, 5, 5)))
            N.append(torch.nn.ConvTranspose3d(in_channels=32, out_channels=3, kernel_size=(1,3,3),  output_padding=(0, 1, 1), stride=(1, 3,3)))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=(1,3,3), output_padding=(0, 1, 1), stride=(1, 3,3)))
            N.append(torch.nn.ReLU())
            # combine depths into 1 
            N.append(torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(3,3,3))) 
            N.append(torch.nn.ReLU())
            N.append(torch.nn.AvgPool3d(kernel_size=(1, 5, 5)))
            N.append(torch.nn.ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=(1,3,3), output_padding=(0, 1, 1), stride=(1, 3,3)))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=(1,3,3), output_padding=(0, 1, 1), stride=(1, 4,4)))
            N.append(torch.nn.ReLU())
            # fit image dimensions
            N.append(torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(1,3,3), dilation=(1,5,13)))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(1,3,3), dilation=(1,5,13)))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(1,3,3), dilation=(1,5,17)))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(1,3,3), dilation=(1,5,16)))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=1))
            N.append(torch.nn.ReLU())
            return torch.nn.Sequential(*N)

        self.block1 = EncoderInitBlock()
        self.block2 = EncoderRepeatBlock()
        self.block3 = EncoderRepeatBlock()
        self.block4 = DecoderBlock()

    def forward(self, x):
        return self.block4(self.block3(self.block2(self.block1(x))))

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