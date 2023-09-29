import torch

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

class CNN_SD_Denoiser(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.shape_types = [
            'Cube',
        ]
        self.material_types = [
            'Standard', 'Glass'
        ]
        self.emission_range = (0.0, 1.0)
        self.color_range = (0.0, 1.0)
        self.input_channels = {
            'noisy_pt_render': 3,
            'normals': 3,
            'depth': 1,
            'albedo': 3,
            'shape': 3,
            'emission': 3,
            'specular': 1
        }
        self.output_channels = {
            'denoised': 3
        }
        
        first_layer = next_power_of_2(self.num_input_channels())
        self.layers = [first_layer,
                       first_layer * 2,
                       self.num_out_channels()
                       ]
        def CNN():
            N = []
            N.append(torch.nn.Conv2d(self.num_input_channels(), 32, kernel_size=5))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.Conv2d(32, 64, kernel_size=5))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.MaxPool2d(kernel_size=3))
            N.append(torch.nn.Conv2d(64, 128, kernel_size=1))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.Dropout(p=0.2))
            N.append(torch.nn.ConvTranspose2d(128, 64, kernel_size=5, output_padding=1, stride=3))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.ConvTranspose2d(64, 8, kernel_size=5))
            N.append(torch.nn.ReLU())
            N.append(torch.nn.ConvTranspose2d(8, 3, kernel_size=3))
            N.append(torch.nn.ReLU())
            return torch.nn.Sequential(*N)
        self.model = CNN()

    def forward(self, x):
        return self.model(x)
        
    def make_input_tensor(self, noisy, normals, depth, albdeo, shape, emission, material):
        assert((noisy.shape[-1] + normals.shape[-1] + depth.shape[-1] + albdeo.shape[-1] + \
               shape.shape[-1] + emission.shape[-1] + material.shape[-1]) == self.num_input_channels())
        tensor = torch.cat([noisy, normals, depth, albdeo, shape, emission, material], dim=2)
        assert (tensor.shape[2] == self.num_input_channels())
        return tensor

    def num_input_channels(self) -> int:
        """
        Returns # of channels of input tensor
        """
        return sum(self.input_channels.values())
    
    def num_out_channels(self) -> int:
        """
        Returns # of channels of output tensor
        """
        return sum(self.output_channels.values())