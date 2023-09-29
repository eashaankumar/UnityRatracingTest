from .network import CNN_SD_Denoiser
import torch

if __name__ == '__main__':
    print("Showing Trainer")

    network = CNN_SD_Denoiser()

    for name, param in network.named_parameters():
        if param.requires_grad:
            print (name, param.data.shape)

    sd_size = (426,240)
    noisy_pt_render = torch.zeros(sd_size + (network.input_channels['noisy_pt_render'],))
    normals = torch.zeros(sd_size + (network.input_channels['normals'],))
    depth = torch.zeros(sd_size + (network.input_channels['depth'],))
    albedo = torch.zeros(sd_size + (network.input_channels['albedo'],))
    shape = torch.zeros(sd_size + (network.input_channels['shape'],))
    emission = torch.zeros(sd_size + (network.input_channels['emission'],))
    specular = torch.zeros(sd_size + (network.input_channels['specular'],))

    print(f'noisy_pt_render: {noisy_pt_render.shape}')
    print(f'normals: {normals.shape}')
    print(f'depth: {depth.shape}')
    print(f'albedo: {albedo.shape}')
    print(f'shape: {shape.shape}')
    print(f'emission: {emission.shape}')
    print(f'specular: {specular.shape}')

    input_tensor = network.make_input_tensor(
                            noisy=noisy_pt_render, 
                            normals=normals,
                            depth=depth,
                            albdeo=albedo,
                            shape=shape,
                            emission=emission,
                            material=specular)
    
    input_tensor = input_tensor.permute(2, 0, 1)

    print(input_tensor.shape)

    optim = torch.optim.Adam(network.parameters(), 1e-3)
    loss = torch.nn.CrossEntropyLoss()
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    out = network(input_tensor[None, :, :, :])
