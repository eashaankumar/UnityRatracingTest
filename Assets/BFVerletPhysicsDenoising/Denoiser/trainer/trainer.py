from .network import CNN_SD_Denoiser
from .data import load_data
import torch
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class LossPlotter:
    def __init__(self, name, xlabel, ylabel) -> None:
        # creating initial data values
        # of x and y
        self.x = np.array([])
        self.y = np.sin(self.x)
        
        plt.ion()
        
        # here we are creating sub plots
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.line1, = self.ax.plot(self.x, self.y)
        
        # setting title
        plt.title(name, fontsize=20)
        
        # setting x-axis label and y-axis label
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
    def add(self, _x, _y):
        # to run GUI event loop
        self.x = np.append(self.x, [_x])
        self.y = np.append(self.y, [_y])
        self.line1.set_xdata(self.x)
        self.line1.set_ydata(self.y)

        # Rescale axes limits
        self.ax.relim()
        self.ax.autoscale()
    
        # drawing updated values
        self.figure.canvas.draw()
    
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        self.figure.canvas.flush_events()

if __name__ == '__main__':
    print("Showing Trainer")

    parser = argparse.ArgumentParser("trainer.data")
    parser.add_argument("--rootpath", help="path to data root directory", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    network = CNN_SD_Denoiser().to(device)

    for name, param in network.named_parameters():
        if param.requires_grad:
            print (name, param.data.shape)

    sd_size = (240,426)

    optim = torch.optim.SGD(network.parameters(), 1e-3)
    loss = torch.nn.CrossEntropyLoss(reduce='sum')
    num_epochs = 200

    train_data = load_data(os.path.join(args.rootpath, 'train'), batch_size=16, num_dataset_threads=1)
    val_data = load_data(os.path.join(args.rootpath, 'val'), batch_size=16, 
                         num_dataset_threads=7)

    loss_plotter = LossPlotter("CNN_SD_Denoiser_Dataset", "epochs", "loss")

    with tqdm(range(num_epochs), unit="epochs") as tepoch:
        for epoch in tepoch:
            total_train_loss = 0
            with tqdm(iter(train_data), unit="batches") as trainloop:
                for i, buffers in enumerate(trainloop):
                    trainloop.set_description(f"Batch {i}")
                    input_tensor = network.make_input_tensor(
                                    noisy=buffers['noisy'],
                                    normals=buffers['normals'],
                                    depth=buffers['depth'],
                                    albedo=buffers['albedo'],
                                    shape=buffers['shape'],
                                    emission=buffers['emission'],
                                    specular=buffers['specular'],
                                    channelIndex=1
                                ).to(device)
                    ground_truth = buffers['converged'].to(device)
                    def assert_tensors(input_tensor, out_tensor):
                        assert input_tensor.shape[0] == out_tensor.shape[0]
                        assert input_tensor.shape[1] == network.num_input_channels()
                        assert out_tensor.shape[1] == network.num_out_channels()
                        assert input_tensor.shape[2] == sd_size[0]
                        assert input_tensor.shape[3] == sd_size[1]
                        assert input_tensor.shape[2] == out_tensor.shape[2]
                        assert input_tensor.shape[3] == out_tensor.shape[3]
                    res = network(input_tensor)
                    l = loss(res, ground_truth)
                    optim.zero_grad()
                    l.backward()
                    optim.step()
                    curr_loss = l.cpu().data
                    total_train_loss += curr_loss
                    trainloop.set_postfix(loss=f"{curr_loss}")
                    loss_plotter.add(epoch * len(train_data) + i, curr_loss)
            #loss_plotter.add(epoch, total_train_loss /  len(train_data))