from .network import CNN_SD_Denoiser
from .data import load_data
import torch
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from inputimeout import inputimeout, TimeoutOccurred
from .utils import *

class LossPlotter:
    def __init__(self, name, xlabel, ylabel, max_points) -> None:
        # creating initial data values
        # of x and y
        self.x = np.array([])
        self.y = np.sin(self.x)

        self.max_points = max_points
        
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
        start = len(self.x) - self.max_points
        self.x = self.x[start:]
        self.y = self.y[start:]
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

class Trainer:
    def __init__(self) -> None:
        pass

    def train(self, train_data, network, loss, optim) -> float:
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
        return total_train_loss / len(train_data)

    def validate(self):
        pass

class TrainingMenu:
    def __init__(self) -> None:
        pass

    def get_menu_options(self):
        return  f"\n\n{bcolors.BOLD}Menu:{bcolors.ENDC}\n" + \
                f"{bcolors.OKBLUE}[1]: Change lr"+ \
                f"{bcolors.ENDC}\n"
    
    def confirm(self, timeout=5, yes='y') -> bool:
        o = inputimeout(prompt=f"Confirm [{yes}/n]", timeout=timeout)
        if (o == yes):
            return True
        return False

    def menu(self, optim, timeout=5, invalid_tries=3):
        try:
            menu = inputimeout(prompt=self.get_menu_options(), timeout=timeout)
            if (menu == "1"):
                new_lr = float(inputimeout(prompt='Enter new lr: ', timeout=20))
                print(f'new lr={new_lr}')
                if (not self.confirm(timeout=20)):
                    return self.menu(timeout=timeout, invalid_tries=invalid_tries)
                for g in optim.param_groups:
                    old_lr = g['lr']
                    g['lr'] = new_lr
                    print(f"{bcolors.OKGREEN}Old LR: {old_lr} New LR: {new_lr}{bcolors.ENDC}")
                return
            else:
                print(f'{bcolors.FAIL}invalid choice{bcolors.ENDC}')
                if (invalid_tries > 0):
                    self.menu(timeout=timeout, invalid_tries=invalid_tries-1)
                return
        except TimeoutOccurred:
            return
        



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


    train_data = load_data(os.path.join(args.rootpath, 'train'), batch_size=16, 
                           num_dataset_threads=7, data_count=1000)
    os.system('CLS')
    val_data = load_data(os.path.join(args.rootpath, 'val'), batch_size=16, 
                         num_dataset_threads=7, data_count=100)
    os.system('CLS')
    loss_plotter = LossPlotter("CNN 240p Denoiser Dataset", "epochs", "loss", max_points=1000)

    trainer = Trainer()

    inputManager = TrainingMenu()

    with tqdm(range(num_epochs), unit="epochs", mininterval=0.01) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch: {epoch}")
            # manage stuff
            if (epoch % 5 == 0):
                inputManager.menu(optim=optim)
                print("Continuing...")
                
            train_loss = trainer.train(train_data=train_data, network=network, loss=loss, optim=optim)
            loss_plotter.add(epoch, train_loss)