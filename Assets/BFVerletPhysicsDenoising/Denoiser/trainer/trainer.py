from .network import CNN_240p_Denoiser, load_model, save_model
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
    def __init__(self, name, xlabel, ylabel, max_points, line1, line2) -> None:
        # creating initial data values
        # of x and y
        self.x = np.array([])
        self.y1 = np.sin(self.x)
        self.y2 = np.sin(self.x)

        self.max_points = max_points
        
        plt.ion()
        
        # here we are creating sub plots
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.line1, = self.ax.plot(self.x, self.y1, label=line1)
        self.line2, = self.ax.plot(self.x, self.y2, label=line2)
        self.ax.legend()
        
        # setting title
        plt.title(name, fontsize=20)
        
        # setting x-axis label and y-axis label
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
    def add(self, _x, _y1, _y2):
        # to run GUI event loop
        self.x = np.append(self.x, [_x])
        self.y1 = np.append(self.y1, [_y1])
        self.y2 = np.append(self.y2, [_y2])
        
        start = len(self.x) - self.max_points
        self.x = self.x[start:]
        self.y1 = self.y1[start:]
        self.y2 = self.y2[start:]
        self.line1.set_xdata(self.x)
        self.line1.set_ydata(self.y1)

        self.line2.set_xdata(self.x)
        self.line2.set_ydata(self.y2)

        # Rescale axes limits
        self.ax.relim()
        self.ax.autoscale()
    
        self.update_display()

    def update_display(self):
        # drawing updated values
        self.figure.canvas.draw()
    
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        self.figure.canvas.flush_events()

        plt.show()

class Trainer:
    def __init__(self) -> None:
        self.network = None
        self.train_data = None
        self.val_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.sd_size = (240,426)
        self.optim = None
        self.loss = None
        self.loss_plotter = LossPlotter("CNN 240p Denoiser Dataset", "epochs", "loss", max_points=1000, line1="Train", line2="Val")
        pass

    def load_model_from_path(self, path):
        model_name = path
        self.network = load_model(model_type=CNN_240p_Denoiser, model_name=model_name)
        self.network.to(self.device)
        self.__init_params()
    
    def save_model(self, path):
        model_name = path
        save_model(self.network, model_type=CNN_240p_Denoiser, model_name=model_name)

    def create_new_model(self):
        self.network = CNN_240p_Denoiser().to(self.device)
        self.__init_params()

    def release_model(self):
        self.network.to('cpu')
        self.network = None

    def __init_params(self):
        self.optim = torch.optim.SGD(self.network.parameters(), 0.02)
        self.loss = torch.nn.CrossEntropyLoss(reduce='sum')

    def load_data(self, rootpath):
        self.train_data = load_data(os.path.join(rootpath, 'train'), batch_size=8, 
                           num_dataset_threads=7, data_count=8000)
        os.system('CLS')
        self.val_data = load_data(os.path.join(rootpath, 'val'), batch_size=8, 
                            num_dataset_threads=7, data_count=500)
        os.system('CLS')
        pass

    def train(self) -> float:
        total_train_loss = 0
        self.network.train()
        with tqdm(iter(self.train_data), unit="batches") as trainloop:
            for i, buffers in enumerate(trainloop):
                self.loss_plotter.update_display()
                trainloop.set_description(f"Train Batch {i}")
                input_tensor = self.network.make_input_tensor(
                                noisy=buffers['noisy'],
                                normals=buffers['normals'],
                                depth=buffers['depth'],
                                albedo=buffers['albedo'],
                                shape=buffers['shape'],
                                emission=buffers['emission'],
                                specular=buffers['specular'],
                                channelIndex=1
                            ).to(self.device)
                ground_truth = buffers['converged'].to(self.device)
                def assert_tensors(input_tensor, out_tensor):
                    assert input_tensor.shape[0] == out_tensor.shape[0]
                    assert input_tensor.shape[1] == self.network.num_input_channels()
                    assert out_tensor.shape[1] == self.network.num_out_channels()
                    assert input_tensor.shape[2] == self.sd_size[0]
                    assert input_tensor.shape[3] == self.sd_size[1]
                    assert input_tensor.shape[2] == out_tensor.shape[2]
                    assert input_tensor.shape[3] == out_tensor.shape[3]
                res = self.network(input_tensor)
                l = self.loss(res, ground_truth)
                self.optim.zero_grad()
                l.backward()
                self.optim.step()
                curr_loss = l.cpu().data
                total_train_loss += curr_loss
                trainloop.set_postfix(loss=f"{curr_loss}")
        return total_train_loss / len(self.train_data)

    def validate(self):
        total_val_loss = 0
        self.network.eval()
        with tqdm(iter(self.val_data), unit="batches") as valloop:
            for i, buffers in enumerate(valloop):
                self.loss_plotter.update_display()
                valloop.set_description(f"Val Batch {i}")
                input_tensor = self.network.make_input_tensor(
                                noisy=buffers['noisy'],
                                normals=buffers['normals'],
                                depth=buffers['depth'],
                                albedo=buffers['albedo'],
                                shape=buffers['shape'],
                                emission=buffers['emission'],
                                specular=buffers['specular'],
                                channelIndex=1
                            ).to(self.device)
                ground_truth = buffers['converged'].to(self.device)
                def assert_tensors(input_tensor, out_tensor):
                    assert input_tensor.shape[0] == out_tensor.shape[0]
                    assert input_tensor.shape[1] == self.network.num_input_channels()
                    assert out_tensor.shape[1] == self.network.num_out_channels()
                    assert input_tensor.shape[2] == self.sd_size[0]
                    assert input_tensor.shape[3] == self.sd_size[1]
                    assert input_tensor.shape[2] == out_tensor.shape[2]
                    assert input_tensor.shape[3] == out_tensor.shape[3]
                res = self.network(input_tensor)
                l = self.loss(res, ground_truth)
                curr_loss = l.cpu().data
                total_val_loss += curr_loss
                valloop.set_postfix(loss=f"{curr_loss}")
        return total_val_loss / len(self.val_data)

class TrainingMenu:
    def __init__(self) -> None:
        pass

    def get_menu_options(self):
        return  f"\n\n{bcolors.BOLD}Menu:{bcolors.ENDC}\n" + \
                f"{bcolors.OKBLUE}[1]: Change lr"+ \
                f"{bcolors.ENDC}\n" + \
                f"Pick an item: "
    
    def confirm(self, timeout=5, yes='y') -> bool:
        o = inputimeout(prompt=f"Confirm [{yes}/n]", timeout=timeout)
        if (o == yes):
            return True
        return False
    
    def _handle_learning_rate(self, optim, timeout=5):
        for g in optim.param_groups:
            olr = g['lr']
            print(f'old lr: {olr}')
        new_lr = float(inputimeout(prompt='Enter new lr: ', timeout=20))
        print(f'new lr={new_lr}')
        if (not self.confirm(timeout=20)):
            return self.menu(timeout=timeout)
        for g in optim.param_groups:
            old_lr = g['lr']
            g['lr'] = new_lr
            print(f"{bcolors.OKGREEN}Old LR: {old_lr} New LR: {new_lr}{bcolors.ENDC}")

    def menu(self, optim, timeout=5):
        try:
            menu = inputimeout(prompt=self.get_menu_options(), timeout=timeout)
            if (menu == "1"):
                self._handle_learning_rate(optim, timeout=timeout)
                return
            else:
                print(f'{bcolors.FAIL}invalid choice{bcolors.ENDC}')
                self.menu(timeout=timeout)
                return
        except TimeoutOccurred:
            return
        



if __name__ == '__main__':
    print("Showing Trainer")

    parser = argparse.ArgumentParser("trainer.data")
    parser.add_argument("--rootpath", help="path to data root directory", type=str, required=True)
    parser.add_argument("--experiment", help="path to experiment root directory", type=str, required=True)
    parser.add_argument("--modelversion", help="version number to save model as", type=str, required=True)
    parser.add_argument("--load_model", help="is this model saved and to be loaded from?", type=bool, required=True, default=False)
    args = parser.parse_args()

    trainer = Trainer()
    if (args.load_model):
        print("loading model")
        trainer.load_model_from_path(os.path.join(args.experiment, f'cnn_240p_den_{args.modelversion}'))
    else:
        print("new model")
        trainer.create_new_model()
        

    # for name, param in network.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data.shape)


    
    num_epochs = 200


    trainer.load_data(args.rootpath)

    inputManager = TrainingMenu()

    with tqdm(range(num_epochs), unit="epochs", mininterval=0.01) as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch: {epoch}")
            # manage stuff
            if (epoch % 5 == 0):
                # save
                if (epoch > 0):
                    #save model
                    model_name = os.path.join(args.experiment, f'cnn_240p_den_{args.modelversion}')
                    trainer.save_model(path=model_name)
                    trainer.release_model()
                    trainer.load_model_from_path(path=model_name)
                    pass

                inputManager.menu(optim=trainer.optim, timeout=10)
                print("Continuing...")
                
            train_loss = trainer.train()

            val_loss = trainer.validate()

            trainer.loss_plotter.add(epoch, train_loss, val_loss)
