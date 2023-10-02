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

class DebugImagePlotter:
    def __init__(self, name, num_images, sd_size):
        
        self.num_images = num_images
        self.num_cols = 3 + 3
        plt.ion()
        self.f, self.axarr = plt.subplots(self.num_images, self.num_cols)
        self.ims = [[None] * self.num_cols] * self.num_images
        assert len(self.ims) == self.num_images
        assert len(self.ims[0]) == self.num_cols
        for x in range(self.num_images):
            for y in range(self.num_cols):
                self.ims[x][y] = self.axarr[x,y].imshow(X=np.random.random((sd_size[0],sd_size[1], 3)))

        plt.title(name, fontsize=20)
        pass

    def add(self, train_images_noisy, train_images_true, train_images_predict,
            val_images_noisy, val_images_true, val_images_predict):
        self.ims[0][0].set_data(train_images_noisy[0].permute(1,2,0))
        # drawing updated values
        self.f.canvas.draw()
    
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        self.f.canvas.flush_events()
        pass

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

        # self.subfigs = self.figure.subfigures(1, 1, wspace=0.07, width_ratios=[1.5, 1.])
        
        # self.img_ax1 = self.subfigs.add_subplot(1, 1, 7)
        

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

class DenoisingAutoencoderTrainer:
    def __init__(self) -> None:
        self.network = None
        self.train_data = None
        self.val_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.sd_size = (240,426)
        self.optim = None
        self.loss = None
        self.loss_plotter = LossPlotter("CNN 240p Denoiser", "epochs", "loss", max_points=1000, line1="Train", line2="Val")
        self.image_debugger = DebugImagePlotter("CNN 240p Denoiser", 5, sd_size=self.sd_size)
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
        self.optim = torch.optim.Adam(self.network.parameters(), 0.0001)
        self.loss = torch.nn.MSELoss()

    def load_data(self, rootpath):
        self.train_data = load_data(os.path.join(rootpath, 'train'), batch_size=8, 
                           num_dataset_threads=10, data_count=100)
        os.system('CLS')
        self.val_data = load_data(os.path.join(rootpath, 'val'), batch_size=8, 
                            num_dataset_threads=10, data_count=10)
        os.system('CLS')
        pass

    def __assert_tensors(self, input_tensor, out_tensor):
        assert input_tensor.shape[0] == out_tensor.shape[0]
        assert input_tensor.shape[1] == self.network.num_input_channels()
        assert out_tensor.shape[1] == self.network.num_out_channels()
        assert input_tensor.shape[2] == self.sd_size[0]
        assert input_tensor.shape[3] == self.sd_size[1]
        assert input_tensor.shape[2] == out_tensor.shape[2]
        assert input_tensor.shape[3] == out_tensor.shape[3]

    def __get_images(self, noisy_batch, ground_truth_batch, pred_batch, count:int):
        train_images_inputs = []
        train_images_true = []
        train_images_predict = []
        for imageI in range(count):
            train_images_inputs.append(noisy_batch[imageI])  
            train_images_true.append(ground_truth_batch[imageI])  
            train_images_predict.append(pred_batch[imageI]) 
        return (train_images_inputs, train_images_true, train_images_predict)

    def train(self) -> float:
        total_train_loss = 0
        self.network.train()
        debug_images = None
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
                res = self.network(input_tensor)
                if (i == 0):
                    debug_images = self.__get_images(  noisy_batch=buffers['noisy'], \
                                        ground_truth_batch=ground_truth, \
                                        pred_batch=res, \
                                        count=min(5, input_tensor.shape[0]))
                    pass
                del input_tensor
                l = self.loss(res, ground_truth)
                del ground_truth
                self.optim.zero_grad()
                l.backward()
                self.optim.step()
                curr_loss = l.cpu().data
                total_train_loss += curr_loss
                trainloop.set_postfix(loss=f"{curr_loss}")
        return total_train_loss / len(self.train_data), debug_images

    def validate(self):
        total_val_loss = 0
        self.network.eval()
        debug_images = None
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
                res = self.network(input_tensor)
                if (i == 0):
                    debug_images = self.__get_images(  noisy_batch=buffers['noisy'], \
                                        ground_truth_batch=ground_truth, \
                                        pred_batch=res, \
                                        count=min(5, input_tensor.shape[0]))
                    pass
                del input_tensor
                l = self.loss(res, ground_truth)
                del ground_truth
                curr_loss = l.cpu().data
                total_val_loss += curr_loss
                valloop.set_postfix(loss=f"{curr_loss}")
        return total_val_loss / len(self.val_data), debug_images

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
                self.menu(timeout=timeout, optim=optim)
                return
        except TimeoutOccurred:
            return
        



if __name__ == '__main__':
    print("Showing Trainer")

    parser = argparse.ArgumentParser("trainer.data")
    parser.add_argument("--rootpath", help="path to data root directory", type=str, required=True)
    parser.add_argument("--experiment", help="path to experiment root directory", type=str, required=True)
    parser.add_argument("--modelversion", help="version number to save model as", type=str, required=True)
    parser.add_argument("--load_model", help="is this model saved and to be loaded from?", type=str, required=True, default='False')
    args = parser.parse_args()

    trainer = DenoisingAutoencoderTrainer()
    if (args.load_model == 'True'):
        print("loading model")
        trainer.load_model_from_path(os.path.join(args.experiment, f'cnn_240p_den_{args.modelversion}'))
    elif (args.load_model == 'False'):
        print("new model")
        trainer.create_new_model()

    else:
        raise Exception(f"Invalid --load_model arg {args.load_model}")
        

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
                
            train_loss, train_images = trainer.train()

            val_loss, valid_images = trainer.validate()

            trainer.loss_plotter.add(epoch, train_loss, val_loss)
            trainer.image_debugger.add(train_images_noisy=train_images[0],
                                       train_images_true=train_images[1],
                                       train_images_predict=train_images[2],
                                       val_images_noisy=valid_images[0],
                                       val_images_true=valid_images[1],
                                       val_images_predict=valid_images[2])
