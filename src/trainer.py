import os
from torch import optim
import time
from torchbearer import Trial
import torchbearer
from torchbearer.callbacks import torch_scheduler

from src.networks import UNet, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, Iternet, AttUIternet, R2UIternet
from src.losses import *


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.augmentation_prob = config.augmentation_prob
        # self.image_size = config.

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.beta_list = (float(self.beta1), float(self.beta2))

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.decay_factor = config.decay_factor
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'UNet':
            self.unet = UNet(n_channels=1, n_classes=1)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=1, output_ch=1, t=self.t)  # TODO: changed for green image channel
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=1, output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3, output_ch=1, t=self.t)
        elif self.model_type == 'Iternet':
            self.unet = Iternet(n_channels=1, n_classes=1)
        elif self.model_type == 'AttUIternet':
            self.unet = AttUIternet(n_channels=1, n_classes=1)
        elif self.model_type == 'R2UIternet':
            self.unet = R2UIternet(n_channels=3, n_classes=1)
        elif self.model_type == 'NestedUNet':
            self.unet = NestedUNet(in_ch=1, out_ch=1)

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr,
                                    betas=tuple(self.beta_list))
        self.unet.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def train(self):
        scheduler = torch_scheduler.StepLR(self.num_epochs_decay, gamma=self.decay_factor)
        loss_plot_plan = os.path.join(self.result_path,
                                      'live_loss_plot%s-%d-%.4f-%d-%.4f.png' % (self.model_type,
                                                                                self.num_epochs,
                                                                                self.lr,
                                                                                self.num_epochs_decay,
                                                                                self.augmentation_prob))
        callbacks = [scheduler]

        # imaging.FromState(torchbearer.X).on_val().cache(16).make_grid().to_pyplot(),
        # 					 imaging.FromState(torchbearer.Y_TRUE).on_val().cache(16).make_grid().to_pyplot(),
        # 					 imaging.FromState(torchbearer.Y_PRED).on_val().cache(16).make_grid().to_pyplot(),
        # 					 imaging.FromState(torchbearer.X).on_test().cache(16).make_grid().to_pyplot(),
        # 					 imaging.FromState(torchbearer.Y_TRUE).on_test().cache(16).make_grid().to_pyplot(),
        # 					 imaging.FromState(torchbearer.Y_PRED).on_test().cache(16).make_grid().to_pyplot(),
        # 					 TensorBoard(write_batch_metrics=True),

        trial = Trial(self.unet, self.optimizer, self.criterion, metrics=['loss', 'binary_acc'],
                      # binary_acc for debugging certain things
                      callbacks=callbacks).to(self.device)
        trial.with_generators(train_generator=self.train_loader,
                              val_generator=self.valid_loader,
                              test_generator=self.test_loader)
        start = time.time()
        history = trial.run(epochs=self.num_epochs, verbose=2)
        stop = time.time()
        train_time = stop - start
        state = self.unet.state_dict()
        unet_path = os.path.join(self.model_path,
                                 '%s-%d-%.4f-%d-%.4f_Index_BCE_Dropout_STAREIndex.pkl' % (self.model_type,
                                                                                          self.num_epochs,
                                                                                          self.lr,
                                                                                          self.num_epochs_decay,
                                                                                          self.augmentation_prob,))
        torch.save(state, unet_path)
        print(history)
        ### Testing
        results = trial.evaluate(data_key=torchbearer.TEST_DATA)
        print("Test result:")
        print(results)

        return history, results
