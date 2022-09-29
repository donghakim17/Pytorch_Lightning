from pytorch_lightning import LightningModule,LightningDataModule
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import wandb
from torch.nn import functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger()



class cifar_model(LightningModule):
    def __init__(self,n_classes=10, n_layer_1=8, n_layer_2=16,n_layer_3=32, lr=1e-3):
        super().__init__()
        in_channel = 3 
        out_channel = [n_layer_1,n_layer_2,n_layer_3]
        module = []
        self.lr = lr
        for dim in out_channel:
            module.append(
                nn.Sequential(nn.Conv2d(in_channel,dim,3,2,1),nn.ReLU()))
            in_channel = dim
        self.model = nn.Sequential(*module)
        self.final = nn.Sequential(nn.Linear(4*4*32,120),nn.Linear(120,n_classes))
    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x,1)
        x = self.final(x)
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        # Log training loss
        self.log('train_loss', loss)
        # Log metrics
        # self.log('train_acc', self.accuracy(logits, y))
        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # Log validation loss (will be automatically averaged over an epoch)
        self.log('valid_loss', loss)

        # Log metrics
        #self.log('valid_acc', self.accuracy(logits, y))
    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        # Log test loss
        self.log('test_loss', loss)

        # Log metrics
        #self.log('test_acc', self.accuracy(logits, y))
    def configure_optimizers(self):
        '''defines model optimizer'''
        return torch.optim.SGD(self.parameters(), lr=self.lr)


class CIFARDataModule(LightningDataModule):

    def __init__(self, data_dir='./', batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''called one ecah GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            cifar_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
            print(len(cifar_train))
            self.cifar_train, self.cifar_val = random_split(cifar_train, [45000, 5000])
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        '''returns training dataloader'''
        mnist_train = DataLoader(self.cifar_train, batch_size=self.batch_size)
        return mnist_train

    def val_dataloader(self):
        '''returns validation dataloader'''
        cifar_val = DataLoader(self.cifar_val, batch_size=self.batch_size)
        return cifar_val

    def test_dataloader(self):
        '''returns test dataloader'''
        cifar_test = DataLoader(self.cifar_test, batch_size=self.batch_size)
        return cifar_test

wandb.login()
wandb_logger = WandbLogger(project='2021317010_김동하_pytorch_lightning_cifar10')
cifar = CIFARDataModule()
model = cifar_model(n_layer_1=8,n_layer_2=16,n_layer_3=32,lr=1e-3)

trainer = Trainer(
    logger=wandb_logger,    # W&B integration
    max_epochs=5            # number of epochs
    )
trainer.fit(model, cifar)
trainer.test(model, datamodule=cifar)
wandb.finish()