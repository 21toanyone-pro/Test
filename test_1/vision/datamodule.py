from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule

class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def setup(self, stage=None):
        self.cifar_train = CIFAR10(root=self.data_dir, train=True, download=True, transform=self.transform)
        self.cifar_test = CIFAR10(root=self.data_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False)