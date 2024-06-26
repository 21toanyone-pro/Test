import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import accuracy_score

class ResNetClassifier(LightningModule):
    def __init__(self):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10) 

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)