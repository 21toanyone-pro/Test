import torch
import pytorch_lightning as pl
from transformers import RobertaForSequenceClassification
from sklearn.metrics import accuracy_score

class RobertaClassifier(pl.LightningModule):
    def __init__(self):
        super(RobertaClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=4)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
        val_loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        labels = batch['labels']
        acc = accuracy_score(labels.cpu(), preds.cpu())
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)
