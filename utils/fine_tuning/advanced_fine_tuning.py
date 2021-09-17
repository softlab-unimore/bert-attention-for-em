import pytorch_lightning as pl
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup, AutoTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import os
import pandas as pd
from core.data_models.em_dataset import EMDataset
from tqdm import tqdm
from utils.data_collector import DataCollector
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'models', 'advanced')


class EMDataModule(pl.LightningDataModule):

    def __init__(self, train_path: str, valid_path: str, test_path: str, model_name: str,
                 tokenization: str = 'sent_pair', label_col: str = 'label', left_prefix: str = 'left_',
                 right_prefix: str = 'right_', max_len: int = 256, verbose: bool = False, categories: list = None,
                 permute: bool = False, seed: int = 42, train_batch_size: int = 32, eval_batch_size: int = 32):
        super().__init__()

        assert isinstance(train_path, str), "Wrong data type for parameter 'train_path'."
        assert isinstance(valid_path, str), "Wrong data type for parameter 'valid_path'."
        assert isinstance(test_path, str), "Wrong data type for parameter 'test_path'."
        assert os.path.exists(train_path), "Train dataset not found."
        assert os.path.exists(valid_path), "Validation dataset not found."
        assert os.path.exists(test_path), "Test dataset not found."

        self.train = pd.read_csv(train_path)
        self.valid = pd.read_csv(valid_path)
        self.test = pd.read_csv(test_path)
        self.model_name = model_name
        self.tokenization = tokenization
        self.label_col = label_col
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix
        self.max_len = max_len
        self.verbose = verbose
        self.categories = categories
        self.permute = permute
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def setup(self):
        AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        self.train_dataset = EMDataset(
            self.train, self.model_name, tokenization=self.tokenization, label_col=self.label_col,
            left_prefix=self.left_prefix, right_prefix=self.right_prefix, max_len=self.max_len, verbose=self.verbose,
            permute=self.permute, seed=self.seed
        )

        self.valid_dataset = EMDataset(
            self.valid, self.model_name, tokenization=self.tokenization, label_col=self.label_col,
            left_prefix=self.left_prefix, right_prefix=self.right_prefix, max_len=self.max_len, verbose=self.verbose,
            permute=self.permute, seed=self.seed
        )

        self.test_dataset = EMDataset(
            self.test, self.model_name, tokenization=self.tokenization, label_col=self.label_col,
            left_prefix=self.left_prefix, right_prefix=self.right_prefix, max_len=self.max_len, verbose=self.verbose,
            permute=self.permute, seed=self.seed
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size)


class MatcherTransformer(pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int = 2,
                 learning_rate: float = 2e-5, max_epochs: int = 10,
                 adam_epsilon: float = 1e-8, warmup_steps: int = 0,
                 weight_decay: float = 0.0, train_batch_size: int = 32,
                 eval_batch_size: int = 32):

        super().__init__()

        if not isinstance(model_name, str):
            raise TypeError("Wrong model name type.")

        # save hyper parameters in the hparams attribute of the model
        self.save_hyperparameters()

        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):

        # input shapes: (batch_size, channel, seq_len)

        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(1)

        if token_type_ids is not None:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask
            )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.hparams.num_labels), labels.view(-1))

        # get hidden states
        hidden_states = outputs[2]

        # get attention maps
        attention = outputs[-1]

        return {'loss': loss, 'logits': logits, 'hidden_states': hidden_states, 'attentions': attention}

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        train_loss = outputs['loss']
        logits = outputs['logits']

        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]

        return {'loss': train_loss, "preds": preds, "labels": labels}

    def training_epoch_end(self, outputs):

        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('train_loss', loss, prog_bar=True)
        f1_scores = f1_score(labels, preds, average=None)
        f1_neg = f1_scores[0]
        f1_pos = f1_scores[1]
        self.log('train_f1_neg', f1_neg)
        self.log('train_f1_pos', f1_pos, prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss = outputs['loss']
        logits = outputs['logits']

        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]

        return {'loss': val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):

        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('val_loss', loss, prog_bar=True)
        f1_scores = f1_score(labels, preds, average=None)
        f1_neg = f1_scores[0]
        f1_pos = f1_scores[1]
        self.log('val_f1_neg', f1_neg)
        self.log('val_f1_pos', f1_pos, prog_bar=True)

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                    (len(train_loader.dataset) // self.hparams.train_batch_size)
                    * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]


def train(model_name: str, num_epochs: int, dm: EMDataModule, out_model_path: str = None, gpus: int = 1):

    print("Starting fine tuning the model...")
    pl.seed_everything(42)

    # fine-tuning the transformer
    model = MatcherTransformer(model_name, max_epochs=num_epochs)
    trainer = pl.Trainer(deterministic=True, gpus=gpus, progress_bar_refresh_rate=30, max_epochs=num_epochs)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    # # check results
    # %load_ext tensorboard
    # %tensorboard --logdir ./lightning_logs

    # save the model
    if out_model_path is not None:
        trainer.save_checkpoint(out_model_path)


def evaluate(out_model_path: str, eval_loader: DataLoader):

    print("Loading pre-trained model...")
    model = MatcherTransformer.load_from_checkpoint(checkpoint_path=out_model_path)

    model.eval()

    preds = torch.empty(0)
    labels = torch.empty(0)
    for test_batch in tqdm(eval_loader):
        input_ids = test_batch['input_ids']
        attention_mask = test_batch['attention_mask']
        token_type_ids = test_batch['token_type_ids']
        batch_labels = test_batch['labels']

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs['logits']
            # hidden_states = outputs['hidden_states']

        batch_preds = torch.argmax(logits, axis=1)
        preds = torch.cat((preds, batch_preds))
        labels = torch.cat((labels, batch_labels))

    average_f1 = f1_score(labels, preds)
    f1_class_scores = f1_score(labels, preds, average=None)
    neg_f1 = f1_class_scores[0]
    pos_f1 = f1_class_scores[1]

    print("Average F1: {}".format(average_f1))
    print("F1 Neg class: {}".format(neg_f1))
    print("F1 Pos class: {}".format(pos_f1))


if __name__ == '__main__':

    fit = True

    conf = {
        'use_case': "Structured_Fodors-Zagats",
        'model_name': 'bert-base-uncased',
        'tok': 'attr_pair',  # 'sent_pair', 'attr', 'attr_pair'
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
    }

    data_collector = DataCollector()
    use_case_dir = data_collector.get_data(conf['use_case'])
    train_path = os.path.join(use_case_dir, "train.csv")
    valid_path = os.path.join(use_case_dir, "valid.csv")
    test_path = os.path.join(use_case_dir, "test.csv")

    dm = EMDataModule(train_path, valid_path, test_path, conf['model_name'], tokenization=conf['tok'],
                      label_col=conf['label_col'], left_prefix=conf['left_prefix'], right_prefix=conf['right_prefix'],
                      max_len=conf['max_len'], verbose=conf['verbose'], permute=conf['permute'])
    dm.setup()

    uc = conf['use_case']
    tok = conf['tok']
    model_name = conf['model_name']
    out_model_path = os.path.join(RESULTS_DIR, f"{uc}_{tok}_tuned")

    if fit:

        num_epochs = 10
        train(model_name, num_epochs, dm, out_model_path=out_model_path, gpus=0)

    else:

        evaluate(out_model_path, dm.test_dataloader())
