from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric
import numpy as np
import os
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score

from core.data_models.em_dataset import EMDataset
from utils.general import get_dataset
from utils.data_collector import DM_USE_CASES
from pathlib import Path
import argparse
import distutils.util

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')


def train(model_name: str, num_epochs: int, train_dataset: EMDataset, val_dataset: EMDataset, train_params: dict,
          out_model_path: str = None):
    print("Starting fine-tuning...")

    assert isinstance(model_name, str), "Wrong data type for parameter 'model_name'."
    assert isinstance(num_epochs, int), "Wrong data type for parameter 'num_epochs'."
    assert isinstance(train_dataset, EMDataset), "Wrong data type for parameter 'train_dataset'."
    assert isinstance(val_dataset, EMDataset), "Wrong data type for parameter 'val_dataset'."
    assert isinstance(train_params, dict), "Wrong data type for parameter 'train_params'."
    if out_model_path is not None:
        assert isinstance(out_model_path, str), "Wrong data type for parameter 'out_model_path'."

    training_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, 'results'),  # output directory
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_params["per_device_train_batch_size"],  # batch size per device during train
        per_device_eval_batch_size=train_params["per_device_eval_batch_size"],  # batch size for evaluation
        warmup_steps=train_params["warmup_steps"],  # number of warmup steps for learning rate scheduler
        weight_decay=train_params["weight_decay"],  # strength of weight decay
        logging_dir=os.path.join(RESULTS_DIR, 'logs'),  # directory for storing logs
        logging_steps=train_params["logging_steps"],
        evaluation_strategy=train_params["evaluation_strategy"],
        seed=train_params["seed"],
    )

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                  num_labels=2,
                                                                  output_attentions=True)

    metric = load_metric("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        model_init=model_init,
        # compute_metrics=compute_metrics,
    )

    trainer.train()

    if out_model_path is not None:
        trainer.save_model(out_model_path)

    return trainer


def evaluate(model_path: str, eval_dataset: EMDataset):
    assert isinstance(model_path, str), "Wrong data type for parameter 'model_path'."
    assert isinstance(eval_dataset, EMDataset), "Wrong data type for parameter 'eval_dataset'."
    assert os.path.exists(model_path), f"No model found at {model_path}."

    print("Starting the inference task...")

    tuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)

    tuned_model.to('cpu')
    tuned_model.eval()

    preds = []
    labels = []
    for features in tqdm(eval_dataset):
        input_ids = features['input_ids'].unsqueeze(0)
        token_type_ids = features['token_type_ids'].unsqueeze(0)
        attention_mask = features['attention_mask'].unsqueeze(0)
        label = features['labels'].tolist()
        labels.append(label)

        outputs = tuned_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        # attns = outputs['attentions']
        pred = torch.argmax(logits, axis=1).tolist()
        preds.append(pred)

    print("F1: {}".format(f1_score(labels, preds)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BERT EM fine-tuning')

    # General parameters
    parser.add_argument('-fit', '--fit', type=lambda x: bool(distutils.util.strtobool(x)), required=True,
                        help='boolean flag for setting training mode')
    parser.add_argument('-use_cases', '--use_cases', nargs='+', required=True, choices=DM_USE_CASES + ['all'],
                        help='the names of the datasets')
    parser.add_argument('-bert_model', '--bert_model', default='bert-base-uncased', type=str,
                        help='the version of the BERT model')
    parser.add_argument('-tok', '--tok', default='sent_pair', type=str, choices=['sent_pair', 'attr_pair'],
                        help='the tokenizer for the EM entries')
    parser.add_argument('-label', '--label_col', default='label', type=str,
                        help='the name of the column in the EM dataset that contains the label')
    parser.add_argument('-left', '--left_prefix', default='left_', type=str,
                        help='the prefix used to identify the columns related to the left entity')
    parser.add_argument('-right', '--right_prefix', default='right_', type=str,
                        help='the prefix used to identify the columns related to the right entity')
    parser.add_argument('-max_len', '--max_len', default=128, type=int,
                        help='the maximum BERT sequence length')
    parser.add_argument('-permute', '--permute', default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for permuting dataset attributes')
    parser.add_argument('-v', '--verbose', default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for the dataset verbose modality')

    # Training parameters
    parser.add_argument('-num_epochs', '--num_epochs', default=10, type=int,
                        help='the number of epochs for training')
    parser.add_argument('-per_device_train_batch_size', '--per_device_train_batch_size', default=8, type=int,
                        help='batch size per device during training')
    parser.add_argument('-per_device_eval_batch_size', '--per_device_eval_batch_size', default=8, type=int,
                        help='batch size for evaluation')
    parser.add_argument('-warmup_steps', '--warmup_steps', default=500, type=int,
                        help='number of warmup steps for learning rate scheduler')
    parser.add_argument('-weight_decay', '--weight_decay', default=0.01, type=float,
                        help='strength of weight decay')
    parser.add_argument('-logging_steps', '--logging_steps', default=10, type=int,
                        help='logging_steps')
    parser.add_argument('-evaluation_strategy', '--evaluation_strategy', default='epoch', type=str,
                        help='evaluation_strategy')
    parser.add_argument('-seed', '--seed', default=42, type=int,
                        help='seed')

    args = parser.parse_args()

    fit = args.fit
    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES

    for use_case in use_cases:

        conf = {
            'use_case': use_case,
            'model_name': args.bert_model,
            'tok': args.tok,
            'label_col': args.label_col,
            'left_prefix': args.left_prefix,
            'right_prefix': args.right_prefix,
            'max_len': args.max_len,
            'permute': args.permute,
            'verbose': args.verbose,
        }

        train_conf = conf.copy()
        train_conf['data_type'] = 'train'
        train_dataset = get_dataset(train_conf)

        val_conf = conf.copy()
        val_conf['data_type'] = 'valid'
        val_dataset = get_dataset(val_conf)

        test_conf = conf.copy()
        test_conf['data_type'] = 'test'
        test_dataset = get_dataset(test_conf)

        out_model_path = os.path.join(RESULTS_DIR, f"{use_case}_{args.tok}_tuned")

        if fit:
            train_params = {
                'per_device_train_batch_size': args.per_device_train_batch_size,
                'per_device_eval_batch_size': args.per_device_eval_batch_size,
                'warmup_steps': args.warmup_steps,
                'weight_decay': args.weight_decay,
                'logging_steps': args.logging_steps,
                'evaluation_strategy': args.evaluation_strategy,
                'seed': args.seed,
            }
            train(args.bert_model, args.num_epochs, train_dataset, val_dataset, train_params=train_params,
                  out_model_path=out_model_path)
        else:
            evaluate(out_model_path, test_dataset)
