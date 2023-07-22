from transformers import AutoModelForSequenceClassification
import numpy as np
import os
from tqdm import tqdm
import torch
from pathlib import Path
import argparse
import distutils.util
import pickle

from core.data_models.em_dataset import EMDataset
from core.data_models.ditto_dataset import DittoDataset
from core.data_models.supcon_dataset import ContrastiveClassificationDataset
from utils.general import get_dataset
from utils.data_collector import DM_USE_CASES, DataCollectorDitto, DataCollectorSupCon
from utils.knowledge import GeneralDKInjector
from core.modeling.ditto import DittoModel
from utils.ditto_utils import tune_threshold, predict
from core.modeling.supcon import ContrastiveClassifierModel
from torch.utils.data import DataLoader
from utils.supcon_utils import DataCollatorContrastiveClassification
from utils.nlp import FastTextModel
from sklearn.metrics import f1_score

PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')


def load_ditto_model(path, lm='roberta'):
    """ Code taken from https://github.com/megagonlabs/ditto/blob/master/matcher.py """
    # load models
    if not os.path.exists(path):
        raise ValueError(f"Model not found: {path}!")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device, lm=lm)

    saved_state = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state['model'])

    return model


def evaluate(tuned_model, eval_dataset: (EMDataset, DittoDataset, ContrastiveClassificationDataset),
             thr=None, collate_fn=None):
    print("Starting the inference task...")

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tuned_model.to(device)
    tuned_model.eval()

    preds = None
    labels = None

    with torch.no_grad():
        for features in tqdm(eval_dataloader):
            input_ids = features['input_ids']

            batch_labels = features['labels'].numpy()

            if isinstance(tuned_model, DittoModel):
                logits = tuned_model(input_ids)
                probs = logits.softmax(dim=1)[:, 1]
                batch_preds = probs.numpy()
                if thr is None:
                    thr = 0.5
                batch_preds[batch_preds >= thr] = 1
                batch_preds[batch_preds < thr] = 0

            elif isinstance(tuned_model, ContrastiveClassifierModel):
                attention_mask = features['attention_mask']
                input_ids_right = features['input_ids_right']
                attention_mask_right = features['attention_mask_right']
                outputs = tuned_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=features['labels'],
                    input_ids_right=input_ids_right, attention_mask_right=attention_mask_right
                )
                logits = outputs[1]
                logits[logits >= 0.5] = 1
                logits[logits < 0.5] = 0
                batch_preds = logits.numpy().flatten()

            else:
                token_type_ids = features['token_type_ids']
                attention_mask = features['attention_mask']
                outputs = tuned_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                batch_preds = torch.argmax(logits, axis=1)

            if labels is None:
                labels = batch_labels
                preds = batch_preds
            else:
                labels = np.concatenate((labels, batch_labels))
                preds = np.concatenate((preds, batch_preds))

    out = {
        'preds': preds,
        'labels': labels,
    }

    return out


def evaluate_simple_bert_model(args, source_uc: str, target_uc: str):
    target_conf = {
        'use_case': target_uc,
        'model_name': args.bert_model,
        'tok': args.tok,
        'label_col': args.label_col,
        'left_prefix': args.left_prefix,
        'right_prefix': args.right_prefix,
        'max_len': args.max_len,
        'permute': args.permute,
        'verbose': args.verbose,
        'typeMask': None,
        'columnMask': None,
        'topk_mask': None
    }

    test_conf = target_conf.copy()
    test_conf['data_type'] = 'test'
    test_dataset = get_dataset(test_conf)

    if args.approach == 'bert':
        model_path = os.path.join(RESULTS_DIR, f"{source_uc}_{args.tok}_tuned")
    elif args.approach == 'sbert':
        model_path = os.path.join(RESULTS_DIR, "sbert", f"{source_uc}_{args.tok}_tuned")
    else:
        raise ValueError("Wrong approach name!")

    tuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)

    res = evaluate(tuned_model, test_dataset, collate_fn=test_dataset.pad)

    f1 = f1_score(res['labels'], res['preds'])
    res['f1'] = f1
    res['source'] = source_uc
    res['target'] = target_uc
    print(f1)

    return res


def evaluate_ditto(source_uc: str, target_uc: str, lm: str, max_len: int):
    ditto_collector = DataCollectorDitto()
    injector = GeneralDKInjector('general')

    valid_path = ditto_collector.get_path(use_case=source_uc, data_type='valid')
    test_path = ditto_collector.get_path(use_case=target_uc, data_type='test')

    valid_path = injector.transform_file(valid_path)
    test_path = injector.transform_file(test_path)

    valid_dataset = DittoDataset(
        path=valid_path, lm=lm, max_len=max_len, verbose=False
    )
    test_dataset = DittoDataset(
        path=test_path, lm=lm, max_len=max_len, verbose=False
    )

    model_path = os.path.join(RESULTS_DIR, "ditto", f"{source_uc}.pt")
    tuned_model = load_ditto_model(model_path)

    # Find the best classifier threshold
    threshold = tune_threshold(valid_dataset, tuned_model)

    res = evaluate(tuned_model, test_dataset, thr=threshold, collate_fn=test_dataset.pad)

    f1 = f1_score(res['labels'], res['preds'])
    res['f1'] = f1
    res['source'] = source_uc
    res['target'] = target_uc
    print(f1)

    return res


def evaluate_supcon(source_uc: str, target_uc: str, lm: str, max_len: int):

    supcon_collector = DataCollectorSupCon()
    test_path = supcon_collector.get_path(use_case=target_uc, data_type='test')

    # Load the test data
    test_dataset = ContrastiveClassificationDataset(
        test_path, dataset_type='test', tokenizer=lm, dataset=target_uc
    )

    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(RESULTS_DIR, "supcon", f"{source_uc}.bin")
    tuned_model = ContrastiveClassifierModel(
        checkpoint_path=model_path,
        len_tokenizer=len(test_dataset.tokenizer),
        model=lm,
        frozen=True,
        device=device
    )

    data_collator = DataCollatorContrastiveClassification(
        tokenizer=test_dataset.tokenizer, max_length=max_len
    )
    res = evaluate(tuned_model, test_dataset, collate_fn=data_collator)

    f1 = f1_score(res['labels'], res['preds'])
    res['f1'] = f1
    res['source'] = source_uc
    res['target'] = target_uc
    print(f1)

    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Out-of-distribution performance evaluation')

    # General parameters
    parser.add_argument('-bert_model', '--bert_model', default='bert-base-uncased', type=str, required=True,
                        choices=['bert-base-uncased', 'sentence-transformers/nli-bert-base', 'roberta-base'],
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
    parser.add_argument('-approach', '--approach', type=str, default='bert', required=True,
                        choices=['bert', 'sbert', 'ditto', 'supcon'],
                        help='the EM approach to use')
    parser.add_argument('-seed', '--seed', default=42, type=int,
                        help='seed')
    parser.add_argument('-out_dir', '--output_dir', type=str,
                        help='the directory where to store the results', required=True)

    args = parser.parse_args()

    use_case_pairs = [
        # Same domain
        ('Structured_Walmart-Amazon', 'Textual_Abt-Buy'),
        ('Textual_Abt-Buy', 'Structured_Walmart-Amazon'),
        ('Structured_DBLP-GoogleScholar', 'Structured_DBLP-ACM'),
        ('Structured_DBLP-ACM', 'Structured_DBLP-GoogleScholar'),
        ('Dirty_DBLP-GoogleScholar', 'Structured_DBLP-ACM'),
        ('Dirty_DBLP-GoogleScholar', 'Dirty_DBLP-ACM'),
        # Different domain
        ('Structured_iTunes-Amazon', 'Structured_DBLP-ACM'),
        ('Structured_iTunes-Amazon', 'Structured_DBLP-GoogleScholar'),
        ('Structured_DBLP-ACM', 'Structured_iTunes-Amazon'),
        ('Structured_DBLP-GoogleScholar', 'Structured_iTunes-Amazon'),
        ('Dirty_iTunes-Amazon', 'Dirty_DBLP-ACM'),
        ('Dirty_DBLP-ACM', 'Dirty_iTunes-Amazon'),
    ]

    for use_case_pair in use_case_pairs:
        source_uc, target_uc = use_case_pair
        print(f'Info. Source dataset={source_uc}, Target dataset={target_uc}')

        # Evaluate
        if args.approach == 'ditto':
            res = evaluate_ditto(
                source_uc=source_uc, target_uc=target_uc, lm=args.bert_model, max_len=args.max_len
            )

        elif args.approach == 'supcon':
            res = evaluate_supcon(
                source_uc=source_uc, target_uc=target_uc, lm=args.bert_model, max_len=args.max_len
            )

        else:
            res = evaluate_simple_bert_model(
                args=args, source_uc=source_uc, target_uc=target_uc,
            )

        # Save the results
        if args.approach == 'bert':
            out_res_name = f"ROB_BERT_{source_uc}_{target_uc}.pickle"
        elif args.approach == 'sbert':
            out_res_name = f"ROB_SBERT_{source_uc}_{target_uc}.pickle"
        elif args.approach == 'ditto':
            out_res_name = f"ROB_DITTO_{source_uc}_{target_uc}.pickle"
        elif args.approach == 'supcon':
            out_res_name = f"ROB_SUPCON_{source_uc}_{target_uc}.pickle"
        else:
            raise ValueError("Wrong approach name!")

        with open(os.path.join(args.output_dir, out_res_name), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
