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

PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')


def get_num_masked_words_in_pair(input_ids, word_ids):
    """ Count how many MASK tokens (id=103) have been inserted in a pair of entity descriptions. """

    masked = []
    for iids, wids in zip(input_ids, word_ids):
        wids = np.array(wids)

        # Get the middle separator
        sep_ixs = np.where(wids == None)[0]
        middle_sep_ix = sep_ixs[1]

        left_ids = (iids[:middle_sep_ix] == 103).nonzero()[:, 0]
        left_words = np.unique(wids[left_ids])

        masked.append(len(left_words))

    return np.array(masked)


def get_num_masked_words(input_ids, word_ids):
    """ Count how many MASK tokens (id=103) have been inserted in an entity description. """

    masked = []
    for iids, wids in zip(input_ids, word_ids):
        wids = np.array(wids)

        left_ids = (iids == 103).nonzero()[:, 0]
        left_words = np.unique(wids[left_ids])

        masked.append(len(left_words))

    return np.array(masked)


def load_ditto_model(path, lm='roberta'):
    """ Code taken from https://github.com/megagonlabs/ditto/blob/master/matcher.py """
    # load models
    if not os.path.exists(path):
        raise ValueError(f"Model not found: {path}!")

    model = DittoModel(device='cpu', lm=lm)

    saved_state = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state['model'])

    return model


def evaluate(tuned_model, eval_dataset: EMDataset, thr=None, collate_fn=None):
    print("Starting the inference task...")

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=len(eval_dataset),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    tuned_model.to('cpu')
    tuned_model.eval()

    preds = None
    labels = None
    masked_tokens = None

    with torch.no_grad():
        for features in tqdm(eval_dataloader):
            input_ids = features['input_ids']

            batch_labels = features['labels'].numpy()

            if 'word_ids' in features:
                batch_masked_tokens = get_num_masked_words_in_pair(input_ids, features['word_ids'])
            else:
                batch_masked_tokens = get_num_masked_words(input_ids, features['word_ids_left'])

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
                batch_preds = logits

            else:
                token_type_ids = features['token_type_ids']
                attention_mask = features['attention_mask']
                outputs = tuned_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                batch_preds = torch.argmax(logits, axis=1)

            if labels is None:
                labels = batch_labels
                preds = batch_preds
                masked_tokens = batch_masked_tokens
            else:
                labels = np.concatenate((labels, batch_labels))
                preds = np.concatenate((preds, batch_preds))
                masked_tokens = np.concatenate((masked_tokens, batch_masked_tokens))

    out = {
        'preds': preds,
        'labels': labels,
        'masked_tokens': masked_tokens,
        'masked_records': masked_tokens > 0
    }

    return out


def evaluate_simple_bert_model(args, use_case: str, token: str, mask_type, topk_mask):
    conf = {
        'use_case': use_case,
        'model_name': args.bert_model,
        'tok': token,
        'label_col': args.label_col,
        'left_prefix': args.left_prefix,
        'right_prefix': args.right_prefix,
        'max_len': args.max_len,
        'permute': args.permute,
        'verbose': args.verbose,
        'typeMask': mask_type,
        'columnMask': args.columnMask,
        'topk_mask': topk_mask
    }

    test_conf = conf.copy()
    test_conf['data_type'] = 'test'
    test_dataset = get_dataset(test_conf)

    if args.approach == 'bert':
        model_path = os.path.join(RESULTS_DIR, f"{use_case}_{args.tok}_tuned")
    elif args.approach == 'sbert':
        model_path = os.path.join(RESULTS_DIR, "sbert", f"{use_case}_{args.tok}_tuned")
    else:
        raise ValueError("Wrong approach name!")

    tuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)

    res = evaluate(tuned_model, test_dataset)

    return res


def evaluate_ditto(use_case: str, lm: str, type_mask: str, topk_mask: int, max_len: int):
    ditto_collector = DataCollectorDitto()
    injector = GeneralDKInjector('general')

    valid_path = ditto_collector.get_path(use_case=use_case, data_type='valid')
    test_path = ditto_collector.get_path(use_case=use_case, data_type='test')

    valid_path = injector.transform_file(valid_path)
    test_path = injector.transform_file(test_path)

    valid_dataset = DittoDataset(
        path=valid_path, lm=lm, typeMask=type_mask, topk_mask=topk_mask, max_len=max_len, verbose=False
    )
    test_dataset = DittoDataset(
        path=test_path, lm=lm, typeMask=type_mask, topk_mask=topk_mask, max_len=max_len, verbose=True
    )

    model_path = os.path.join(RESULTS_DIR, "ditto", f"{use_case}.pt")
    tuned_model = load_ditto_model(model_path)

    # Find the best classifier threshold
    threshold = tune_threshold(valid_dataset, tuned_model)

    res = evaluate(tuned_model, test_dataset, thr=threshold, collate_fn=test_dataset.pad)

    # from sklearn.metrics import f1_score
    # print(f1_score(res['labels'], res['preds']))

    return res


def evaluate_supcon(use_case: str, lm: str, type_mask: str, topk_mask: int):
    def get_posneg():
        if lm == ['Structured_Walmart-Amazon', 'Dirty_Walmart-Amazon']:
            return 10
        elif lm in ['Structured_Beer', 'Structured_Amazon-Google', 'Textual_Abt-Buy', 'Structured_Fodors-Zagats']:
            return 9
        elif lm == ['Structured_DBLP-ACM', 'Structured_DBLP-GoogleScholar', 'Dirty_DBLP-ACM',
                    'Dirty_DBLP-GoogleScholar']:
            return 8
        elif lm == ['Structured_iTunes-Amazon', 'Dirty_iTunes-Amazon']:
            return 7

    supcon_collector = DataCollectorSupCon()
    test_path = supcon_collector.get_path(use_case=use_case, data_type='test')

    # Load the test data
    test_dataset = ContrastiveClassificationDataset(
        test_path, dataset_type='test', tokenizer=lm, dataset=use_case
    )

    # Load the model
    model_path = os.path.join(RESULTS_DIR, "supcon", f"{use_case}.bin")
    tuned_model = ContrastiveClassifierModel(
        checkpoint_path=model_path,
        len_tokenizer=len(test_dataset.tokenizer),
        model=lm,
        frozen=True,
        pos_neg=get_posneg(),
        device='cpu'
    )

    data_collator = DataCollatorContrastiveClassification(
        tokenizer=test_dataset.tokenizer, typeMask=type_mask, topk_mask=topk_mask
    )
    res = evaluate(tuned_model, test_dataset, collate_fn=data_collator)

    # from sklearn.metrics import f1_score
    # print(f1_score(res['labels'], res['preds']))

    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Word masking')

    # General parameters
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
    parser.add_argument('-typeMask', '--typeMask', default='off', type=str,
                        choices=['off', 'random', 'selectCol', 'maskSyn', 'maskSem'], help='mask typing')
    parser.add_argument('-columnMask', '--columnMask', default='0', type=str,
                        help='list attributes to mask')
    parser.add_argument('-max_len', '--max_len', default=128, type=int,
                        help='the maximum BERT sequence length')
    parser.add_argument('-permute', '--permute', default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for permuting dataset attributes')
    parser.add_argument('-v', '--verbose', default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='boolean flag for the dataset verbose modality')
    parser.add_argument('-approach', '--approach', type=str, default='bert',
                        choices=['bert', 'sbert', 'ditto', 'supcon'],
                        help='the EM approach to use')
    parser.add_argument('-seed', '--seed', default=42, type=int,
                        help='seed')

    args = parser.parse_args()

    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES

    for use_case in use_cases:
        for token in ['sent_pair']:  # 'attr_pair', 'sent_pair'
            for modeMask in ['maskSyn']:  # 'off', 'random', 'maskSem', 'maskSyn'
                for topk_mask in [3]:  # None, 3
                    if modeMask == 'selectCol' and token == 'sent_pair':
                        continue
                    print(f'Info. Dataset={use_case}, tok={token}, modeMask={modeMask}, topk_mask={topk_mask}')

                    # Evaluate
                    if args.approach == 'ditto':
                        res = evaluate_ditto(
                            use_case=use_case, lm=args.bert_model, type_mask=modeMask, topk_mask=topk_mask,
                            max_len=args.max_len
                        )

                    elif args.approach == 'supcon':
                        res = evaluate_supcon(
                            use_case=use_case, lm=args.bert_model, type_mask=modeMask, topk_mask=topk_mask
                        )

                    else:
                        res = evaluate_simple_bert_model(
                            args=args, use_case=use_case, token=token, mask_type=modeMask, topk_mask=topk_mask
                        )

                    # Save the results
                    res.update({'data': use_case, 'tok': token, 'mask': modeMask, 'topk_mask': topk_mask})
                    if args.approach == 'bert':
                        out_res_name = os.path.join(f"INFERENCE_{use_case}_{token}_{modeMask}_{topk_mask}")
                    elif args.approach == 'sbert':
                        out_res_name = os.path.join(f"INFERENCE_SBERT_{use_case}_{token}_{modeMask}_{topk_mask}")
                    elif args.approach == 'ditto':
                        out_res_name = os.path.join(f"INFERENCE_DITTO_{use_case}_{token}_{modeMask}_{topk_mask}")
                    else:
                        raise ValueError("Wrong approach name!")
                    with open(f'{out_res_name}.pickle', 'wb') as handle:
                        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
