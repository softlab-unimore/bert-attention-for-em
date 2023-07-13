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
from utils.general import get_dataset
from utils.data_collector import DM_USE_CASES, DataCollectorDitto
from utils.knowledge import GeneralDKInjector
from core.modeling.ditto import DittoModel


PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')


def get_num_masked_words(input_ids, word_ids):
    word_ids = np.array(word_ids)
    sep_ixs = np.where(word_ids == None)[0]
    middle_sep_ix = sep_ixs[1]

    left_ids = (input_ids[:middle_sep_ix] == 103).nonzero()[:, 0]
    left_words = np.unique(word_ids[left_ids])

    return len(left_words)


def load_ditto_model(path, lm='roberta'):
    """ Code taken from https://github.com/megagonlabs/ditto/blob/master/matcher.py """
    # load models
    if not os.path.exists(path):
        raise ValueError(f"Model not found: {path}!")

    model = DittoModel(device='cpu', lm=lm)

    saved_state = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state['model'])

    return model


def evaluate(model_path: str, eval_dataset: EMDataset):
    assert isinstance(model_path, str), "Wrong data type for parameter 'model_path'."
    assert isinstance(eval_dataset, (EMDataset, DittoDataset)), "Wrong data type for parameter 'eval_dataset'."
    assert os.path.exists(model_path), f"No model found at {model_path}."

    print("Starting the inference task...")

    if isinstance(eval_dataset, DittoDataset):
        tuned_model = load_ditto_model(model_path)
    else:
        tuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)

    tuned_model.to('cpu')
    tuned_model.eval()

    # FIXME: test for DITTO
    threshold = tune_threshold(config, model, hp)

    # run prediction
    predict(hp.input_path, hp.output_path, config, model,
            summarizer=summarizer,
            max_len=hp.max_len,
            lm=hp.lm,
            dk_injector=dk_injector,
            threshold=threshold)
    exit(1)

    preds = []
    labels = []
    masked_tokens = []
    for features in tqdm(eval_dataset):
        input_ids = features['input_ids'].unsqueeze(0)
        token_type_ids = features['token_type_ids'].unsqueeze(0)
        attention_mask = features['attention_mask'].unsqueeze(0)
        label = features['labels'].tolist()
        labels.append(label)

        masked = get_num_masked_words(input_ids[0], features['word_ids'])
        masked_tokens.append(masked)

        if isinstance(tuned_model, DittoModel):
            logits = tuned_model(input_ids)
            probs = logits.softmax(dim=1)[:, 1]
            pred = [1 if p > 0.5 else 0 for p in probs]
        else:
            outputs = tuned_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            pred = torch.argmax(logits, axis=1).tolist()
        preds.append(pred)

    out = {
        'preds': np.array(preds),
        'labels': np.array(labels),
        'masked_tokens': np.array(masked_tokens),
        'masked_records': np.array(masked_tokens) > 0
    }

    return out


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
    parser.add_argument('-approach', '--approach', type=str, default='bert', choices=['bert', 'sbert', 'ditto'],
                        help='the EM approach to use')

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

    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES

    for use_case in use_cases:
        for token in ['sent_pair']:  # 'attr_pair', 'sent_pair'
            for modeMask in [None, 'random', 'maskSem', 'maskSyn']:  # 'off', 'random', 'maskSem', 'maskSyn'
                for topk_mask in [3]:  # None, 3
                    if modeMask == 'selectCol' and token == 'sent_pair':
                        continue
                    print(f'Info. Dataset={use_case}, tok={token}, modeMask={modeMask}, topk_mask={topk_mask}')

                    # Get masked test data
                    if args.approach != 'ditto':
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
                            'typeMask': modeMask,
                            'columnMask': args.columnMask,
                            'topk_mask': topk_mask
                        }

                        test_conf = conf.copy()
                        test_conf['data_type'] = 'test'
                        test_dataset = get_dataset(test_conf)

                    else:
                        ditto_collector = DataCollectorDitto()
                        test_path = ditto_collector.get_path(use_case=use_case, data_type='test')
                        injector = GeneralDKInjector('general')
                        test_path = injector.transform_file(test_path)
                        test_dataset = DittoDataset(
                            path=test_path, lm=args.bert_model, typeMask=modeMask, topk_mask=topk_mask
                        )

                    if args.approach == 'bert':
                        out_model_path = os.path.join(RESULTS_DIR, f"{use_case}_{args.tok}_tuned")
                    elif args.approach == 'sbert':
                        out_model_path = os.path.join(RESULTS_DIR, "sbert", f"{use_case}_{args.tok}_tuned")
                    elif args.approach == 'ditto':
                        out_model_path = os.path.join(RESULTS_DIR, "ditto", f"{use_case}.pt")
                    else:
                        raise ValueError("Wrong approach name!")

                    res = evaluate(out_model_path, test_dataset)
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
