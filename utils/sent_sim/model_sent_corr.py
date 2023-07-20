from transformers import AutoModel, AutoModelForSequenceClassification
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats.stats import pearsonr
import pickle
from pathlib import Path
import argparse

from core.data_models.em_dataset import EMDataset
from core.data_models.ditto_dataset import DittoDataset
from core.data_models.supcon_dataset import ContrastiveClassificationDataset
from utils.general import get_dataset
from utils.data_collector import DM_USE_CASES, DataCollectorDitto, DataCollectorSupCon
from utils.knowledge import GeneralDKInjector
from core.modeling.ditto import DittoModel
from core.modeling.supcon import ContrastiveClassifierModel
from utils.supcon_utils import DataCollatorContrastiveClassification

PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union


def remove_stopwords(text, stopwords):
    if len(stopwords) == 0:
        return text
    return ' '.join([w for w in text.split() if w not in stopwords])


def get_left_right_sent_embeddings(embeddings, end_left, start_right, end_right):
    left_embs = [embeddings[i][1:end_left[i]] for i in range(len(embeddings))]
    right_embs = [embeddings[i][start_right[i]:end_right[i]] for i in range(len(embeddings))]

    # Mean pooling
    left_sent_emb = torch.cat([x.mean(0).unsqueeze(0) for x in left_embs])
    right_sent_emb = torch.cat([x.mean(0).unsqueeze(0) for x in right_embs])

    return left_sent_emb, right_sent_emb


def get_sentence_correlation(tuned_model, eval_dataset: (EMDataset, DittoDataset, ContrastiveClassificationDataset),
                             collate_fn=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tuned_model.to(device)
    tuned_model.eval()

    loader = DataLoader(eval_dataset, batch_size=32, num_workers=4, shuffle=False, collate_fn=collate_fn)

    labels = None
    jac_sims = None
    cos_sims = None
    for features in tqdm(loader):
        input_ids = features['input_ids']
        batch_labels = features["labels"].numpy()

        # Sentence similarity
        stopwords = eval_dataset.stopwords if hasattr(eval_dataset, 'stopwords') else []
        batch_jac_sims = np.array(
            [
                jaccard_similarity(
                    remove_stopwords(sent1, stopwords).split(),
                    remove_stopwords(sent2, stopwords).split()
                )
                for sent1, sent2 in zip(features['sent1'], features['sent2'])]
        )

        # Forward
        with torch.no_grad():
            if isinstance(tuned_model, DittoModel):
                outputs = tuned_model.bert(input_ids)

                # Get indices for left and right sentences
                # Ditto uses a RoBERTa encoder
                # RoBERTa-based encoders use the two </s></s> special tokens to separate the two sentences
                # (their token id is 2)
                sep_ixs = torch.where(input_ids == 2)[1].reshape(-1, 3)
                end_left = sep_ixs[:, 0]
                start_right = sep_ixs[:, 1] + 1
                end_right = sep_ixs[:, 2]

                left_sent_emb, right_sent_emb = get_left_right_sent_embeddings(
                    embeddings=outputs['last_hidden_state'],
                    end_left=end_left,
                    start_right=start_right,
                    end_right=end_right
                )

            elif isinstance(tuned_model, ContrastiveClassifierModel):
                attention_mask = features['attention_mask']
                input_ids_right = features['input_ids_right']
                attention_mask_right = features['attention_mask_right']

                output_left = tuned_model.encoder(input_ids, attention_mask)
                left_token_embeddings = output_left['last_hidden_state']
                left_sent_emb = mean_pooling(left_token_embeddings, attention_mask)

                output_right = tuned_model.encoder(input_ids_right, attention_mask_right)
                right_token_embeddings = output_right['last_hidden_state']
                right_sent_emb = mean_pooling(right_token_embeddings, attention_mask_right)

            else:
                if hasattr(tuned_model, 'base_model'):
                    # If the model is a sequence classifier then get only the encoder
                    encoder = tuned_model.base_model
                else:
                    # The model is already a pretrained encoder
                    encoder = tuned_model

                token_type_ids = features['token_type_ids']
                attention_mask = features['attention_mask']
                outputs = encoder(
                    input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
                )

                # Get indices for left and right sentences
                # BERT-based encoders use the ['SEP'] special token to separate the two sentences (its token id is 102)
                sep_ixs = torch.where(input_ids == 102)[1].reshape(-1, 2)
                end_left = sep_ixs[:, 0]
                start_right = end_left + 1
                end_right = sep_ixs[:, 1]

                left_sent_emb, right_sent_emb = get_left_right_sent_embeddings(
                    embeddings=outputs['last_hidden_state'],
                    end_left=end_left,
                    start_right=start_right,
                    end_right=end_right
                )

        batch_cos_sims = cosine_similarity(X=left_sent_emb, Y=right_sent_emb).diagonal()

        if labels is None:
            labels = batch_labels
            jac_sims = batch_jac_sims
            cos_sims = batch_cos_sims
        else:
            labels = np.concatenate((labels, batch_labels))
            jac_sims = np.concatenate((jac_sims, batch_jac_sims))
            cos_sims = np.concatenate((cos_sims, batch_cos_sims))

    res = {
        'stats': {
            'match': pearsonr(jac_sims[labels == 1], cos_sims[labels == 1])[0],
            'non-match': pearsonr(jac_sims[labels == 0], cos_sims[labels == 0])[0],
            'all': pearsonr(jac_sims, cos_sims)[0]
        },
        'raw': {
            'labels': labels,
            'jac_sims': jac_sims,
            'cos_sims': cos_sims
        }
    }

    return res


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


def ditto_sentence_similarity(use_case, lm, max_len):
    ditto_collector = DataCollectorDitto()
    injector = GeneralDKInjector('general')

    test_path = ditto_collector.get_path(use_case=use_case, data_type='test')
    test_path = injector.transform_file(test_path)

    test_dataset = DittoDataset(
        path=test_path, lm=lm, max_len=max_len, verbose=True
    )

    model_path = os.path.join(RESULTS_DIR, "ditto", f"{use_case}.pt")
    tuned_model = load_ditto_model(model_path)

    results = get_sentence_correlation(
        tuned_model=tuned_model, eval_dataset=test_dataset, collate_fn=test_dataset.pad
    )
    print(results["stats"])

    return results


def supcon_sentence_similarity(use_case, lm, max_len):
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(RESULTS_DIR, "supcon", f"{use_case}.bin")
    tuned_model = ContrastiveClassifierModel(
        checkpoint_path=model_path,
        len_tokenizer=len(test_dataset.tokenizer),
        model=lm,
        frozen=True,
        pos_neg=get_posneg(),
        device=device
    )

    data_collator = DataCollatorContrastiveClassification(
        tokenizer=test_dataset.tokenizer, max_length=max_len
    )
    results = get_sentence_correlation(
        tuned_model=tuned_model, eval_dataset=test_dataset, collate_fn=data_collator
    )
    print(results["stats"])

    return results


def simple_bert_sentence_similarity(use_case, bert_model, tok, max_len, train_type):
    conf = {
        'use_case': use_case,
        'model_name': bert_model,
        'tok': tok,
        'label_col': "label",
        'left_prefix': "left_",
        'right_prefix': "right_",
        'max_len': max_len,
        'permute': False,
        'verbose': False,
        'typeMask': None,
        'columnMask': None
    }

    test_conf = conf.copy()
    test_conf['data_type'] = 'test'
    test_dataset = get_dataset(test_conf)

    if train_type == 'ft':
        if bert_model == 'bert-base-uncased':
            model_name = os.path.join(RESULTS_DIR, f"{use_case}_{tok}_tuned")
        elif bert_model == 'sentence-transformers/nli-bert-base':
            model_name = os.path.join(RESULTS_DIR, "sbert", f"{use_case}_{tok}_tuned")
        else:
            raise NotImplementedError()
        tuned_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        tuned_model = AutoModel.from_pretrained(bert_model)

    results = get_sentence_correlation(
        tuned_model=tuned_model, eval_dataset=test_dataset, collate_fn=test_dataset.pad
    )
    print(results["stats"])

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sentence correlation')

    parser.add_argument('-use_cases', '--use_cases', nargs='+', required=True, choices=DM_USE_CASES + ['all'],
                        help='the names of the datasets')
    parser.add_argument('-train_type', '--train_type', default='pt', choices=['ft', 'pt'],
                        help='Whether to use the pretrained (pt) or the fine-tuned (ft) model.')
    parser.add_argument('-bert_model', '--bert_model', default='bert-base-uncased', type=str, required=True,
                        choices=['bert-base-uncased', 'sentence-transformers/nli-bert-base', 'roberta-base'],
                        help='the version of the BERT model')
    parser.add_argument('-tok', '--tok', default='sent_pair', type=str, choices=['sent_pair', 'attr_pair'],
                        help='the tokenizer for the EM entries')
    parser.add_argument('-max_len', '--max_len', default=128, type=int,
                        help='the maximum BERT sequence length')
    parser.add_argument('-approach', '--approach', type=str, default='bert', required=True,
                        choices=['bert', 'sbert', 'ditto', 'supcon'],
                        help='the EM approach to use')
    parser.add_argument('-out_dir', '--output_dir', type=str,
                        help='the directory where to store the results', required=True)

    args = parser.parse_args()

    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES
    train_type = args.train_type
    bert_model = args.bert_model
    tok = args.tok
    max_len = args.max_len
    approach = args.approach

    out_results = {}
    for use_case in use_cases:
        print(f"USE CASE: {use_case}")

        if args.approach == 'ditto':
            results = ditto_sentence_similarity(use_case, bert_model, max_len)
        elif args.approach == 'supcon':
            results = supcon_sentence_similarity(use_case, bert_model, max_len)
        else:
            results = simple_bert_sentence_similarity(use_case, bert_model, tok, max_len, train_type)

        out_results[use_case] = results

    if approach in ['bert', 'sbert']:
        model_name = f'{approach}_{train_type}'
    else:
        model_name = approach
    with open(os.path.join(args.output_dir, f'{model_name}_sent_corr.pickle'), 'wb') as fp:
        pickle.dump(out_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
