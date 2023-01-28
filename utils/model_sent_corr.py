from transformers import AutoModel, AutoModelForSequenceClassification
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats.stats import pearsonr
import pickle

from core.data_models.em_dataset import EMDataset
from utils.general import get_dataset
from utils.data_collector import DM_USE_CASES
from pathlib import Path
import argparse

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')


def mean_pooling(token_embeddings, attention_mask):
    """ Mean Pooling - Take attention mask into account for correct averaging """
    sent_embs = []
    for i in range(len(token_embeddings)):
        emb_i = token_embeddings[i].unsqueeze(0)
        mask_i = attention_mask[i].unsqueeze(0)
        input_mask_expanded = mask_i.unsqueeze(-1).expand(emb_i.size()).float()
        sent_embs.append(
            torch.sum(emb_i * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        )

    return torch.cat(sent_embs).detach().cpu().numpy()


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union


def get_sentence_correlation(model_name, eval_dataset: EMDataset, train_type: str):
    assert isinstance(eval_dataset, EMDataset), "Wrong data type for parameter 'eval_dataset'."

    if train_type == 'pt':
        tuned_model = AutoModel.from_pretrained(model_name)
    elif train_type == 'ft':
        tuned_model = AutoModelForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)
    else:
        raise ValueError("Wrong train type!")

    tuned_model.to('cpu')
    tuned_model.eval()

    loader = DataLoader(eval_dataset, batch_size=32, num_workers=4, shuffle=False)

    labels = None
    jac_sims = None
    cos_sims = None
    for features in tqdm(loader):
        input_ids = features['input_ids']
        token_type_ids = features['token_type_ids']
        attention_mask = features['attention_mask']
        batch_labels = features["labels"].numpy()
        batch_jac_sims = np.array([jaccard_similarity(sent1.split(), sent2.split())
                                   for sent1, sent2 in zip(features['sent1'], features['sent2'])])
        sep_ixs = torch.where(input_ids == 102)[1][::2]

        outputs = tuned_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        if train_type == 'pt':
            embeddings = outputs[0]  # last hidden states
        else:
            embeddings = outputs[1][11]  # last hidden states

        left_embs = [torch.split(embeddings[i], sep_ixs[i])[0] for i in range(embeddings.shape[0])]
        right_embds = [torch.cat(torch.split(embeddings[i], sep_ixs[i])[1:]) for i in range(embeddings.shape[0])]
        left_attn_mask = [torch.split(attention_mask[i], sep_ixs[i])[0] for i in range(attention_mask.shape[0])]
        right_attn_mask = [torch.cat(torch.split(attention_mask[i], sep_ixs[i])[1:]) for i in
                           range(attention_mask.shape[0])]

        # Perform pooling
        left_sent_emb = mean_pooling(left_embs, left_attn_mask)
        right_sent_emb = mean_pooling(right_embds, right_attn_mask)
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sentence correlation')

    parser.add_argument('-use_cases', '--use_cases', nargs='+', required=True, choices=DM_USE_CASES + ['all'],
                        help='the names of the datasets')
    parser.add_argument('-train_type', '--train_type', default='pt', required=True, choices=['ft', 'pt'],
                        help='Whether to use the pretrained (pt) or the fine-tuned (ft) model.')
    parser.add_argument('-bert_model', '--bert_model', default='bert-base-uncased', type=str,
                        choices=['bert-base-uncased', 'sentence-transformers/nli-bert-base'],
                        help='the version of the BERT model')
    parser.add_argument('-tok', '--tok', default='sent_pair', type=str, choices=['sent_pair', 'attr_pair'],
                        help='the tokenizer for the EM entries')

    args = parser.parse_args()

    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES
    tok = args.tok
    bert_model = args.bert_model
    train_type = args.train_type
    MODEL_DIR = "C:\\Users\\matte\\PycharmProjects\\bertAttention\\results\\models"


    out_results = {}
    m_name = None
    for use_case in use_cases:
        print(f"USE CASE: {use_case}")

        conf = {
            'use_case': use_case,
            'model_name': bert_model,
            'tok': tok,
            'label_col': "label",
            'left_prefix': "left_",
            'right_prefix': "right_",
            'max_len': 128,
            'permute': False,
            'verbose': False,
        }

        test_conf = conf.copy()
        test_conf['data_type'] = 'test'
        test_dataset = get_dataset(test_conf)

        if train_type == 'ft':
            if bert_model == 'bert-base-uncased':
                model_name = os.path.join(MODEL_DIR, f"{use_case}_{tok}_tuned")
                m_name = 'bert_ft'
            elif bert_model == 'sentence-transformers/nli-bert-base':
                model_name = os.path.join(MODEL_DIR, "sbert", f"{use_case}_{tok}_tuned")
                m_name = 'sbert_ft'
            else:
                raise NotImplementedError()
        else:
            model_name = bert_model
            if bert_model == 'sentence-transformers/nli-bert-base':
                m_name = 'sbert_pt'
            elif bert_model == 'bert-base-uncased':
                m_name = 'bert_pt'

        results = get_sentence_correlation(model_name, test_dataset, train_type)
        print(results["stats"])

        out_results[use_case] = results

    assert m_name is not None
    with open(f'{m_name}_sent_corr.pickle', 'wb') as fp:
        pickle.dump(out_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
