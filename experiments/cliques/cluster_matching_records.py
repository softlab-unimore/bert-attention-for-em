import pandas as pd
import networkx as nx
import os
from pathlib import Path
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import f1_score
import argparse

from utils.data_collector import DM_USE_CASES, DataCollectorWDC, DataCollector, DataCollectorSupCon
from utils.general import get_dataset
from core.data_models.em_dataset import EMDataset
from core.data_models.supcon_dataset import ContrastiveClassificationDataset
from core.modeling.supcon import ContrastiveClassifierModel
from utils.supcon_utils import DataCollatorContrastiveClassification

PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')


class IntegratedDataPreparationComponent(object):
    """
    This class manages the preparation of an integrated data source.
    """

    def __init__(self, data, left_id_col, right_id_col, match_label_col, out_entity_col):
        """
        This method initializes the main variables related to an integrated data source.

        :param data: Pandas DataFrame object
        :param left_id_col: the column name that contains the identifier of the left record of each matching pair
        :param right_id_col: the column name that contains the identifier of the right record of each matching pair
        :param out_entity_col: the column name that contains the identifier of the output entities
        :param match_label_col: the column name that contains the label (match/non-match) of each matching pair
        """

        self.left_id_col = left_id_col
        self.right_id_col = right_id_col
        self.match_label_col = match_label_col
        self.entity_id_col = out_entity_col
        self.data = data[data[match_label_col] == 1]

    def _cluster_record_ids_by_entity(self):
        """
        This method transitively extends the matching pairs between records in order to obtain clusters of records that
        refer to the same real-world entity.
        This method organizes the matching pairs with a graph data structure and then performs their transitive
        extension by applying a standard connected component algorithm.
        :return: a list of sets, where each set contains the record ids that refer to the same real-world entity
        """

        # STEP 1: creation of the graph of matches

        # organize ONLY THE MATCHES in a graph data structure
        # get unique left and right matches record ids and prefix them with the "name" of the data source of origin
        # these extended ids are the names of the nodes of the graph
        left_matches = ["left_{}".format(l) for l in self.data[self.left_id_col].unique()]
        right_matches = ["right_{}".format(r) for r in self.data[self.right_id_col].unique()]
        unique_matches = left_matches + right_matches

        # create the graph of matches
        matches_graph = nx.Graph()
        for node in unique_matches:  # nodes
            matches_graph.add_node(node)
        for match_pair in self.data[[self.left_id_col, self.right_id_col]].values:  # edges
            matches_graph.add_edge("left_{}".format(match_pair[0]), "right_{}".format(match_pair[1]))

        # plot graph of matches
        # fig = plt.figure(figsize=(20, 10))
        # edges = matches_graph.edges()
        # pos = nx.spring_layout(matches_graph)
        # nx.draw_networkx_nodes(matches_graph, pos, node_size = 200)
        # nx.draw_networkx_labels(matches_graph, pos)
        # nx.draw_networkx_edges(matches_graph, pos, edgelist=edges, arrows=False)

        # STEP 2: find clusters of records by entity

        # apply connected components algorithm
        cluster_entities = []
        ccs = nx.connected_components(matches_graph)
        for cc in ccs:
            cluster_entities.append(cc)

        return cluster_entities

    def group_records_by_entity(self):
        """
        This method performs two task:
        1. transitively extends the matching pairs between records in order to obtain clusters of records that
        refer to the same real-world entity
        2. converts a list of sets, where each set contains the record ids that refer to the same real-world
        entity, in a tabular format.

        :return: Pandas dataframe object that contains records labelled by entity id
        """

        def add_entity_id(x, id_map):
            lid = f'left_{x[self.left_id_col]}'
            rid = f'right_{x[self.right_id_col]}'
            assert id_map[lid] == id_map[rid]
            return id_map[lid]

        cluster_entities = self._cluster_record_ids_by_entity()
        entity_map = {}
        for i in range(len(cluster_entities)):
            cluster = cluster_entities[i]
            for node in cluster:
                if node in entity_map:
                    raise ValueError("Overlap!")
                entity_map[node] = i
        entity_ids = self.data.apply(lambda x: add_entity_id(x, entity_map), axis=1)
        out_data = self.data.copy()
        out_data[self.entity_id_col] = entity_ids.values

        # Filter out the groups that contain less than 3 records
        entity_counts = entity_ids.value_counts()
        target_entity_ids = list(entity_counts[entity_counts > 2].index)
        if len(target_entity_ids) > 0:
            out_data = out_data[out_data[self.entity_id_col].isin(target_entity_ids)]
            out_data = out_data.sort_values(by=self.entity_id_col)
        else:
            out_data = pd.DataFrame()

        return out_data


def convert_matching_pairs_to_integrated_dataset(data, left_id, right_id, match_label, out_entity_col_id):
    """
    This function converts a list of matching pairs into an integrated dataset.

    :param data: Pandas DataFrame object containing the dataset
    :param left_id: the foreign key column in the matching pairs data that refers to the records of the first dataset
    :param right_id: the foreign key column in the matching pairs data that refers to the records of the second dataset
    :param match_label: the column in the matching pairs data that identifies as match or non-match each matching pair
    :return: first dataset, second dataset, integrated dataset
    """

    integration_component = IntegratedDataPreparationComponent(data, left_id, right_id, match_label, out_entity_col_id)
    integrated_dataset = integration_component.group_records_by_entity()

    return integrated_dataset


def get_preds(tuned_model, eval_dataset, collate_fn=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tuned_model.to(device)
    tuned_model.eval()

    loader = DataLoader(eval_dataset, batch_size=32, num_workers=4, shuffle=False, collate_fn=collate_fn)

    preds = None
    labels = None
    with torch.no_grad():
        for features in tqdm(loader):
            input_ids = features['input_ids']
            attention_mask = features['attention_mask']
            batch_labels = features['labels'].numpy()

            if isinstance(tuned_model, ContrastiveClassifierModel):
                input_ids_right = features['input_ids_right']
                attention_mask_right = features['attention_mask_right']
                outputs = tuned_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=features['labels'],
                    input_ids_right=input_ids_right, attention_mask_right=attention_mask_right
                )
                logits = outputs[1]
                logits[logits >= 0.5] = 1
                logits[logits < 0.5] = 0
                pred = logits.numpy().flatten()

            else:
                token_type_ids = features['token_type_ids']
                outputs = tuned_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                pred = torch.argmax(logits, axis=1).numpy().reshape(-1)

            if preds is None:
                preds = pred
                labels = batch_labels
            else:
                preds = np.concatenate((preds, pred))
                labels = np.concatenate((labels, batch_labels))

    return labels, preds


def supcon_clique_effectiveness(uc, lm, data_type, max_len):
    supcon_collector = DataCollectorSupCon()
    data_path = supcon_collector.get_path(use_case=uc, data_type=data_type)

    # Load the data
    subset_dataset = None
    dataset = uc
    size = None
    if supcon_collector.is_wdc_dataset(uc):
        size = uc.split('_')[0].lower()
        subset_dataset = uc.split('_')[1].lower()
        dataset = 'lspc'
    dataset = ContrastiveClassificationDataset(
        data_path, dataset_type=data_type, tokenizer=lm, dataset=dataset, sub_dataset=subset_dataset, size=size
    )

    drop_cols = ['cluster_id_left', 'cluster_id_right', 'pair_id', 'features_left', 'features_right', 'category_left',
                 'category_right', 'identifiers_left', 'identifiers_right', 'keyValuePairs_left', 'keyValuePairs_right',
                 'specTableContent_left', 'specTableContent_right']
    df = dataset.raw_data
    for drop_col in drop_cols:
        if drop_col in df.columns:
            df = df.drop(drop_cols, axis=1)

    integrated_data = convert_matching_pairs_to_integrated_dataset(
        data=df,
        left_id='id_left',
        right_id='id_right',
        match_label='label',
        out_entity_col_id='entity_id'
    )

    num_cliques = len(integrated_data['entity_id'].unique()) if len(integrated_data) > 0 else 0
    print(uc, num_cliques)

    if num_cliques > 0:
        dataset = Subset(dataset, indices=list(integrated_data.index))

        # Load the model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join(RESULTS_DIR, "supcon", f"{uc}.bin")
        model = ContrastiveClassifierModel(
            checkpoint_path=model_path,
            len_tokenizer=len(dataset.dataset.tokenizer),
            model=lm,
            frozen=True,
            device=device
        )

        data_collator = DataCollatorContrastiveClassification(
            tokenizer=dataset.dataset.tokenizer, max_length=max_len
        )
        labels, preds = get_preds(model, dataset, collate_fn=data_collator)
        assert len(labels) == len(preds)

        acc = 0
        for eid, group_preds in pd.Series(preds).groupby(integrated_data['entity_id'].values):
            if (group_preds == 0).sum() > 0:
                acc += 1
        print(f"ACC: {acc}/{num_cliques}={acc / num_cliques}")
    else:
        print("No cliques!")


def simple_bert_clique_effectiveness(uc, bench, lm, data_type, approach, max_len):
    if bench == 'wdc':
        collector = DataCollectorWDC()
        data_path = collector.get_path(use_case=uc, data_type=data_type)
    elif bench == 'dm':
        collector = DataCollector()
        data_path = collector.get_path(use_case=uc, data_type=data_type)
    else:
        raise ValueError("Wrong benchmark name!")

    integrated_data = convert_matching_pairs_to_integrated_dataset(
        data=pd.read_csv(data_path),
        left_id='left_id',
        right_id='right_id',
        match_label='label',
        out_entity_col_id='entity_id'
    )

    num_cliques = len(integrated_data['entity_id'].unique()) if len(integrated_data) > 0 else 0
    print(uc, num_cliques)

    if num_cliques > 0:
        conf = {
            'use_case': uc,
            'model_name': lm,
            'tok': 'sent_pair',
            'label_col': 'label',
            'left_prefix': 'left_',
            'right_prefix': 'right_',
            'max_len': max_len,
            'permute': False,
            'verbose': False,
            'bench': bench
        }

        data_conf = conf.copy()
        data_conf['data_type'] = data_type
        dataset = get_dataset(data_conf)
        dataset = Subset(dataset, indices=list(integrated_data.index))

        if approach == 'bert':
            model_path = os.path.join(RESULTS_DIR, f"{uc}_sent_pair_tuned")
        elif approach == 'sbert':
            model_path = os.path.join(RESULTS_DIR, 'sbert', f"{uc}_sent_pair_tuned")
        else:
            raise ValueError("Approach not found!")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        labels, preds = get_preds(model, dataset, collate_fn=dataset.dataset.pad)

        acc = 0
        for eid, group_preds in pd.Series(preds).groupby(integrated_data['entity_id'].values):
            if (group_preds == 0).sum() > 0:
                acc += 1
        print(f"ACC: {acc}/{num_cliques}={acc / num_cliques}")
    else:
        print("No cliques!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Clique effectiveness')
    parser.add_argument('-approach', '--approach', type=str, default='bert', required=True,
                        choices=['bert', 'sbert', 'ditto', 'supcon'],
                        help='the EM approach to use')
    parser.add_argument('-bert_model', '--bert_model', default='bert-base-uncased', type=str, required=True,
                        choices=['bert-base-uncased', 'sentence-transformers/nli-bert-base', 'roberta-base'],
                        help='the version of the BERT model')
    args = parser.parse_args()

    approach = args.approach
    lm = args.bert_model
    max_len = 128
    use_cases = [
        # ('Structured_DBLP-GoogleScholar', 'valid', 'dm'),
        # ('Dirty_DBLP-GoogleScholar', 'valid', 'dm'),
        # ('Structured_DBLP-GoogleScholar', 'test', 'dm'),
        # ('Dirty_DBLP-GoogleScholar', 'test', 'dm'),
        ('Large_Computers', 'valid', 'wdc'),
        # ('Large_Cameras', 'valid', 'wdc'),
        # ('Large_Shoes', 'valid', 'wdc'),
        # ('Large_Watches', 'valid', 'wdc')
    ]

    for uc, data_type, bench in use_cases:

        if approach in ['bert', 'sbert']:
            simple_bert_clique_effectiveness(uc, bench, lm, data_type, approach, max_len)
        elif approach == 'supcon':
            supcon_clique_effectiveness(uc, lm, data_type, max_len)
        else:
            raise ValueError("Approach not found!")




