from utils.general import get_dataset
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import f1_score
from core.data_models.em_dataset import EMDataset
from pathlib import Path
import os
import pickle
from multiprocessing import Process


PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'pretrain')


def compute_features(dataset: EMDataset, bert_model, tokenizer: str, featurizer: str, save_path: str = None):
    """
    This method extracts some features from the input dataset by combining the embeddings of the left and right entities
    extracted from the input BERT model.
    Two methods for combining the embeddings are available:
    - concat: it concatenates the left and right embeddings
    - diff: it computes the difference between the left and the right embeddings
    """

    assert isinstance(dataset, EMDataset)
    assert isinstance(tokenizer, str)
    assert tokenizer in ['sent_pair', 'attr_pair']
    assert isinstance(featurizer, str)
    assert featurizer in ['concat', 'diff']
    if save_path is not None:
        assert isinstance(save_path, str)

    features = []
    for row in tqdm(dataset):
        input_ids = row['input_ids'].unsqueeze(0)
        attention_mask = row['attention_mask'].unsqueeze(0)
        token_type_ids = row['token_type_ids'].unsqueeze(0)

        output = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        states = output.hidden_states
        # stack and sum the last 4 layers
        output = torch.stack([states[i] for i in [-4, -3, -2, -1]]).sum(0).squeeze()

        try:
            pad_idx = list(attention_mask[0].numpy()).index(0)
        except ValueError:
            pad_idx = len(attention_mask[0].numpy())

        sep_idx = list(token_type_ids[0].numpy()).index(1) - 1
        left_emb = output[1:sep_idx].mean(0)
        right_emb = output[sep_idx + 1: pad_idx].mean(0)

        if featurizer == 'concat':
            merge_emb = torch.cat([left_emb, right_emb])
        else:   # diff
            merge_emb = left_emb - right_emb

        features.append(merge_emb.detach().numpy())

    features = np.array(features)

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(features, f)

    return features


def train_model(X, y, save_path=None, hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=20000,
                verbose=True):

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, max_iter=max_iter,
                        verbose=verbose)
    mlp.fit(X, y)

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(mlp, f)

    return mlp


def eval_model(model, X, y):
    y_pred = model.predict(X)
    return f1_score(y, y_pred)


def run_experiment(conf, featurizer_method, experiment, bert_model, res_dir):

    use_case = conf['use_case']
    out_dir = os.path.join(res_dir, use_case)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file_name = f'{use_case}_{conf["tok"]}_{featurizer_method}.pkl'
    train_save_path = os.path.join(out_dir, f'DATA_train_{out_file_name}')
    test_save_path = os.path.join(out_dir, f'DATA_test_{out_file_name}')
    model_save_path = os.path.join(out_dir, f'MODEL_{out_file_name}')

    train_conf = conf.copy()
    train_conf['use_case'] = use_case
    train_conf['data_type'] = 'train'
    train_dataset = get_dataset(train_conf)
    y_train = train_dataset.get_complete_data()['label'].values

    test_conf = conf.copy()
    test_conf['use_case'] = use_case
    test_conf['data_type'] = 'test'
    test_dataset = get_dataset(test_conf)
    y_test = test_dataset.get_complete_data()['label'].values

    if experiment == 'compute_features':
        compute_features(train_dataset, bert_model, tokenizer=conf['tok'], featurizer=featurizer_method,
                         save_path=train_save_path)
        compute_features(test_dataset, bert_model, tokenizer=conf['tok'], featurizer=featurizer_method,
                         save_path=test_save_path)

    elif experiment == 'train':

        # load train features
        X_train = pickle.load(open(train_save_path, "rb"))
        # train the model
        train_model(X_train, y_train, save_path=model_save_path)

    elif experiment == 'eval':

        # load test features
        X_test = pickle.load(open(test_save_path, "rb"))
        # load the trained model
        classifier = pickle.load(open(model_save_path, "rb"))
        # evaluate the model on the test set
        f1 = eval_model(classifier, X_test, y_test)
        print(f"F1({use_case}, {conf['tok']}, {featurizer_method}) = {f1}")

    else:
        raise ValueError("Wrong experiment name.")


if __name__ == '__main__':

    # [BEGIN] INPUT PARAMETERS
    # use_cases = ["Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
    #              "Structured_Amazon-Google", "Structured_Walmart-Amazon",
    #              "Structured_Beer", "Structured_iTunes-Amazon",
    #              "Structured_Fodors-Zagats", "Textual_Abt-Buy",
    #              "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
    #              "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]
    use_cases = ["Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM", "Dirty_DBLP-ACM", "Dirty_DBLP-GoogleScholar"]
    # use_cases = ["Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer", "Structured_iTunes-Amazon"]
    # use_cases = ["Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_Walmart-Amazon"]

    conf = {
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',  # 'sent_pair', 'attr', 'attr_pair',
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
    }

    featurizer_method = 'diff'   # 'concat', 'diff'

    experiment = 'eval'     # 'compute_features', 'train', 'eval'

    # [END] INPUT PARAMETERS

    if experiment == 'compute_features':
        bert_model = AutoModel.from_pretrained(conf['model_name'], output_hidden_states=True)
    else:
        bert_model = None

    processes = []
    for use_case in use_cases:
        uc_conf = conf.copy()
        uc_conf['use_case'] = use_case
        p = Process(target=run_experiment,
                    args=(uc_conf, featurizer_method, experiment, bert_model, RESULTS_DIR,))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    # for use_case in use_cases:
    #     run_experiment(uc_conf, featurizer_method, RESULTS_DIR)
