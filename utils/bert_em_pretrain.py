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
import argparse
from utils.data_collector import DM_USE_CASES
import distutils.util

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
        else:  # diff
            merge_emb = left_emb - right_emb

        features.append(merge_emb.detach().numpy())

    features = np.array(features)

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(features, f)

    return features


def train_model(X, y, train_params, save_path=None):
    mlp = MLPClassifier(**train_params)
    mlp.fit(X, y)

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(mlp, f)

    return mlp


def eval_model(model, X, y):
    y_pred = model.predict(X)
    return f1_score(y, y_pred)


def run_experiment(conf, featurizer_method, experiment, bert_model, train_params, res_dir):
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
        train_model(X_train, y_train, train_params, save_path=model_save_path)

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

    parser = argparse.ArgumentParser(description='BERT EM pretrain')

    # General parameters
    parser.add_argument('-use_cases', '--use_cases', nargs='+', required=True, choices=DM_USE_CASES + ['all'],
                        help='the names of the datasets')
    parser.add_argument('-experiment', '--experiment', type=str, required=True,
                        choices=['compute_features', 'train', 'eval'],
                        help='the experiment to run: 1) "compute_features" for calculating entity embeddings, \
                        2) "train" for training a MLP model on the entity embeddings, 3) "eval" for run the inference')
    parser.add_argument('-featurizer_method', '--featurizer_method', default='diff', type=str,
                        choices=['diff', 'concat'],
                        help='method used for elaborating the pre-trained BERT embeddings. \
                        1) "diff": the embeddings representing left and right entities are subtracted, \
                        2) "concat": the embeddings representing left and right entities are concatenated')
    parser.add_argument('-multi_process', '--multi_process', default=False,
                        type=lambda x: bool(distutils.util.strtobool(x)),
                        help='multi process modality')
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
    parser.add_argument('-hidden_layer_sizes', '--hidden_layer_sizes', default=(100,), type=tuple,
                        help='sizes of the MLP hidden layers')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                        help='MLP learning rate')
    parser.add_argument('-max_iter', '--max_iter', default=20000, type=int,
                        help='MLP training max iterations')
    parser.add_argument('-mlp_verbose', '--mlp_verbose', default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='MLP verbose modality')
    parser.add_argument('-seed', '--seed', default=42, type=int,
                        help='MLP random state')

    args = parser.parse_args()

    # [BEGIN] INPUT PARAMETERS
    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES

    conf = {
        'model_name': args.bert_model,
        'tok': args.tok,
        'label_col': args.label_col,
        'left_prefix': args.left_prefix,
        'right_prefix': args.right_prefix,
        'max_len': args.max_len,
        'permute': args.permute,
        'verbose': args.verbose,
    }

    train_params = {
        'hidden_layer_sizes': args.hidden_layer_sizes,
        'learning_rate_init': args.learning_rate,
        'max_iter': args.max_iter,
        'verbose': args.mlp_verbose,
        'random_state': int(args.seed),
    }

    featurizer_method = args.featurizer_method
    experiment = args.experiment

    # [END] INPUT PARAMETERS

    if experiment == 'compute_features':
        bert_model = AutoModel.from_pretrained(conf['model_name'], output_hidden_states=True)
    else:
        bert_model = None

    if args.multi_process:
        processes = []
        for use_case in use_cases:
            uc_conf = conf.copy()
            uc_conf['use_case'] = use_case
            p = Process(target=run_experiment,
                        args=(uc_conf, featurizer_method, experiment, bert_model, train_params, RESULTS_DIR,))
            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    else:
        for use_case in use_cases:
            uc_conf = conf.copy()
            uc_conf['use_case'] = use_case
            run_experiment(uc_conf, featurizer_method, experiment, bert_model, train_params, RESULTS_DIR)
