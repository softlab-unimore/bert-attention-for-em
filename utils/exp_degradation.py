import pickle
import shutil

import pandas as pd
from transformers import AutoModelForSequenceClassification
import numpy as np
import os
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
import copy
from core.data_models.em_dataset import EMDataset
from utils.general import get_dataset
from utils.data_collector import DM_USE_CASES
from pathlib import Path
import argparse
import distutils.util
import csv
from sklearn.metrics import precision_score
import re
from utils.GMASK.utils_glue import (convert_examples_to_features,
                        output_modes, processors)
import random
from utils.GMASK.group_mask import interpret, mostImpWord
from utils.GMASK.pytorch_transformers import (BertConfig,
                                          BertForSequenceClassification, BertTokenizer)
from utils.GMASK import explain

from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)


PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def createOrdImp(args, model, tokenizer, eval_dataset, listInfo, listAttrName):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    count = 0
    model.eval()
    listTokenImp = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        set_seed(args)
        count += 1
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

        _, pred = logits.max(dim=1)

        # explain model prediction
        input_words = []
        for btxt in batch[0].data[0]:
            btxt = int(btxt)
            if tokenizer.ids_to_tokens[btxt] != '[PAD]' and tokenizer.ids_to_tokens[btxt] != '[SEP]' and tokenizer.ids_to_tokens[btxt] != '[CLS]':
                input_words.append(tokenizer.ids_to_tokens[btxt])
        args.input_words = input_words
        listTokenImp.append([mostImpWord(args, model, inputs, pred),input_words])
    return listTokenImp

def createImportanceWord(use_case):
    args.task_name = 'general'
    args.output_dir = '..\\results\\models\\'+ use_case +'_sent_pair_tuned'
    args.data_dir = '..\\data\\'+use_case.split('_')[0]+'\\'+use_case.split('_')[1]
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    # Load a trained model and vocabulary that you have fine tuned
    model = model_class.from_pretrained('..\\results\\models\\'+ use_case +'_sent_pair_tuned')
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    model.to(args.device)
    # Test
    test_dataset, listInfo, listAttr = explain.load_and_cache_examples(args, args.task_name, tokenizer, type='test')
    listTokenImp = createOrdImp(args, model, tokenizer, test_dataset, listInfo, listAttr)
    return listTokenImp


def evaluate(args, importantWord, model_path: str, eval_dataset: EMDataset):
    assert isinstance(model_path, str), "Wrong data type for parameter 'model_path'."
    assert isinstance(eval_dataset, EMDataset), "Wrong data type for parameter 'eval_dataset'."
    assert os.path.exists(model_path), f"No model found at {model_path}."

    print("Starting the inference task...")

    tuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)

    tuned_model.to('cpu')
    tuned_model.eval()

    preds = []
    labels = []
    predsMatch = []
    predsNoMatch = []
    lineDet = 0

    for features in tqdm(eval_dataset):
        input_ids = features['input_ids'].unsqueeze(0)
        columnMaskInt = list(map(int, re.findall('(\d+)', columnMask)))
        listAttr = [x for x in range(len(importantWord[0]))]
        diff = list(set(listAttr) - set(columnMaskInt))
        listInput = input_ids.tolist()[0]
        listTokenIds = []
        listAttentions = []
        unUpdate = [0] + [i for i, val in enumerate(listInput) if val == 102]
        if len(diff) > 0:
            wordRemoved = []
            for idxRem in diff:
                wordRemoved = wordRemoved + importantWord[lineDet][idxRem]

            countOrigValue = 0
            keepToken = []
            valueTok = 0
            valueAtt = 1
            for idx, tok in enumerate(listInput):
                if idx in unUpdate:
                    keepToken.append(tok)
                    if idx >= unUpdate[1] and idx < unUpdate[2]:
                        listTokenIds.append(valueTok)
                        listAttentions.append(valueAtt)
                        valueTok = 1
                    elif idx >= unUpdate[2]:
                        listTokenIds.append(valueTok)
                        listAttentions.append(valueAtt)
                        valueTok = 0
                        valueAtt = 0

                else:
                    if not countOrigValue in wordRemoved:
                        keepToken.append(tok)

                    countOrigValue+=1
                    listTokenIds.append(valueTok)
                    listAttentions.append(valueAtt)
            for i in range(len(listTokenIds)-len(keepToken)):
                keepToken.append(0)
            input_ids = torch.as_tensor([keepToken])
            token_type_ids = torch.as_tensor([listTokenIds])
            attention_mask = torch.as_tensor([listAttentions])

        else:
            token_type_ids = features['token_type_ids'].unsqueeze(0)
            attention_mask = features['attention_mask'].unsqueeze(0)

        label = features['labels'].tolist()
        labels.append(label)

        outputs = tuned_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        # attns = outputs['attentions']
        pred = torch.argmax(logits, axis=1).tolist()
        preds.append(pred)

        if label == 1:
            predsMatch.append(pred[0])
        else:
            predsNoMatch.append(pred[0])
        lineDet +=1

    precZer = predsNoMatch.count(0)/len(predsNoMatch)
    precUn = predsMatch.count(1)/len(predsMatch)

    print("Precision 0: " + str(precZer))
    print("Precision 1: " + str(precUn))
    print("Precision Total: " + str(precision_score(labels, preds)))
    print("F1: {}".format(f1_score(labels, preds)))
    return f1_score(labels, preds), precZer, precUn, precision_score(labels, preds)

def getDictAttr(listAttr):
    dictVal = {}
    for nameAttr in listAttr:
        if 'left' in nameAttr and nameAttr != 'left_id':
            dictVal[nameAttr.split('_')[1]] = 0
    return dictVal


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def defineListImpWord(numberAttr, listImpWord):
    listIndexWord = []
    for listInt in listImpWord:
        listIdxWord = copy.deepcopy(listInt[0])
        listIdxWord = np.argsort(listIdxWord)
        listIdxWord = list(listIdxWord)
        listIdxWord.reverse()
        sublist_size = int(truncate(len(listIdxWord)/numberAttr))
        sublists = [listIdxWord[i:i+sublist_size] for i in range(0, len(listIdxWord), sublist_size)]
        if len(sublists) > numberAttr:
            sublists[-2].extend(sublists.pop())
        listIndexWord.append(sublists)

    return listIndexWord



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
    parser.add_argument('-typeMask', '--typeMask', default='off', type=str,
                        help='mask typing')
    parser.add_argument('-columnMask', '--columnMask', default='0', type=str,
                        help='list attributes to mask')
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
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--gpu', default=-1, type=int, help='0:gpu, -1:cpu')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
    parser.add_argument("--max_seq_length", default=100, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    args = parser.parse_args()

    fit = args.fit
    use_cases = args.use_cases
    if use_cases == ['all']:
        use_cases = DM_USE_CASES

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.gpu > -1:
        args.device = "cuda"
    else:
        args.device = "cpu"

    args.n_gpu = 1


    for use_case in use_cases:
        out_model_path = os.path.join(RESULTS_DIR, f"{use_case}_{args.tok}_tuned")
        f = open(out_model_path+'\\resultsLerfMerf.csv', 'w')
        writer = csv.writer(f)
        header = ['Dataset', 'Attribue Keep', 'F1']
        writer.writerow(header)

        dictDegr = {'listMrf' : {0:[],1:[],'f1':[]},'listLrf' : {0:[],1:[], 'f1':[]}}
        analys = 'listMrf'

        pathFile = '..\\data\\'+use_case.split('_')[0]+'\\'+use_case.split('_')[1]+'\\test.csv'
        df = pd.read_csv(pathFile)
        getDict = getDictAttr(list(df.head()))
        listCompl = []

        for value in range(len(getDict.keys())):
            listCompl.append(str(list(range(value+1))))
        listCompl.reverse()
        listCompl.append('Change')
        i = 0
        for value in range(len(getDict.keys())):
            listCompl.append(str(list(range(value,len(getDict.keys()),1))))

        for columnMask in listCompl:
            if columnMask != 'Change':
                print('Info. Dataset= ' + use_case + ', attributeKeep= ' + columnMask)
                conf = {
                    'use_case': use_case,
                    'data_type': 'train',
                    'model_name': args.bert_model,
                    'tok': 'sent_pair',
                    'label_col': args.label_col,
                    'left_prefix': args.left_prefix,
                    'right_prefix': args.right_prefix,
                    'max_len': args.max_len,
                    'permute': args.permute,
                    'verbose': args.verbose,
                    'typeMask': "off",
                    'columnMask': columnMask,
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


                file_names = ['added_tokens.json', 'special_tokens_map.json', 'vocab.txt']
                for file_name in file_names:
                    source_file = os.path.join('..\\utils\\GMASK', file_name)
                    dest_file = os.path.join(out_model_path, file_name)
                    if not os.path.exists(dest_file):
                        shutil.copy(source_file, dest_file)

                importantWord = createImportanceWord(use_case)
                listIndexImpWord = defineListImpWord(len(getDict.keys()), importantWord)
                f1Obt, precZer, precUn, precTot = evaluate(args, listIndexImpWord, out_model_path, test_dataset)
                dictDegr[analys][0].append(precZer)
                dictDegr[analys][1].append(precUn)
                dictDegr[analys]['f1'].append(f1Obt)
                writer.writerow([use_case,'[' + columnMask + ']', str(f1Obt)])
            else:
                analys = 'listLrf'
        file = open(out_model_path+'\\resultDataset_'+use_case+'_WORD.pkl', 'wb')
        pickle.dump(dictDegr, file)
