from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import jsonlines
from collections import defaultdict
import pickle

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm

from utils.GMASK.pytorch_transformers import (BertConfig,
                                                     BertForSequenceClassification, BertTokenizer)

from utils.GMASK.utils_glue import (convert_examples_to_features,
                        output_modes, processors)

from utils.GMASK.group_mask import interpret, mostImpWord

os.environ["CUDA_VISIBLE_DEVICES"]="0"

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

def revealAttr(all_inputs_ids):

    listInfo = []
    for listInput in all_inputs_ids:
        attr = 0
        listAttr = []
        listPos = []
        for idx in range(len(listInput)):
            if listInput[idx] == 0:
                break
            if listInput[idx] == 1405:
                attr+=1
                listPos.append(idx)
                continue
            if listInput[idx] == 101:
                continue
            if listInput[idx] != 102:
                listAttr.append(attr)
        listInfo.append({'listPos':listPos, 'listAttr':listAttr})
    return listInfo

def removeSeparetor(listaPos, listDirty):
    for idx in range(len(listDirty)):
        removed = 0
        listTor = list(listDirty[idx])
        for listIdx in listaPos[idx]['listPos']:
            del listTor[listIdx-removed]
            removed+=1
            listTor.append(torch.tensor(0))
            # listDirty[idx] = torch.cat([listDirty[idx][:listIdx],listDirty[idx][listIdx+1:].append(0),])
        listDirty[idx] = torch.tensor([listTor])
    return listDirty



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
    size = 0
    count = 0
    acc = 0
    fileobject = open('..\\utils\\GMASK\\interpretation.txt', 'w')
    fileobject1 = open('..\\utils\\GMASK\\sort_index.txt', 'w')
    model.eval()
    testSentence = 0
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





def test_explain(args, model, tokenizer, eval_dataset, listInfo, listAttrName):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    size = 0
    count = 0
    acc = 0
    fileobject = open('..\\utils\\GMASK\\interpretation.txt', 'w')
    fileobject1 = open('..\\utils\\GMASK\\sort_index.txt', 'w')
    fileobject2 = open(eval_output_dir +'\\groupFind'+args.model_type.capitalize()+'.txt', 'w')
    model.eval()
    testSentence = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        set_seed(args)
        count += 1
        
        batch = tuple(t.to(args.device) for t in batch)

        fileobject.write(str(count))
        fileobject.write('\n')
        fileobject1.write(str(count))
        fileobject1.write('\n')
        input_words = []
        for btxt in batch[0].data[0]:
            btxt = int(btxt)
            if tokenizer.ids_to_tokens[btxt] != '[PAD]' and tokenizer.ids_to_tokens[btxt] != '[SEP]' and tokenizer.ids_to_tokens[btxt] != '[CLS]':
                input_words.append(tokenizer.ids_to_tokens[btxt])
                fileobject.write(tokenizer.ids_to_tokens[btxt])
                fileobject.write(' ')
                fileobject1.write(tokenizer.ids_to_tokens[btxt])
                fileobject1.write(' ')
        fileobject.write(' >> ')
        fileobject1.write(' >> ')

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

        _, pred = logits.max(dim=1)

        fileobject.write(str(inputs['labels'].detach().cpu().numpy()[0]))
        fileobject1.write(str(inputs['labels'].detach().cpu().numpy()[0]))
        fileobject.write(' >> ')
        fileobject1.write(' >> ')

        fileobject.write(str(int(pred)))
        fileobject1.write(str(int(pred)))
        fileobject.write(' >> ')
        fileobject1.write(' >> ')

        # explain model prediction

        args.input_words = input_words
        output, z_prob, t_prob, x_ids, z_max_idx, somVal = interpret(args, model, inputs, pred)
        listGroupWord = []
        listGroupAttr = []
        for createListGroup in range(len(z_prob[0][0])):
            listGroupWord.append([])
            listGroupAttr.append([])
        for idx in range(len(z_max_idx)):
            if x_ids[idx] < somVal:
                valDetect = x_ids[idx]-1
            else:
                valDetect = x_ids[idx]-2


            wordTop = input_words[valDetect]
            attrTop = listAttrName[listInfo[testSentence]['listAttr'][valDetect]]
            listGroupWord[z_max_idx[idx]].append(wordTop)
            listGroupAttr[z_max_idx[idx]].append(attrTop)


        testSentence +=1
        acc += (output == pred).sum().float()
        size += len(pred)

        batch_text = batch[0][0]
        for idx in x_ids:
            idx = int(idx)
            fileobject.write(tokenizer.ids_to_tokens[int(batch_text[idx])])
            fileobject.write(' ')
            fileobject1.write(str(int(idx)))
            fileobject1.write(' ')
        fileobject.write(' >> ')
        fileobject1.write(' >> ')
        fileobject.write(str(output.cpu().numpy()[0]))
        fileobject.write('\n')
        fileobject1.write(str(output.cpu().numpy()[0]))
        fileobject.write('\n')


        fileobject2.write(str(count) + '\n')
        listString = []
        for group in listGroupWord:
            listString.append('[' + ','.join(group) + ']')
        fileobject2.write(','.join(listString))
        fileobject2.write('\n')
        listString = []
        for group in listGroupAttr:
            listString.append('[' + ','.join(group) + ']')
        fileobject2.write(','.join(listString))
        fileobject2.write('\n')

    acc /= size

    return acc


def load_and_cache_examples(args, task, tokenizer, type):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        # open a file, where you stored the pickled data
        file = open(args.data_dir+'/'+'listAttr.pkl', 'rb')
        # dump information to that file
        listAttr = pickle.load(file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        docs = defaultdict(dict)
        # reader = jsonlines.Reader(open(os.path.join(args.data_dir, 'docs.jsonl')))
        # for line in reader:
            # docs[line["docid"]] = line["document"]

        if type == 'train':
            examples = processor.get_train_examples(args.data_dir, docs)
        elif type == 'dev':
            examples = processor.get_dev_examples(args.data_dir, docs)
        else:
            examples, listAttr = processor.get_test_examples(args.data_dir, docs)
            file = open(args.data_dir+'/'+'listAttr.pkl', 'wb')
            pickle.dump(listAttr, file)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ['roberta']),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    listInfo = revealAttr(all_input_ids)
    all_input_ids = removeSeparetor(listInfo, all_input_ids)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_input_mask = removeSeparetor(listInfo, all_input_mask)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_segment_ids = removeSeparetor(listInfo, all_segment_ids)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    all_evidence = torch.tensor([f.evidence for f in features], dtype=torch.long)
    all_evidence = removeSeparetor(listInfo, all_evidence)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_evidence)
    return dataset, listInfo, listAttr


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='./dataset/esnli', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--task_name", default='esnli', type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default='./output/enli', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=100, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=10.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.gpu > -1:
        args.device = "cuda"
    else:
        args.device = "cpu"
    args.n_gpu = 1

    # Set seed
    set_seed(args)

    # get current path
    args.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.data_dir)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)

    # Load a trained model and vocabulary that you have fine tuned
    model = model_class.from_pretrained(args.output_dir)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    model.to(args.device)

    # Test
    test_dataset, listInfo, listAttr = load_and_cache_examples(args, args.task_name, tokenizer, type='test')
    post_acc = test_explain(args, model, tokenizer, test_dataset, listInfo, listAttr)