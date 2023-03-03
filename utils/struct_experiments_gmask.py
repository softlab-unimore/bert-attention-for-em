import argparse
import logging
import os
import shutil
from utils.GMASK.pytorch_transformers import (BertConfig,
                                          BertForSequenceClassification, BertTokenizer)

from utils.GMASK.utils_glue import (output_modes, processors)

from utils.data_collector import DM_USE_CASES

from structure_quality_evaluation import createJSON, getQuality


os.environ["CUDA_VISIBLE_DEVICES"]="0"

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

def creationArgs():
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
    parser.add_argument('--gpu', default=-1, type=int, help='0:gpu, -1:cpu')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
    parser.add_argument("-f", "--file", required=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    import os
    from utils.GMASK import explain
    args = creationArgs()
    if args.data_dir == 'all':
        nameData = DM_USE_CASES
    else:
        nameData = [args.data_dir]

    for listDataset in nameData:
        typeData = listDataset.split('_')[0]
        nameDataset = listDataset.split('_')[1]
        args.data_dir = '..\\data\\'+typeData+'\\'+nameDataset
        args.task_name = 'general'
        args.output_dir = '..\\results\\models\\'+typeData+'_'+nameDataset+'_sent_pair_tuned'

        typeExtr = args.model_type.capitalize()


        file_names = ['added_tokens.json', 'special_tokens_map.json', 'vocab.txt']
        for file_name in file_names:
            source_file = os.path.join('..\\utils\\GMASK', file_name)
            dest_file = os.path.join(args.output_dir, file_name)
            if not os.path.exists(dest_file):
                shutil.copy(source_file, dest_file)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        if args.gpu > -1:
            args.device = "cuda"
        else:
            args.device = "cpu"

        args.n_gpu = 1
        # Set seed
        explain.set_seed(args)
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
        test_dataset, listInfo, listAttr = explain.load_and_cache_examples(args, args.task_name, tokenizer, type='test')
        post_acc = explain.test_explain(args, model, tokenizer, test_dataset, listInfo, listAttr)
        createJSON(typeData+'_'+nameDataset,typeExtr)
        getQuality(typeData+'_'+nameDataset, typeExtr)