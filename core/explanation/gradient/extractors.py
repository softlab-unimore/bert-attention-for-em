# Code adapted from https://github.com/koren-v/Interpret

import torch
from torch.nn.functional import softmax
import numpy as np
from tqdm import tqdm
import matplotlib
from torch import nn
from utils.bert_utils import get_entity_pair_attr_idxs, get_sent_pair_word_idxs
from utils.result_collector import BinaryClassificationResultsAggregator
import pathlib
import pickle
import os


class BaseGradientExtractor:
    def __init__(self,
                 model,
                 criterion,
                 tokenizer,
                 show_progress=True,
                 **kwargs):

        """
        :param model: nn.Module object - can be HuggingFace's model or custom one.
        :param criterion: torch criterion used to train your model.
        :param tokenizer: HuggingFace's tokenizer.
        :param show_progress: bool flag to show tqdm progress bar.
        :param kwargs:
            encoder: string indicates the HuggingFace's encoder, that has 'embeddings' attribute. Used
                if your model doesn't have 'get_input_embeddings' method to get access to encoder embeddings
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.show_progress = show_progress
        self.kwargs = kwargs
        # to save outputs in saliency_interpret
        self.batch_output = None
        self.max_len = 128
        if 'max_len' in self.kwargs:
            self.max_len = self.kwargs['max_len']
        if 'special_tokens' in self.kwargs:
            self.special = self.kwargs['special_tokens']

    def _get_gradients(self, batch):
        # set requires_grad to true for all parameters, but save original values to
        # restore them later
        original_param_name_to_requires_grad_dict = {}
        for param_name, param in self.model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True
        embedding_gradients = []
        hooks = self._register_embedding_gradient_hooks(embedding_gradients)

        loss = self.forward_step(batch)

        self.model.zero_grad()
        loss.backward()

        for hook in hooks:
            hook.remove()

        # restore the original requires_grad values of the parameters
        for param_name, param in self.model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

        return embedding_gradients[0]

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        """
        Registers a backward hook on the
        Used to save the gradients of the embeddings for use in get_gradients()
        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.
        """

        def hook_layers(module, grad_in, grad_out):
            embedding_gradients.append(grad_out[0])

        backward_hooks = []
        embedding_layer = self.get_embeddings_layer()
        backward_hooks.append(embedding_layer.register_backward_hook(hook_layers))
        return backward_hooks

    def get_embeddings_layer(self):
        if hasattr(self.model, "get_input_embeddings"):
            embedding_layer = self.model.get_input_embeddings()
        else:
            encoder_attribute = self.kwargs.get("encoder")
            assert encoder_attribute, "Your model doesn't have 'get_input_embeddings' method, thus you " \
                                      "have provide 'encoder' key argument while initializing SaliencyInterpreter object"
            embedding_layer = getattr(self.model, encoder_attribute).embeddings
        return embedding_layer

    def colorize(self, instance, skip_special_tokens=False):

        special_tokens = self.special_tokens

        word_cmap = matplotlib.cm.Blues
        prob_cmap = matplotlib.cm.Greens
        template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
        colored_string = ''
        # Use a matplotlib normalizer in order to make clearer the difference between values
        normalized_and_mapped = matplotlib.cm.ScalarMappable(cmap=word_cmap).to_rgba(instance['grad'])
        for word, color in zip(instance['tokens'], normalized_and_mapped):
            if word in special_tokens and skip_special_tokens:
                continue
            # handle wordpieces
            word = word.replace("##", "") if "##" in word else ' ' + word
            color = matplotlib.colors.rgb2hex(color[:3])
            colored_string += template.format(color, word)
        colored_string += template.format(0, "    Label: {} |".format(instance['label']))
        prob = instance['prob']
        color = matplotlib.colors.rgb2hex(prob_cmap(prob)[:3])
        colored_string += template.format(color, "{:.2f}%".format(instance['prob'] * 100)) + '|'
        return colored_string

    @property
    def special_tokens(self):
        """
        Some tokenizers don't have 'eos_token' and 'bos_token' attributes.
        So needed we some trick to get them.
        """
        if self.tokenizer.bos_token is None or self.tokenizer.eos_token is None:
            special_tokens = self.tokenizer.build_inputs_with_special_tokens([])
            special_tokens_ids = self.tokenizer.convert_ids_to_tokens(special_tokens)
            self.tokenizer.bos_token, self.tokenizer.eos_token = special_tokens_ids

        special_tokens = self.tokenizer.eos_token, self.tokenizer.bos_token
        return special_tokens

    def forward_step(self, batch):
        """
        If your model receive inputs in another way or you computing not
         like in this example simply override this method. It should return the batch loss
        :param batch: batch returned by dataloader
        :return: torch.Tensor: batch loss
        """
        entity_pair = (batch[0], batch[1])
        input_ids = batch[2]['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = batch[2]["attention_mask"].unsqueeze(0).to(self.device)
        token_type_ids = batch[2]["token_type_ids"].unsqueeze(0).to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs['logits']
        label = torch.argmax(logits, dim=1)
        batch_losses = self.criterion(logits, label)
        loss = torch.mean(batch_losses)

        self.batch_output = [input_ids, entity_pair, outputs]

        return loss

    def update_output(self):
        """
        You can also override this method if you want to change the format
         of outputs. (e.g. store just gradients)
        :return: batch_output
        """

        input_ids, entity_pair, outputs, grads = self.batch_output

        probs = softmax(outputs['logits'], dim=-1)
        probs, labels = torch.max(probs, dim=-1)

        tokens = [
            self.tokenizer.convert_ids_to_tokens(input_ids_)
            for input_ids_ in input_ids
        ]

        embedding_grads = grads.sum(dim=2)
        # norm for each sequence
        norms = torch.norm(embedding_grads, dim=1, p=1)
        # normalizing
        for i, norm in enumerate(norms):
            embedding_grads[i] = torch.abs(embedding_grads[i]) / norm

        batch_output = []

        iterator = zip(tokens, probs, embedding_grads, labels)

        for example_tokens, example_prob, example_grad, example_label in iterator:
            example_dict = dict()
            # as we do it by batches we has a padding so we need to remove it
            example_tokens = [t for t in example_tokens if t != self.tokenizer.pad_token]
            example_dict['tokens'] = example_tokens
            example_dict['grad'] = example_grad.cpu().tolist()[:len(example_tokens)]
            example_dict['label'] = example_label.item()
            example_dict['prob'] = example_prob.item()
            batch_output.append(example_dict)

        return batch_output


class GradientExtractor(BaseGradientExtractor):
    """
    This class extracts Integrated Gradients (https://arxiv.org/abs/1703.01365)
    """

    def __init__(self,
                 model,
                 criterion,
                 tokenizer,
                 num_steps=20,
                 show_progress=True,
                 **kwargs):
        super().__init__(model, criterion, tokenizer, show_progress, **kwargs)
        # Hyperparameters
        self.num_steps = num_steps

    def saliency_interpret(self, test_dataloader):

        instances_with_grads = []
        iterator = tqdm(test_dataloader) if self.show_progress else test_dataloader

        for batch in iterator:
            # we will store there batch outputs such as gradients, probability, tokens
            # so as each of them are used in different places, for convenience we will create
            # it as attribute:
            self.batch_output = []
            self._integrate_gradients(batch)
            batch_output = self.update_output()
            instances_with_grads.extend(batch_output)

        return instances_with_grads

    def _register_forward_hook(self, alpha, embeddings_list):
        """
        Register a forward hook on the embedding layer which scales the embeddings by alpha. Used
        for one term in the Integrated Gradients sum.
        We store the embedding output into the embeddings_list when alpha is zero.  This is used
        later to element-wise multiply the input by the averaged gradients.
        """

        def forward_hook(module, inputs, output):
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach())

            # Scale the embedding by alpha
            output.mul_(alpha)

        embedding_layer = self.get_embeddings_layer()
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _integrate_gradients(self, batch):

        ig_grads = None

        # List of Embedding inputs
        embeddings_list = []

        # Exclude the endpoint because we do a left point integral approximation
        for alpha in np.linspace(0, 1.0, num=self.num_steps, endpoint=False):
            # Hook for modifying embedding value
            handle = self._register_forward_hook(alpha, embeddings_list)

            grads = self._get_gradients(batch)
            handle.remove()

            # Running sum of gradients
            if ig_grads is None:
                ig_grads = grads
            else:
                ig_grads = ig_grads + grads

        # Average of each gradient term
        ig_grads /= self.num_steps

        # Gradients come back in the reverse order that they were sent into the network
        embeddings_list.reverse()

        # Element-wise multiply average gradient by the input
        ig_grads *= embeddings_list[0]

        self.batch_output.append(ig_grads)


class EntityGradientExtractor(object):
    grad_agg_fns = {'sum': np.sum, 'max': np.max, 'avg': np.mean, 'median': np.median}

    def __init__(self, model, tokenizer, text_unit: str, special_tokens: bool = False, show_progress: bool = True):

        assert isinstance(text_unit, str), "Wrong data type for parameter 'text_unit'."
        assert isinstance(special_tokens, bool), "Wrong data type for parameter 'special_tokens'."
        text_units = ['tokens', 'words', 'attrs']
        assert text_unit in text_units, f"Wrong text_unit: {text_unit} not in {text_units}."
        assert isinstance(show_progress, bool), "Wrong data type for parameter 'show_progress'."

        self.tokenizer = tokenizer
        self.grad_extractor = GradientExtractor(
            model,
            nn.CrossEntropyLoss(),
            tokenizer,
            show_progress=show_progress,
        )
        self.text_unit = text_unit
        self.special_tokens = special_tokens
        self.grads_history = []

    def get_text_unit_grads(self, data: dict, grad: list, special_token_data: dict = None):
        assert isinstance(data, dict), "Wrong data type for parameter 'data'."
        params = ['left_names', 'right_names', 'left_idxs', 'right_idxs']
        assert all([p in data for p in params]), "Wrong data format for parameter 'data'."
        assert isinstance(grad, list), "Wrong data type for parameter 'grad'."
        if special_token_data is not None:
            assert isinstance(special_token_data, dict), "Wrong data type for parameter 'special_token_data'."
            assert all([p in special_token_data for p in
                        ['names', 'idxs']]), "Wrong data format for parameter 'special_token_data'."
            assert all([len(special_token_data[p]) == 3 for p in
                        ['names', 'idxs']]), "Wrong data format for parameter 'special_token_data'."

        assert int(round(np.sum(grad))) == 1

        l = data['left_names']
        r = data['right_names']
        l_idxs = data['left_idxs']
        r_idxs = data['right_idxs']
        if len(l_idxs) > 1:
            l = l[:len(l_idxs)]
            r = r[:len(r_idxs)]

        if special_token_data is None:
            all_units = l + r
        else:
            special_tokens = special_token_data['names']
            all_units = [special_tokens[0]] + l + [special_tokens[1]] + r + [special_tokens[2]]

        if self.text_unit == 'tokens':
            l_grad = grad[l_idxs[0][0]:l_idxs[0][1]]
            r_grad = grad[r_idxs[0][0]:r_idxs[0][1]]
        else:
            # aggregate the gradient scores that refer to the same unit (i.e., word or attribute)
            l_grad = {}
            r_grad = {}
            for grad_agg_fn_name, grad_agg_fn in EntityGradientExtractor.grad_agg_fns.items():
                l_grad[grad_agg_fn_name] = [grad_agg_fn(grad[l_idx[0]: l_idx[1]]) for l_idx in l_idxs]
                r_grad[grad_agg_fn_name] = [grad_agg_fn(grad[r_idx[0]: r_idx[1]]) for r_idx in r_idxs]

        if special_token_data is None:
            if isinstance(l_grad, dict):
                all_grad = {k: l_grad[k] + r_grad[k] for k in l_grad}
            else:
                all_grad = l_grad + r_grad
        else:
            special_idxs = special_token_data['idxs']
            special_grads = np.array(grad)[special_idxs]
            if isinstance(l_grad, dict):
                all_grad = {k: [special_grads[0]] + l_grad[k] + [special_grads[1]] + r_grad[k] + [special_grads[2]] for
                            k in l_grad}
            else:
                all_grad = [special_grads[0]] + l_grad + [special_grads[1]] + r_grad + [special_grads[2]]

        out_data = {
            'all': all_units,
            'all_grad': all_grad,
            'left': l,
            'left_grad': l_grad,
            'right': r,
            'right_grad': r_grad
        }

        return out_data

    @staticmethod
    def check_extracted_grad(grads_data: list):
        assert isinstance(grads_data, list), "Wrong data type for parameter 'grad_data'."
        assert len(grads_data) > 0, "Empty grad data."

        params = ['pred', 'prob', 'label', 'grad']
        grad_params = ['all', 'all_grad', 'left', 'left_grad', 'right', 'right_grad']
        error_msg = "Wrong data format for parameter 'grad_data'."

        for grad_data in grads_data:
            if grad_data is not None:
                assert isinstance(grad_data, dict), error_msg
                assert all([p in grad_data for p in params]), error_msg
                for key in grad_data:
                    if key in ['pred', 'label', 'prob']:
                        assert isinstance(grad_data[key], (int, float)), error_msg
                    else:
                        assert all([p in grad_data[key] for p in grad_params]), error_msg

    def extract(self, data, max_len=128, out_path=None):

        grads_data = self.grad_extractor.saliency_interpret(data)

        out_grad_data = []

        for i in range(len(data)):
            left_entity, right_entity, features = data[i]
            grad_data = grads_data[i]
            tokens = grad_data['tokens']
            grad = grad_data['grad']
            pred = grad_data['label']
            prob = grad_data['prob']
            sep_idx = tokens.index('[SEP]')

            if self.text_unit == 'tokens':
                data_for_grad = {
                    'left_names': [f'l_{t}' for t in tokens[1:sep_idx]],  # remove [CLS]
                    'right_names': [f'r_{t}' for t in tokens[sep_idx + 1: -1]],  # remove two [SEP]
                    'left_idxs': [(1, sep_idx)],
                    'right_idxs': [(sep_idx + 1, len(tokens) - 1)]
                }

            elif self.text_unit == 'words':
                sent1 = ' '.join([str(v) for v in left_entity])
                sent2 = ' '.join([str(v) for v in right_entity])

                left_word_idxs, right_word_idxs, _ = get_sent_pair_word_idxs(sent1, sent2, self.tokenizer, max_len)
                right_word_idxs = [(sep_idx + rw_idx[0], sep_idx + rw_idx[1]) for rw_idx in right_word_idxs]

                data_for_grad = {
                    'left_names': [f'l_{w}' for w in sent1.split()],
                    'right_names': [f'r_{w}' for w in sent2.split()],
                    'left_idxs': left_word_idxs,
                    'right_idxs': right_word_idxs
                }

            else:
                data_for_grad = get_entity_pair_attr_idxs(left_entity, right_entity, self.tokenizer, max_len)
                # no attribute gradients have been extracted. this happens when the attribute values have been truncated
                # as they have exceeded the max sentence length
                if data_for_grad is None:
                    out_grad_data.append(None)
                    continue

            if not self.special_tokens:
                special_token_data = None
            else:
                special_token_data = {
                    'names': ['[CLS]', '[SEP]', '[SEP]'],
                    'idxs': [0, sep_idx, len(tokens) - 1]
                }

            gradients = self.get_text_unit_grads(data_for_grad, grad, special_token_data)

            record_out_data = {
                'pred': pred,
                'prob': prob,
                'label': features['labels'].item(),
                'grad': gradients
            }

            out_grad_data.append(record_out_data)

        self.grads_history.append(out_grad_data)

        if out_path is not None:
            out_dir_path = out_path.split(os.sep)
            out_dir = os.sep.join(out_dir_path[:-1])
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            with open(f'{out_path}.pkl', 'wb') as f:
                pickle.dump(out_grad_data, f)

        return out_grad_data


class AggregateAttributeGradient(object):

    def __init__(self, grads_data: list, target_categories: list = ['all']):
        EntityGradientExtractor.check_extracted_grad(grads_data)

        self.grads_data = grads_data
        self.agg_metrics = ['mean']
        self.target_categories = target_categories

    def aggregate(self, metric: str):
        assert isinstance(metric, str), "Wrong data type for parameter 'metric'."
        assert metric in self.agg_metrics, f"Wrong metric: {metric} not in {self.agg_metrics}."

        all_data = []
        all_labels = None
        left_data = []
        left_labels = None
        right_data = []
        right_labels = None
        for grad_data in self.grads_data:
            item = {
                'label': grad_data['label'],
                'pred': grad_data['pred']
            }
            grads = grad_data['grad']

            if all_labels is not None:
                assert grads['all'] == all_labels
                assert grads['left'] == left_labels
                assert grads['right'] == right_labels
            else:
                all_labels = grads['all']
                left_labels = grads['left']
                right_labels = grads['right']

            all_item = item.copy()
            all_item['grad'] = grads['all_grad']
            all_data.append(all_item)
            left_item = item.copy()
            left_item['grad'] = grads['left_grad']
            left_data.append(left_item)
            right_item = item.copy()
            right_item['grad'] = grads['right_grad']
            right_data.append(right_item)

        all_aggregator = BinaryClassificationResultsAggregator('grad', target_categories=self.target_categories)
        all_aggregator.add_batch_data(all_data)
        if metric == 'mean':
            all_agg_data = all_aggregator.aggregate(metric)
        else:
            raise NotImplementedError()

        left_aggregator = BinaryClassificationResultsAggregator('grad', target_categories=self.target_categories)
        left_aggregator.add_batch_data(left_data)
        if metric == 'mean':
            left_agg_data = left_aggregator.aggregate(metric)
        else:
            raise NotImplementedError()

        right_aggregator = BinaryClassificationResultsAggregator('grad', target_categories=self.target_categories)
        right_aggregator.add_batch_data(right_data)
        if metric == 'mean':
            right_agg_data = right_aggregator.aggregate(metric)
        else:
            raise NotImplementedError()

        out_data = {}
        for cat in self.target_categories:

            if all_agg_data[cat] is None:
                out_data[cat] = None
                continue

            if metric == 'mean':
                all_grad = all_agg_data[cat]['mean']
                all_error_grad = all_agg_data[cat]['std']
                left_grad = left_agg_data[cat]['mean']
                left_error_grad = left_agg_data[cat]['std']
                right_grad = right_agg_data[cat]['mean']
                right_error_grad = right_agg_data[cat]['std']
            else:
                raise NotImplementedError()

            out_data[cat] = {
                'all': all_labels,
                'all_grad': all_grad,
                'all_error_grad': all_error_grad,
                'left': left_labels,
                'left_grad': left_grad,
                'left_error_grad': left_error_grad,
                'right': right_labels,
                'right_grad': right_grad,
                'right_error_grad': right_error_grad,
            }

        return out_data
