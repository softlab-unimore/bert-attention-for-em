# Analyzing How BERT Performs Entity Matching
State-of-the-art Entity Matching (EM) approaches rely on transformer architectures, such as *BERT*, for generating  highly contextualized embeddings of terms. The embeddings  are then used to predict whether pairs of entity descriptions refer to the same real-world entity. BERT-based EM models demonstrated to be effective, but act as black-boxes for the users, who have limited insight into the motivations behind their decisions.
In this repo, we perform a multi-facet analysis of the components of pre-trained and fine-tuned BERT architectures applied to an EM task.

## Library

### Requirements

- Python: Python 3.*
- Packages: requirements.txt

### Installation

```bash
$ virtualenv -p python3 venv

$ source venv/bin/activate

$ pip install -r requirements.txt

```

## Experiments

### Create BERT-based EM models
This means to create a binary classifier on top of the BERT model.
For purely demonstrative purposes, below we will train the EM model only on the dataset Structured_Fodors-Zagats.

**Option 1**: pre-trained EM model. Only the classification layer is fine-tuned on the EM task.
```python
  python -m utils.bert_em_pretrain --use_cases Structured_Fodors-Zagats --tok sent_pair --experiment compute_features
  python -m utils.bert_em_pretrain --use_cases Structured_Fodors-Zagats --tok sent_pair --experiment train
```

**Option 2**: fine-tuned EM model. Both the BERT architecture and the classification layer are fine-tuned on the EM task.
```python
  python -m utils.bert_em_fine_tuning --fit True --use_cases Structured_Fodors-Zagats --tok sent_pair
```
The model will be stored in the directory *results/models/*.

### Experiment Sec. 4.1 (Tab. 2)
Pre-trained EM model
```python
  python -m utils.bert_em_pretrain --use_cases all --tok sent_pair --experiment eval
  python -m utils.bert_em_pretrain --use_cases all --tok attr_pair --experiment eval
```
	
Fine-tuned EM model
```python
  python -m utils.bert_em_fine_tuning --fit True --use_cases all --tok sent_pair
  python -m utils.bert_em_fine_tuning --fit True --use_cases all --tok attr_pair
```

### Experiment Sec. 4.2 (Fig. 1)
```python
  python -m experiments.fine_tuning_impact_on_attention.py --use_cases all
```

### Experiment Sec. 4.3 (Fig. 2)
```python
  python -m experiments.fine_tuning_impact_on_embeddings.py --use_cases all
```

### Experiment Sec. 5.1 (Fig. 3)		

## License
[MIT License](LICENSE)
