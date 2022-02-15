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
