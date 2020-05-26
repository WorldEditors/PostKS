## Quick Start

Realization for paper: "Learning to Select Knowledge for Response Generation in Dialog Systems", Rongzhong Lian, Min Xie, Fan Wang, Jinhua Peng, Hua Wu, IJCAI 2019
(PostKS)

### Requirement
* PyTorch 0.4.1+
* Python 3.5+

### Data Preparation
Put train/valid/test data files in `data/` folder with the same prefix:
* `$prefix.train`
* `$prefix.valid`
* `$prefix.test`

Adding embeddings if you need a warm start

We provide only a small size dataset for demonstration

### Training
    python run_seq2seq.py : Seq2Seq
    python run_knowledge.py : PostKS with HGFU

### Testing
    python run_seq2seq.py --test --ckpt $model_state_file
