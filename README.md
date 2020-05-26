## Quick Start

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
