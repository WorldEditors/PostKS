

## Quick Start

### Data Preparation
Put train/valid/test data files in `data/` folder with the same prefix:
* `$prefix.train`
* `$prefix.valid`
* `$prefix.test`

### Training
    python run_seq2seq.py

### Testing
    python run_seq2seq.py --test --ckpt $model_state_file


