# ParaBridge

An enhancement on LangBridge meant to properly handle paragraph-sized questions through an LSTM which aggregates sentence embeddings.

## Usage

```
pip install -r requirements.txt

export NLTK_DATA=/workspace/CACHE/NLTK_DATA

export HF_HOME="/workspace/CACHE/HF_HOME/"

python scripts/train_eval.py --config ./config.json
```
