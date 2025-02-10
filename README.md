# ParaBridge

An enhancement on LangBridge meant to properly handle paragraph-sized questions through an LSTM which aggregates sentence embeddings.

## Usage

```
pip install -r requirements.txt

export HF_HOME="/workspace/HF_HOME/"

python scripts/train_eval.py --config ./config.json
```
