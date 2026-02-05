# kkc_pt (kana -> kanji candidate generator in PyTorch)

## Create source file from `range3/wiki40b-ja`

## Install

```bash
pip install datasets

# Sudachi で reading 作る：
pip install sudachipy sudachidict_core

# MeCab（fugashi）で reading 作る：
pip install fugashi unidic-lite
```

## Usage

```bash
# 品質優先（推奨：unknown を含む文は捨てる + coverage 1.0）：
python make_pairs_from_wiki40b_ja.py --out pairs.jsonl --split train --streaming --max_lines 200000 --max_len 120 --analyzer sudachi --sudachi_mode C --min_coverage 1.0 --drop_unknown

# MeCab 版：

python make_pairs_from_wiki40b_ja.py --out pairs.jsonl --split train --streaming --max_lines 200000 --max_len 120 --analyzer mecab --min_coverage 1.0 --drop_unknown

```

## Input format
pairs.jsonl (one sample per line):
{"id":"...", "reading_hira":"きょうはみずをのんだ", "surface":"今日は水を飲んだ"}

## Install
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

## Train
python train.py --pairs pairs.jsonl --out_dir out --device cpu --epochs 5 --batch_size 16 --max_src_len 128 --max_tgt_len 128

Outputs:
out/model.pt
out/src_vocab.json
out/tgt_vocab.json

## Inference (beam search top-k)
python infer.py --model_dir out --text きょうはみずをのんだ --beam 8 --topk 5 --device cpu
