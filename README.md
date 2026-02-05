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

### Create user defined source file

```bash
# 1件追加
python make_user_pairs.py --out user_pairs.jsonl --add --reading_hira まぐろをかいたい --surface マグロを解体

# inputs.txt から一括追加（タブ区切り）
python make_user_pairs.py --out user_pairs.jsonl --from_txt inputs.txt --sep tab

# 上書きしたい

python make_user_pairs.py --out user_pairs.jsonl --from_txt inputs.txt --sep tab --overwrite


```
#### inputs.txt (前後の文脈なし)

```bash

いぬをかいたい	犬を飼いたい
ねこをかいたい	猫を飼いたい
まぐろをかいたい	マグロを解体
ていをなす	体をなす
きかいがとまる	機械が止まる
せいどがひくい	精度が低い
かたがこる	肩がこる

```

#### user_inputs_ctx.txt (前後の文脈あり)

```bash
犬を	かいたい		飼いたい
猫を	かいたい		飼いたい
マグロを	かいたい		解体
	てい	をなす	体
```

#### 前後の文脈の考慮した input.txt

```bash
left<TAB>reading_hira<TAB>right<TAB>surface

# 2列（従来）
たなか	田中

# 3列（right なし）
私は	きょうは		今日は

# 3列（left なし）
	きょうは	仕事だ	今日は

# 4列（left/right 両方あり、空もOK）
私は	きょうは	仕事だ	今日は

```

```bash
# iki40B から context-aware データ生成（span モードがデフォルト）
python make_pairs_from_wiki40b_ja.py --out pairs_ctx.jsonl --split train --streaming --max_lines 200000 --analyzer sudachi --sudachi_mode C --drop_unknown

```

```bash

# 2) 学習 （前後文脈あり）
python train.py --pairs pairs_ctx.jsonl --out_dir out_ctx --device cpu --epochs 5 --batch_size 16 --max_src_len 192 --max_tgt_len 64


python train.py --pairs pairs_ctx.jsonl user_pairs.jsonl --out_dir out_ctx --device cpu --epochs 5 --batch_size 16 --max_src_len 192 --max_tgt_len 64
```


```bash
# 推論（前後文脈あり）
python infer.py --model_dir out_ctx --text なまえは --left 私の --right 中野です --topk 5

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
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python train.py --pairs pairs.jsonl --out_dir out --device cpu --epochs 5 --batch_size 16 --max_src_len 128 --max_tgt_len 128

python train.py --pairs pairs.jsonl user_pairs.jsonl --out_dir out --device cpu --epochs 5 --batch_size 16 --max_src_len 128 --max_tgt_len 128
```

Outputs:
out/model.pt
out/src_vocab.json
out/tgt_vocab.json

## Inference (beam search top-k)
python infer.py --model_dir out --text きょうはみずをのんだ --beam 8 --topk 5 --device cpu
