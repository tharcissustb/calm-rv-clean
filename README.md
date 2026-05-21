# Datasets

This repository does **not** distribute the original datasets due to licensing and size constraints.  
To reproduce the experiments, please download the datasets from the official sources listed below.

## PHEME (Rumour Verification)
- **Source**: Kochkina et al., figshare, 2018
- **URL**: https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078
- **Format**: Thread‑based (source tweet + replies), 9 events, 2,402 threads
- **Labels**: true, false, unverified

## RumourEval17
- **Source**: Derczynski et al., SemEval 2017
- **URL**: https://alt.qcri.org/semeval2017/task8/
- **Format**: Thread‑based, 297 threads
- **Labels**: true, false, unverified

## Chinese CED Dataset
- **Source**: Song et al., IEEE TKDE 2019
- **URL**: https://github.com/thunlp/Chinese_Rumor_Dataset
- **Format**: Thread‑based Weibo posts, 1,315 examples
- **Labels**: rumor / non‑rumor

## LIAR (Standalone claims)
- **Source**: Wang, ACL 2017
- **URL**: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
- **Format**: Single political claims, 12,791 examples
- **Labels**: true, false, unverified (after mapping)

## Augmented Data
- Generated using BART and synonym replacement (WordNet)
- The augmentation scripts are provided in `scripts/`.
- Run `scripts/generate_synonym_aug.py` to create the synonym‑augmented JSONL file.
- Run `scripts/generate_hard_augmented.py` (or the original BART pipeline) to create the BART‑augmented file.

All training scripts expect the JSONL files to be placed in `data/` with the following names:
- `pheme.jsonl`
- `rumoureval17.jsonl`
- `chinese_ced.jsonl`
- `liar.jsonl`
