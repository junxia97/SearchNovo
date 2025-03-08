# Bridging the Gaps between Database Search and De Novo Peptide Sequencing with SearchNovo (ICLR 2025)
<p>
  <a href="https://github.com/pytorch/pytorch"> <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" height="22px"></a>
  <a href="https://github.com/Lightning-AI/pytorch-lightning"> <img src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" height="22px"></a>
<p>

<p align="center" width="100%">
  <img src='./images/fig1.png' width="600%">
</p>

## Introduction
SearchNovo is a new framework that combines the strengths of database search and de novo sequencing for peptide identification from mass spectrometry data. It retrieves similar spectra from databases and uses a fusion mechanism to guide the generation of target peptide sequences. This approach addresses the limitations of both methods, such as identifying novel peptides and dealing with missing peaks, resulting in improved performance across benchmark datasets.


## Installation
This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell
conda env create -f searchnovo.yaml
conda activate searchnovo
```


## Preprocessing Dataset 
To preprocess the dataset, run the following command:
```shell
python tests/preprocess.py --output_file processed.parquet --config_path config_path --data_dir data_dir
```

## Train a new  model 
To train a model from scratch, run:
```shell
python tests/main.py --mode train --data_path processed_parquet_path --config_path config_path
```


## Sequence mass spectra
To sequence the mass spectra with Searchnovo, use the following command:
```shell
python tests/main.py --mode denovo --data_path processed_parquet_path --ckpt_path ckpt_path --denovo_output_path csv_path --config_path config_path
```

## Citation
```
@inproceedings{
xia2025bridging,
title={Bridging the Gap between Database Search and {\textbackslash}emph\{De Novo\} Peptide Sequencing with SearchNovo},
author={Jun Xia and Sizhe Liu and Jingbo Zhou and Shaorong Chen and hongxin xiang and Zicheng Liu and Yue Liu and Stan Z. Li},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=SjMtxqdQ73}
}
```
