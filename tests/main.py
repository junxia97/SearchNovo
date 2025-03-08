import sys
import os
import csv
import argparse

current_dir = os.path.abspath(os.getcwd())
sys.path.append(current_dir)
from novobench.datasets import CustomDataset, NineSpeciesDataset
from novobench.models.searchnovo import SearchnovoRunner
from novobench.data.base import SpectrumData
from novobench.utils.config import Config
import polars as pl



def train(config_file, data_dir, model_file):
    config = Config(config_file, "casanovo")
    df = pl.read_parquet(data_dir)
    train_df = df.filter(pl.col("label") == "train")
    val_df = df.filter(pl.col("label") == "val")
    train = SpectrumData(train_df)
    val = SpectrumData(val_df)
    model = SearchnovoRunner(config, model_file)
    model.train(train, val)



def denovo(config_file,data_dir, model_file, saved_path):
    config = Config(config_file, "casanovo")
    df = pl.read_parquet(data_dir)
    test_df = df.filter(pl.col("label") == "test")
    test = SpectrumData(test_df)
    model = SearchnovoRunner(config, model_file, saved_path)
    model.denovo(test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--data_path", type=str,required=True)
    parser.add_argument("--ckpt_path", type=str,default=None)
    parser.add_argument("--denovo_output_path", type=str,default='')
    parser.add_argument("--config_path", type=str,required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train(args.config_path, args.data_path, args.ckpt_path)
    elif args.mode == "denovo":
        denovo(args.config_path, args.data_path, args.ckpt_path, args.denovo_output_path) 
    else:
        raise ValueError("Invalid mode!")
