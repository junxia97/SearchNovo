import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from novobench.datasets import CustomDataset
from novobench.models.searchnovo import SearchnovoRunner
from novobench.transforms.base import BaseTransform
from novobench.data.base import SpectrumData
from novobench.utils.config import Config
import numpy as np
import polars as pl
from matchms.similarity import CosineGreedy
from matchms import Spectrum
from multiprocessing import Pool, cpu_count
import pandas as pd
import argparse



# TODO inefficient : ~ 1 day for a dataset

class FindMostSimilarSpectrumDev(BaseTransform):

    def __call__(self, data: SpectrumData, num_chunks) -> SpectrumData:
        full_df = data.df.to_pandas()
        chunks = np.array_split(full_df, num_chunks)
        search_df = full_df[full_df["search"] == 1]
        with Pool(processes=num_chunks) as pool:
            results = pool.map(self.process_chunk, [(chunk, search_df) for chunk in chunks])
        combined_df = pl.DataFrame(pd.concat(results))
        data.set_df(combined_df)
        return data

    def process_chunk(self, args):
        chunk, full_df = args
        most_similar_peptides = []
        most_similar_scores = []
        for i, row in chunk.iterrows():
            mz_array = row["mz_array"]
            intensity_array = row["intensity_array"]
            precursor_mz = row["precursor_mz"]

            max_similarity = -np.inf
            most_similar_peptide = None

            filtered_df = full_df[abs(full_df["precursor_mz"] - precursor_mz) < 20]
            if len(filtered_df) < 5:
                filtered_df = full_df[abs(full_df["precursor_mz"] - precursor_mz) < 50]
                if len(filtered_df) < 5:
                    filtered_df = full_df

            for j, other_row in filtered_df.iterrows():
                if i == j:
                    continue
                other_mz_array = other_row["mz_array"]
                other_intensity_array = other_row["intensity_array"]
                other_precursor_mz = other_row["precursor_mz"]
                peptide = other_row["modified_sequence"]
                similarity = self.similarity_func(mz_array, intensity_array, precursor_mz, other_mz_array, other_intensity_array, other_precursor_mz)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_peptide = peptide

            most_similar_peptides.append(most_similar_peptide)
            most_similar_scores.append(max_similarity)


        most_similar_scores = [float(score) for score in most_similar_scores]
        return chunk.assign(ref_sequence=most_similar_peptides, similarity_score=most_similar_scores)


    def similarity_func(self, mz_array, intensity_array, precursor_mz, other_mz_array, other_intensity_array, other_precursor_mz):
        # TODO: these 4 lines are unncessary
        mz_array = [float(i) for i in mz_array]
        other_mz_array = [float(i) for i in other_mz_array]
        other_intensity_array = [float(i) for i in other_intensity_array]
        intensity_array = [float(i) for i in intensity_array]

        reference = Spectrum(mz=np.array(list(mz_array)), intensities=np.array(list(intensity_array)), metadata={"precursor_mz": precursor_mz})
        query = Spectrum(mz=np.array(list(other_mz_array)), intensities=np.array(list(other_intensity_array)), metadata={"precursor_mz": other_precursor_mz})
        cosine_greedy = CosineGreedy(tolerance=0.2)
        score = cosine_greedy.pair(reference, query)
        
        return score['score']


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Find the most similar spectra using multiprocessing.")
    
    # Add arguments for number of chunks and file paths
    parser.add_argument('--num_chunks', type=int, default=4, help="Number of chunks to divide the data into for multiprocessing.")
    parser.add_argument('--data_dir', type=str, default="", help="Directory where data files are located.")
    parser.add_argument('--train_file', type=str, default="train.parquet", help="Name of the train file.")
    parser.add_argument('--val_file', type=str, default="valid.parquet", help="Name of the validation file.")
    parser.add_argument('--test_file', type=str, default="test.parquet", help="Name of the test file.")
    parser.add_argument('--output_file', type=str, default="searchnovo_processed.parquet", help="Output file to save the processed DataFrame.")
    parser.add_argument("--config_path", type=str,required=True)
    
    args = parser.parse_args()
    
    # Mapping of dataset parts to filenames
    file_mapping = {
        "train": args.train_file,
        "valid": args.val_file,
        "test": args.test_file,
    }

    # Initialize dataset with parsed arguments
    dataset = CustomDataset(args.data_dir, file_mapping) 
    
    config = Config(args.config_path, "casanovo")

    # data = dataset.load_data(transform = SearchnovoRunner.preprocessing_pipeline(config))
    data = dataset.load_data()

    # Access the train, validation, and test DataFrames
    train_df = data.get_train()._df
    val_df = data.get_valid()._df
    test_df = data.get_test()._df
    
    train_df = train_df.with_columns([
        pl.lit(1).alias("search"),
        pl.lit("train").alias("label")
    ])

    val_df = val_df.with_columns([
        pl.lit(1).alias("search"),
        pl.lit("val").alias("label")
    ])

    test_df = test_df.with_columns([
        pl.lit(0).alias("search"),
        pl.lit("test").alias("label")
    ])

    combined_df = pl.concat([train_df, val_df, test_df])
    combined_data = SpectrumData(combined_df)
    t = FindMostSimilarSpectrumDev()
    combined_data = t(combined_data, args.num_chunks)

    print(combined_data.df)
    combined_data.df.write_parquet(args.output_file)





if __name__ == "__main__":
    main()