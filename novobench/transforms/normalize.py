import spectrum_utils.spectrum as sus
from novobench.transforms.base import BaseTransform
from novobench.data.base import SpectrumData
import numpy as np
import polars as pl

class ScaleIntensity(BaseTransform):
    """
    A transformation class that adjusts the m/z range of spectral data.
    
    Attributes:
        min_mz (float): The minimum m/z value to include in the spectrum.
        max_mz (float): The maximum m/z value to include in the spectrum.
    """

    def __call__(self, data: SpectrumData) -> SpectrumData:
        updated_mz_arrays = []
        updated_intensity_arrays = []

        for precursor_mz, precursor_charge, mz_array, int_array in zip(data.precursor_mz, data.precursor_charge, data.mz_array, data.intensity_array):
            spectrum = sus.MsmsSpectrum("", precursor_mz, precursor_charge, mz_array.to_numpy().astype(np.float32), int_array.to_numpy().astype(np.float32))
            spectrum.scale_intensity("root", 1)
            intensities = spectrum.intensity / np.linalg.norm(spectrum.intensity)
            updated_mz_arrays.append(pl.Series(spectrum.mz))
            updated_intensity_arrays.append(pl.Series(spectrum.intensity))

        data.set_df(data.df.with_columns([pl.Series("mz_array", updated_mz_arrays, dtype=pl.List(pl.Float32)), 
                              pl.Series("intensity_array", updated_intensity_arrays, dtype=pl.List(pl.Float32))]))
        return data