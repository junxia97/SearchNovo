import spectrum_utils.spectrum as sus
from novobench.transforms.base import BaseTransform
from novobench.data.base import SpectrumData
import numpy as np
import polars as pl

class SetRangeMZ(BaseTransform):
    def __init__(self, min_mz: float = 50.0, max_mz: float = 2500.0):
        super().__init__()
        self.min_mz = min_mz
        self.max_mz = max_mz

    def __call__(self, data: SpectrumData) -> SpectrumData:
        updated_mz_arrays = []
        updated_intensity_arrays = []

        for precursor_mz, precursor_charge, mz_array, int_array in zip(data.precursor_mz, data.precursor_charge, data.mz_array, data.intensity_array):
            spectrum = sus.MsmsSpectrum("", precursor_mz, precursor_charge, mz_array.to_numpy().astype(np.float32), int_array.to_numpy().astype(np.float32))
            spectrum.set_mz_range(self.min_mz, self.max_mz)
            mz = spectrum.mz
            intensity = spectrum.intensity
            if len(spectrum.mz) == 0:
                print('value error')
                mz = [0,]
                intensity = [1,]
            updated_mz_arrays.append(pl.Series(mz))
            updated_intensity_arrays.append(pl.Series(intensity))

        data.set_df(data.df.with_columns([pl.Series("mz_array", updated_mz_arrays, dtype=pl.List(pl.Float32)), 
                              pl.Series("intensity_array", updated_intensity_arrays, dtype=pl.List(pl.Float32))]))
        return data


class RemovePrecursorPeak(BaseTransform):
    def __init__(self, remove_precursor_tol: float = 2.0):
        super().__init__()
        self.remove_precursor_tol = remove_precursor_tol


    def __call__(self, data: SpectrumData) -> SpectrumData:
        updated_mz_arrays = []
        updated_intensity_arrays = []

        for precursor_mz, precursor_charge, mz_array, int_array in zip(data.precursor_mz, data.precursor_charge, data.mz_array, data.intensity_array):
            spectrum = sus.MsmsSpectrum("", precursor_mz, precursor_charge, mz_array.to_numpy().astype(np.float32), int_array.to_numpy().astype(np.float32))
            spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")
            mz = spectrum.mz
            intensity = spectrum.intensity
            if len(spectrum.mz) == 0:
                print('value error')
                mz = [0,]
                intensity = [1,]
            updated_mz_arrays.append(pl.Series(mz))
            updated_intensity_arrays.append(pl.Series(intensity))

        data.set_df(data.df.with_columns([pl.Series("mz_array", updated_mz_arrays, dtype=pl.List(pl.Float32)), 
                              pl.Series("intensity_array", updated_intensity_arrays, dtype=pl.List(pl.Float32))]))
        return data



class FilterIntensity(BaseTransform):
    def __init__(self, min_intensity: float = 0.01, n_peaks: int = 200):
        super().__init__()
        self.n_peaks = n_peaks
        self.min_intensity = min_intensity


    def __call__(self, data: SpectrumData) -> SpectrumData:
        updated_mz_arrays = []
        updated_intensity_arrays = []

        for precursor_mz, precursor_charge, mz_array, int_array in zip(data.precursor_mz, data.precursor_charge, data.mz_array, data.intensity_array):
            spectrum = sus.MsmsSpectrum("", precursor_mz, precursor_charge, mz_array.to_numpy().astype(np.float32), int_array.to_numpy().astype(np.float32))
            spectrum.filter_intensity(self.min_intensity, self.n_peaks)
            mz = spectrum.mz
            intensity = spectrum.intensity
            if len(spectrum.mz) == 0:
                print('value error')
                mz = [0,]
                intensity = [1,]
            updated_mz_arrays.append(pl.Series(mz))
            updated_intensity_arrays.append(pl.Series(intensity))

        data.set_df(data.df.with_columns([pl.Series("mz_array", updated_mz_arrays, dtype=pl.List(pl.Float32)), 
                              pl.Series("intensity_array", updated_intensity_arrays, dtype=pl.List(pl.Float32))]))
        return data