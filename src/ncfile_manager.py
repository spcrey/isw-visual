from glob import glob
import os
import netCDF4 as nc

import numpy as np

class NcfileReader:
    def __call__(self, file_path: str, fea_name: str) -> np.ndarray:
        file_path = os.path.relpath(file_path)
        data = nc.Dataset(file_path, "r").variables[fea_name][:]
        return data

class NcfileSearcher:
    def __call__(self, file_folder: str, file_name_patern: str) -> list[str]:
        file_paths = glob(os.path.join(file_folder, file_name_patern))
        return file_paths

class NcfileManager:
    _file_reader = NcfileReader()
    _file_searcher = NcfileSearcher()

    def __init__(self, file_folder: str, file_name_patern: str, fea_name: str, t0: int=None, t1: int=None) -> None:
        file_paths = self._file_searcher(file_folder, file_name_patern)
        file_paths.sort()
        file_paths = file_paths[t0: t1]
        self._times = [int(file_path.split(".")[-3 if file_path[-7:-3] == "glob" else -2]) for file_path in file_paths]
        self._fea_name = fea_name
        self._file_paths = file_paths

    def get_time_indexs(self):
        return list(range(len(self._times)))

    def __len__(self):
        return len(self._times)
    
    def __getitem__(self, index):
        grid_data = self._file_reader(self._file_paths[index], self._fea_name)
        time = self._times[index]
        return time, grid_data
    