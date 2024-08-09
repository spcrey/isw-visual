from glob import glob
import json
import os
import argparse
import sys

import numpy as np

sys.path.append("src")

from ocean_data import SpaceDataArray, AmpPointData
from ncfile_manager import NcfileManager
from multiprocessing_manager import MultiProcessingManager
from coord_generator import XyCoordGenerator

class ParamReciver:
    def __call__(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--x0", type=int)
        parser.add_argument("--x1", type=int)
        parser.add_argument("--y0", type=int)
        parser.add_argument("--y1", type=int)
        parser.add_argument("--nop", default=1, type=int,
                help="the number of processes")
        parser.add_argument("--data_folder", type=str, metavar="data_folder",
                help="the folder path of dataset file")
        parser.add_argument("--output_folder", type=str,
                help="the folder path of saving json file and json file's name is \"data.json\"")
        args, _ = parser.parse_known_args() 
        return args

class SpaceDataArrayUpdater:
    def __init__(self, sapce_data_array: SpaceDataArray, xy_coord) -> None:
        self._sapce_data_array = sapce_data_array
        self._xy_coord = xy_coord

    def update(self, time: int, amp_grid_data: np.ndarray):
        for (x, y) in self._xy_coord:
            amp_point_data = amp_grid_data[y,x]
            if amp_point_data:
                self._sapce_data_array.add_point_data(x, y, AmpPointData(time, amp_point_data))
        
    def save_to_json(self, file_folder: str):
        os.makedirs(file_folder, exist_ok=True)
        with open(os.path.join(file_folder, "data.json"), "w") as file:
            json.dump(self._sapce_data_array.json(), file, indent=4)

def main():
    MultiProcessingManager()
    param_reciver = ParamReciver()
    param = param_reciver()
    ncfile_manager = NcfileManager(param.data_folder, "Amp1.*.nc", "Amp1")
    xy_coord_generator = XyCoordGenerator()
    xy_coord = xy_coord_generator(param.x0, param.y0, param.x1, param.y1)
    sapce_data_array = SpaceDataArray()
    space_data_array_updater = SpaceDataArrayUpdater(sapce_data_array, xy_coord)
    for (time, grid_data) in ncfile_manager:
        space_data_array_updater.update(time, grid_data)
    space_data_array_updater.save_to_json(param.output_folder)

if __name__ == "__main__":
    main()
