from abc import ABC, abstractmethod
from datetime import datetime
import json
import multiprocessing
import argparse
import os
import sys

from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.append("src")

from multiprocessing_manager import MultiProcessingManager
from ncfile_manager import NcfileManager
import odvpkg.ocean_data_visual as odv


class DrawDoubleHcFrame(odv.DrawHccFrameTemplate):
    # HC: heat and contour
    def __init__(self, level_num: int, aspect: float=1, cmap: str=None, heat_interpolation: str=None, 
            contour_color: str="black"):
        heat_image_adder = odv.HeatFrameAdder(1.0, aspect, cmap, heat_interpolation)
        contourf_image_adder = odv.NoneFrameAdder()
        contour_image_adder = odv.ContourFrameAdder(1.0, contour_color, level_num)
        super().__init__(heat_image_adder, contourf_image_adder, contour_image_adder)

    def __call__(self, ax: Axes, frame: np.ndarray, vmin: float, vmax: float):
        heat_frame, contour_frame = frame
        heat_image = self._heat_image_adder(heat_frame, ax, vmin, vmax)
        contourf_image = self._contourf_image_adder(contour_frame, ax, vmin, vmax)
        self._contour_image_adder(contour_frame, ax, np.min(contour_frame), np.max(contour_frame))
        image = heat_image or contourf_image
        return image

class ParamReciver:
    def _recieve(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--fea_name", default="temp", type=str, choices=["temp", "eta", "u", "v", "ut"],
                        help="which feature to visualize, include \"temp\" and \"eta\"")
        parser.add_argument("--t0", default=None, type=int, metavar="t0")
        parser.add_argument("--t1", default=None, type=int, metavar="t1")
        parser.add_argument("--z0", default=None, type=int, metavar="z0")
        parser.add_argument("--z1", default=None, type=int, metavar="z1")
        parser.add_argument("--y0", default=None, type=int, metavar="y0")
        parser.add_argument("--y1", default=None, type=int, metavar="y1")
        parser.add_argument("--x0", default=None, type=int, metavar="x0")
        parser.add_argument("--x1", default=None, type=int, metavar="x1")
        parser.add_argument("--nop", default=1, type=int,
                            help="the number of processes")
        parser.add_argument("--yx_aspect", default=1.76, type=float,
                            help="aspect of width and height when feature is eta")
        parser.add_argument("--zx_aspect", default=20, type=float,
                            help="aspect of width and height when feature is temp, uv")
        parser.add_argument("--image_type", default="svg", type=str, choices=["svg", "jpg", "png"])
        parser.add_argument("--data_folder", type=str, metavar="data_folder",
                            help="the folder path of dataset file")
        parser.add_argument("--image_folder", type=str, metavar="image_folder",
                            help="the folder path of saving image and image name is \"image_xx\"")
        args, _ = parser.parse_known_args() 
        return args
    
    def _adapt(self, args: argparse.Namespace):
        datas_shape = (24, 90, 960, 1696)
        args.t0 = args.t0 if args.t0 else 0
        args.z0 = args.z0 if args.z0 else 0
        args.y0 = args.y0 if args.y0 else 0
        args.x0 = args.x0 if args.x0 else 0
        args.t1 = args.t1 if args.t1 else datas_shape[0]
        args.z1 = args.z1 if args.z1 else datas_shape[1]
        args.y1 = args.y1 if args.y1 else datas_shape[2]
        args.x1 = args.x1 if args.x1 else datas_shape[3]

    def __call__(self) -> argparse.Namespace:
        args = self._recieve()
        self._adapt(args)
        return args

class Colorbar:
    def __init__(self, colors: list[str], vmin: float, vmax: float) -> None:
        self._colors = colors
        self.vmin = vmin
        self.vmax = vmax

    def get_camp(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list("cmap", self._colors, N=100)

    def save(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "colorbar.json"), "w") as file:
            json.dump({
                "colors": self._colors,
                "vmin": self.vmin,
                "vmax": self.vmax
            }, file, indent=4)

class ImageRecorder:
    def __init__(self, image_grnerator: odv.ImageGenerator, colorbar: Colorbar, image_folder: str) -> None:
        self._image_generator = image_grnerator
        self._colorbar = colorbar
        self._image_folder = image_folder

    def __call__(self, data: np.ndarray, image_name) -> None:
        self._image_generator(data, self._image_folder, image_name, vmin=self._colorbar.vmin, vmax=self._colorbar.vmax)

class DistCalcor:
    def __call__(self, x0: int, y0: int, x1: int, y1: int) -> int:
        return int(np.sqrt((x1 - x0)**2 + (y1 - y0)**2))

class Cropper:
    def __init__(self, x0: int, x1: int, y0: int, y1: int) -> None:
        self._x0 = x0
        self._x1 = x1
        self._y0 = y0
        self._y1 = y1

    def __call__(self, data: np.ndarray) -> np.ndarray:
        data = data[self._y0: self._y1, self._x0: self._x1]
        data = np.flip(data, axis=0)
        return data
    
class Gradientor:
    def __call__(self,  data: np.ndarray) -> np.ndarray:
        data = np.gradient(data, 1, axis=1)
        data = data[2:-2, 2:-2]
        return data

class SlashInterpolator:
    _dist_calcor = DistCalcor()

    def __init__(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> None:
        self.xy_dist = self._dist_calcor(x0, y0, x1, y1)
        self._x0 = x0
        self._x1 = x1
        self._y0 = y0
        self._y1 = y1
        self._z0 = z0
        self._z1 = z1

    def __call__(self, data: np.ndarray) -> np.ndarray:
        data_shape = data.shape
        x_coords = np.linspace(self._x0, self._x1 -1, self.xy_dist)
        y_coords = np.linspace(self._y0, self._y1 -1, self.xy_dist)
        interp_func = RegularGridInterpolator(
            (np.arange(data_shape[0]), np.arange(data_shape[1]), np.arange(data_shape[2])), 
            data)
        coordss = np.zeros((self._z1-self._z0, self.xy_dist, 3))
        for i in range(len(coordss)):
            for j in range(len(coordss[i])):
                coordss[i][j][0] = self._z0 + i
                coordss[i][j][1] = y_coords[j]
                coordss[i][j][2] = x_coords[j]
        data = interp_func(coordss)
        return data

class ZeroValueTransformer:
    def __init__(self, transform_value: float) -> None:
        self._transform_value = transform_value
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        data[data==0] = self._transform_value
        return data

class Visualer(ABC):
    def __init__(self, ncfile_manager: NcfileManager, image_recorder: ImageRecorder) -> None:
        self._ncfile_manager = ncfile_manager
        self._image_recorder = image_recorder

    def get_time_indexs(self) -> list[int]:
        return self._ncfile_manager.get_time_indexs()

    @abstractmethod
    def run(self) -> None:
        pass

class TempVisualer(Visualer):
    def __init__(self, ncfile_manager: NcfileManager, image_recorder: ImageRecorder, 
                zero_value_transformer: ZeroValueTransformer, slash_interpolator: SlashInterpolator) -> None:
        super().__init__(ncfile_manager, image_recorder)
        self._slash_interpolator = slash_interpolator
        self._zero_value_transformer = zero_value_transformer

    def run(self, time_indexs: list[int]) -> None:
        for time_index in time_indexs:
            _, data = self._ncfile_manager[time_index]
            data = self._slash_interpolator(data)
            data = self._zero_value_transformer(data)
            self._image_recorder(data, f"image_{str(time_index).rjust(2, '0')}")

class UtVisualer(Visualer):
    def __init__(self, u_ncfile_manager: NcfileManager, temp_ncfile_manager: NcfileManager, image_recorder: ImageRecorder, 
                zero_value_transformer: ZeroValueTransformer, slash_interpolator: SlashInterpolator) -> None:
        super().__init__(u_ncfile_manager, image_recorder)
        self._u_ncfile_manager = u_ncfile_manager
        self._temp_ncfile_manager = temp_ncfile_manager
        self._slash_interpolator = slash_interpolator
        self._zero_value_transformer = zero_value_transformer

    def run(self, time_indexs: list[int]) -> None:
        for time_index in time_indexs:
            _, u_data = self._u_ncfile_manager[time_index]
            _, temp_data = self._temp_ncfile_manager[time_index]
            temp_data = self._slash_interpolator(temp_data)
            u_data = self._slash_interpolator(u_data)
            temp_data = self._zero_value_transformer(temp_data)
            self._image_recorder((u_data, temp_data), f"image_{str(time_index).rjust(2, '0')}")

class UvVisualer(Visualer):
    def __init__(self, ncfile_manager: NcfileManager, image_recorder: ImageRecorder, 
                slash_interpolator: SlashInterpolator) -> None:
        super().__init__(ncfile_manager, image_recorder)
        self._slash_interpolator = slash_interpolator

    def run(self, time_indexs: list[int]) -> None:
        for time_index in time_indexs:
            _, data = self._ncfile_manager[time_index]
            data = self._slash_interpolator(data)
            self._image_recorder(data, f"image_{str(time_index).rjust(2, '0')}")

class EtaVisualer(Visualer):
    def __init__(self, ncfile_manager: NcfileManager, image_recorder: ImageRecorder, 
                cropper: Cropper, gradientor: Gradientor) -> None:
        super().__init__(ncfile_manager, image_recorder)
        self._cropper = cropper
        self._gradientor = gradientor

    def run(self, time_indexs: list[int]) -> None:
        for time_index in time_indexs:
            _, data = self._ncfile_manager[time_index]
            data = self._cropper(data)
            data = self._gradientor(data)
            self._image_recorder(data, f"image_{str(time_index).rjust(2, '0')}")

class VisualRuner:
    def __init__(self, visualer: Visualer) -> None:
        self._visualer = visualer
        self._time_indexs = self._visualer.get_time_indexs()
        self._multi_process_manager = MultiProcessingManager()

    def run(self) -> None:
        self._multi_process_manager.run(self._visualer.run, self._time_indexs)

class VisualerFactory:
    _dist_calcor = DistCalcor()

    def __call__(self, param) -> Visualer:
        if param.fea_name == "eta":
            ncfile_manager = NcfileManager(param.data_folder, "stateEta.*.glob.nc", "Eta", param.t0, param.t1 + 1)
            cropper = Cropper(param.x0, param.x1, param.y0, param.y1)
            gradientor = Gradientor()
            colors = ["#0000FF", "#FFFFFF"]
            colors = ["#4b0095", "#0c7ff2", "#fefefe", "#fa8b04"]
            colorbar = Colorbar(colors, vmin=-0.01, vmax=0.01)
            colorbar.save(param.image_folder)
            draw_fun = odv.DrawHeatFrame(param.yx_aspect, colorbar.get_camp())
            figsize = (20 * (param.x1 - param.x0 - 2) / (param.y1 - param.y0 - 2), 20)
            image_generator = odv.PurseImageGenerator(figsize, draw_fun, param.image_type)
            image_recorder = ImageRecorder(image_generator, colorbar, param.image_folder)
            return EtaVisualer(ncfile_manager, image_recorder, cropper, gradientor)
            
        elif param.fea_name == "temp":
            ncfile_manager = NcfileManager(param.data_folder, "stateT.*.glob.nc", "Temp", param.t0, param.t1 + 1)
            zero_value_transformer = ZeroValueTransformer(15)
            slash_interpolator = SlashInterpolator(param.x0, param.x1, param.y0, param.y1, param.z0, param.z1)
            colors = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
            colorbar = Colorbar(colors, vmin=0.0, vmax=30.0)
            colorbar.save(param.image_folder)
            draw_fun = odv.DrawCcFrame(33, param.zx_aspect, colorbar.get_camp())
            figsize = (20 * slash_interpolator.xy_dist / (param.z1 - param.z0), 20)
            image_generator = odv.PurseImageGenerator(figsize, draw_fun, param.image_type)
            image_recorder = ImageRecorder(image_generator, colorbar, param.image_folder)
            return TempVisualer(ncfile_manager, image_recorder, zero_value_transformer, slash_interpolator)
        
        elif param.fea_name == "ut":
            u_ncfile_manager = NcfileManager(param.data_folder, "stateU.*.glob.nc", "U", param.t0, param.t1 + 1)
            temp_ncfile_manager = NcfileManager(param.data_folder, "stateT.*.glob.nc", "Temp", param.t0, param.t1 + 1)
            zero_value_transformer = ZeroValueTransformer(15)
            slash_interpolator = SlashInterpolator(param.x0, param.x1, param.y0, param.y1, param.z0, param.z1)
            colors = ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"]
            colorbar = Colorbar(colors, vmin=-1.0, vmax=1.0)
            colorbar.save(param.image_folder)
            draw_fun = DrawDoubleHcFrame(33, param.zx_aspect, colorbar.get_camp())
            figsize = (20 * slash_interpolator.xy_dist / (param.z1 - param.z0), 20)
            image_generator = odv.PurseImageGenerator(figsize, draw_fun, param.image_type)
            image_recorder = ImageRecorder(image_generator, colorbar, param.image_folder)
            return UtVisualer(u_ncfile_manager, temp_ncfile_manager, image_recorder, zero_value_transformer, slash_interpolator)
        
        elif param.fea_name == "u":
            ncfile_manager = NcfileManager(param.data_folder, "stateU.*.glob.nc", "U", param.t0, param.t1 + 1)
            slash_interpolator = SlashInterpolator(param.x0, param.x1, param.y0, param.y1, param.z0, param.z1)
            colors = ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"]
            colorbar = Colorbar(colors, vmin=-1.0, vmax=1.0)
            colorbar.save(param.image_folder)
            draw_fun = odv.DrawHeatFrame(param.zx_aspect, colorbar.get_camp())
            figsize = (20 * slash_interpolator.xy_dist / (param.z1 - param.z0), 20)
            image_generator = odv.PurseImageGenerator(figsize, draw_fun, param.image_type)
            image_recorder = ImageRecorder(image_generator, colorbar, param.image_folder)
            return UvVisualer(ncfile_manager, image_recorder, slash_interpolator)
        
        elif param.fea_name == "v":
            ncfile_manager = NcfileManager(param.data_folder, "V.*.glob.nc", "V", param.t0, param.t1 + 1)
            slash_interpolator = SlashInterpolator(param.x0, param.x1, param.y0, param.y1, param.z0, param.z1)
            colors = ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"]
            colorbar = Colorbar(colors, vmin=-1.0, vmax=1.0)
            colorbar.save(param.image_folder)
            draw_fun = odv.DrawHeatFrame(param.zx_aspect, colorbar.get_camp())
            figsize = (20 * slash_interpolator.xy_dist / (param.z1 - param.z0), 20)
            image_generator = odv.PurseImageGenerator(figsize, draw_fun, param.image_type)
            image_recorder = ImageRecorder(image_generator, colorbar, param.image_folder)
            return UvVisualer(ncfile_manager, image_recorder, slash_interpolator)
        
        else:
            raise f"no visualer named {param.fea_name}"

class TimeRecorder:
    def __init__(self) -> None:
        self._start_time = datetime.now()

    def finish(self) -> None:
        print(f"sum time={datetime.now() - self._start_time}")

def main():
    multi_process_manager = MultiProcessingManager()
    time_recorder = TimeRecorder()
    param_reciver = ParamReciver()
    param = param_reciver()
    multi_process_manager.set_nop(param.nop)
    visualer_factory = VisualerFactory()
    visualer = visualer_factory(param)
    runer = VisualRuner(visualer)
    runer.run()
    time_recorder.finish()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
    