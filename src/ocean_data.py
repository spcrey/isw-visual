from abc import ABC, abstractmethod

class OceanData(ABC):
    @abstractmethod
    def json(self) -> dict:
        pass

class OceanDataArray(OceanData):
    @abstractmethod
    def __getitem__(self, index: int) -> OceanData:
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def sort(self) -> None:
        pass

class PointData(OceanData):
    def __init__(self, time: int) -> None:
        self._time = time

    def get_time(self) -> int:
        return self._time
    
class AmpPointData(PointData):
    def __init__(self, time: int, amp: float) -> None:
        super().__init__(time)
        self._amp = amp
    
    def json(self) -> dict:
        return {
            "time": self._time,
            "amp": self._amp,
        }

class TimeDataArray(OceanDataArray):
    def __init__(self, x: int, y: int) -> None:
        self._array = []
        self._x = x
        self._y = y

    def get_x(self) -> int:
        return self._x
    
    def get_y(self) -> int:
        return self._y

    def sort(self) -> None:
        self._array = sorted(self._array, key=lambda data: data.get_time())

    def __len__(self) -> int:
        return len(self._array)

    def __getitem__(self, index: int) -> PointData:
        self._array[index]

    def add(self, data: PointData) -> None:
        self._array.append(data)

    def json(self) -> dict:
        return {
            "x": self._x,
            "y": self._y,
            "data": [data.json() for data in self._array]
        }

class SpaceDataArray(OceanDataArray):
    def __init__(self) -> None:
        self._array = []

    def sort(self) -> None:
        self._array = sorted(self._array, key=lambda data: data.get_x())
        self._array = sorted(self._array, key=lambda data: data.get_y())
        for data in self._array:
            data.sort()

    def _get_time_data_array(self, x: int, y: int) -> TimeDataArray:
        for data in self._array:
            if data.get_x() == x and data.get_y() == y:
                return data
        time_data_array = TimeDataArray(x, y)
        self.add(time_data_array)
        return time_data_array
    
    def add_point_data(self, x: int, y: int, data: PointData):
        time_data_array = self._get_time_data_array(x, y)
        time_data_array.add(data)

    def __len__(self):
        return len(self._array)

    def __getitem__(self, index: int):
        self._array[index]

    def add(self, data: TimeDataArray):
        self._array.append(data)

    def json(self) -> dict:
        return {
            "data": [data.json() for data in self._array],
        }

def main():
    sapce_data_array = SpaceDataArray()

    time_data_array = TimeDataArray(x=2, y=40)
    point_data = AmpPointData(20, 4)
    time_data_array.add(point_data)
    point_data = AmpPointData(18, 5)
    time_data_array.add(point_data)
    sapce_data_array.add(time_data_array)
    
    time_data_array = TimeDataArray(x=3, y=40)
    point_data = AmpPointData(20, 7)
    time_data_array.add(point_data)
    point_data = AmpPointData(22, 9)
    time_data_array.add(point_data)
    sapce_data_array.add(time_data_array)

    sapce_data_array.add_point_data(3, 41, AmpPointData(22, 7))

    sapce_data_array.sort()
    print(sapce_data_array.json())

if __name__ == "__main__":
    main()
