from abc import ABC, abstractmethod
import time
import multiprocessing
from typing import Any

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

class TimeProcessingRuner(ABC):
    @abstractmethod
    def run(self, fun, time_indexs):
        pass

class NoneProcessingRuner(TimeProcessingRuner):
    def run(self, fun, time_indexs):
        pass
    
class SingleProcessingRuner(TimeProcessingRuner):
    def run(self, fun, time_indexs):
        fun(time_indexs)

class TimeIndexsSplitor:
    def __call__(self, time_indexs, num):
        multi_time_indexs = [[] for _ in range(num)]
        for i, time_index in enumerate(time_indexs):
            multi_time_indexs[i%num].append(time_index)
        return multi_time_indexs

class MultiProcessingRuner(TimeProcessingRuner):
    _time_indexs_splitor = TimeIndexsSplitor()

    def __init__(self, nop: int) -> None:
        self._nop = nop

    def run(self, fun, time_indexs):
        multi_time_indexs = self._time_indexs_splitor(time_indexs, self._nop)
        procs = []
        for time_indexs in multi_time_indexs:
            proc = multiprocessing.Process(target=fun, args=(time_indexs, ))
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join() 

@singleton
class MultiProcessingManager:
    def __init__(self) -> None:
        multiprocessing.freeze_support()
        self._time_processing_runer = SingleProcessingRuner()

    def set_nop(self, nop: int):
        if nop:
            self._time_processing_runer = MultiProcessingRuner(nop)
        else:
            self._time_processing_runer = SingleProcessingRuner()

    def run(self, fun, time_indexs):
        self._time_processing_runer.run(fun, time_indexs)

class CC:
    def fun(self, time_indexs: list[int]):
        for time_index in time_indexs:
            time.sleep(1)
            print(time_index)

class DD:
    def __init__(self) -> None:
        self._mp_manager = MultiProcessingManager()
        self._cc = CC()

    def run(self):
        self._mp_manager.run(self._cc.fun, list(range(25)))

def main():
    mp_manager = MultiProcessingManager()
    mp_manager.set_nop(5)
    dd = DD()
    dd.run()

if __name__ == "__main__":
    main()
    