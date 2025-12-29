import time

class Profiler:
    def __init__(self):
        self.t1 = 0
        self.t2 = 0
        self.name = ""

    def start(self, name=""):
        self.t1 = time.perf_counter(), time.process_time()
        self.name = name

    def end(self):
        self.t2 = time.perf_counter(), time.process_time()
        print(f"{self.name + " |"} Real time: {self.t2[0] - self.t1[0]:.2f} seconds", flush=True)
        print(f"{self.name + " |"} CPU time: {self.t2[1] - self.t1[1]:.2f} seconds", flush=True)