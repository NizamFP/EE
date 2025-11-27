import threading
import time
import resource
import psutil
import os

class MemoryMonitor:
    def __init__(self, limit_mb):
        self.limit_bytes = limit_mb * 1024 * 1024
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.running = False
        self.memory_exceeded = False

    def _monitor(self):
        while self.running:
            mem = self.process.memory_info().rss
            self.peak_memory = max(self.peak_memory, mem)
            
            # check memory limit
            if mem > self.limit_bytes:
                self.memory_exceeded = True
                os._exit(1)  # hard kill process if exceeded
            
            time.sleep(0.002)  # 2ms sampling interval

    def __enter__(self):
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        resource.setrlimit(resource.RLIMIT_DATA, (self.limit_bytes, hard))

        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.thread.join()

        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        resource.setrlimit(resource.RLIMIT_DATA, (hard, hard))

        return False

    def get_peak_usage(self):
        return self.peak_memory / (1024 * 1024)
