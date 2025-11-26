import resource
import psutil
import os

class MemoryMonitor:  
    def __init__(self, limit_mb):
        self.limit_bytes = limit_mb * 1024 * 1024
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.process.memory_info().rss
        self.peak_memory = 0
        self.memory_exceeded = False    
        
    def __enter__(self):
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)

        resource.setrlimit(
            resource.RLIMIT_DATA,
            (self.limit_bytes, hard)
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)

        resource.setrlimit(
            resource.RLIMIT_DATA,
            (hard, hard)
        )
        if exc_type is MemoryError:
            self.memory_exceeded = True
        return False
    
    def get_current_usage(self):
        current = self.process.memory_info().rss
        usage_mb = (current - self.baseline_memory) / (1024 * 1024)
        self.peak_memory = max(self.peak_memory, usage_mb)
        return usage_mb
    
    def get_peak_usage(self):
        return self.peak_memory