from __future__ import annotations

import threading
import time

import psutil

from monitor.server.models import SystemMetric

try:
    import pynvml

    pynvml.nvmlInit()
    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False


def get_system_metrics() -> SystemMetric:
    """Collect current GPU/CPU metrics."""
    metric = SystemMetric()

    # CPU + RAM
    metric.cpu_util = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    metric.ram_used = mem.used / (1024**3)
    metric.ram_total = mem.total / (1024**3)

    # GPU (first device)
    if _GPU_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            name = pynvml.nvmlDeviceGetName(handle)

            metric.gpu_util = util.gpu
            metric.gpu_mem_used = mem_info.used / (1024**3)
            metric.gpu_mem_total = mem_info.total / (1024**3)
            metric.gpu_temp = temp
            metric.gpu_name = name if isinstance(name, str) else name.decode()

            # Power draw (milliwatts → watts)
            try:
                metric.gpu_power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                metric.gpu_power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            except Exception:
                pass

            # Fan speed (not available on passively-cooled GPUs like DGX Spark)
            try:
                metric.gpu_fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except Exception:
                metric.gpu_fan_speed = -1.0

            # Clock speeds (MHz)
            try:
                metric.gpu_clock_sm = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_SM
                )
                metric.gpu_clock_mem = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM
                )
            except Exception:
                pass
        except Exception:
            pass

    return metric


class SystemMonitorThread(threading.Thread):
    """Background thread that periodically collects system metrics."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.latest: SystemMetric = SystemMetric()
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.latest = get_system_metrics()
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
