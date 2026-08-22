"""Process-level resource sampling shared by the VLM benchmarks.

Peak RSS is the number that decides whether a model fits an 8 GB SBC, and
it cannot be read after the fact — allocation peaks during load and during
the first inference, then the allocator hands pages back. So RSS is sampled
on a background thread for the whole duration of the phase being measured.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class Phase:
    """Timing and memory for one measured phase."""

    name: str
    wall_s: float = 0.0
    cpu_s: float = 0.0
    rss_start_mb: float = 0.0
    rss_peak_mb: float = 0.0
    rss_end_mb: float = 0.0
    notes: dict = field(default_factory=dict)

    @property
    def rss_delta_mb(self) -> float:
        """Resident growth attributable to this phase."""
        return self.rss_end_mb - self.rss_start_mb

    @property
    def cpu_utilization(self) -> float:
        """CPU-seconds per wall-second — >1.0 means multiple cores were busy."""
        return self.cpu_s / self.wall_s if self.wall_s > 0 else 0.0


class ResourceProbe:
    """Context manager that measures wall time, CPU time, and peak RSS.

    Sampling runs on a daemon thread at ``interval_s``. The interval is a
    tradeoff: too coarse and a short allocation spike is missed, too fine
    and the sampler itself perturbs the measurement. 20 ms is fine enough
    to catch a multi-hundred-MB session load without measurably competing
    with inference for CPU.

    Example:
        >>> with ResourceProbe("load") as p:
        ...     session = make_session()
        >>> p.phase.rss_peak_mb
    """

    def __init__(self, name: str, interval_s: float = 0.02) -> None:
        self._proc = psutil.Process()
        self._interval_s = interval_s
        self.phase = Phase(name=name)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0.0

    def _sample(self) -> None:
        """Poll RSS until asked to stop, retaining the maximum."""
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss / 1e6
                self._peak = max(self._peak, rss)
            except psutil.Error:
                return
            self._stop.wait(self._interval_s)

    def __enter__(self) -> ResourceProbe:
        rss = self._proc.memory_info().rss / 1e6
        self.phase.rss_start_mb = rss
        self._peak = rss
        cpu = self._proc.cpu_times()
        self._cpu0 = cpu.user + cpu.system
        self._t0 = time.perf_counter()

        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.phase.wall_s = time.perf_counter() - self._t0
        cpu = self._proc.cpu_times()
        self.phase.cpu_s = (cpu.user + cpu.system) - self._cpu0

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        self.phase.rss_end_mb = self._proc.memory_info().rss / 1e6
        self.phase.rss_peak_mb = max(self._peak, self.phase.rss_end_mb)


def system_info() -> dict:
    """Describe the machine, so a result file is interpretable later."""
    import platform

    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(vm.total / 1e9, 1),
    }
