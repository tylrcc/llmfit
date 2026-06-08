"""Detect the host's compute resources and derive a memory budget for models.

The detection (which shells out to ``sysctl`` / ``nvidia-smi``) is kept separate
from the pure math (:func:`memory_budget_gb`) so the budgeting logic can be
unit-tested without any particular hardware.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class GPU:
    name: str
    vram_gb: float
    kind: str  # "nvidia", "apple", or "unknown"


@dataclass
class Hardware:
    os: str
    arch: str
    chip: str
    cpu_cores: int
    total_ram_gb: float
    available_ram_gb: float
    apple_silicon: bool
    gpu: GPU | None

    @property
    def is_unified_memory(self) -> bool:
        return self.apple_silicon


def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _total_ram_bytes() -> int:
    system = platform.system()
    if system == "Darwin":
        raw = _sysctl("hw.memsize")
        if raw.isdigit():
            return int(raw)
    # Linux / generic
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _available_ram_bytes(total: int) -> int:
    try:
        import psutil  # optional; gives a live "available" reading
        return int(psutil.virtual_memory().available)
    except Exception:
        return int(total * 0.75)  # conservative fallback


def _cpu_cores() -> int:
    import os
    return os.cpu_count() or 1


def _detect_nvidia() -> GPU | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not out:
        return None
    name, _, mem = out[0].partition(",")
    try:
        vram_gb = float(mem.strip()) / 1024.0  # MiB -> GiB
    except ValueError:
        vram_gb = 0.0
    return GPU(name.strip(), round(vram_gb, 1), "nvidia")


def detect() -> Hardware:
    system = platform.system()
    arch = platform.machine()
    apple_silicon = system == "Darwin" and arch == "arm64"

    total = _total_ram_bytes()
    available = _available_ram_bytes(total)

    if system == "Darwin":
        chip = _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
    else:
        chip = platform.processor() or arch

    gpu = _detect_nvidia()
    if gpu is None and apple_silicon:
        # Unified memory: the GPU can address most of system RAM.
        gpu = GPU(chip, round(total / 1e9 * 0.75, 1), "apple")

    return Hardware(
        os=system,
        arch=arch,
        chip=chip,
        cpu_cores=_cpu_cores(),
        total_ram_gb=round(total / 1e9, 1),
        available_ram_gb=round(available / 1e9, 1),
        apple_silicon=apple_silicon,
        gpu=gpu,
    )


def memory_budget_gb(hw: Hardware) -> float:
    """Memory (GB) we consider safely usable for a model + its runtime.

    * Discrete NVIDIA GPU: bounded by VRAM (the fast path).
    * Apple Silicon: unified memory, so ~72% of total RAM, leaving headroom
      for the OS and other apps.
    * CPU-only: ~60% of total RAM (slower, and the OS needs room).
    """
    if hw.gpu and hw.gpu.kind == "nvidia":
        return round(hw.gpu.vram_gb, 1)
    if hw.apple_silicon:
        return round(hw.total_ram_gb * 0.72, 1)
    return round(hw.total_ram_gb * 0.60, 1)


def parse_param_size(text: str) -> float | None:
    """Pull a parameter count in billions out of strings like '7.6B', '70b'."""
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])", text)
    if not m:
        return None
    value = float(m.group(1))
    return value / 1000.0 if m.group(2).lower() == "m" else value
