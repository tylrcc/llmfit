"""A live, htop-style view of system memory/CPU and models currently loaded in
Ollama (which ones are resident, how much memory, and GPU vs CPU placement).
"""

from __future__ import annotations

import time

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .hardware import Hardware
from .ollama import Ollama, OllamaError


def _bar(fraction: float, width: int = 24) -> Text:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    color = "green" if fraction < 0.7 else "yellow" if fraction < 0.9 else "red"
    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * (width - filled), style="grey37")
    bar.append(f" {fraction * 100:4.0f}%")
    return bar


def _render(hw: Hardware, client: Ollama) -> Group:
    try:
        import psutil
        vm = psutil.virtual_memory()
        mem_used = vm.used / 1e9
        mem_frac = vm.percent / 100.0
        cpu_frac = psutil.cpu_percent(interval=None) / 100.0
    except Exception:
        mem_used, mem_frac, cpu_frac = 0.0, 0.0, 0.0

    sys_tbl = Table.grid(padding=(0, 2))
    sys_tbl.add_column(justify="right", style="bold")
    sys_tbl.add_column()
    sys_tbl.add_row("RAM", _bar(mem_frac).append(
        f"   {mem_used:.1f} / {hw.total_ram_gb:.0f} GB", style="dim"))
    sys_tbl.add_row("CPU", _bar(cpu_frac))

    models = Table(expand=True, box=None, pad_edge=False, header_style="dim")
    models.add_column("loaded model", style="cyan", no_wrap=True)
    models.add_column("size", justify="right")
    models.add_column("placement", justify="left")
    models.add_column("unloads in", justify="right", style="dim")
    try:
        loaded = client.loaded()
    except OllamaError:
        loaded = []
    if not loaded:
        models.add_row("[dim]— no models loaded —[/]", "", "", "")
    else:
        for m in loaded:
            gpu_pct = m.gpu_fraction * 100
            placement = (f"[green]{gpu_pct:.0f}% GPU[/]" if gpu_pct >= 99 else
                         f"[yellow]{gpu_pct:.0f}% GPU[/] / {100 - gpu_pct:.0f}% CPU")
            models.add_row(m.name, f"{m.size_gb:.1f} GB", placement,
                           _expiry(m.expires_at))

    return Group(
        Panel(sys_tbl, title=f"[bold]{hw.chip}[/]  ·  {hw.cpu_cores} cores",
              border_style="blue"),
        Panel(models, title="[bold]Ollama — resident models[/]",
              border_style="magenta"),
        Text("press Ctrl-C to quit", style="dim"),
    )


def _expiry(iso: str) -> str:
    if not iso:
        return ""
    try:
        from datetime import datetime
        # Ollama returns RFC3339; trim fractional seconds / zone for parsing.
        cleaned = iso.split(".")[0].rstrip("Z")
        dt = datetime.fromisoformat(cleaned)
        secs = (dt - datetime.now()).total_seconds()
        if secs <= 0:
            return "now"
        if secs < 60:
            return f"{secs:.0f}s"
        return f"{secs / 60:.0f}m"
    except Exception:
        return ""


def run(hw: Hardware, client: Ollama, interval: float = 1.0) -> None:
    with Live(_render(hw, client), refresh_per_second=4, screen=False) as live:
        try:
            while True:
                time.sleep(interval)
                live.update(_render(hw, client))
        except KeyboardInterrupt:
            pass
