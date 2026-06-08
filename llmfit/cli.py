"""llmfit command-line interface."""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import __version__, backends as bk, catalog, monitor
from .fit import can_run, fit_installed, recommend
from .hardware import Hardware, detect, memory_budget_gb
from .ollama import Ollama, OllamaError

console = Console()
err = Console(stderr=True)

_STATUS_STYLE = {
    catalog.COMFORTABLE: ("green", "✓ comfortable"),
    catalog.TIGHT: ("yellow", "~ tight"),
    catalog.MAXED: ("dark_orange", "! maxed out"),
    catalog.WONT_FIT: ("red", "✗ won't fit"),
}


def _status_text(status: str) -> Text:
    style, label = _STATUS_STYLE.get(status, ("white", status))
    return Text(label, style=style)


def _hardware_panel(hw: Hardware, budget: float) -> None:
    gpu = hw.gpu
    if gpu and gpu.kind == "nvidia":
        gpu_line = f"{gpu.name}  ·  {gpu.vram_gb:.0f} GB VRAM"
    elif gpu and gpu.kind == "apple":
        gpu_line = f"{gpu.name} GPU  ·  unified memory"
    else:
        gpu_line = "no discrete GPU detected (CPU inference)"

    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold cyan", justify="right")
    t.add_column()
    t.add_row("chip", hw.chip)
    t.add_row("cores", str(hw.cpu_cores))
    t.add_row("memory", f"{hw.total_ram_gb:.0f} GB total  ·  "
                        f"{hw.available_ram_gb:.0f} GB free now")
    t.add_row("gpu", gpu_line)
    t.add_row("model budget", f"[bold]{budget:.0f} GB[/]  "
                             f"[dim](safe to spend on a model)[/]")
    console.print(t)


@click.group(invoke_without_command=True,
             context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="llmfit")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Right-size local LLMs for your hardware.

    \b
    llmfit            full report: your hardware + what fits
    llmfit can 32b    can I run a 32B model? at what quantization?
    llmfit models     fit status of your installed Ollama models
    llmfit bench      measure real tokens/sec for a model
    llmfit top        live monitor of system + loaded models
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(scan)


@main.command()
def scan() -> None:
    """Show your hardware and which models fit (the default)."""
    hw = detect()
    budget = memory_budget_gb(hw)
    console.print("\n[bold]🖥  Your machine[/]")
    _hardware_panel(hw, budget)

    # Installed models, if Ollama is reachable.
    client = Ollama()
    if client.is_up():
        installed = client.installed()
        if installed:
            console.print("\n[bold]📦 Installed models[/]")
            tbl = Table(box=None, pad_edge=False)
            tbl.add_column("model", style="cyan", no_wrap=True)
            tbl.add_column("est. to run", justify="right")
            tbl.add_column("fit")
            for m in installed:
                f = fit_installed(m, budget)
                tbl.add_row(m.name, f"{f.footprint_gb:.1f} GB", _status_text(f.status))
            console.print(tbl)
    else:
        console.print(f"\n[dim]Ollama not reachable at {client.host} — "
                      "skipping installed models. Start it with `ollama serve`.[/]")

    # Catalog recommendations.
    console.print("\n[bold]💡 You can comfortably run[/] "
                  "[dim](≈ Q4_K_M estimates)[/]")
    tbl = Table(box=None, pad_edge=False)
    tbl.add_column("model", style="cyan")
    tbl.add_column("params", justify="right")
    tbl.add_column("est. to run", justify="right")
    tbl.add_column("fit")
    tbl.add_column("", style="dim")
    for f in recommend(budget):
        tbl.add_row(f.label, f"{f.params_b:g}B", f"{f.footprint_gb:.1f} GB",
                    _status_text(f.status), f.detail)
    console.print(tbl)
    console.print("\n[dim]Estimates, not guarantees. Measure real speed with "
                  "`llmfit bench <model>`.[/]\n")


@main.command(name="can")
@click.argument("target")
def can_cmd(target: str) -> None:
    """Answer 'can I run TARGET?', e.g. `llmfit can 32b` or `llmfit can 70b`."""
    hw = detect()
    budget = memory_budget_gb(hw)
    ans = can_run(target, budget)
    if ans.params_b is None:
        err.print(f"[red]error[/] couldn't read a parameter size from "
                  f"'{target}'. Try something like `32b` or `13b`.")
        sys.exit(1)

    if ans.best_quant:
        head = (f"[bold green]Yes[/] — a {ans.params_b:g}B model fits on your "
                f"{budget:.0f} GB budget at [bold]{ans.best_quant}[/] or smaller.")
    else:
        head = (f"[bold red]No[/] — a {ans.params_b:g}B model won't fit your "
                f"{budget:.0f} GB budget, even at the smallest quantization.")
    console.print("\n" + head + "\n")

    tbl = Table(box=None, pad_edge=False)
    tbl.add_column("quant", style="cyan")
    tbl.add_column("est. to run", justify="right")
    tbl.add_column("fit")
    for quant, fp, status in ans.options:
        tbl.add_row(quant, f"{fp:.1f} GB", _status_text(status))
    console.print(tbl)
    console.print()


@main.command()
def models() -> None:
    """Show fit status for every installed Ollama model."""
    hw = detect()
    budget = memory_budget_gb(hw)
    client = Ollama()
    try:
        installed = client.installed()
    except OllamaError as exc:
        err.print(f"[red]error[/] {exc}")
        sys.exit(1)
    if not installed:
        console.print("[dim]No models installed. Pull one with "
                      "`ollama pull llama3.1`.[/]")
        return
    tbl = Table(title=f"installed models  ·  budget {budget:.0f} GB")
    tbl.add_column("model", style="cyan", no_wrap=True)
    tbl.add_column("params", justify="right")
    tbl.add_column("quant")
    tbl.add_column("on disk", justify="right")
    tbl.add_column("est. to run", justify="right")
    tbl.add_column("fit")
    for m in installed:
        f = fit_installed(m, budget)
        tbl.add_row(m.name, (f"{f.params_b:g}B" if f.params_b else "?"),
                    m.quant or "?", f"{m.size_gb:.1f} GB",
                    f"{f.footprint_gb:.1f} GB", _status_text(f.status))
    console.print(tbl)


@main.command()
@click.argument("model", required=False)
@click.option("-n", "--tokens", default=128, show_default=True,
              help="How many tokens to generate for the measurement.")
@click.option("--backend", default="ollama", show_default=True,
              help=f"Runtime to benchmark: {', '.join(bk.BACKEND_CHOICES)}.")
@click.option("--url", "base_url", default=None, metavar="URL",
              help="Override the backend base URL (e.g. http://localhost:8080/v1).")
def bench(model: str | None, tokens: int, backend: str, base_url: str | None) -> None:
    """Measure real tokens-per-second for a MODEL on any local runtime.

    Defaults to Ollama; use --backend llamacpp / mlx / lmstudio / vllm / openai
    (with --url) to benchmark an OpenAI-compatible server instead.
    """
    try:
        kind, label, endpoint = bk.resolve(backend, base_url)
    except OllamaError as exc:
        err.print(f"[red]error[/] {exc}")
        sys.exit(1)

    if kind == "ollama":
        client = Ollama()
        if not client.is_up():
            err.print(f"[red]error[/] Ollama not reachable at {client.host}. "
                      "Start it with `ollama serve`.")
            sys.exit(1)
        if not model:
            # Smallest generative model (skip embedding-only models).
            candidates = [m for m in client.installed()
                          if "embed" not in m.name.lower()]
            if not candidates:
                err.print("[red]error[/] no generative models installed to benchmark.")
                sys.exit(1)
            model = candidates[-1].name
            console.print(f"[dim]no model given — benchmarking smallest: {model}[/]")
    else:
        client = bk.OpenAICompatBench(base_url=endpoint, label=label)
        if not client.is_up():
            err.print(f"[red]error[/] {label} not reachable at {endpoint}. "
                      "Start your server, e.g. `llama-server -m model.gguf "
                      "--port 8080` or `mlx_lm.server --port 8080`.")
            sys.exit(1)
        if not model:
            served = client.models()
            if not served:
                err.print(f"[red]error[/] {label} reports no served model; "
                          "pass one explicitly.")
                sys.exit(1)
            model = served[0]
            console.print(f"[dim]no model given — using served: {model}[/]")

    with console.status(f"[bold]benchmarking[/] {model} on {label} ..."):
        try:
            r = client.benchmark(model, num_tokens=tokens)
        except OllamaError as exc:
            err.print(f"[red]error[/] {exc}")
            sys.exit(1)

    tbl = Table(box=None)
    tbl.add_column(style="bold cyan", justify="right")
    tbl.add_column()
    tbl.add_row("model", model)
    tbl.add_row("backend", f"{label}  [dim]{endpoint}[/]")
    if kind == "ollama":
        tbl.add_row("load time", f"{r.load_seconds:.2f} s")
        tbl.add_row("prompt (prefill)", f"{r.prompt_tps:.1f} tok/s "
                                       f"[dim]({r.prompt_tokens} tokens)[/]")
    else:
        tbl.add_row("time to first token", f"{r.load_seconds:.2f} s")
    tbl.add_row("generate (decode)", f"[bold green]{r.eval_tps:.1f} tok/s[/] "
                                    f"[dim]({r.eval_tokens} tokens)[/]")
    console.print(tbl)


@main.command()
@click.option("--interval", default=1.0, show_default=True,
              help="Refresh interval in seconds.")
def top(interval: float) -> None:
    """Live monitor: system memory/CPU and models resident in Ollama."""
    hw = detect()
    client = Ollama()
    monitor.run(hw, client, interval=interval)


if __name__ == "__main__":
    main()
