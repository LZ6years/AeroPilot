# Repository instructions

Read [AGENT.md](AGENT.md) before changing this repository; it contains the
project's architecture, ownership boundaries and development rules.

## Remote generated files

On the wangzequan Linux server, put newly generated evaluation results, logs,
frozen snapshots and diagnostic files under `$DATADIR`, grouped by dataset or
purpose, never directly under `~`. The verified DATADIR is
`/home/Newdisk/wangzequan`; non-interactive SSH may not inherit it, so launchers
must explicitly resolve/set it. Do not move active runs or edit their frozen
paths while they are running. Preserve existing source, credentials and assets.

## Required local Python environment

For this Windows workspace, run all project Python commands in the **Conda
`flyagent` environment**, using this explicit interpreter:

```text
F:\conda_env\flyagent\python.exe
```

This applies to tests, CLI commands, manual probes, benchmark runners, diagnostic
scripts and background jobs. Do not assume that the current shell has activated
the environment. Do not use bare `python`, `pip`, `pytest`, or the base interpreter
`F:\Anaconda\python.exe` for project execution.

PowerShell examples, from the repository root:

```powershell
& 'F:\conda_env\flyagent\python.exe' -m pytest -q
& 'F:\conda_env\flyagent\python.exe' -m src.cli
& 'F:\conda_env\flyagent\python.exe' tests/manual/your_probe.py --help
```

- When dependency changes are authorized, use this interpreter with `-m pip`.
- Python launchers should pass `sys.executable` to their child processes, after
  the parent has been started with the interpreter above. This also applies to
  frozen-source experiment children.
- For PowerShell background launches, set `Start-Process -FilePath` to the
  explicit interpreter above, with `-WindowStyle Hidden`; do not resolve it
  through `Get-Command python` or the current PATH.
- Before a real perception/GPU experiment, verify and record `sys.executable`,
  `torch.__version__`, `torch.version.cuda`, and `torch.cuda.is_available()` in
  its diagnostics. If CUDA is unavailable, report it instead of silently
  switching environments or labeling the experiment as GPU inference.
- If this interpreter is missing or unusable, report the environment problem;
  do not silently fall back to base or reinstall packages. On another machine,
  use the user-specified equivalent `flyagent` environment. Remote simulator
  servers retain their separately configured remote environment.

Verified on 2026-09-27: `flyagent` contained PyTorch `2.6.0+cu126`, CUDA 12.6,
and successfully executed a CUDA tensor operation on the RTX 3060 Laptop GPU.
The base environment contained CPU-only PyTorch `2.5.1+cpu`. These are diagnostic
observations, not dependency version pins. Earlier experiments launched through
base must not be presented as GPU measurements of the `flyagent` environment.
