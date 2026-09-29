# Diagnostics tests

These tests exercise profilers, GPUs, networks, simulators, browsers, or long
experiment workflows. They are excluded from the default product regression
suite and must be invoked explicitly:

```powershell
& 'F:\conda_env\flyagent\python.exe' -m pytest -q tests/diagnostics
```

Individual tests may still require the environment indicated by their marker
or module documentation.
