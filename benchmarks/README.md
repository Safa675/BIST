# Benchmarks

This directory stores Phase 6 integrated benchmark reports.

Expected files:

- `phase6_latest.json`: latest run output
- `phase6_baseline.json`: approved baseline used for regression checks

Generate latest:

```bash
python -m Models.common.benchmarking_cli
```

Create/update baseline:

```bash
python -m Models.common.benchmarking_cli --write-baseline
```

