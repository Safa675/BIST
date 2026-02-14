# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- pip (Python package installer)

## Installing from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/OWNER/bist-quant.git
cd bist-quant
pip install -e ".[dev]"
```

This installs the package in editable mode with all development dependencies.

## Optional Dependencies

### Data Quality

For data validation with Pandera:

```bash
pip install -e ".[data-quality]"
```

### Live Data Fetchers

For live data fetching from various sources:

```bash
pip install -e ".[fetchers]"
```

### Documentation

For building documentation:

```bash
pip install -e ".[docs]"
```

## Package Structure

The package is distributed as `bist-quant` but imported as `Models`:

```python
# Distribution name: bist-quant
# Import name: Models

from Models import PortfolioEngine
from Models.signals import build_signal
from Models.common.utils import cross_sectional_rank
```

This naming convention is maintained for backward compatibility with existing codebases.

## Verifying Installation

```python
import Models
print(Models.get_available_signals())
```

You should see a list of available signal names.

## Running Tests

```bash
pytest -q tests
```

## Building the Package

```bash
python -m build
```

This creates wheel and source distributions in the `dist/` directory.
