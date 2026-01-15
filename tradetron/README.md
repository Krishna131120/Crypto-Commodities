# Tradetron Integration

This folder contains all Tradetron-related code and documentation for paper trading commodities.

## Contents

- **`TRADETRON_COMPLETE_GUIDE.md`** - **START HERE!** Complete step-by-step guide with ALL steps
- **`tradetron_client.py`** - Main API client for sending signals to Tradetron
- **`test_tradetron.py`** - Test script to verify your setup
- **`__init__.py`** - Package initialization

## Quick Start

1. **Read the complete guide:** `TRADETRON_COMPLETE_GUIDE.md` - Follow every step
2. Get your API token from Tradetron dashboard
3. Add to `.env` file: `TRADETRON_API_TOKEN=your-token-here`
4. Test: `python tradetron/test_tradetron.py`
5. Use in your code: `from tradetron import TradetronClient`

## Usage

```python
from tradetron import TradetronClient
from trading.execution_engine import ExecutionEngine

# Initialize Tradetron client
client = TradetronClient()  # Loads token from .env

# Use with ExecutionEngine
engine = ExecutionEngine(client=client, ...)
```

## Documentation

- **📖 MAIN GUIDE**: `TRADETRON_COMPLETE_GUIDE.md` - Complete step-by-step guide with every detail

## Notes

- All Tradetron-related code should be added to this folder
- The client implements the `BrokerClient` interface for compatibility
- Tradetron uses signal-based execution (not direct orders)
- For paper trading, use "TT-PaperTrading" broker (no external broker needed)