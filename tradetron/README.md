# Tradetron Integration

This folder contains all Tradetron-related code and documentation for paper trading commodities.

## Contents

- **`tradetron_client.py`** - Main API client for sending signals to Tradetron
- **`test_tradetron.py`** - Test script to verify your setup
- **`HOW_TRADETRON_WORKS.md`** - **START HERE!** Clarifies how Tradetron works and answers common questions
- **`TRADETRON_PAPER_TRADING_GUIDE.md`** - Detailed setup and usage guide
- **`TRADETRON_SIMPLE_EXPLANATION.md`** - Simple step-by-step explanation
- **`__init__.py`** - Package initialization

## Quick Start

1. Get your API token from Tradetron dashboard
2. Add to `.env` file: `TRADETRON_API_TOKEN=your-token-here`
3. Test: `python tradetron/test_tradetron.py`
4. Use in your code: `from tradetron import TradetronClient`

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

- **⚠️ READ FIRST**: `HOW_TRADETRON_WORKS.md` - Explains how Tradetron works, answers "Do I need Angel One?" and clarifies the API token location
- **Simple Guide**: `TRADETRON_SIMPLE_EXPLANATION.md` - Easy step-by-step instructions
- **Detailed Guide**: `TRADETRON_PAPER_TRADING_GUIDE.md` - Comprehensive documentation

## Notes

- All Tradetron-related code should be added to this folder
- The client implements the `BrokerClient` interface for compatibility
- Tradetron uses signal-based execution (not direct orders)
