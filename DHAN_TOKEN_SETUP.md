# DHAN Token Setup and Daily Refresh Guide

## Where to Store Your Token

You have **3 options** to store your DHAN access token and client ID:

### Option 1: Environment Variables (Recommended for Testing)

**PowerShell:**
```powershell
$env:DHAN_ACCESS_TOKEN="your_token_here"
$env:DHAN_CLIENT_ID="1107954503"
```

**Windows Command Prompt:**
```cmd
set DHAN_ACCESS_TOKEN=your_token_here
set DHAN_CLIENT_ID=1107954503
```

**Note:** These are temporary and reset when you close the terminal.

### Option 2: .env File (Recommended for Production)

Create a file named `.env` in your project root with:

```
DHAN_ACCESS_TOKEN="your_token_here"
DHAN_CLIENT_ID="1107954503"
```

**Or PowerShell style:**
```
$env:DHAN_ACCESS_TOKEN="your_token_here"
$env:DHAN_CLIENT_ID="1107954503"
```

**Important:** Add `.env` to `.gitignore` to avoid committing your tokens!

### Option 3: Direct in Code (Not Recommended)

You can pass tokens directly when creating the client:
```python
from trading.dhan_client import DhanClient

client = DhanClient(
    access_token="your_token_here",
    client_id="1107954503"
)
```

## Daily Token Refresh

### Manual Refresh

Run the refresh script:
```bash
python refresh_dhan_token.py
```

This will:
1. Load your current token from `.env` or environment variables
2. Refresh it using DHAN RenewToken API
3. Update your `.env` file automatically
4. Show you the new expiry time

### Automated Refresh (Windows Task Scheduler)

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Daily at 9:00 AM (before market opens)
4. Action: Start a program
5. Program: `python`
6. Arguments: `refresh_dhan_token.py`
7. Start in: `C:\Users\pc44\Desktop\Krishna Crypto\Crypto-Commodities`

### Automated Refresh (PowerShell Script)

Create `refresh_token_daily.ps1`:
```powershell
cd "C:\Users\pc44\Desktop\Krishna Crypto\Crypto-Commodities"
.venv\Scripts\Activate.ps1
python refresh_dhan_token.py
```

Then schedule it with Task Scheduler.

## Token Expiry

- **Validity:** 24 hours
- **Refresh:** Use RenewToken API (requires valid token)
- **If Expired:** Generate new token from web.dhan.co

## Getting a New Token (If Expired)

1. Go to https://web.dhan.co
2. Login to your account
3. Navigate to **Profile** → **DhanHQ Trading APIs**
4. Click **Generate Access Token**
5. Copy the token
6. Update your `.env` file or environment variables

## Verification

Test your token:
```bash
python test_dhan_api_connection.py
```

This will show you:
- Token expiry status
- API connection status
- Account details (if token is valid)
