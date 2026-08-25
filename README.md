# regime-trader

HMM-based market regime detection and volatility-adaptive allocation, traded through Alpaca.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
copy config\credentials.yaml.example config\credentials.yaml
```

Fill in `.env` and `config/credentials.yaml` with your Alpaca API credentials, then run:

```
python main.py
```
