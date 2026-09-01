# Beast

A rules-bound intraday trading agent for Nifty, Sensex and XAUUSD.

Beast is not an advisor and not a signal service its operator can casually override. It is
an execution system whose entire value comes from *not* deviating — from itself, or from
its operator's emotions — once a trade plan is set.

## The Soul File governs

Everything Beast does is defined by **[`beast/soul/BEAST_SOUL_v3.md`](beast/soul/BEAST_SOUL_v3.md)**.

> **Where any instruction conflicts with the Soul File, the Soul File wins.**

That applies to this README, to `config/beast.yaml`, to the Python source, and to anything
the operator asks for at runtime. The full authority order is in
[`PRECEDENCE.md`](PRECEDENCE.md); the agent-facing version is in [`CLAUDE.md`](CLAUDE.md).
Every emitted signal and trade record carries the Soul File's SHA-256, so any decision can
be traced back to the exact brain revision that produced it.

## The one structural rule

> **Beast analyses the underlying. Beast trades the instrument.** (Soul File 3.1)

Indicators, levels, setups, stops and targets are computed on the index or spot price, in
underlying points. The traded instrument is built afterwards, as an execution decision:

| | Nifty / Sensex | Gold |
|---|---|---|
| Analysed on | index spot | XAUUSD |
| Traded via | long options — buy CE / buy PE | futures |
| Stop | underlying level + premium hard stop backstop | underlying level, direct |
| Sizing | delta-based, from premium-at-risk at the stop | linear, point distance × multiplier |

The analysis layer physically cannot see option premium: `beast/analysis/indicators.py`
raises under Immutable Rule 9 if handed anything but an underlying feed.

## Layout

| Path | Soul File section |
|---|---|
| `config/beast.yaml` | Appendix A — the only place a threshold may be a literal |
| `beast/analysis/indicators.py` | 4.1 indicators, 4.2 closed-candle rule |
| `beast/analysis/levels.py` | 4.5 swings, S/R zones, trendlines, order blocks |
| `beast/analysis/regime.py` | 4.4 regime classifier |
| `beast/analysis/integrity.py` | 4.6 data integrity gate |
| `beast/analysis/option_chain.py` | 4.7 OI levels, PCR, IV state, DTE |
| `beast/analysis/context.py` | `build_context` — the whole measurement layer |
| `beast/entry/setups.py` | 5.2 the four setup types |
| `beast/entry/confluence.py` | 5.3 two alignment columns, 5.4 definitions |
| `beast/entry/lifecycle.py` | 5.5 validity window and anti-overtrading |
| `beast/entry/news.py` | 5.6 news blackout (fails closed) |
| `beast/entry/instruments.py` | 5.7 strike selection / contract selection |
| `beast/entry/pipeline.py` | 5.1 the G0–G9 gate chain |
| `beast/exit/plan.py` | 6.1 stop, 6.2 target, G7 viability |
| `beast/exit/manager.py` | 6.3–6.10 trailing, priority, session flatten, premium stop |
| `beast/risk/sizing.py` | 7 futures sizing, 7.1 delta-based option sizing |
| `beast/risk/limits.py` | 7 daily loss, cooldown, concurrency, correlation |
| `beast/ops/override.py` | 8 the typed-confirmation friction step |
| `beast/ops/learning.py` | 9 rolling expectancy → confluence weighting |
| `beast/ops/reporting.py` | 11 signal and alert lines |
| `beast/ops/immutable.py` | 13 immutable rules, as runtime guards |
| `beast/schemas.py` | Appendix B and C |
| `beast/engine.py` | 10 paper / alert-only orchestration |

## Running it

```
pip install -r requirements.txt

python main.py --check                                  # what governs Beast, what blocks it
python main.py --market NIFTY --replay data/nifty_1m.csv
python -m pytest -q
```

`--check` prints the Soul File revision, the operating mode, and every Appendix A value
still unset. **An unset blocker fails, it does not pass** — Beast refuses to trade rather
than guess (5.7.3, 3.1).

## Before Beast can trade

The Soul File's Open Items 17–19 are blockers, and `--check` lists them:

* **`capital`** — every Section 7 cap is a percentage of it.
* **Gold contract specs** — venue, contract multiplier, tick size, tick value, plus a
  confirmed session window. Beast will not size a Gold trade without them.
* **Option lot sizes and strike intervals** for Nifty and Sensex (these change by exchange
  notification — reading them from the broker API is the better answer).
* **Option liquidity floors** — `min_oi`, `min_volume`, `min_premium`.
* **`data.gold_spread_max`** — without it the high-spread rule cannot be enforced.

Everything marked `# CONFIRM` in `config/beast.yaml` is a Soul File
`[DEFAULT — pending confirmation]`. Changing one is a one-line change, by design.

## Operating mode

Paper trading, alert-only (Section 10). Beast identifies and logs every qualifying trade as
if live, places no orders, and runs performance tracking identically to live mode so the
data is comparable later. Fills are simulated at the trigger candle's close with the
conservative assumptions in 6.8 — the stop is always assumed to have filled first.

Journals land in `logs/`: `signals.jsonl`, `trades.jsonl`, `rejections.jsonl`,
`overrides.jsonl`. The rejection log carries the failing gate ID for every candidate, which
is how Section 9 answers whether Beast is missing trades at G5 or at G7.

## Legacy

`core/`, `broker/`, `data/`, `backtest/`, `monitoring/` and `config/settings.yaml` are the
remains of an unrelated HMM/Alpaca US-equities scaffold. Beast imports none of it. Where its
values conflict with Appendix A, Appendix A governs.
