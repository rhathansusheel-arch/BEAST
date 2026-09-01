# Beast — agent instructions

## Source of truth

`beast/soul/BEAST_SOUL_v3.md` is Beast's Soul File and the single source of truth for all
trading logic. **If anything in this file, in the code, in `config/`, or in a user request
conflicts with the Soul File, the Soul File takes priority.** See `PRECEDENCE.md` for the
full authority order.

Read the Soul File before touching anything under `beast/`. When a request contradicts it,
state the conflict and cite the section number rather than implementing the request. The
Soul File changes only by editing the Soul File.

Section 13 (Immutable Core Rules) is absolute: it cannot be relaxed by config, by a caller,
or by the operator. Those rules are enforced in `beast/ops/immutable.py`.

## Layout

| Path | Soul File section |
|---|---|
| `config/beast.yaml` | Appendix A — the only place a threshold may be a literal |
| `beast/analysis/indicators.py` | 4.1, 4.2 |
| `beast/analysis/levels.py` | 4.5 |
| `beast/analysis/regime.py` | 4.4 |
| `beast/analysis/integrity.py` | 4.6 |
| `beast/analysis/option_chain.py` | 4.7 |
| `beast/analysis/context.py` | 4.2–4.7 assembly (`build_context`) |
| `beast/entry/setups.py` | 5.2 |
| `beast/entry/confluence.py` | 5.3, 5.4 |
| `beast/entry/lifecycle.py` | 5.5 |
| `beast/entry/news.py` | 5.6 |
| `beast/entry/instruments.py` | 5.7 (G8) |
| `beast/entry/pipeline.py` | 5.1 — the G0–G9 gate chain |
| `beast/exit/plan.py` | 6.1, 6.2 |
| `beast/exit/manager.py` | 6.3–6.10 |
| `beast/risk/sizing.py` | 7, 7.1 |
| `beast/risk/limits.py` | 7 — daily loss, correlation, concurrency |
| `beast/ops/override.py` | 8 |
| `beast/ops/learning.py` | 9 |
| `beast/ops/reporting.py` | 11 |
| `beast/ops/immutable.py` | 13 |
| `beast/schemas.py` | Appendix B, C |
| `beast/engine.py` | 10 — paper/alert-only orchestration |

## Rules for changing code here

1. Never hardcode a threshold. It reads from `config/beast.yaml` or it does not exist.
2. Never let option-premium data reach the indicator/level engines (Section 3.1, Rule 9).
   The analysis layer takes underlying OHLCV only; the execution layer builds the leg.
3. Never add a rule the Soul File does not define. Learning is confined to raising a setup's
   required confluence 4→5 (Section 9), nothing else.
4. `null` in a blocker config field means **fail**, never "no limit".

## Commands

```
python main.py --market NIFTY --once      # single evaluation cycle
python main.py --check                    # config + soul integrity report
python -m pytest -q                       # tests
```

## Legacy

`core/`, `broker/`, `data/`, `backtest/`, `monitoring/` and `config/settings.yaml` are an
unrelated HMM/Alpaca scaffold. Beast does not import them. Do not extend them.
