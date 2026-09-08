# Beast — regime-aware intraday trading agent

Beast is a rules-bound intraday trading agent for **Nifty 50 / Sensex** (traded as
long weekly index options) and **XAUUSD** (traded as futures). Its behaviour is
defined entirely by a specification document — the *soul file*
(`beast_brain_update_v3.md`) — and this repository is that document turned into
executable code.

The design premise, in the document's words:

> *"I trade the plan, not the feeling. My job is not to be right on every trade —
> my job is to execute the process with zero emotional drift, every single time,
> in every market I touch."*

**Current operational mode: paper trading / alert-only.** Beast identifies and
logs every qualifying trade as if live, places no real orders, and runs
performance tracking identically to how live mode will, so the data is comparable
later (soul file §10).

---

## The one structural rule

> **Beast analyses the underlying. Beast trades the instrument.**

Every indicator, level, setup, stop and target is computed on the **underlying
index or spot price**, in underlying points. Option premium never enters the
analysis layer — theta decay and IV shifts distort premium charts and would
produce false signals. The option leg is constructed *after* a valid underlying
signal exists, as an execution decision.

That split is enforced by the module layout:

| Layer | Modules | Knows about |
|---|---|---|
| **Analysis** | `data/feature_engineering`, `core/levels`, `core/confluence`, `core/hmm_engine`, `core/regime_strategies` | Underlying prices only |
| **Decision** | `core/signal_generator`, `core/exit_manager`, `core/risk_manager` | Underlying points, R-multiples |
| **Execution** | `core/instrument_selector`, `broker/*` | Strikes, premiums, contracts, lots |

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt

cp .env.example .env                               # add your API keys
cp config/credentials.yaml.example config/credentials.yaml   # optional alternative

python main.py --check                             # validate config, list blockers
python main.py --no-dashboard                      # run in paper mode
```

`--check` is the right first command. It prints which config values are still
unset and therefore which trades Beast will refuse — see **Blockers** below.

### Other commands

```bash
python main.py                     # live loop with the rich dashboard
python main.py --dry-run           # full pipeline, no position opened
python main.py --once              # a single evaluation cycle, then exit
python main.py --train-only        # fit the volatility models and exit
python main.py --backtest    data/samples/nifty_1m_sample.csv --market NIFTY50
python main.py --stress-test data/samples/nifty_1m_sample.csv --market NIFTY50
python main.py --compare     data/samples/nifty_1m_sample.csv --market NIFTY50
python main.py --dashboard         # print the persisted state and exit
python main.py --review            # section 9 weekly performance review
streamlit run monitoring/streamlit_app.py   # browser dashboard
pytest -q                          # 798 tests
```

`--dashboard` prints `state_snapshot.json` plus today's journal. It does **not**
attach to a running process: Beast exposes no IPC, and putting a control socket
on the trading host to serve a status pane is not a trade worth making. For a
live view, run Beast in the foreground — the dashboard is on by default.

`--compare` reports in **R**, not currency. Read it with the caveat it prints:
buy-and-hold on an intraday series carries overnight gap risk Beast never takes.

---

## The runtime (`main.py`)

**Startup**, in this order and not another: config → brokers and account →
market hours → volatility models (loaded or trained, then **frozen** for the
session) → risk manager and capital → positions synced **from the broker** →
`state_snapshot.json` → feeds → system state → `System online`.

Positions are synced before the snapshot is read on purpose. Reading the
snapshot first invites treating it as the position list, and it is not: it
records what Beast *believed* at its last shutdown, which is wrong by
construction if anything filled afterwards and absent entirely after a hard
kill. The broker is the authority; the snapshot only ever raises a question.

**The loop runs two clocks**, because §4.2 does:

- **Live-price path** — every cycle. Stops, targets, trails, the §6.10 premium
  backstop, and the §6.7 flatten sequence. Runs even when entries are paused,
  and even when the *bar* feed is down: quote and history are separate endpoints
  that fail independently, so a dead bar feed falls back to a direct quote. Only
  when neither is available does it escalate — a live position whose stop cannot
  be evaluated is the worst state in the module, and the resting stop at the
  broker is then the only protection left.
- **Closed-bar path** — entries, on the §4.3 cascade (bias 15M Indian / 30M
  Gold, setup 5M/15M, trigger 1M). `vol_state` recomputes once per **bias** bar,
  not once per cycle; scoring a bias-TF model on every trigger tick would inflate
  the consecutive-bar count into a confirmation that never happened.

**Three circuit breakers**, none of which touches an open position: the §7
session pause (daily cap or three consecutive losses, cleared only by a new
session), a feed pause after 3 consecutive failures, and a runner halt after 10
consecutive unhandled errors.

**Shutdown** on SIGINT/SIGTERM releases the terminal, closes feeds, **leaves
positions open**, writes the snapshot, prints the session summary. Positions are
deliberately not closed: their stops are resting, and flattening on every
shutdown would turn a routine restart — a deploy, a config change, a reboot —
into a realised loss on every open trade. What shutdown owes the operator is to
say loudly what it is leaving behind, which it does.

**Retries** (`broker/retry.py`) are three attempts with jittered exponential
backoff, and they **refuse** to retry a call marked non-idempotent. A submit that
timed out may already have reached the exchange; a blind resend is how one
position becomes two. Retries are for reads until the ops layer supplies
deterministic client order ids.

---

## How a trade happens

### The entry pipeline (§5.1)

Ten gates, in order, short-circuiting. The first failure rejects the candidate
and is logged with its gate ID, so §9 can see *where* signals die rather than
only what was taken.

| Gate | Name | Passes when |
|---|---|---|
| G0 | Session | Inside the window, past the opening-range guard, before the last-entry cutoff |
| G1 | Data integrity | Feed fresh, spread within limit, gap handled |
| G2 | No-trade conditions | Not in a news blackout or a loss-limit pause |
| G3 | Regime / bias | Direction permitted under the current regime |
| G4 | Setup detection | A valid Setup 1–4 instance exists and is unexpired |
| G5 | Confluence | ≥ 4 of 6 indicators aligned (≥ 5 counter-bias) |
| G6 | Trigger | The setup's trigger fired inside its validity window |
| G7 | Trade viability | Stop within bounds; R:R target feasible against the next opposing level |
| G8 | Instrument | A tradable option strike, or a futures contract outside rollover |
| G9 | Risk & portfolio | Size computed; concurrent, correlation and loss-limit checks clear |

**One signal per bar.** When two setups qualify on the same trigger close, Beast
takes the one with the tighter structural stop; ties break toward the higher
confluence count.

### The three-tier cascade (§4.3)

| Role | Indian | Gold | Purpose |
|---|---|---|---|
| Bias / regime | 15M | 30M | Directional bias and regime filter. Never generates entries. |
| Setup | 5M | 15M | Level mapping, setup detection, indicator confluence. |
| Trigger | 1M | 1M | Precision entry trigger and invalidation. |

All evaluation happens on **closed candles only** — the one exception being
stop, target and trailing-stop triggers, which evaluate on live price. A stop is
a stop.

### The four setups (§5.2)

1. **Trendline breakout/breakdown** — trend-continuation column
2. **Reversal at support/resistance** — reversal column; the only setup permitted
   counter to the bias regime, and then only at 5-of-6 on a Tier A level
3. **Order block retest** — trend-continuation column
4. **Indicator confluence trend continuation** — trend-continuation column;
   permitted only in trending regimes

### Confluence (§5.3–5.4)

Six indicators on the setup timeframe: **ADX+DI, Stochastic, MACD, RSI, Bollinger
Bands, session VWAP**. ATR is a utility for buffers and sizing and is *never*
counted. Each indicator has **two** alignment rules — trend-continuation and
reversal — and the column is selected by setup type; the two are never mixed
within one count. §5.4's machine-checkable definitions ("fresh crossover",
"expanding histogram", "walking the band") are implemented as testable predicates
in `core/confluence.py`, not re-interpreted at each call site.

A **conflict rule** applies: if the opposing direction registers 3+ aligned
indicators, the signal is rejected regardless of the primary count. A 4–3 split
is disagreement, not confluence.

### Exits (§6)

Every trade is entered with a complete exit plan computed *before* the signal is
emitted. An entry whose exit plan cannot be constructed is not an entry.

- **Stop** — structural (trendline swing / rejection wick / OB far edge /
  pullback extreme) plus a `0.25 × ATR` buffer, clamped to `[0.5, 2.5] × ATR`.
  Above the maximum the trade is **rejected**, never sized down.
- **Target** — fixed `2.0R`. If an opposing Tier A level blocks it, the trade is
  rejected rather than re-targeted (configurable).
- **Trail** — arms at `+1.0R` (`+0.7R` on expiry day), moving the stop to
  breakeven + costs in one step, then ATR-chandelier with a ratchet that only
  ever tightens.
- **Premium hard stop** (options only) — exit at 65% of entry premium as a
  backstop for when delta/IV break the mapping. It can only cut risk short,
  never widen it.
- **Session flatten** — no position is ever carried past the hard-flat time.

---

## Option handling

### Strike selection (§5.7)

Runs only after a valid underlying signal with a complete price plan exists.
Nearest weekly expiry; target delta **0.55** within a **0.45–0.65** band; long CE
for bullish, long PE for bearish. **No option selling, no spreads, ever** — the
risk model in §7 is built on a bounded, known loss per trade, and short options
would silently break every cap in it.

Every liquidity filter must pass: open interest, session volume, bid-ask spread,
both sides quoted, premium floor. **An unset filter is treated as failing, not
passing.**

### Sizing (§7.1)

Options break the linear formula, because the loss at the stop is not the premium
paid — it is the *premium decline* when the underlying reaches the stop:

```
premium_loss_per_unit = underlying_stop_distance × delta
raw_lots              = (capital × risk_pct) ÷ (premium_loss_per_unit × lot_size)
lots                  = floor(raw_lots × vol_factor)
```

Three caps sit on top and the binding one wins: a **premium outlay cap** (10% of
capital), a **whole-lot floor** (`lots < 1` is rejected, never rounded up), and
**no delta re-reading** mid-trade. `vol_factor` is clamped at 1.0, so volatility
adjustment can only ever *reduce* exposure.

### Option chain as context (§4.7)

The chain is a **level and context source, never a signal source** — it never
adds to or subtracts from the 4-of-6 count. Max call/put OI strikes enter the
same Tier A level pool as price structure and can therefore reject trades at G7.
Where an OI level lands within `0.25 × ATR` of a price-structure level, the merged
zone gets `+2` strength: the chart and the positioning agree.

---

## Risk (§7)

| | Nifty | Sensex | Gold |
|---|---|---|---|
| Risk per trade | **5%** (v3.1) | 3% | 2% |
| Daily loss cap | **15%** (v3.1) | 10% | 5% |
| Alt trigger | 3 consecutive losses | 3 consecutive losses | 3 consecutive losses |
| Max concurrent | 3 (Indian, shared) | 3 (Indian, shared) | 1 |

v3.1 raised Nifty risk from 3% to 5% and its daily cap from 10% to 15%, and made
the cap per-instrument rather than one combined Indian number. Note that the
outlay cap absorbs most of that increase in practice: at
`max_premium_outlay_pct: 0.10` the outlay cap binds before the risk cap on most
legs, so effective Nifty risk stays well under 5% (soul file open question D).

One caveat, recorded as `DECISIONS.md` D-07: the caps are per-instrument but the
session **pause** switch is still family-scoped, so whichever of Nifty and Sensex
trips first pauses both. That is stricter than v3.1 intends, never looser.

The pause lasts the rest of the session — no exceptions, no
"one more trade to win it back". Nifty and Sensex are correlated: same-direction
positions count as **one** against the cap and share a single trade's risk
allocation; opposite-direction simultaneous positions are not permitted.

---

## Two regime classifiers, and only one of them decides anything

`core/hmm_engine.py` holds `RuleRegimeClassifier`, which implements §4.4 exactly
— ADX, DI and the Bollinger basis on the bias timeframe. **This is
authoritative.** Gate G3 consults this and nothing else.

It also still holds `HMMRegimeEngine`, the first-generation volatility overlay.
**That one is disabled** (`hmm.enabled: false`) and superseded by
`core/regime/`. Two defects make it unfit to run: it infers with
`predict_proba`, which is forward-backward smoothing and therefore look-ahead
biased, and it labels states with a directional component (`low_vol_up`) on a
model with no directional mandate. It is disabled rather than deleted only
because Phase 1 of the regime build may not touch the entry pipeline that still
imports it; both go in Phase 2 (`DECISIONS.md` D-09).

The separation between a rule classifier that decides and a statistical layer
that does not exists because §9 bounds what learning may do: *"Beast does not
invent new setup types on its own. Learning is confined to how strictly it
applies the setups already defined."* A statistical model silently vetoing
trades would be exactly the unconstrained learning the document rules out.

The same bound applies to `core/learning.py`: the **only** value it can hand back
to the pipeline is a raised confluence requirement (4 → 5) for a setup+market pair
whose expectancy has gone negative over a rolling 30-trade sample, reverting when
expectancy recovers. It cannot touch a single cap in §7.

---

## Discipline enforcement (§8)

Beast will not accept a manual close on a live position unless it has hit its
stop, target or trailing stop. Any other close request requires an **exact typed
confirmation** — `CONFIRM OVERRIDE: closing against plan` — every time, with no
fatigue exception. Two actions have no confirmation at all and are simply
refused: **widening a stop after entry**, and **adding to a position mid-trade**.

Every accepted override is logged with the trade context and the outcome the
original plan would have produced, and the weekly report states what the
overrides cost or saved in R.

---

## Blockers — values Beast refuses to trade without

Several config values ship as `null` **on purpose**. The soul file is explicit
that an unset threshold is treated as *failing*, not passing, so the affected
trades are rejected at G8/G9 with a logged reason rather than falling back to a
guessed default.

| Config key | Blocks | Open item |
|---|---|---|
| `instruments.{nifty,sensex}.lot_size` / `.strike_interval` | All index option trades | 18 |
| `options.min_oi` / `min_volume` / `min_premium` | All index option trades | 19 |
| `instruments.gold.venue` / `contract_multiplier` / `tick_size` / `tick_value` | All gold trades | 17 |
| `data.gold_spread_max` | Gold entries (the high-spread rule is unenforceable) | 15 |

Run `python main.py --check` to see the live list. Lot sizes and strike ladders
change by exchange notification, so `prefer_broker_contract_master: true` lets
Beast read them from the Zerodha contract master at startup and leave config as
the override.

---

## Project layout

```
beast-trader/
├── config/
│   ├── beast_config.yaml       # Appendix A + regime: - the ONLY place a threshold is literal
│   └── credentials.yaml.example
├── core/
│   ├── config.py                # loader; require() turns unset values into refusals
│   ├── schemas.py               # Appendix B (Signal) and C (Trade Log)
│   ├── session.py               # §3, §6.7 session windows and flatten sequence
│   ├── levels.py                # §4.5 swings, S/R zones, trendlines, order blocks
│   ├── confluence.py            # §5.3 two-mode engine + §5.4 definitions
│   ├── hmm_engine.py            # §4.4 rule classifier (authoritative); legacy overlay, disabled
│   ├── regime/                  # the volatility layer - vol_state, NOT §4.4's regime
│   │   ├── contracts.py         #   VolState + ModelMetadata - the whole public surface
│   │   ├── vol_features.py      #   session-aware features + the causal z-score scaler
│   │   ├── hmm_engine.py        #   BIC selection + FILTERED (forward-only) inference
│   │   └── stability.py         #   confirmation, flicker, stale-hold, size_multiplier
│   ├── regime_strategies.py     # §5.2 the four setups, detection and triggers
│   ├── option_chain.py          # §4.7 OI levels, PCR, IV state, max pain
│   ├── instrument_selector.py   # §5.7 G8 - strike / contract selection
│   ├── risk_manager.py          # §7, §7.1 sizing, loss limits, correlation
│   ├── exit_manager.py          # §6 plan construction and the exit state machine
│   ├── signal_generator.py      # §5.1 the ten-gate pipeline + §11 reason line
│   ├── learning.py              # §9 bounded self-learning
│   ├── session_state.py         # state_snapshot.json - risk counters across a restart
│   ├── override.py              # §8 the friction step
│   └── ai_analyst.py            # Claude narration - advisory only, never decides
├── broker/
│   ├── zerodha_client.py        # Nifty/Sensex spot + weekly option chain
│   ├── paper_broker.py          # the simulated venue - XAUUSD and all paper-mode fills
│   ├── order_executor.py        # the only place an order can be sent
│   ├── retry.py                 # bounded backoff; refuses to retry a write
│   └── position_tracker.py      # live positions -> Appendix C records
├── data/
│   ├── market_data.py           # feed assembly + CSV replay provider
│   ├── feature_engineering.py   # §4.1 indicators (Wilder-smoothed)
│   └── news_calendar.py         # §5.6 blackout, fail-closed
├── monitoring/
│   ├── logger.py                # 4 rotating JSON streams + the runtime context
│   ├── dashboard.py             # the terminal view (rich)
│   ├── streamlit_app.py         # the browser view - read-only, file-backed
│   ├── alerts.py, journal.py
├── backtest/
│   ├── backtester.py, performance.py, stress_test.py
├── docs/
│   └── soul-v3.2-proposed-diff.md   # amendments awaiting operator approval
├── tests/
├── DECISIONS.md                 # every unsettled decision, one line each
└── main.py
```

### Where the build prompt's tree maps onto this one

The regime build prompt describes a greenfield tree. This repository already
existed, so the modules were left where they are rather than renamed to match a
diagram — renaming twenty modules breaks every import to deliver no behavioural
change (`DECISIONS.md` D-05, D-06). The correspondence:

| Prompt's path | Here |
|---|---|
| `core/indicators.py` | `data/feature_engineering.py` |
| `core/level_engine.py` | `core/levels.py` |
| `core/regime_classifier.py` | `core/hmm_engine.py::RuleRegimeClassifier` |
| `core/setups.py` | `core/regime_strategies.py` |
| `core/entry_pipeline.py` | `core/signal_generator.py` |
| `core/journal.py` | `monitoring/journal.py` |
| `core/news_calendar.py` | `data/news_calendar.py` |
| `options/chain_analysis.py` | `core/option_chain.py` |
| `options/leg_builder.py` | `core/instrument_selector.py` |
| `data/bar_aggregator.py` | `data/feature_engineering.py::resample_ohlc` |
| `core/regime/*` | same — this is the only new package |

`core/session_state.py` now exists. Still queued for Phase 2: the ops layer -
the full reconcile (verify a resting stop covers every broker position before
anything else), `KILL`-flag polling, the heartbeat writer and the
`clean_shutdown` marker.

---

## The volatility layer (`core/regime/`)

Two concepts, two names, and conflating them would be the fastest way to get a
volatility model quietly steering direction:

| Name | Values | Authority |
|---|---|---|
| `regime` | `TREND_UP \| TREND_DOWN \| RANGE` | Soul file §4.4, rule-based ADX/DI/BB on the bias TF. Governs which setups are permitted. |
| `vol_state` | `CALM \| NORMAL \| TURBULENT \| UNKNOWN` | The HMM. Governs nothing except how much smaller the next position is. |

**The HMM is a volatility classifier.** It does not predict direction, generate
signals, or pick setups, and it does not replace, modify or vote on §4.4.

Three invariants, each asserted in code rather than documented and hoped for:

1. **It can only shrink.** `size_multiplier` lives in `(0, 1.0]` — enforced in
   `VolState.__post_init__`, so an out-of-range value cannot be constructed —
   and combines with §7's `vol_factor` by `min()`, floored at
   `risk.vol_factor_floor`. Never by product: the two measure the same thing, so
   multiplying double-counts volatility, and the product's floor would be
   `0.5 x 0.5 = 0.25`, silently overriding a soul-file value.
2. **A veto only blocks.** Nothing in the layer can permit a candidate the
   G0–G9 chain rejected.
3. **Inference is filtered.** The forward algorithm alone, in log space:
   `P(state_t | observations_1..t)`. No `predict()`, no `predict_proba()` — both
   run forward-*backward* and revise past states with future bars, which is
   look-ahead bias that makes a backtest look excellent and live trading look
   like a different system. `tests/test_look_ahead.py` asserts the filtered
   posterior differs from the smoothed one everywhere except the final bar,
   where the two must agree exactly.

Labels are sorted by **mean realized volatility, ascending** — never by mean
return. `CRASH / BEAR / BULL / EUPHORIA` are directional labels on a model with
no directional mandate, and once the string `"BULL"` exists in a log somebody
will eventually read it as bias. Return-sorted labels are also unstable across
retrains: this week's `BULL` is not next week's.

One model per market. Nifty, Sensex and Gold have different sessions and
different volatility distributions, and Sensex carries a 15-minute feed delay
(§4.6, §12) that the other two do not, so every Sensex `VolState` is tagged
`data_delay_minutes: 15`. Volume features are per-market: XAUUSD spot is OTC and
its feed "volume" is the provider's tick count, a measure of publishing rate
rather than traded size, so `regime.features.use_volume.gold` is `false` and a
training run whose matrix disagrees with that policy fails rather than fits.

### Not yet wired in

The layer computes and logs and **changes nothing**. Gate `G10_VOL_STATE` is not
in the chain and `effective_size_factor` is not called by the risk manager,
because both are Section 7 and Section 5.1 behaviour and the soul file is the
single source of truth for those. The amendments are drafted in
`docs/soul-v3.2-proposed-diff.md` and await approval. Wiring them in before
approval would be exactly the "sizing input that exists only in code" that the
governance model exists to prevent.

---

## Monitoring

**Four rotating JSON streams** (`monitoring/logger.py`), 10 MB x 30 backups each:

| Stream | Holds |
|---|---|
| `main.log` | Everything. The complete ordered session record. |
| `trades.log` | Signals, fills, closed trades — the Appendix B/C stream. |
| `alerts.log` | Anything the operator was paged about. |
| `regime.log` | `vol_state` and §4.4 regime transitions, model events. |

`main.log` is a superset, not a partition — reconstructing a session from four
disjoint files means merging by timestamp and hoping the clocks agree.

Every record carries the same **runtime context**: timestamp, `vol_state` and its
probability, §4.4 regime, equity, open positions, and the day's P&L. It is
injected by a logging filter rather than passed at each call site, so it cannot
drift. That matters when reading a log afterwards: the question is almost never
*what happened* but *what was true when it happened*. A stale-feed warning at
14:31 reads very differently when the same line says equity was down 12% and a
position was open.

**Two views, both read-only:**

- `monitoring/dashboard.py` — the terminal, on by default, refreshing every 5s.
  Six panels: volatility & regime, portfolio, positions, recent signals, risk
  status with colour-coded bars, system.
- `monitoring/streamlit_app.py` — `streamlit run monitoring/streamlit_app.py`.
  Reads `state_snapshot.json`, the journal (opened `mode=ro`, so it can never
  take a write lock on a database the trading loop is writing) and the log
  streams. It does **not** attach to the process — see `--dashboard` above for
  why — and it has no button that places, cancels or closes anything. §8 puts
  every manual deviation behind an explicit confirmation step with full logging;
  a web button is exactly the frictionless override that section prevents.

**What the views deliberately do not show.** No `BULL`/`BEAR` labels — those are
directional labels on a model with no directional mandate, and a badge is read as
bias just as surely as a log line. No allocation or leverage — Beast sizes each
trade at a fixed % of capital and Immutable Rule 10 permits long CE/PE only, so
neither number has a referent. No drawdown-from-peak — §7 defines exactly two
session-stopping conditions (the per-instrument daily cap and three consecutive
losses), and showing a third would imply a rule nothing enforces. What is shown
instead is **open risk**: what the live positions lose if every stop fills.

**Alerts** (`monitoring/alerts.py`) carry the soul file's named conditions plus
eight operational ones: `VOL_STATE_CHANGE`, `REGIME_CHANGE`, `CIRCUIT_BREAKER`,
`LARGE_PNL`, `FEED_DOWN`, `API_LOST`, `MODEL_RETRAINED`, `FLICKER_EXCEEDED`.
Rate limited to one per (kind, market) per 15 minutes, so a wide spread on Gold
never suppresses a loss-limit pause on Nifty. Critical kinds bypass the limit —
for a dead feed or a lost broker session the second occurrence is as informative
as the first, because it means the condition did not clear.

---

## The AI layer

`core/ai_analyst.py` calls Claude (`claude-opus-5`, adaptive thinking) to write
the operator-facing narration, the weekly review, and the rejection summary.

**It never makes a trading decision.** It is called *after* a signal is fully
formed, with the signal as input, and cannot change any field. Every call is
wrapped so that an API failure, a refusal or a timeout degrades to the
deterministic text — Beast must keep trading when Claude is down.
`ai.advisory_only` is validated at startup and must stay `true`.

Set `ANTHROPIC_API_KEY`, or authenticate once with `ant auth login`. Set
`ai.enabled: false` to run entirely offline.

---

## Testing

```bash
pytest -q                                  # 708 tests
pytest tests/test_look_ahead.py            # the ones that decide whether backtests mean anything
pytest tests/test_size_multiplier_invariant.py   # the volatility layer can only shrink
```

`tests/test_look_ahead.py` is the most important file in the suite. It now
verifies five distinct forms of look-ahead:

1. An indicator value never changes when later bars arrive.
2. A confirmed swing never appears before its confirmation candles have closed.
3. The replay provider never hands the engine a bar that had not closed by the
   evaluation timestamp.
4. A volatility **feature** at bar T does not change when later bars arrive.
5. The filtered **posterior** at bar T is identical whether computed from
   `data[0:T]` or `data[0:T+100]` — and differs from `predict_proba`'s smoothed
   posterior everywhere except the final bar, where the two must agree exactly.
   That final-bar equality is what proves the divergence is smoothing rather
   than a bug in the forward recursion.

If any of those fail, every performance figure in the project is fiction.

`tests/test_size_multiplier_invariant.py` is a property test over every
combination of vol state, confidence, flicker condition and `vol_factor`: the
effective size factor never exceeds 1.0 and never falls below
`risk.vol_factor_floor`, and no configuration lets a veto approve anything.

---

## Still open

Soul file v3.1 ends with 34 items flagged for operator confirmation, plus the
`# CONFIRM` keys the new `regime:` block adds. Every one is a single line in
`config/beast_config.yaml`, and nothing is duplicated in code.

Both of v3.1's own blockers are now closed: **Conflict 1 is resolved as path B**
(analysis stays on the underlying, Immutable Rule 9 stands, §4.7 widens instead)
and **Conflict 2** in favour of complete simulated execution. Both are written up
for formal amendment in `docs/soul-v3.2-proposed-diff.md`, which is unapproved.

The remaining items that change behaviour most:

- **Infeasible-target policy** (`exit.target_infeasible_policy`) — currently
  *reject*; the alternative targets the blocking level with a 1.5R floor. This is
  the single biggest driver of how many trades Beast takes.
- **Setup 1 trigger mode** — `break_close` catches more moves and eats more
  fakeouts; `retest` is the reverse.
- **Target delta 0.55 / band 0.45–0.65** — the biggest driver of how much
  underlying edge survives into premium.
- **Premium hard stop at 35%** — too tight and it fires ahead of valid underlying
  stops; too loose and it stops being a backstop.
- **Expiry-day block** — ATM/ITM only, 13:30 last entry, wider delta band, faster
  trail. Or turn expiry-day trading off entirely until there is data.
- **Sensex as options** — assumed to mirror Nifty. If it is traded as futures it
  moves to the Gold execution path with no change to its analysis.
