# Decisions log

Every design decision the soul file and the build prompts did not settle, one
line each, with the reasoning compressed to what a reviewer needs to disagree
with it. Newest section last.

Authority order, unchanged: **soul file > this log > code**. Anything here that
turns out to contradict `beast-soul-file-v3.1.md` is wrong by definition and
should be raised, not reconciled locally.

---

## Phase 0/1 — regime layer (2026-09-08)

Built against `beast-soul-file-v3.1.md` and
`beast-regime-layer-claude-code-prompt.md` (revision 2).

### Operator decisions taken as given

| # | Decision | Source |
|---|---|---|
| D-01 | **Conflict 1 resolves as path B.** Analysis stays on the underlying for every market. Immutable Rule 9 stands unedited, instruction-block item 0 is reverted, and §4.7 widens instead so OI/IV/delta/liquidity carry more weight as chain-derived inputs to the underlying-based engine. §7.1's delta-based sizing is untouched. | Operator, 2026-09-08 |
| D-02 | Target repository is `Desktop\BEAST\beast-trader`, not the parallel `Rhathan Projects\BEAST` tree. | Operator |
| D-03 | The four ops-layer scripts (`alerts.py`, `watchdog.py`, `killswitch.py`, `preflight.py`) will be authored here rather than supplied. Acceptance criterion 6 of the ops prompt ("byte-identical to what I gave you") is therefore void and becomes "reviewed". Not yet started — that is Phase 2 work. | Operator |
| D-04 | Conflict 2 is treated as resolved in favour of complete simulated execution: instruction item 6 is operative, `paper_full_execution: true` is in Appendix A, and §10's "alert-only" paragraph is stale text. Recorded in the v3.2 diff for formal replacement. | Soul file v3.1 + prompt §0 |

### Deviations from the build prompt, and why

| # | Decision | Reasoning |
|---|---|---|
| D-05 | **Phase 0 did not scaffold a greenfield tree.** The prompt's §4 tree and §9's "no logic, every method body `raise NotImplementedError`" assume an empty repository. This one already holds ~14,400 lines built from soul file v3.0, with 139 passing tests. Rewriting working modules into stubs would destroy a working system to satisfy a diagram. Phase 0 was therefore executed as: honour §1's four consequences, update the config to v3.1, add what is genuinely missing, and leave working code alone. **This is the largest deviation in this log — see the module map below.** |
| D-06 | Existing module names were **not** renamed to §4's tree (`data/feature_engineering.py` stays put rather than becoming `core/indicators.py`, and so on). Renaming twenty modules breaks every import and every test to deliver no behavioural change. The map is in `README.md`. |
| D-07 | v3.1's `daily_loss_cap` is per-instrument (`{nifty: 0.15, sensex: 0.10, gold: 0.05}`) and the config now says so, but `RiskManager`'s session-pause state is still keyed by **family** (`indian`/`gold`). So Nifty's 15% and Sensex's 10% share one pause switch and whichever trips first pauses both. That is **stricter** than v3.1's per-instrument reading, never looser. Splitting the state per instrument is a risk-manager change, which Phase 1 is forbidden to make; it is queued for Phase 2. Soul-file open question B. |
| D-08 | `capital` lives at `risk.capital`, not as a top-level key. The prompt's §10 blocker table names it `capital`; the existing code already reads `risk.capital` throughout. Keeping one location beat renaming for cosmetics. Proposed for Appendix A under `risk:` in the v3.2 diff. |
| D-09 | The legacy `HMMRegimeEngine` in `core/hmm_engine.py` was **disabled, not deleted**, and its `hmm:` config block survives alongside the new `regime:` block. Phase 1 may not touch the entry pipeline, which still imports it. With `hmm.enabled: false` it returns an inert state and nothing it computes reaches a signal, a log or a sizing decision. Delete both in Phase 2. This is the one config duplication in the file and it is temporary. |
| D-10 | The build prompt's look-ahead test compares `predict_vol_state_filtered(data[0:400])[-1]` with `...(data[0:500])[400]`. Those are different bars — `data[0:400]` ends at index 399. The test as written would fail on a correct implementation. `tests/test_look_ahead.py` asserts the invariant the prompt *names* (index 399) and says so in the docstring. Not a case of adjusting a test to pass: the corrected test is strictly stronger, and it is also asserted across four cut points. |

### Design choices inside the regime layer

| # | Decision | Reasoning |
|---|---|---|
| D-11 | Emission log-densities and log-transitions are floored at `EMISSION_LOG_FLOOR = -1e4`. EM can collapse a state's covariance or drive a transition to exactly zero, and `log(0) = -inf` makes that state permanently unreachable — the forward recursion multiplies, so a posterior that hits exactly zero can never recover. The floor keeps the arithmetic finite without changing behaviour: a floored state is still indistinguishable from impossible against any rival with a sane density. |
| D-12 | Multi-bar returns are rolling **sums of gap-excluded one-bar returns**, not `log(close/close.shift(k))`. The prompt says to exclude gap-spanning returns from all return-based features, but a literal reading kills the 20-bar feature outright: an Indian 15M session is 25 bars, so nearly every 20-bar window straddles a boundary. The prompt's own wording resolves it — "rolling windows may span sessions; the individual return observations feeding them may not." |
| D-13 | `sma50_slope` and `volume_trend` are `(x - x.shift(5)) / (5 * scale)`, scaled by price and by mean volume respectively. The prompt says "slope" without defining it. Scaling makes the feature comparable across Nifty at 24,000, Sensex at 81,000 and Gold at 2,500; an unscaled slope would mostly encode the instrument's price level. |
| D-14 | RSI enters as a **z-score against a 50-bar rolling window**, not as a raw level. §4.1's 30/50/70 are entry thresholds; handing the model the raw level invites it to learn them and blur the line between the volatility layer and the entry layer. |
| D-15 | Feature windows (1/5/20 returns, 20-bar vol, 5/20 ratio, 50-bar slope, 200-bar distance, 10/20 ROC, 50-bar volume z, 10-bar volume trend) are module constants in `vol_features.py`, not config. They are regime-layer internals with no Appendix A home. The indicator **periods** (ADX, RSI, ATR) are a different matter and are read from Appendix A, never hardcoded. |
| D-16 | An unconfirmed state still reports its current bucket in `VolState.label`, with `is_confirmed: False` and the uncertainty multiplier applied. Reporting `UNKNOWN` until confirmation would hide information the operator wants on the dashboard, and the multiplier already carries the safety. |
| D-17 | `size_multiplier` bounds are enforced in `VolState.__post_init__`, so an out-of-range multiplier cannot be constructed at all. Checking at the point of use would leave the invariant dependent on every caller remembering it. |
| D-18 | `veto` on the `VolState` carries only the direction-independent half of the decision. `veto.on_turbulent_counter_bias` needs the proposed direction and the §4.4 regime, neither of which belongs in this layer, so `stability.veto_for_signal(vol_state, is_counter_bias)` combines them at gate G10. |
| D-19 | The scaler is persisted with the model. Applying a different scaler to the same model is the same class of bug as changing the feature set, and the feature hash would not catch it. |
| D-20 | `regime.hmm.random_state` and `regime.paths.model_dir` were added to the prompt's §6 schema. Determinism (same data, same states) and a model directory were both assumed by the prompt without being specified. |
| D-21 | `rolling_window_bars: 25000` was added, read only when `retrain_window: rolling`. The prompt offers `expanding | rolling` as a `# CONFIRM` without giving the rolling window a length. |
| D-22 | The look-ahead test fixture forces `covariance_type: diag`, while the shipped config says `full`. On the ~900 synthetic bars a test can afford, a full covariance over 14 features is singular — 105 free covariance parameters per state against a few hundred effective observations. At the real 12,500-bar minimum `full` is comfortably identified. Fitting a degenerate model in the test would have been testing the degeneracy, not the filter. **Worth watching in Phase 2:** if real 15M history is thinner than 12,500 bars, `full` should drop to `diag` rather than the bar minimum being lowered. |
| D-23 | `broker/alpaca_client.py` was deleted rather than left dormant, and XAUUSD routes to the new `broker/paper_broker.py`. Prompt §1 requires no Alpaca; leaving a dead adapter that can trade none of Beast's three markets would read as coverage that does not exist. |
| D-24 | `tests/test_vol_stability.py` is an addition to the prompt's three-module Phase 1 list. Shipping `StabilityTracker` with no tests would leave confirmation, flicker and the stale-hold budget unexercised — the three guards that separate a volatility layer from a number that resizes positions at random. |

### Open, and deliberately not decided here

- Every `# CONFIRM` in the new `regime:` block: multipliers, `stale_max_bars`,
  `zscore_lookback`, `carry_across_sessions`, the three veto switches, the
  walk-forward window sizes, and the per-market `min_train_bars`.
- Soul-file open items 17, 18 and 19 remain BLOCKERs: gold contract specs,
  option lot sizes and strike intervals, option liquidity floors. All still
  `null`, all still treated as failing.
- The v3.2 amendments in `docs/soul-v3.2-proposed-diff.md` are unapproved.
  `regime.enabled` ships `true` but the layer is not wired into sizing or the
  gate chain, so it currently changes nothing; the wiring must not land before
  those amendments are approved.


---

## main.py — startup, loop, shutdown (2026-09-08)

Built to an operator-supplied runtime spec. Four items in that spec contradicted
the soul file and were **not** implemented as written. Each is substituted with
the soul file's equivalent below, per the standing rule that the soul file
outranks every other instruction and a contradiction is reported rather than
reconciled locally.

| # | Spec said | Built instead | Why |
|---|---|---|---|
| D-25 | "connect to Alpaca, verify account" | Zerodha for Nifty/Sensex, `PaperBroker` for XAUUSD | Alpaca trades US equities and crypto. It can reach neither XAUUSD nor Indian index options, which are the only three instruments in scope. The adapter was deleted in Phase 0 (D-23) at the operator's direction. |
| D-26 | "StrategyOrchestrator: target allocation per symbol" | The G0–G9 gate chain, one signal per bar (§5.1) | Beast's decision unit is one trade at a time through an ordered short-circuiting gate chain, sized by fixed % risk. It does not allocate a portfolio and has nothing to rebalance; the regime prompt's §1 orders the allocation block deleted for exactly this reason. |
| D-27 | "each bar close, default 5-min bars" | Two clocks: a live-price exit path every cycle, and a closed-bar entry path on the §4.3 cascade (bias 15M/30M, setup 5M/15M, trigger 1M) | §4.2: *"Beast evaluates only on closed candles. Exception: stop-loss, target and trailing-stop triggers evaluate on live price, not candle close. A stop is a stop."* A single 5-minute bar-close loop would evaluate stops once every five minutes. |
| D-28 | "if model >7 days old or missing, retrain" | `regime.hmm.retrain_interval_days` (7) triggers a retrain; `max_model_age_days` (10) refuses the model outright | Same behaviour at the default, but the number lives in Appendix A rather than as a literal in logic, per the soul file's config discipline. Two thresholds because "time to refresh" and "too old to use" are different questions. |

### Decisions inside the runtime

| # | Decision | Reasoning |
|---|---|---|
| D-29 | A dead **bar** feed falls back to a direct **quote** to keep managing an open position, and escalates only when neither is available. Quote and history are different endpoints that fail independently, so a dead bar feed does not mean there is no price. Without this the exit path stopped exactly when the feed did — the opposite of §4.6, which blocks entries and leaves exits running. Caught by `tests/test_runtime.py::TestFeedHealth`. |
| D-30 | Entries pause after **3** consecutive feed failures, not the first. One miss is a hiccup; three is a feed that is not there. Exits are never gated on this. |
| D-31 | The runner halts after **10** consecutive unhandled loop errors. A loop that throws every cycle is not trading, it is paging, and an operator woken by the tenth identical alert is worse off than one woken by the first and told Beast stopped. Positions are left open on that path too. |
| D-32 | Startup **waits** for a session rather than exiting when nothing is open; `--exit-if-closed` inverts it. Starting early and idling is harmless; exiting at 09:10 means nobody is watching the 09:15 open. |
| D-33 | Positions are synced from the broker **before** the snapshot is read. Reading the snapshot first invites treating it as the position list, which it is not — it records what Beast believed at its last shutdown, wrong by construction if anything filled afterwards and absent entirely after a hard kill. |
| D-34 | Snapshot recovery restores risk counters only when the snapshot's session day matches today's **and** the mode matches. A restart must not reset the daily loss cap; a Friday snapshot must not carry Friday's losses into Monday; paper P&L must never seed a live session's cap. |
| D-35 | Retrains are deferred until every market is closed, even when due mid-session. §3 rule 5 and the model-freeze rule: swapping a model mid-session moves `vol_state` for reasons that are not in the market, and positions before and after would have been sized under different models. |
| D-36 | `broker/retry.py` **refuses** to retry a call marked non-idempotent, raising rather than resending. A submit that timed out may already have reached the exchange, and a blind resend is how one position becomes two. Retries are for reads until the ops layer supplies deterministic client order ids. |
| D-37 | `--dashboard` renders the persisted state (`state_snapshot.json` + today's journal), not a live attachment to a running process. Beast exposes no IPC, and adding a control socket to the trading host to serve a status pane is not a trade worth making. Stated in the command's own output so nobody mistakes it for live. |
| D-38 | `--compare` reports in **R**, not currency, and prints the caveat: buy-and-hold on an intraday series carries overnight gap risk Beast never takes, and pays spread twice in total against Beast's twice per trade. It is a sanity check on whether the rules add anything, not a verdict. |
| D-39 | `--dry-run` runs the entire pipeline including journalling and narration, skipping only `positions.open()`, and logs what it would have opened. Suppressing the journal too would make a dry run untestable against a real one. |

### Still not built here

The ops layer's reconcile is the gap this file names but does not fill.
`_step_sync_positions` detects which markets hold a position and reports
disagreements; it does **not** verify that every broker position has a resting
stop covering its full quantity, place one when it does not, restore targets
from the trade plan, rehydrate trail state under the 6.3 ratchet rule, cancel
orphan orders, or force SAFE mode on a position with no plan in the database.
Until that exists a position found at startup is reported loudly and left alone
rather than half-managed. That is Phase 2, together with `KILL`-flag polling,
the heartbeat writer and the `clean_shutdown` marker.


---

## monitoring/ package (2026-09-08)

Built to an operator-supplied spec. Five items in it contradicted the soul file
and were substituted rather than implemented as written.

| # | Spec said | Built instead | Why |
|---|---|---|---|
| D-40 | `BULL (72%)` in the REGIME panel | `CALM \| NORMAL \| TURBULENT \| UNKNOWN`, shown **beside** section 4.4's `TREND_UP / TREND_DOWN / RANGE` on a separate line | Directional labels on a model with no directional mandate. The regime prompt's §2.2 forbids them by name, and the reason is the display as much as the log: a "BULL" badge on a volatility read gets traded as a bias. The two concepts also keep two names — merging them into one "REGIME" field is the specific mistake the naming rule exists to prevent. |
| D-41 | `$105,230`, `SPY` | Rupees, and Nifty/Sensex/XAUUSD | Beast trades Indian index options and gold. There is no US equity anywhere in its scope. |
| D-42 | `Allocation: 95% \| Leverage: 1.25x`, "Rebalance 60%→95%" | **Open risk**: what the live positions lose if every stop fills, as a % of capital, plus concurrent-position counts against the §7 caps | Beast sizes each trade at a fixed % of capital and holds long options and futures outright. It has no target allocation and nothing to rebalance towards, and Immutable Rule 10 (long CE/PE only) means it cannot be levered. A leverage figure would be a number with no referent. |
| D-43 | `From Peak: 1.2%/10%` | Removed | §7 defines exactly two session-stopping conditions: the per-instrument daily loss cap and three consecutive losses. A drawdown-from-peak limit is not in the soul file and nothing in the engine enforces one, so displaying it would imply a rule that does not exist. |
| D-44 | `Daily DD: 0.3%/3%` | Daily loss against the actual per-instrument cap (Nifty 15%, Sensex 10%, Gold 5%) | 3% is not a Beast number. |

### Decisions inside monitoring

| # | Decision | Reasoning |
|---|---|---|
| D-45 | Rotation is **10 MB x 30 backups**, not "30 days". A logger cannot know how many days 300 MB covers — a quiet week and a volatile one differ by an order of magnitude — and a size-and-count policy has a bound a disk-space alert can be written against. Section 9's history is in the SQLite journal, so rotating a log away loses telemetry, never trade records. |
| D-46 | `main.log` receives **every** record; the other three streams are filtered views, not a partition. Reconstructing a session from four disjoint files means merging by timestamp and hoping the clocks agree. |
| D-47 | The six required context fields are injected by a logging filter from a process-wide `RuntimeContext` the runner refreshes once per cycle, not passed at each call site. Six extra arguments on every log call would drift out of sync within a week. The context is refreshed even with no dashboard attached — a headless run is exactly the one whose log gets read afterwards. |
| D-48 | `set_runtime_context` ignores unknown keys rather than raising. A caller passing a field this build does not carry should lose the field, not the log line. |
| D-49 | Only **confirmed** `vol_state` transitions raise an alert. An unconfirmed state moves with the posterior and would page several times an hour for a market that never changed regime — which is the flicker the stability layer exists to absorb, and re-emitting it as an alert would undo that work. |
| D-50 | `CIRCUIT_BREAKER`, `FEED_DOWN` and `API_LOST` are **critical** and bypass the 15-minute rate limit, and circuit breakers alert on **both** edges. For these the second occurrence is as informative as the first, because it means the condition did not clear; and an operator told entries stopped but never told they resumed will assume Beast is still halted. |
| D-51 | `LARGE_PNL` fires on crossing each **quarter of the daily cap**, not on an absolute number. Banded so it fires once per band rather than every cycle past a line, and expressed against the cap because "down 45,000" means nothing without knowing the cap is 75,000. |
| D-52 | `FEED_DOWN` carries different text depending on whether a position is open. With none it is an inconvenience; with one, in-process stop evaluation has lost its input and the resting broker stop is the only protection left — which the operator must be told, not left to infer. |
| D-53 | The Streamlit app is **read-only and file-backed**. It reads `state_snapshot.json`, the journal (opened `mode=ro` so it cannot take a write lock on a database the trading loop is writing), and the JSON log streams. It does not attach to the process: adding a network listener to the host holding the broker credentials, so a browser tab can be one cycle fresher, is not a trade worth making. |
| D-54 | The Streamlit app has **no** buttons that place, cancel or close anything. §8 puts every manual deviation behind an explicit confirmation step with full logging; a web button is precisely the frictionless override that section exists to prevent. Stopping Beast is `Ctrl-C` on the process, deliberately. |
| D-55 | The dashboard collapses per-market risk rows to one row per **family** before summing. Nifty and Sensex share a single `MarketRiskState`, so summing the rows as they arrive double-counts the Indian session's P&L. |

---

## Gold on MetaTrader 5 — contract specs and sizing (2026-09-14)

The soul file names the gold instrument keys as if gold were a futures
contract. The venue Beast has is a MetaTrader 5 demo account, which serves
XAUUSD as a **spot CFD priced in lots**. These are not the same object, and
the mapping between them is recorded here rather than reinterpreted in code.
It is also proposed as a Section 5.7.6 / Appendix A amendment in
`docs/soul-v3.2-proposed-diff.md`, Group C.

| # | Decision | Why |
|---|---|---|
| D-56 | `instruments.gold.trade: cfd` selects a new G8 path, `core/instrument_selector.py::CfdSelector`, and the **futures-named keys hold CFD fields**: `contract_multiplier` ← `trade_contract_size` (ounces per lot), `tick_size` ← `trade_tick_size`, `tick_value` ← `trade_tick_value` **per one lot in the account currency**. A CFD has no contract list and no expiry, so the leg is `FuturesLeg(contract=<broker symbol>, expiry=None, contracts=1, volume_lots=<broker volume>)` and P&L is `points × contract_multiplier × volume_lots`. `trade: futures` keeps the unchanged COMEX/MCX path. | The alternative — pretending the CFD is a futures contract with a synthetic expiry — would feed the rollover logic a date that does not exist and hide the unit change. Naming the mapping makes it reviewable. **What breaks on a broker change:** every value under `instruments.gold` is that broker's; `venue` records which one, and `_fill_gold_specs` refuses to apply values that differ from the previous session's. |
| D-57 | Gold volume is `floor((capital × 2% × vol_factor) / loss_per_lot / volume_step) × volume_step`, capped at `volume_max`, **rejected below `volume_min`**, where `loss_per_lot = (stop_distance / tick_size) × tick_value`. Section 7.1's whole-lot floor (`lots < 1` rejected) is **not** applied. | 7.1's floor is an index-options rule; `0.01` lots is a legitimate gold size. The spirit is kept exactly — round down, never up, refuse rather than inflate — on the broker's own grid. `risk_per_trade` and `daily_loss_cap` are untouched. |
| D-58 | `instruments.gold.account_currency` must be declared and must equal `account_info().currency`, or gold is refused at G9 with a logged reason. No conversion rate is assumed or read. | `trade_tick_value` is in the account currency; `risk.capital` is rupees (D-41). Dividing INR risk by a USD loss-per-lot oversizes by roughly the USD/INR rate — ~88×. Whether gold gets its own capital pool or a declared rate is an operator decision, deliberately not made here. |
| D-59 | A stop closer than `stops_level_points × point` is rejected **at G8**, with the reason logged, before sizing. The stop is never widened. | MT5 rejects such an order with retcode 10016 at `order_send`; discovering it there is the wrong moment. Section 8 forbids widening a stop after the fact, and a planned stop is no different. |
| D-60 | `data.gold_spread_max` is in **price units** (USD per ounce, compared directly to `ask − bid`), set from the p90 of a spread sample taken with `scripts/mt5_symbol_specs.py` during the hours Beast trades, with those hours recorded beside the value. It stays `null` — and gold stays refused at G1 — until such a sample exists. | Points vs price here is a 100× error waiting to happen; the unit is declared at the key and converted nowhere. A guessed ceiling either gates gold out permanently or lets the rollover blowout through. |
| D-61 | `_fill_gold_specs` (main.py) reads every `null` gold key from the live symbol at startup when `prefer_broker_contract_master` is true, logs each value with `(broker)` or `(config)`, never overwrites a non-null config value, leaves a `null` as `null` when MT5 is unreachable or reports zero, and on a later read compares against the first read — a change is logged as an error and raised as a circuit-breaker alert, **not applied**. | The Zerodha branch already works this way for lot sizes (open item 18). A broker read that fails must not become a default; a broker that changed the symbol between sessions is something the operator must be told, not something Beast resizes around. |
| D-62 | XAUUSD routes to `MT5Adapter` (`broker.routing`), which in `mode: paper` prices fills from the live MT5 tick and transmits nothing. `broker/paper_broker.py` is **no longer in the XAUUSD path**; D-23 and the README line describing it as the XAUUSD venue are stale. Fills on the demo ledger require `mode: live`, `broker.paper_trading: false` and `account_mode: demo`, and that switch is not made here. | Recorded because the prompt asked which was true. Routing was not changed in this task. |

---

## Phase 2 — startup reconcile and the ops layer (2026-09-14)

D-03 listed `alerts.py` as an unbuilt ops script; that was stale — it was
built with the monitoring package and is used here as is.

| # | Decision | Why |
|---|---|---|
| D-63 | `_step_sync_positions` was a live bug: it read `self.positions` (the in-process tracker, empty on a fresh start) and never asked the venue, so `_broker_positions` was always `[]` and `compare_positions` reported "broker holds nothing" unconditionally. Replaced by `core/reconcile.py`, which asks each connected adapter directly (`open_positions()`, `pending_orders()`, filtered by magic) and treats a venue that cannot be asked as **unknown** — entries refused, operator paged — never as flat. | Belief is what a hard kill invalidates. A reconcile that reads belief is not a reconcile. |
| D-64 | Stop semantics are the adapter's, stated in code: on MT5 the SL is a field of the position and covers its whole volume, so "is a stop resting" is `sl != 0` and an undersized stop cannot occur there; `BrokerPosition.stop_volume` exists for a venue that models stops as orders. An absent stop is placed at the plan level before any other startup step completes. A **wider** resting stop is flagged SAFE and left untouched — section 8 refuses a widening outright, and "correcting" toward the plan would be a loosening in disguise if the resting one is the truth. A **tighter** one is kept: 6.3's ratchet only tightens, so it is legitimate trail state. SAFE-mode stops go at `entry ± ops.safe_mode_stop_atr_multiple × ATR(setup TF)`, clamped inside any existing stop and outside `stops_level`, falling back to `instruments.gold.safe_mode_fallback_points`; with neither, CRITICAL and nothing placed. | Never widen; never invent a distance. |
| D-65 | `state_snapshot.json` is also written every `ops.snapshot_every_seconds` with `in_progress: true`, atomically, so the dashboard shows the live session. `clean_exit` keeps its meaning — it is only true on a clean shutdown; a live snapshot older than two minutes is shown red. | `save_snapshot` was called only in `shutdown()` and on the error path, so the dashboard showed last night. |
| D-66 | The KILL flag's default and safe mode is `halt`: entries off, exits running, Beast stays up. `flatten` demands `CONFIRM OVERRIDE: closing against plan` at the command line, stores it in the flag, and the loop routes every close through `core/override.py` again with that phrase — one friction step, not a second path — then exits 3. The flag is read at the top of every cycle, preflight refuses to start with one present, and the watchdog refuses to restart through one. | A stop is still a stop when the operator wants no *new* risk; section 8 is not to be bypassed by a file. |
| D-67 | The watchdog restarts `beast.service` only on a stale heartbeat, only after re-reading once, never on `clean_shutdown: true`, a KILL flag, or a deliberate exit code, and stops entirely after `ops.max_restarts` in `ops.restart_window_minutes` — raising CRITICAL and staying up to report. An open position in the last heartbeat makes every restart alert CRITICAL and names the position and its stop. It holds no broker handle and cannot touch an order. | Flapping a trading process against a live venue is worse than being down. Trading decisions stay in Beast. |
| D-68 | `main.py` exit codes: 0 clean, 1 startup failure, 2 deliberate halt (error streak), 3 KILL, 10 uncaught crash. `beast.service` has `RestartPreventExitStatus=0 1 2 3`. | So systemd and the watchdog can tell "it stopped on purpose" from "it died". |
| D-69 | Every order's comment is `beast:<market>:<session_day>:<plan_id>` (first 8 hex of the signal id; ≤ 31 chars), set once in `OrderExecutor` so every venue sees the same key. The reconcile matches on it first, then direction + entry time within 10 minutes, then direction alone at a confidence below `ops.reconcile_match_threshold` — which is SAFE mode. | Deterministic from the signal, so a resend carries the same tag and a position can be traced to its plan by comment alone. |
| D-70 | Preflight is `ExecStartPre` and fails closed: a check that cannot run (no `timedatectl`, no MT5) is a FAIL, not a skip. A REAL account under `mode: paper`, `BEAST_ALLOW_LIVE=1` without `account_mode: live`, a stale KILL flag, a drifting clock or a full disk each refuse the start. | The point is to refuse when the answer is unknown. |
| D-71 | `Signal.from_dict` rebuilds a signal from its Appendix B JSON so a matched position can be adopted by the tracker with its plan and trail. Trail state is not persisted anywhere, so it is reconstructed from the one fact the venue keeps — the resting stop: tighter than plan ⇒ the trail armed and the ratchet is at that stop. The restored stop is asserted never looser than the venue's. A payload that cannot be rebuilt leaves the position protected at the venue with entries paused, not half-managed. | Rehydrate under the ratchet or not at all. |

---

## The live dashboard (2026-09-14)

| # | Decision | Why |
|---|---|---|
| D-72 | The display payload **extends `heartbeat.json`** (schema v2) rather than adding a `live_state.json`. One writer, one file, one cadence; the watchdog reads `written_at`, `clean_shutdown`, `positions[].stop_present_at_broker` and `system.last_error` from it and the dashboard reads the rest. The trading loop never reads it. It is written at the end of every `tick()`, unconditionally, including a cycle that raised and a cycle in which no market was open, through `write_json_atomic` (same-directory temp, fsync, `os.replace`, `allow_nan=False`, a `BeastJSONEncoder` that stamps tzinfo and raises on NaN). Any failure is logged at WARNING at most once per five minutes, counted, and ignored. | Two files doing the same job is how one goes stale unnoticed; a dashboard that freezes exactly when something goes wrong is worse than none; a full disk must stop the dashboard, not the trader. |
| D-73 | `state_snapshot.json` keeps its **recovery** meaning and is also written every `ops.snapshot_interval_seconds` and immediately when the recovery-relevant state changes - a position opened or closed, a stop ratcheted or trail armed, a session pause tripped or cleared, a session rolled, an entry pause set or lifted - detected by fingerprint rather than by instrumenting each site. Every such write carries `clean_exit: false` and `in_progress: true`; only `shutdown()` writes `clean_exit: true`. `restore_counters` and `compare_positions` are unchanged. | A hard kill should lose seconds of recovery state, not a session, without a cosmetic file becoming safety-critical. |
| D-74 | The Streamlit app is split into two `st.fragment(run_every=monitoring.dashboard_refresh_seconds)` fragments: the live one reads `heartbeat.json` fresh (never cached); the journal one calls `@st.cache_data(ttl=monitoring.dashboard_journal_ttl_seconds)` queries keyed on the session day. SQLite is opened `mode=ro` with `PRAGMA query_only = ON`, a lock is reported not raised, and never `cache_resource`d. Log streams are read from the end, `monitoring.dashboard_log_tail_kb` at most, first partial line discarded. Malformed live JSON renders DOWN, never the previous value. | The previous app read the whole 10 MB log on every refresh and re-ran every journal query at 5 s; the fix is the cadence split, not a longer TTL. |
| D-75 | `.streamlit/config.toml` binds `address = "127.0.0.1"` and is the single source of truth; the unit passes no `--server.address`. `0.0.0.0` is forbidden. The unit runs as `beastview`, a separate user with read access to the state and journal and write access to its own home only, under `ProtectSystem=strict`. Access is an SSH tunnel or the existing nginx basic-auth proxy on :80. | Streamlit has no authentication; binding it publicly puts the journal on the box that holds the broker credentials. |
| D-76 | The status pill's thresholds are multiples of `loop_interval_seconds` carried in the file itself, not a fixed number of seconds: LIVE under 2x, LAGGING to 6x, STALE beyond; DOWN for no or unparseable file; HALTED on a KILL flag; STOPPED on `clean_shutdown`. STALE and DOWN dim or withhold every number. | The loop interval is 20 s for gold and would differ for another market; a threshold in seconds would be wrong for one of them. |
| D-77 | The header's account kind reads **UNVERIFIED (cfg: DEMO)** until the MT5 handshake has confirmed it this session. A blank `SessionFacts` has `is_demo=False`, and the first version of the writer rendered that as LIVE. | The page must never guess the one thing the account-mode guard exists to prove. |
| D-78 | The host clock must be in `sessions.timezone`; preflight refuses to start otherwise, and the VPS is set to `Asia/Kolkata` with `timedatectl`. The live-state writer stamps from `datetime.now(timezone.utc)` and converts, never by labelling a naive `datetime.now()` as the session zone. | Found by the dashboard: a file written seconds earlier read as 5.5 h old on the UTC VPS. `SessionClock` and `_trim_open_bar` make the same host-clock assumption, so on a UTC host every session window, the last-entry cutoff and the flatten time were 5.5 h off - silently. Making Beast tz-aware throughout is a larger change than this task allows; making the assumption explicit and enforced is not. |
