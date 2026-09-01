# Instruction for Claude Code

Update the core decision-making engine ("the brain") of the Beast trading agent using the specification below.

This is Beast's **soul file** — the single source of truth for how it decides to enter, hold, and exit trades. Implement or update the agent's logic so it matches this document exactly, in particular:

0. **Instrument architecture (Sections 3.1, 4.7, 5.7):** this is the structural change in v3. **All analysis runs on the underlying index/spot — never on the option premium chart.** Signals, levels, indicators, stops and targets are computed in underlying points. The traded instrument is then constructed as a separate step: a **long option leg** for Nifty/Sensex, a **futures contract** for Gold. Build this as a clean two-layer separation — an *analysis layer* that is instrument-agnostic, and an *execution layer* that translates a price plan into the tradable instrument. Do not let option premium data leak into the indicator engine.
1. **Technical Analysis Framework (Section 4):** exact indicator parameters, the closed-candle evaluation rule, the three-tier timeframe cascade (bias → setup → trigger), the regime classifier, and the algorithmic definitions for swing points, S/R zones, trendlines and order blocks. These definitions are shared infrastructure — Sections 5 and 6 both call into them, so build them once as a level/context engine, not inline per setup. **Section 4.7 adds the option-chain layer for Nifty/Sensex** — OI-derived levels feed the same Tier A level pool as price structure.
2. **Entry logic (Section 5):** the 10-gate entry pipeline in 5.1 (implement it as an ordered, short-circuiting gate chain — a signal that fails any gate is rejected and logged with the failing gate ID), the four setup types with their per-setup detection/trigger/invalidation rules in 5.2, and the 4-of-6 indicator confluence engine in 5.3. Note that 5.3 has *two* alignment modes per indicator — Trend-Continuation and Reversal — and the correct column must be selected based on which setup type is being evaluated. Do not use a single flat rule per indicator; implement both modes and the setup-type-based selection logic. Section 5.4 defines the ambiguous terms in that table ("fresh crossover", "expanding histogram", "at/near the level", "walking the band") in machine-checkable form — use those definitions, do not re-interpret them.
3. **Exit logic (Section 6):** stop placement from structure + ATR buffer, fixed R:R target, trailing stop activation and ratchet mechanics, session-close flatten sequence, and the exit-priority resolution rules. No manual early exit outside the defined trigger conditions. **Section 6.10 adds the option-specific layer** — the underlying-level stop stays primary, with a premium hard stop as a safety net for when delta/IV break the mapping.
4. **Risk management (Section 7):** per-market risk-per-trade, daily loss limits (both the % cap and the 3-consecutive-loss trigger), volatility-adjusted position sizing, the correlation rule for simultaneous Nifty/Sensex exposure, and the **delta-based sizing formula for option legs** (7.1) — sizing from premium-at-risk at the underlying stop, not from premium paid.
5. **Override/discipline enforcement (Section 8):** the explicit-confirmation friction step on any manual override attempt, with full logging.
6. **Operational mode (Section 10):** current phase is paper trading / alert-only — log every qualifying signal as if live, place no real orders, but run performance tracking identically to how live mode will.

Treat Sections 1–3 and 9, 11–13 as governing context/config rather than new logic to build, but keep them in sync — market hours, timeframes, communication style, and the immutable rules should be reflected in how the agent is structured and how it reports signals.

**Config discipline:** every value marked `[DEFAULT — pending confirmation]` must be surfaced in the single config block in Appendix A, not hardcoded in logic. Appendix A is the only place a threshold may be literal; everything else reads from it. Changing any flagged default must be a one-line change.

**Schemas:** Appendix B defines the Signal object and Appendix C the Trade Log record. Emit and persist exactly these fields — Section 9's learning loop and Section 11's reporting both read from them.

---

# BEAST — Trading Agent Soul File

> This document defines who Beast is, what it is permitted to do, and the rules it may never break.
> It is the single source of truth for Beast's behavior across all markets it trades.

**Version:** 3.0 — instrument architecture resolved: Nifty/Sensex traded as **long options**, Gold as **futures**. Adds Section 3.1 (instrument specs), 4.7 (option chain analysis), 5.7 (strike selection), 6.10 (option exit handling) and 7.1 (delta-based sizing). Analysis framework from v2 is unchanged and remains common to both.
**Version 2.0** — Sections 4, 5 and 6 expanded from principle to executable specification.

---

## 1. Identity

**Name:** Beast
**Role:** Fully autonomous intraday trading agent
**Markets:** Indian intraday (Nifty, Sensex) and XAUUSD (Gold)
**Mode:** Autonomous execution, governed entirely by this document — no discretionary deviation, including by its own operator.

Beast is not an advisor. Beast is not a signal service the operator can casually override. Beast is a rules-bound execution system whose entire value comes from *not* deviating — from itself, or from its operator's emotions — once a trade plan is set.

**Core identity statement (for the agent to internalize):**
> "I trade the plan, not the feeling. My job is not to be right on every trade — my job is to execute the process with zero emotional drift, every single time, in every market I touch."

---

## 2. Prime Directive

1. Follow this document's logic identically across both markets. The *analysis framework* does not change based on which market is open — only the market-specific parameters below do.
2. Never take a trade that does not satisfy the entry criteria in Section 5.
3. Never hold, add to, or exit a trade outside the rules in Section 6, regardless of who is asking.
4. Preserve capital before pursuing profit. A missed trade costs nothing. A rule violation costs trust in the whole system.

---

## 3. Market Scope & Trading Hours

| Market | Analysis instrument | Traded instrument | Trading Window (IST) | Last New Entry | Flatten Begins | Hard Flat |
|---|---|---|---|---|---|---|
| Nifty | Nifty 50 index (spot) | **Weekly index options — long CE / long PE** | 9:15 AM – 3:30 PM | 3:00 PM | 3:20 PM | 3:25 PM |
| Sensex | Sensex index (spot) | **Weekly index options — long CE / long PE** | 9:15 AM – 3:30 PM | 3:00 PM | 3:20 PM | 3:25 PM |
| Gold | XAUUSD price | **Gold futures contract** | 5:00 AM – 9:00 PM | 8:30 PM | 8:50 PM | 8:55 PM |

*Last-entry, flatten-begin and hard-flat times are `[DEFAULT — pending confirmation]`.*

**Opening-range guard `[DEFAULT — pending confirmation]`:** no new entries in the first 15 minutes of the Indian session (9:15–9:30 AM). Session VWAP and opening-range levels are not yet meaningful before that. XAUUSD has no equivalent guard — its 5:00 AM window start is not a liquidity event.

### 3.1 Instrument Architecture

This is the single most important structural rule in the document:

> **Beast analyses the underlying. Beast trades the instrument.**
> Every indicator, level, setup, stop and target in Sections 4, 5 and 6 is computed on the **underlying index or spot price**. Option premium charts are never used for analysis — their indicators are distorted by theta decay and IV shifts and will produce false signals. The option leg is constructed *after* a valid underlying signal exists, as an execution decision, not an analysis one.

| | Nifty / Sensex | Gold |
|---|---|---|
| Analysed on | Index spot (points) | XAUUSD price |
| Traded via | Long options (buy CE for long bias, buy PE for short bias) | Futures contract |
| P&L relationship to underlying | **Non-linear** — delta, theta, vega all intervene | **Linear** — 1 point of underlying = fixed currency value |
| Stop defined as | Underlying level, executed as a premium exit + premium hard-stop backstop (6.10) | Underlying level, direct |
| Risk per trade equals | Premium lost when underlying reaches the stop (7.1) — *not* the full premium paid | Point distance × contract value |
| Extra pre-trade checks | Option chain: strike liquidity, OI levels, IV state, expiry proximity (4.7, 5.7) | Contract rollover proximity |
| Direction expression | Long CE / Long PE. **No option selling, no spreads, no naked short options.** `[DEFAULT — pending confirmation]` | Long / short futures, both permitted |

**Why long-options-only is the default:** Beast's entire risk model in Section 7 is built on a bounded, known loss per trade. Short option positions have unbounded risk that cannot be expressed as a fixed % of capital, and would silently break every cap in Section 13. If option selling is ever wanted, it needs its own risk section — it is not a config toggle.

**Flagged for confirmation:** Sensex is assumed to follow the same long-options treatment as Nifty (weekly index options). Confirm — if Sensex is traded as futures instead, it moves to the Gold execution path with no change to its analysis.

**Flagged for confirmation:** the Gold futures venue and contract are not yet specified. Tick size, lot/contract multiplier, margin and the exact session window all depend on whether this is COMEX GC, MCX Gold/Gold Mini, or a broker CFD written on the futures. The 5:00 AM – 9:00 PM IST window carried over from v1 matches neither COMEX nor MCX exactly, so confirm both. Contract specs are stubbed in Appendix A and Beast will refuse to size a Gold trade until they are populated.

No positions are opened outside these windows. Any position still open at window close is flattened according to the exit logic in Section 6.7 — Beast does not hold Indian intraday positions overnight, and does not hold XAUUSD positions past the 9 PM cutoff.

---

## 4. Technical Analysis Framework

The same core toolkit is applied in both markets, with one addition for XAUUSD. This section defines the shared measurement layer. Sections 5 and 6 do not compute levels themselves — they consume what this section produces.

### 4.1 Indicator Parameters

All parameters are `[DEFAULT — pending confirmation]` and live in Appendix A.

| Indicator | Parameters | Role |
|---|---|---|
| ADX + DI | ADX(14), +DI(14), −DI(14) | Trend strength + directional bias |
| Stochastic | %K(14), smoothing 3, %D(3) | Momentum / overbought-oversold |
| MACD | 12 / 26 / 9 (EMA-based) | Trend & momentum confirmation |
| RSI | RSI(14) | Momentum / divergence |
| Bollinger Bands | 20-period SMA basis, 2.0 standard deviations | Volatility / mean-reversion zones |
| Session VWAP | Anchored to that market's session open (9:15 IST Indian; 5:00 IST XAUUSD) | Intraday fair value reference |
| ATR *(utility)* | ATR(14) on the setup timeframe | Buffers, sizing, stop distance — **not a confluence indicator, never counted in the 4-of-6** |

**Threshold defaults** (also `[DEFAULT — pending confirmation]`): ADX trend filter 20, Stochastic 20/80, RSI 30/50/70.

### 4.2 Evaluation Cadence & the Closed-Candle Rule

- Beast evaluates **only on closed candles.** No entry decision is ever made from an in-progress candle. This eliminates repainting and makes backtest and live behaviour identical.
- **Exception:** stop-loss, target and trailing-stop triggers evaluate on **live price**, not candle close. A stop is a stop.
- Cadence:
  - **Bias timeframe** context recomputed on each bias-TF close.
  - **Setup timeframe** levels and setup detection recomputed on each setup-TF close.
  - **Trigger timeframe** entry-trigger check runs on each trigger-TF close.
- Indicator values used for confluence are read **from the setup timeframe** unless a rule explicitly names another timeframe. One timeframe per confluence count — never mix a 15M MACD with a 1M RSI in the same tally.

### 4.3 Timeframe Cascade

| Role | Indian (Nifty/Sensex) | XAUUSD | Purpose |
|---|---|---|---|
| **Bias / regime** | 15M | 30M | Directional bias and regime filter only. Never generates entries. |
| **Setup** | 5M | 15M | Level mapping, setup detection, indicator confluence count. |
| **Trigger** | 1M | 1M / 5M | Precision entry trigger and invalidation reference. |

**Assumption flagged for confirmation:** the 30M timeframe is treated as XAUUSD-only, used for bias/context rather than entry triggers on the lower timeframes. If 30M should also apply to Nifty/Sensex, say so and this updates in one line.

**Cascade rule:** a setup is only actionable if it was identified on the setup TF, is not contradicted by the bias TF per 4.4, and is triggered on the trigger TF. All three must agree in sequence. A trigger-TF signal with no setup-TF setup behind it is noise and is discarded without logging.

### 4.4 Regime Classifier (bias timeframe)

On each bias-TF close, Beast classifies the market into exactly one regime:

| Regime | Condition (bias TF) | Permitted setups |
|---|---|---|
| `TREND_UP` | ADX ≥ 20 **and** +DI > −DI **and** close > BB middle band | Long: 1, 3, 4. Short: 2 only (counter-trend reversal, see below). |
| `TREND_DOWN` | ADX ≥ 20 **and** −DI > +DI **and** close < BB middle band | Short: 1, 3, 4. Long: 2 only. |
| `RANGE` | ADX < 20, or the DI/BB conditions disagree | Setups 2 and 3 only. Setup 1 and 4 are suppressed. `[DEFAULT — pending confirmation]` |

**Counter-bias reversals:** Setup 2 taken against the prevailing bias-TF regime requires **5 of 6** confluence instead of 4, and the level being reversed at must be a bias-TF level (Section 4.5 tier A). `[DEFAULT — pending confirmation]`

### 4.5 Level Engine — Algorithmic Definitions

The agent must not "eyeball" levels. Each object below has a deterministic definition, a validity lifetime, and an invalidation condition.

**Swing point (fractal):** a candle whose high is the highest of the 2 candles either side of it (swing high), or whose low is the lowest of the 2 either side (swing low). Confirmed only after the 2 following candles close. Lookback window: 100 candles on the relevant TF. `[DEFAULT — pending confirmation: fractal N = 2]`

**Support/Resistance zone:**
- Formed where **≥ 2 swing points of the same type** cluster within `0.15 × ATR(14)` of each other on the setup TF.
- Zone width = `0.25 × ATR(14)`, centred on the mean of the clustered swings.
- **Tier A (major):** derived from the bias TF, or from a session-structural price — prior-day high/low, prior-day close, current-session high/low, and for Gold the overnight (pre-05:00 IST) high/low. **For Nifty/Sensex, option-chain OI levels (4.7) are also Tier A** and enter the same pool, competing on the same strength score.
- **Tier B (minor):** derived from the setup TF only.
- **Strength score** = number of touches + 1 per prior rejection from the zone. Reported on the signal for transparency; does not gate the trade.
- **Invalidation:** a zone is dead once price closes beyond it by `> 0.5 × ATR` on the setup TF. A dead resistance zone becomes a candidate support zone (flip), retaining Tier but resetting strength to 1.

**Trendline:**
- Requires **≥ 3 confirmed swing points** of the same type (lows for an up-trendline, highs for a down-trendline), fitted by least squares.
- Valid only if the maximum perpendicular deviation of any anchor point from the fitted line is `≤ 0.20 × ATR`, and no candle has **closed** beyond the line between the first and last anchor.
- "Previously respected" = the ≥3 touches condition above. Wicks through the line do not break it; closes do.
- **Invalidation:** one close beyond the line by `≥ 0.10 × ATR` on the setup TF retires the trendline (and may generate a Setup 1 signal — see 5.2).

**Order block:**
- An **impulse** is: ≥ 3 consecutive same-direction setup-TF candle closes, **or** a move of `≥ 1.5 × ATR` within 5 candles — and in either case the move must break a prior confirmed swing point (structure break).
- The **order block** is the last opposing-direction candle immediately before that impulse.
- **Zone = the candle body (open→close).** `[DEFAULT — pending confirmation; alternative is full high→low wick range]`
- **Freshness:** an OB is *fresh* until price has traded into its zone once. After the first retest it is marked *used* and generates no further signals. `[DEFAULT — pending confirmation]`
- **Invalidation:** a setup-TF close beyond the far edge of the zone kills the OB permanently.
- **Expiry:** an untouched OB expires at the end of the session in which it formed (Indian) or after 24 hours (XAUUSD). `[DEFAULT — pending confirmation]`

**Rejection candle** (used by Setups 2 and 3): a candle satisfying **at least one** of —
- wick on the tested side ≥ 50% of the candle's total range, and close back inside the zone; or
- an engulfing candle in the reversal direction (body engulfs the prior candle's body); or
- a close back inside the zone after any wick pierce, combined with an RSI divergence at the level (per 5.4).

### 4.6 Data Integrity Gate

Before any evaluation cycle, Beast checks:
- **Staleness:** if the most recent candle close is older than 2× the trigger-TF interval, mark the feed `STALE` and suppress new entries for that instrument until fresh data resumes.
- **Sensex delay:** all Sensex evaluation carries a `data_delay_minutes: 15` tag (Section 12). Sensex signals are flagged on every emission and Sensex is excluded from any trigger-TF logic tighter than 5M. `[DEFAULT — pending confirmation]`
- **XAUUSD spread:** if spread > `SPREAD_MAX` (Appendix A), emit HIGH SPREAD ALERT, no entry. Existing positions are unaffected — exits still honour their triggers.
- **Gap detection:** if the session opens more than `1.0 × ATR` from the prior close, all levels are recomputed before the first entry is permitted.
- **Option chain freshness (Nifty/Sensex):** the chain snapshot must be no older than **3 minutes**, or option-derived levels are marked stale and dropped from the Tier A pool for that cycle. Price-structure levels continue to work — a stale chain degrades context, it does not halt trading. `[DEFAULT — pending confirmation]`

### 4.7 Option Chain Analysis — Nifty / Sensex Only

The option chain is a **context and level source**, not a signal source. It never adds to or subtracts from the 4-of-6 confluence count in 5.3 — that engine stays exactly as defined, on six indicators, on the underlying. What the chain does is (a) contribute Tier A levels, (b) inform the target-feasibility check at G7, and (c) constrain which instrument is bought at G8.

**Chain snapshot:** pulled on each setup-TF close for the nearest weekly expiry, covering ATM ± **10 strikes**. `[DEFAULT — pending confirmation]`

**4.7.1 OI-derived levels (feed into 4.5 as Tier A)**

| Level | Definition | Reads as |
|---|---|---|
| **Max Call OI strike** | Strike with the highest call open interest in the snapshot range | Resistance — call writers defend it |
| **Max Put OI strike** | Strike with the highest put open interest | Support — put writers defend it |
| **Second Max Call / Put OI** | Next-highest on each side | Secondary resistance / support |
| **Max OI-change strike (intraday)** | Largest OI *addition* since session open, call side and put side | Intraday level being actively built; often the day's operative boundary |
| **Max Pain** | Strike at which total option-holder value is minimised | Magnet/context only — never a trade trigger `[DEFAULT — pending confirmation]` |

Zone width for an OI level is `± 0.25 × strike_interval` (so ±12.5 pts on a 50-pt Nifty ladder), not ATR-based, because these are fixed price ladders rather than swing clusters.

**Level convergence bonus:** where an OI level lands within `0.25 × ATR` of a price-structure Tier A level, the merged zone's strength score gets **+2**. This is the highest-quality level Beast can identify — the chart and the positioning agree.

**4.7.2 OI change interpretation (context tag on every Nifty/Sensex signal)**

Computed from simultaneous price and OI change on the underlying's dominant strikes since session open:

| Price | OI | Tag | Bias implication |
|---|---|---|---|
| Up | Up | `LONG_BUILDUP` | Bullish, fresh money |
| Down | Up | `SHORT_BUILDUP` | Bearish, fresh money |
| Up | Down | `SHORT_COVERING` | Bullish but weaker — a squeeze, not accumulation |
| Down | Down | `LONG_UNWINDING` | Bearish but weaker |

This tag is **recorded on every signal** (Appendix B) and used by Section 9 for learning. It is **not** a gate by default — but it is the natural first candidate if the operator later wants a chain-based filter, and it needs trade history before that decision is worth making. `[DEFAULT — pending confirmation: `oi_tag_as_gate: false`]`

**4.7.3 PCR (Put-Call Ratio)**

Total put OI ÷ total call OI across the snapshot range, recorded as context. Extremes (> 1.5 or < 0.6) are logged as a `PCR_EXTREME` flag. Contrarian at extremes, meaningless in the middle — which is exactly why it is logged rather than traded. `[DEFAULT — pending confirmation]`

**4.7.4 IV state**

- **ATM IV** and **IV percentile** (rank vs the trailing 30 sessions) are captured on each snapshot.
- `IV_ELEVATED` flag when IV percentile > **80**. Long option positions bought into elevated IV are exposed to IV crush the moment the event passes — this compounds with the news blackout in 5.6, which is the primary defence.
- `IV_CRUSH_RISK` flag when a scheduled high-impact event sits inside the trade's expected holding window. `[DEFAULT — pending confirmation]`

**4.7.5 Expiry proximity**

`days_to_expiry` (DTE) is computed each cycle and drives strike selection in 5.7:

| DTE | Regime | Effect |
|---|---|---|
| ≥ 2 | Normal | Standard strike rules (5.7) |
| 1 | Pre-expiry | Standard rules, theta flagged |
| 0 | **Expiry day** | ATM-or-ITM only, earlier last-entry cutoff, see 5.7.4 |

---

## 5. Entry Logic

Beast trades four setup types. A trade is only valid when **at least one setup type is present AND indicator confluence supports the direction** — and only after every gate in 5.1 has passed.

### 5.1 The Entry Pipeline

Gates run in order and short-circuit. The first failing gate rejects the candidate; the rejection is recorded with the gate ID so Section 9 can learn where signals die.

| Gate | Name | Passes when |
|---|---|---|
| **G0** | Session | Inside the trading window, after the opening-range guard, before the last-entry cutoff (Section 3). |
| **G1** | Data integrity | Feed fresh, spread within limit, gap handled (4.6). |
| **G2** | No-trade conditions | Not inside a news blackout window; not inside a post-loss-limit pause (5.5, 7). |
| **G3** | Regime / bias | Proposed direction permitted under the current regime (4.4). |
| **G4** | Setup detection | A valid Setup 1–4 instance exists on the setup TF (5.2), and is not expired or already traded. |
| **G5** | Confluence | ≥ 4 of 6 indicators aligned in the proposed direction using the correct column (5.3), or ≥ 5 for counter-bias reversals. |
| **G6** | Trigger | The setup's trigger condition fires on the trigger TF within the validity window (5.2, 5.5). |
| **G7** | Trade viability | Structural stop within min/max distance; R:R target feasible against the next opposing level — **including OI-derived Tier A levels for Nifty/Sensex** (6.1, 6.2, 4.7.1). |
| **G8** | Instrument selection | **Nifty/Sensex:** a tradable option leg exists — expiry, strike, delta band, liquidity and spread all pass (5.7). **Gold:** contract specs loaded and not inside a rollover window. Pass-through for futures otherwise. |
| **G9** | Risk & portfolio | Position size computed (7.1 for options, 7 for futures); concurrent-position cap, correlation rule and daily loss state all clear. |
| **→** | **Emit** | Signal object (Appendix B) is emitted and logged. In paper mode, no order is placed. |

Pseudocode:

```
on trigger_tf_close(instrument):
    ctx = build_context(instrument)          # 4.2–4.6
    if not gate_G0(ctx): return reject(G0)
    if not gate_G1(ctx): return reject(G1)
    if not gate_G2(ctx): return reject(G2)

    for setup in active_setups(instrument):   # 5.2, freshest first
        direction = setup.direction
        if not gate_G3(ctx, direction, setup): continue
        mode = REVERSAL if setup.type == 2 else TREND_CONTINUATION
        count = confluence_count(ctx, direction, mode)     # 5.3
        required = 5 if setup.counter_bias else 4
        if count < required: continue
        if not setup.trigger_fired(ctx): continue
        plan = build_trade_plan(setup, ctx)   # stop, target, trail — Section 6 — ALL IN UNDERLYING POINTS
        if not plan.viable: continue          # G7

        leg = select_instrument(plan, ctx)    # G8 — 5.7 for options, contract lookup for futures
        if leg is None: continue              # no tradable strike / contract → reject, log G8 reason

        sized = size_position(plan, leg, ctx) # G9 — 7.1 delta-based for options, linear for futures
        if not sized.permitted: continue
        return emit_signal(setup, plan, leg, sized, count)

    return no_signal
```

**One signal per bar.** If two setups qualify on the same trigger-TF close, Beast takes the one with the **tighter structural stop** (better R per unit risk); ties break toward the higher confluence count, then toward the setup type with the better trailing 30-trade expectancy from Section 9. Beast never fires two entries on the same instrument from one candle.

### 5.2 Setup Types — Detection, Trigger, Invalidation

Each setup specifies four things: how it is *detected* on the setup TF, what *triggers* the entry on the trigger TF, where the *stop* sits, and what *disqualifies* it.

---

**Setup 1 — Trendline Breakout/Breakdown**
*Confluence column: Trend-Continuation.*

- **Detect:** a valid trendline per 4.5 (≥ 3 anchor touches, unbroken) exists, and a setup-TF candle closes beyond it by `≥ 0.10 × ATR`.
- **Momentum requirement:** the breaking candle's range is `≥ 1.2 × ATR` **or** the next candle continues in the break direction. Volatility expansion stands in for volume where reliable volume is unavailable (index derivatives, spot gold). `[DEFAULT — pending confirmation]`
- **Trigger (two modes, config-selected — default `break_close`):**
  - `break_close`: enter on the close of the first trigger-TF candle that closes beyond the line following the setup-TF break.
  - `retest`: wait for price to return to the broken line (now flipped S↔R) and print a rejection candle there; enter on that candle's close. Retest must occur within the validity window (5.5) or the setup expires.
- **Stop:** beyond the last opposing swing point on the far side of the broken line, plus `0.25 × ATR` buffer.
- **Disqualifiers:** an opposing Tier A level sits closer than `1.0 R` from entry (no room to target); the break occurs inside the news blackout; the trendline has fewer than 3 anchors.

---

**Setup 2 — Reversal at Support/Resistance**
*Confluence column: Reversal. This is the only setup permitted counter to the bias regime, at 5-of-6.*

- **Detect:** price trades into a Tier A or Tier B zone (4.5) and prints a **rejection candle** (4.5) on the setup TF, closing back inside the zone.
- **Trigger:** on the trigger TF, price closes back above (bullish) / below (bearish) the zone edge with the rejection candle's extreme intact — i.e. the extreme has not been exceeded before the trigger closes.
- **Stop:** beyond the rejection wick's extreme, plus `0.25 × ATR` buffer.
- **Disqualifiers:** the zone has already been tested twice this session and failed both times (5.5 re-entry cap); the zone is Tier B and the trade is counter-bias; price arrived at the zone in a single impulse candle `> 2.5 × ATR` (momentum blowthrough risk — wait for a second test). `[DEFAULT — pending confirmation]`

---

**Setup 3 — Order Block Retest**
*Confluence column: Trend-Continuation.*

- **Detect:** a fresh, unexpired order block (4.5) exists in the direction of the impulse that created it, and price re-enters its zone.
- **Trigger:** a rejection or continuation candle closes inside the zone in the impulse direction on the trigger TF.
- **Stop:** beyond the far edge of the OB zone, plus `0.25 × ATR` buffer.
- **Disqualifiers:** OB already marked *used*; OB has expired; a setup-TF close has already gone beyond the far edge (OB dead); price entered the zone in the direction *opposite* the impulse with a structure break in between.

---

**Setup 4 — Indicator Confluence Trend Continuation**
*Confluence column: Trend-Continuation. Permitted only in `TREND_UP` / `TREND_DOWN` regimes.*

- **Detect:** on the setup TF, ADX ≥ 20 **and rising** (current ADX > ADX 3 candles ago), DI aligned with the regime, and price is in a **pullback** — defined as a retracement to the Session VWAP, the BB middle band, or the most recent setup-TF swing, without breaking the prior structural swing in the trend direction.
- **Trigger:** a resumption candle on the trigger TF — a close in the trend direction that reclaims the pullback reference level (VWAP / BB mid / swing) it pulled back to.
- **Stop:** beyond the pullback's extreme swing point, plus `0.25 × ATR` buffer.
- **Disqualifiers:** the pullback has retraced more than 61.8% of the prior impulse leg (trend health in question); ADX is falling; the trade would be taken against the bias TF. `[DEFAULT — pending confirmation on the 61.8% figure]`

---

### 5.3 Confluence Requirement

> **Minimum 4 of 6 indicators must be aligned in the desired trade direction before Beast pulls the trigger.** (5 of 6 for counter-bias Setup 2 — see 4.4.)

Alignment logic is context-dependent: **Setup 2 (Reversal at S/R)** uses the *Reversal Alignment* column below, since it looks for exhaustion/turn signals at an extreme. **Setups 1, 3, and 4 (Trendline Break, Order Block Retest, Trend Continuation)** use the *Trend-Continuation Alignment* column, since they trade with an established directional move. Beast selects the column matching the setup type it is currently evaluating — never mixes the two for a single confluence count.

| Indicator | Trend-Continuation Alignment (Setups 1, 3, 4) | Reversal Alignment (Setup 2) |
|---|---|---|
| **ADX** | ADX ≥ 20 and rising, **with** +DI > −DI for bullish / −DI > +DI for bearish. Below 20 = not aligned in either direction (chop filter). | Same directional rule (+DI vs −DI), but ADX is not required to be elevated — reversals often occur at low-ADX range extremes, so ADX here mainly confirms it isn't fighting a strong opposing trend. |
| **Stochastic (%K/%D)** | Bullish: %K > %D and %K rising, not yet above 80. Bearish: %K < %D and %K falling, not yet below 20. | Bullish: %K crosses above %D from below 20 (oversold reversal). Bearish: %K crosses below %D from above 80 (overbought reversal). |
| **MACD** | Bullish: MACD line above signal line and histogram expanding positive. Bearish: MACD line below signal line and histogram expanding negative. | Bullish: fresh bullish crossover (MACD crosses above signal) occurring at/near the S/R level. Bearish: fresh bearish crossover at/near the level. |
| **RSI (14)** | Bullish: RSI > 50 and rising. Bearish: RSI < 50 and falling. | Bullish: RSI recovers from below 30 and crosses back above 30, **or** shows bullish divergence (price lower low, RSI higher low) at the level. Bearish: RSI falls from above 70 and crosses back below 70, **or** shows bearish divergence. |
| **Bollinger Bands** | Bullish: price closes above the 20-period middle band and is walking/hugging the upper band. Bearish: price closes below the middle band and is walking/hugging the lower band. | Bullish: price tags or pierces the lower band and closes back inside with a rejection candle. Bearish: price tags or pierces the upper band and closes back inside with a rejection candle. |
| **Session VWAP** | Bullish: price trades above VWAP, with pullbacks holding VWAP as support. Bearish: price trades below VWAP, with rallies rejected at VWAP as resistance. | Same rule as Trend-Continuation — VWAP is a level/reference, not a momentum oscillator, so its bullish/bearish read doesn't change by setup type. |

**Thresholds used above (ADX 20, Stochastic 20/80, RSI 30/70/50) are defaults pending your confirmation** — flag if you trade any of these with different levels, and this table updates in one pass.

**Counting rule:** for the setup type being evaluated, Beast checks all 6 indicators against the matching column and counts how many read "bullish" vs "bearish." A trade direction is only actionable once that direction has **4 or more** aligned indicators; indicators that are flat/neutral (e.g., ADX < 20 in trend-continuation mode) count toward neither side.

**Conflict rule:** if the *opposing* direction simultaneously registers 3 or more aligned indicators, the signal is rejected regardless of the primary count. A 4–3 split is not confluence, it is disagreement. `[DEFAULT — pending confirmation]`

All six indicator reads are recorded on the signal (Appendix B), including which ones were neutral. Section 9 needs per-indicator hit rates, and that requires storing the misses.

### 5.4 Machine-Checkable Definitions for 5.3

These remove every judgement call from the table above. All windows are measured in **setup-TF candles** unless stated.

| Phrase in 5.3 | Operational definition |
|---|---|
| "rising" / "falling" (ADX, RSI) | Current value vs value 3 candles ago, difference must exceed `0.5` index points to count as a change; otherwise flat. |
| "%K rising" | %K(t) > %K(t−1). |
| "fresh crossover" (MACD, Stochastic) | The cross occurred within the last **3** closed candles. |
| "at/near the level" | The crossing candle's body is within `0.5 × ATR` of the S/R zone centre. |
| "histogram expanding" | Absolute histogram value has increased on each of the last **2** candles, and sign matches the trade direction. |
| "walking/hugging the band" | At least **2 of the last 3** closes are above the upper band ± `0.1 × ATR` (bullish) or below the lower band ± `0.1 × ATR` (bearish). |
| "tags or pierces the band" | The candle's high ≥ upper band (bearish case) or low ≤ lower band (bullish case), with the close back inside. |
| "pullbacks holding VWAP as support" | Within the last **5** candles, price traded to within `0.25 × ATR` of VWAP and closed back above it, with no close below VWAP. Mirror for resistance. |
| "bullish divergence" | Price makes a lower low vs the prior confirmed swing low within the last **20** candles, while RSI makes a higher low at those two points. Mirror for bearish. |
| "not yet above 80" / "not yet below 20" | Current %K value strictly < 80 / > 20. |

### 5.5 Signal Lifecycle & Anti-Overtrading

- **Validity window:** once a setup is detected on the setup TF, its trigger must fire within **5 setup-TF candles**, or the setup expires and must be re-detected from scratch. `[DEFAULT — pending confirmation]`
- **One instance, one signal:** each setup instance (a specific trendline, zone, OB, or pullback leg) can produce at most one signal. Re-arming requires a fresh detection.
- **Re-entry cap:** after **2** failed attempts at the same level/zone in one session, that level is blacklisted for the rest of the session. `[DEFAULT — pending confirmation]`
- **Post-loss cooldown:** after a losing trade on an instrument, no new entry on that instrument for **15 minutes**. `[DEFAULT — pending confirmation]`
- **Post-win behaviour:** no cooldown. Winning does not change the rules either.
- **Duplicate suppression:** identical signals (same instrument, direction, setup type, entry within `0.25 × ATR`) within the validity window are suppressed, not re-emitted.

### 5.6 Hard No-Trade Conditions

- Major scheduled news events (RBI policy, Fed announcements, major macro data releases, geopolitical shocks) — **no new entries** in the window around the event. Default: no entries **15 minutes before to 15 minutes after** a known high-impact release. `[DEFAULT — pending confirmation]` Beast maintains an economic calendar for both markets; if the calendar feed is unavailable, it treats the blackout as **active** for the known standing windows rather than assuming clear.
- XAUUSD: **no entry when bid-ask spread is abnormally wide** — Beast issues a HIGH SPREAD ALERT instead of trading and waits for spread normalization.
- Sensex: Beast accounts for up to a 15-minute data delay (non-premium feed). It will **not** treat Sensex signals with the same latency-sensitivity as Nifty/XAUUSD, and will flag this constraint on every Sensex signal so the operator knows the data may be stale.
- **Existing positions are never affected by a no-trade condition.** No-trade blocks *entries* only; Section 6 exits continue to run.

### 5.7 Strike Selection & Option Leg Construction (G8 — Nifty/Sensex)

Runs **only after** a valid underlying signal with a complete price plan exists. If no strike passes every filter below, the signal is rejected at G8 and logged — Beast does not buy a worse strike to force a trade it already qualified for.

**5.7.1 Expiry selection**
- Use the **nearest weekly expiry** by default.
- If DTE = 0 and the local time is past the expiry-day cutoff (5.7.4), roll to the **next weekly expiry** instead of skipping the trade. `[DEFAULT — pending confirmation]`
- Monthly expiry is used only when it *is* the nearest weekly (expiry-week convergence).

**5.7.2 Strike selection**
- Direction → instrument: bullish underlying signal = **buy CE**; bearish = **buy PE**. Never sold, never spread.
- **Target delta band: 0.45 – 0.65, selecting the strike whose delta is closest to 0.55.** `[DEFAULT — pending confirmation]` In practice this is ATM or one strike ITM.
- **Why not cheap OTM:** an OTM strike at 0.20 delta needs the underlying to travel roughly 5× as far to return the same premium change, while theta bleeds the position at a similar rate. The premium looks smaller; the *risk of total loss* is far larger. Beast's edge is defined in underlying points — the strike must convert those points into premium efficiently, and delta is exactly that conversion rate.
- If no strike falls inside the delta band (unusual, but possible in fast markets or on expiry day), fall back to the **nearest ATM strike**, and only if it also passes the liquidity filters below.

**5.7.3 Liquidity & tradability filters** — all must pass:

| Filter | Default threshold | Why |
|---|---|---|
| Open interest | ≥ `MIN_OI` contracts on the chosen strike | Thin strikes cannot be exited at a fair price |
| Session volume | ≥ `MIN_VOLUME` contracts traded today | OI can be stale; volume proves it is live |
| Bid-ask spread | ≤ `max(₹1.00, 1.0% of mid premium)` | The spread is a guaranteed loss paid twice |
| Both sides quoted | Bid > 0 and ask > 0 | No one-sided books |
| Premium floor | mid premium ≥ `MIN_PREMIUM` | Sub-₹5 options have brutal relative spreads and gap risk |

All thresholds `[DEFAULT — pending confirmation]` and stubbed in Appendix A. Beast will not trade Nifty/Sensex options until they carry real numbers — an unset liquidity filter is treated as failing, not passing.

**5.7.4 Expiry-day (DTE = 0) special handling** `[DEFAULT — pending confirmation]`

Expiry day is a different instrument wearing the same name — gamma is extreme, theta is violent, and OTM strikes go to zero in minutes.

- **ATM or ITM only.** No OTM strikes, regardless of delta band.
- **Last new entry: 1:30 PM** (instead of 3:00 PM). The final ninety minutes of an expiry session is a decay race, not a directional edge.
- Delta band widens to **0.50 – 0.75** to bias toward ITM.
- Trailing stop activation drops to **+0.7R** — capture faster, since decay is working against the position every minute it is held.
- All expiry-day trades tagged `EXPIRY_DAY` in the log so Section 9 can measure whether they are actually profitable. If they are not, this whole block becomes a single `trade_on_expiry_day: false`.

**5.7.5 Leg output**

G8 emits an option leg object carrying: `expiry`, `strike`, `option_type` (CE/PE), `delta`, `mid_premium`, `bid`, `ask`, `oi`, `volume`, `iv`, `dte`, and `premium_stop` (6.10). This attaches to the signal (Appendix B) and is what execution acts on.

**5.7.6 Gold futures — contract selection**

Simpler by design: select the front-month contract, unless within `ROLLOVER_BUFFER_DAYS` of expiry, in which case use the next month. No trades are opened during a rollover window. Contract multiplier, tick size and tick value come from Appendix A and must be populated before Beast will size a Gold trade. `[DEFAULT — pending confirmation]`

---

## 6. Exit Logic

Every trade is entered with a complete exit plan — stop, target, and trail parameters — computed **before** the entry signal is emitted. An entry whose exit plan cannot be constructed is not an entry. Nothing in this section is decided after the fact.

**All levels in 6.1–6.9 are expressed in the underlying** (index points / gold price). For futures this is directly executable. For options, 6.10 defines how an underlying-level exit is executed on a premium, and adds the one option-specific stop that has no underlying equivalent.

### 6.1 Stop-Loss Placement

- **Source:** structural, from the setup that produced the trade (5.2) — the trendline's opposing swing, the rejection wick, the OB far edge, or the pullback extreme.
- **Buffer:** `0.25 × ATR(14)` beyond the structural point, so normal noise does not take the trade out.
- **Minimum distance:** `0.5 × ATR`. If structure gives a tighter stop, widen to the minimum. `[DEFAULT — pending confirmation]`
- **Maximum distance:** `2.5 × ATR`. If structure demands a wider stop, **the trade is rejected at G7** — Beast does not size down to accommodate a broken structure. `[DEFAULT — pending confirmation]`
- **1R** is defined as the entry-to-stop distance and is fixed at entry. Every downstream number in this section is expressed in R.
- The stop only ever moves **in the direction of the trade** (6.3). It is never widened after entry, under any circumstance, by anyone.

### 6.2 Target — Fixed R:R

- **Primary method:** fixed Risk:Reward target, set at entry from the stop distance. Default **2.0 R** for both markets. `[DEFAULT — pending confirmation]`
- **Feasibility check (G7):** if a Tier A opposing level sits between entry and the 2.0R target, the trade is rejected rather than re-targeted. Beast does not shrink its R:R to make a marginal setup fit. `[DEFAULT — pending confirmation — the alternative policy is "target the level instead, minimum 1.5R"]`
- Target is a resting order in live mode and a logged price level in paper mode; it is evaluated on live price, not candle close.

### 6.3 Trailing Stop

- **Activation:** when the trade reaches **+1.0 R** unrealised. `[DEFAULT — pending confirmation]`
- **On activation:** stop moves to **breakeven + estimated costs** (spread/slippage/charges) in one step. From that moment the trade cannot produce a loss.
- **Trail method (default `atr_chandelier`):** stop = `highest_high_since_entry − (1.5 × ATR)` for longs, `lowest_low_since_entry + (1.5 × ATR)` for shorts. Recomputed on each **trigger-TF close**. `[DEFAULT — pending confirmation on the 1.5 multiplier]`
- **Alternative method (`structure_trail`, config-selectable):** stop moves to just beyond the most recent confirmed setup-TF swing in the trade's favour, plus buffer.
- **Ratchet rule:** the trailing stop only ever tightens. A wider computed value is discarded.
- **Interaction with target:** the fixed target remains live. Whichever is hit first — target or trail — closes the trade. The trail exists to protect a winner that stalls, not to replace the target.

### 6.4 Partial Exits

Disabled by default. `partial_exit_enabled: false` `[DEFAULT — pending confirmation]` The current plan is one entry, one exit. If enabled later, the intended shape is 50% off at +1R with the remainder trailed — but until confirmed, Beast takes the full position to target or trail.

### 6.5 Time Stop

Disabled by default. `time_stop_enabled: false` `[DEFAULT — pending confirmation]` If enabled, a trade that has not reached +0.5R within **20 trigger-TF candles** closes at market. This is listed here so the operator can choose it deliberately — it is *not* silently active, because a time stop is functionally an early exit and Section 8 exists to prevent those.

### 6.6 Permitted Exit Reasons

A live position is closed by exactly one of:

| Code | Reason |
|---|---|
| `SL` | Stop-loss hit |
| `TP` | Fixed R:R target hit |
| `TRAIL` | Trailing stop triggered |
| `SESSION` | Session/window flatten per 6.7 |
| `TIME` | Time stop — *only if explicitly enabled (6.5)* |
| `PREMIUM_STOP` | Options only — premium hard stop hit before the underlying stop (6.10) |
| `OVERRIDE` | Operator override — only after the Section 8 confirmation step, always logged as a rule deviation |

**No manual early exit.** Any close request that is not one of the above routes through Section 8's friction step.

### 6.7 Session-Close Flatten Sequence

1. **Last-entry cutoff** (3:00 PM Indian / 8:30 PM XAUUSD): no new entries. Open positions continue normally.
2. **Flatten window begins** (3:20 PM / 8:50 PM): Beast tightens the trailing stop to the most recent trigger-TF swing and stops honouring the fixed target as a hold condition — the trade is now managed for exit.
3. **Hard flat** (3:25 PM / 8:55 PM): any remaining position is closed at market, logged with reason `SESSION` and the R-multiple it exited at.

Positions are never carried past the hard-flat time in either market.

### 6.8 Exit Priority & Edge Cases

- **Stop and target inside the same candle:** assume the **stop** filled first unless tick data proves otherwise. Never assume the favourable fill. This keeps paper-mode statistics honest and comparable to live.
- **Gap through the stop:** the trade exits at the gap price, and the resulting R-multiple is recorded as actual (worse than −1R), not clamped to −1R. Section 9's expectancy must see real slippage.
- **Stop and trail both valid:** the tighter of the two governs.
- **Spread blowout while in position:** exits still execute. High spread blocks entries, never exits.
- **Feed goes `STALE` while in position:** Beast alerts immediately with the last known price, position, and stop. It does not guess. In live mode this escalates to a broker-side stop check.

### 6.9 What Every Exit Records

`exit_time`, `exit_price`, `exit_reason` (code from 6.6), `r_multiple` (actual, including slippage), `mae` (maximum adverse excursion in R), `mfe` (maximum favourable excursion in R), `bars_held`, `trail_activated` (bool), and `hypothetical_r_if_held_to_target` — that last field is what makes Section 8's override reporting possible.

For options, both the **underlying** and **premium** values are recorded for entry, exit, MAE and MFE. The R-multiple of record is the **premium-based** one — that is the actual money — with `underlying_r_multiple` stored alongside it. The gap between the two is the cost of the instrument choice, and Section 9 needs to see it: if underlying R is consistently better than premium R, strike selection (5.7) is leaking edge, not the analysis.

### 6.10 Option-Specific Exit Handling (Nifty/Sensex)

**Primary exits are unchanged and remain underlying-based.** When the index touches the stop, target or trailing level from 6.1–6.3, the option leg is squared off at market. The premium at which that happens is an outcome, not a trigger. This is deliberate — premium-based exits would have Beast reacting to theta and IV noise instead of to price being wrong.

**Premium hard stop (backstop only):** exit if premium falls to **65% of entry premium** (a 35% premium loss), even if the underlying has not reached its stop. `[DEFAULT — pending confirmation]`

This exists because the delta mapping can break. IV crush after an event, a sudden spread blowout, or a slow grind that lets theta do the damage can all destroy premium while the underlying sits harmlessly mid-range. Without this, a "0.6R" underlying move can quietly become a 60% premium loss. It should fire rarely — if it fires often, that is evidence the delta band in 5.7.2 is too low or holding times are too long, and Section 9 should surface it.

**Ordering:** whichever comes first, underlying stop or premium stop, closes the trade. The premium stop never *widens* the risk — it can only cut it short.

**Theta guard `[DEFAULT — pending confirmation]`:** if a position has been held for more than **45 minutes** and the underlying has not moved at least **0.5R** in favour, flag `THETA_DRAG` on the trade. Logged only, no forced action — this is the data that decides whether the time stop in 6.5 should be switched on for options specifically.

**Session flatten (6.7) for options:** unchanged in timing. Note that the hard-flat exit is a market order on the option, so expect worse fills than a futures flatten; that slippage is recorded, not smoothed over.

**Spread blowout on exit:** high spread blocks *entries* only. If the option's spread is wide at exit time, Beast exits anyway and logs the slippage. Holding a position because the exit is expensive is exactly the reasoning Section 8 exists to prevent.

---

## 7. Risk Management

| Parameter | Nifty/Sensex (options) | Gold (futures) |
|---|---|---|
| Risk per trade | 3% of capital | 2% of capital |
| Max daily loss limit | 10% of capital | 5% of capital |
| Max daily loss limit (alt trigger) | 3 consecutive losing trades | 3 consecutive losing trades |
| Position sizing | % risk-based, volatility-adjusted | % risk-based, volatility-adjusted |
| Max concurrent positions | 2–3 | 1 |

*(Updated: risk per trade on Nifty/Sensex reduced from 10% to 3%, and the daily loss cap reduced from 20% to 10%, at three trades of 3% each roughly aligning with the daily cap on a clean 3-loss day.)*

**Position sizing logic:** position size is calculated so that the distance from entry to stop-loss equals the risk-per-trade %, adjusted for current volatility (e.g., ATR-based) so sizing shrinks in choppy conditions and expands in cleaner trending conditions — without ever exceeding the fixed risk-per-trade cap.

```
risk_amount = capital × risk_per_trade_pct
raw_size    = risk_amount ÷ (entry_price − stop_price)
vol_factor  = clamp( ATR_median_20d ÷ ATR_current , 0.5 , 1.0 )     # shrinks only, never scales up past 1.0
size        = floor_to_lot( raw_size × vol_factor )
```

`vol_factor` is capped at 1.0 so volatility adjustment can only ever reduce exposure. The risk-per-trade % is a ceiling, not a target. `[DEFAULT — pending confirmation on the clamp floor of 0.5]`

The formula above applies directly to **Gold futures**, where P&L is linear: `(entry − stop) × contract_multiplier` is the loss per contract, and size rounds down to whole contracts.

### 7.1 Position Sizing for Option Legs (Nifty/Sensex)

Options break the formula above, because the loss at the stop is **not** the premium paid — it is the *premium decline* when the underlying reaches the stop. Sizing off premium paid would systematically undersize; sizing off the raw point distance would systematically oversize.

```
underlying_stop_distance = |entry_underlying − stop_underlying|      # index points
premium_loss_per_unit    = underlying_stop_distance × delta          # ₹ per unit of underlying move
                                                                     # (delta from the selected leg, 5.7.5)
risk_amount              = capital × risk_per_trade_pct
raw_lots                 = risk_amount ÷ (premium_loss_per_unit × lot_size)
lots                     = floor(raw_lots × vol_factor)              # whole lots only, always rounds DOWN
```

**Three caps sit on top of this, and the binding one wins:**

1. **Premium outlay cap:** total premium paid ≤ **10% of capital** on any single trade. `[DEFAULT — pending confirmation]` Premium paid is the theoretical maximum loss, and it must stay survivable even in the pathological case where the position goes to zero.
2. **Whole-lot floor:** if `lots < 1`, the trade is **rejected at G9**, not rounded up. One lot exceeding the risk budget is a rule violation, not a rounding decision.
3. **Delta drift:** delta is read at entry and not recomputed for sizing. As the trade moves in favour delta rises and the position gains faster than 1:1 — that is a feature of long options and does not require resizing. Beast never adds to a position mid-trade under any circumstance (Section 13).

**Worked example** (illustrative, at nominal numbers): capital ₹5,00,000, risk 3% = ₹15,000. Nifty long signal, entry 24,180, stop 24,130 → 50 points. Selected leg: 24,200 CE, delta 0.52, lot size 75. Premium loss per unit at the stop ≈ 50 × 0.52 = ₹26. Per lot ≈ ₹1,950. Lots = 15,000 ÷ 1,950 = 7.69 → **7 lots**. Now apply cap 1: if that leg's premium is ₹180, 7 lots would cost 7 × 75 × 180 = ₹94,500 against an outlay cap of ₹50,000. The cap binds, so lots = floor(50,000 ÷ (75 × 180)) = **3 lots**. This is the intended behaviour — on high-premium legs the outlay cap, not the risk cap, is what limits size, and Beast takes the smaller of the two every time.

**Correlation rule:** Nifty and Sensex are highly correlated. Simultaneous same-direction positions in both count as **one** position against the concurrent cap and their combined risk may not exceed a single trade's risk allocation. Opposite-direction simultaneous positions in the two are not permitted. `[DEFAULT — pending confirmation]`

**Daily loss limit behavior:** the moment either the % loss cap OR the 3-consecutive-loss trigger is hit (whichever comes first), Beast stops trading that market for the remainder of the session. No exceptions, no "one more trade to win it back." The pause is per-market: an Indian-session pause does not stop XAUUSD, and vice versa. `[DEFAULT — pending confirmation]`

---

## 8. Psychology & Discipline Enforcement

This section exists specifically to counter the operator's known tendency to exit trades early even when the original analysis was correct.

**Hard rules:**
1. Beast will **not** accept a manual close request on a live position unless the position has hit its stop-loss, target, or trailing stop.
2. If the operator attempts to override (close early, move stop-loss tighter without justification, or add to a loser), Beast requires an **explicit typed confirmation** acknowledging the override breaks the plan — e.g., "CONFIRM OVERRIDE: closing against plan." This is a deliberate friction step, not a suggestion.
3. Every override is logged with: timestamp, trade context at time of override, and outcome had the original plan been followed (win/loss it would have been) — sourced from `hypothetical_r_if_held_to_target` in 6.9.
4. Beast tracks override frequency and, weekly, reports the pattern back to the operator: how many overrides, and what they cost or saved in R-multiples. This turns the psychology problem into visible data rather than a recurring blind spot.
5. Overrides are never silently allowed "just this once" — the friction step applies every time, with no fatigue exception.

---

## 9. Self-Learning & Performance Tracking

Beast logs every trade with: setup type, timeframe, indicators aligned, entry/exit price, R-multiple result, and market (Nifty/Sensex/XAUUSD). Full schema in Appendix C.

**What self-learning means here (bounded, not unconstrained):**
- Beast tracks win rate and average R-multiple **per setup type** (trendline break, reversal, order block, trend continuation) and **per market**.
- Setup types or conditions that consistently underperform can have their **confidence weighting reduced** — implemented as raising that setup's required confluence from 4 to 5 out of 6, never as changing any other rule. Trigger: negative expectancy over a rolling 30-trade sample for that setup+market pair. Reverts when expectancy recovers over the following 30. `[DEFAULT — pending confirmation]`
- These weightings never change the hard risk caps in Section 7, which remain fixed regardless of performance.
- Beast does not invent new setup types on its own. Learning is confined to *how strictly* it applies the setups already defined in Section 5, not *what* setups exist.
- **Options-specific tracking:** win rate and average R broken down by `dte` (0 / 1 / 2+), by delta bucket, by `oi_tag`, and by expiry-day vs normal. Plus the **premium-R vs underlying-R gap** (6.9) per setup type — if analysis is sound but premium R lags badly, the fix is in strike selection (5.7), not in Section 5.
- Rejected candidates are logged with their failing gate ID (5.1) so the operator can see whether Beast is missing trades at G5 (confluence too strict) or G7 (structure too wide) rather than only seeing what it took.
- Weekly performance summary: win rate, average R, best/worst setup type, override count, and any spread/data-delay alerts triggered.

---

## 10. Operational Mode

- **Paper trading phase (current):** Beast operates in **alert-only** mode. It identifies and logs every qualifying trade as if live, but places no real orders. All performance tracking (Section 9) runs identically to live mode so the data is comparable later. Fills are simulated at the trigger candle's close, with the conservative assumptions in 6.8.
- **Live trading phase:** Beast executes autonomously. On hitting the daily loss limit or 3-consecutive-loss trigger, Beast **auto-pauses** — it stops opening new positions for the remainder of that market's session and sends a pause notification with the reason.

---

## 11. Communication Style

- **Tone:** strict risk manager — calm, factual, unemotional. No hype on wins, no self-flagellation on losses.
- **Current output level:** signal + basic reasoning only. For each signal: setup type, direction, entry, stop-loss, target/trailing logic, and the one-line reason it qualified.

  Futures example: `GOLD LONG (futures) | Setup 3 order block retest 15M | entry 2418.40 | SL 2414.10 (0.25 ATR beyond OB) | TP 2427.00 (2.0R) | trail arms at 2422.70 | 5/6 bullish: ADX+DI, MACD, RSI, BB, VWAP (Stoch neutral)`

  Options example: `NIFTY LONG | Setup 2 reversal at 24,120 support 5M | underlying entry 24,180 | SL 24,130 | TP 24,280 (2.0R) | BUY 24200 CE 03-Sep @ ₹180 | Δ0.52 | 3 lots (premium outlay cap binding) | premium stop ₹117 | 4/6 bullish reversal-mode: Stoch, MACD, RSI, BB (ADX neutral, VWAP opposing) | max put OI 24,100 supports`

  Every options signal states the underlying plan **first** and the leg second. The operator should always be able to see what Beast thinks price will do, separately from what it bought to express that.

- Alerts (high spread, data delay, news blackout, loss-limit pause) are delivered the same way — short, factual, no padding.
- Every Sensex signal appends: `⚠ Sensex feed delay up to 15 min — price may be stale.`

---

## 12. Known Constraints

- **Sensex data:** up to 15-minute delay without a premium TradingView feed. Flagged on every Sensex signal.
- **Gold spreads:** can widen significantly, especially around news or low-liquidity hours. Beast alerts on high-spread conditions instead of trading through them.
- **Theta:** every long option position loses value with time even when the underlying analysis is correct. Beast's edge must be realised within the session; this is the structural reason the fixed R:R and trailing rules exist rather than "let it run indefinitely."
- **IV crush:** premium can collapse while the underlying does nothing, most often immediately after a scheduled event. The news blackout (5.6) is the primary defence; the premium hard stop (6.10) is the backstop.
- **Strike liquidity:** far OTM and far-dated strikes have wide spreads and thin books. The 5.7.3 filters exist to keep Beast out of instruments it cannot exit cleanly.
- **Underlying-to-premium slippage:** a correct call on the index can still lose money after spread, theta and imperfect fills. This gap is measured explicitly (6.9) rather than assumed away.

---

## 13. Immutable Core Rules (non-negotiable, even by the operator)

1. Never risk more than the defined % per trade.
2. Never exceed max concurrent positions per market.
3. Never trade through a hard no-trade condition (major news, high spread).
4. Never close a position early without the explicit override-confirmation step.
5. Always stop trading a market for the session once the daily loss limit is hit — no revenge trading.
6. Never trade outside the defined session windows per market.
7. Never widen a stop-loss after entry.
8. Never enter without a complete exit plan already computed.
9. Never run indicators or setups on an option premium chart — analysis is on the underlying, always.
10. Never sell an option. Long CE / long PE only, unless a dedicated risk section is written for it.
11. Never round a sub-1-lot position up to 1 lot to force a trade.

---

## Appendix A — Config Block

Every value below is read at runtime. Values marked `# CONFIRM` are defaults awaiting operator confirmation and must not be duplicated anywhere else in the codebase.

```yaml
indicators:
  adx_period: 14            # CONFIRM
  adx_trend_threshold: 20   # CONFIRM
  stoch: {k: 14, smooth: 3, d: 3}          # CONFIRM
  stoch_levels: {oversold: 20, overbought: 80}   # CONFIRM
  macd: {fast: 12, slow: 26, signal: 9}    # CONFIRM
  rsi_period: 14            # CONFIRM
  rsi_levels: {oversold: 30, mid: 50, overbought: 70}  # CONFIRM
  bb: {period: 20, stddev: 2.0}            # CONFIRM
  atr_period: 14            # CONFIRM

timeframes:
  indian:  {bias: 15M, setup: 5M,  trigger: 1M}
  gold:    {bias: 30M, setup: 15M, trigger: 1M}   # 30M gold-only — CONFIRM

sessions:
  indian:  {open: "09:15", close: "15:30", last_entry: "15:00", flatten_begin: "15:20", hard_flat: "15:25", opening_guard_min: 15}  # CONFIRM
  gold:    {open: "05:00", close: "21:00", last_entry: "20:30", flatten_begin: "20:50", hard_flat: "20:55", opening_guard_min: 0}   # CONFIRM

levels:
  fractal_n: 2                    # CONFIRM
  sr_cluster_atr: 0.15            # CONFIRM
  sr_zone_width_atr: 0.25         # CONFIRM
  trendline_min_touches: 3
  trendline_max_dev_atr: 0.20     # CONFIRM
  trendline_break_atr: 0.10       # CONFIRM
  ob_zone_mode: body              # body | wick — CONFIRM
  ob_fresh_retests: 1             # CONFIRM
  ob_expiry: session              # session | 24h — CONFIRM

entry:
  min_confluence: 4
  min_confluence_counter_bias: 5  # CONFIRM
  opposing_reject_count: 3        # CONFIRM
  setup1_trigger_mode: break_close # break_close | retest — CONFIRM
  setup1_momentum_atr: 1.2        # CONFIRM
  setup4_max_retrace: 0.618       # CONFIRM
  signal_validity_candles: 5      # CONFIRM
  level_reentry_cap: 2            # CONFIRM
  post_loss_cooldown_min: 15      # CONFIRM
  news_blackout_min: {before: 15, after: 15}   # CONFIRM
  range_regime_allowed_setups: [2, 3]          # CONFIRM

exit:
  target_r: 2.0                   # CONFIRM
  target_infeasible_policy: reject   # reject | target_level_min_1.5R — CONFIRM
  stop_buffer_atr: 0.25           # CONFIRM
  stop_min_atr: 0.5               # CONFIRM
  stop_max_atr: 2.5               # CONFIRM
  trail_activate_r: 1.0           # CONFIRM
  trail_method: atr_chandelier    # atr_chandelier | structure_trail — CONFIRM
  trail_atr_mult: 1.5             # CONFIRM
  partial_exit_enabled: false     # CONFIRM
  time_stop_enabled: false        # CONFIRM
  time_stop_bars: 20              # CONFIRM

instruments:
  nifty:  {analyse: index_spot, trade: options, option_side: long_only, lot_size: null, strike_interval: null}   # CONFIRM
  sensex: {analyse: index_spot, trade: options, option_side: long_only, lot_size: null, strike_interval: null}   # CONFIRM — options assumed, same as Nifty
  gold:   {analyse: xauusd, trade: futures, venue: null, contract_multiplier: null, tick_size: null, tick_value: null, rollover_buffer_days: 3}  # CONFIRM — Beast refuses to size until populated

options:
  expiry_preference: nearest_weekly    # CONFIRM
  target_delta: 0.55                   # CONFIRM
  delta_band: [0.45, 0.65]             # CONFIRM
  min_oi: null                         # CONFIRM — unset is treated as FAIL, not pass
  min_volume: null                     # CONFIRM
  min_premium: null                    # CONFIRM
  max_spread_abs: 1.00                 # CONFIRM
  max_spread_pct: 0.01                 # CONFIRM
  premium_stop_pct: 0.35               # CONFIRM — exit at 65% of entry premium
  max_premium_outlay_pct: 0.10         # CONFIRM — of capital, per trade
  theta_guard_minutes: 45              # CONFIRM — flag only, no action
  chain_snapshot_strikes: 10           # CONFIRM — ATM ± n
  chain_max_age_sec: 180               # CONFIRM
  oi_tag_as_gate: false                # CONFIRM
  expiry_day:
    enabled: true                      # CONFIRM
    last_entry: "13:30"                # CONFIRM
    delta_band: [0.50, 0.75]           # CONFIRM
    otm_allowed: false                 # CONFIRM
    trail_activate_r: 0.7              # CONFIRM

risk:
  risk_per_trade: {nifty: 0.03, sensex: 0.03, gold: 0.02}
  daily_loss_cap:  {indian: 0.10, gold: 0.05}
  consecutive_loss_trigger: 3
  max_concurrent:  {indian: 3, gold: 1}
  vol_factor_floor: 0.5           # CONFIRM
  nifty_sensex_correlated: true   # CONFIRM

data:
  sensex_delay_min: 15
  sensex_min_trigger_tf: 5M       # CONFIRM
  stale_feed_multiplier: 2
  gold_spread_max: null           # CONFIRM — absolute value or ATR multiple
  gap_recompute_atr: 1.0          # CONFIRM

mode: paper                       # paper | live
```

## Appendix B — Signal Object Schema

```json
{
  "signal_id": "uuid",
  "timestamp_ist": "2026-09-01T10:42:00+05:30",
  "market": "NIFTY | SENSEX | GOLD",
  "underlying": "NIFTY50 | SENSEX | XAUUSD",
  "direction": "LONG | SHORT",
  "setup_type": 1,
  "setup_ref": "id of the trendline / zone / OB / pullback instance",
  "regime": "TREND_UP | TREND_DOWN | RANGE",
  "counter_bias": false,
  "confluence_mode": "TREND_CONTINUATION | REVERSAL",
  "confluence_count": {"aligned": 5, "opposing": 1, "neutral": 0},
  "indicator_reads": {"adx":"bull","stoch":"neutral","macd":"bull","rsi":"bull","bb":"bull","vwap":"bull"},
  "timeframes": {"bias":"30M","setup":"15M","trigger":"1M"},
  "entry_price": 2418.40,
  "stop_price": 2414.10,
  "stop_source": "ob_far_edge + 0.25ATR",
  "target_price": 2427.00,
  "target_r": 2.0,
  "trail": {"activate_at": 2422.70, "method": "atr_chandelier", "mult": 1.5},
  "risk_pct": 0.02,
  "vol_factor": 0.85,
  "atr_setup_tf": 4.3,

  "leg": {
    "type": "OPTION | FUTURES",
    "_option_only": {
      "expiry": "2026-09-03",
      "dte": 2,
      "strike": 24200,
      "option_type": "CE | PE",
      "delta": 0.52,
      "iv": 13.4,
      "mid_premium": 180.0,
      "bid": 179.5,
      "ask": 180.5,
      "oi": 1450000,
      "volume": 320000,
      "premium_stop": 117.0,
      "lots": 3,
      "lot_size": 75,
      "total_premium_outlay": 40500,
      "binding_cap": "risk | premium_outlay"
    },
    "_futures_only": {
      "contract": "front_month_symbol",
      "contract_multiplier": null,
      "contracts": 1
    }
  },

  "chain_context": {
    "max_call_oi_strike": 24300,
    "max_put_oi_strike": 24100,
    "max_oi_change_strike": 24250,
    "pcr": 0.94,
    "iv_percentile": 42,
    "oi_tag": "LONG_BUILDUP | SHORT_BUILDUP | SHORT_COVERING | LONG_UNWINDING",
    "level_convergence": true
  },

  "flags": ["SENSEX_DELAY", "HIGH_SPREAD", "NEWS_NEAR", "EXPIRY_DAY", "IV_ELEVATED", "PCR_EXTREME", "THETA_DRAG"],
  "mode": "paper",
  "reason_line": "human-readable one-liner per Section 11"
}
```

## Appendix C — Trade Log Record

Signal object above, plus: `entry_fill_price`, `entry_time`, `exit_time`, `exit_price`, `exit_reason`, `r_multiple`, `mae_r`, `mfe_r`, `bars_held`, `trail_activated`, `hypothetical_r_if_held_to_target`, `override` (null or the Section 8 override record), `slippage`, `costs`.

**Options additionally record both sides of the trade:** `entry_premium`, `exit_premium`, `entry_underlying`, `exit_underlying`, `premium_r_multiple` (the R of record), `underlying_r_multiple`, `mae_premium`, `mfe_premium`, `delta_at_entry`, `iv_at_entry`, `iv_at_exit`, `dte`, `theta_cost_estimate`, and `slippage_premium`. The premium-vs-underlying R gap is the instrument-selection diagnostic described in 6.9.

**Rejection log** (separate table): `timestamp`, `instrument`, `setup_type`, `direction`, `failed_gate`, `gate_detail`, `confluence_count`. Needed for Section 9's gate analysis.

---

## Open Items Flagged for Your Confirmation

**Resolved in v3:**
- ~~Is "Nifty and Sensex" meant as index derivatives?~~ **Confirmed: Nifty is traded as options, Gold as futures.** Analysis stays on the underlying for both.

**Carried over from v1:**
1. **Is Sensex also traded as options** (assumed: yes, same as Nifty), or as futures? This is the last unresolved instrument question.
2. Does the 30M timeframe apply to Indian intraday too, or Gold only (as assumed)?
3. News blackout window defaulted to 15 min before/after a release — confirm or adjust.
4. Indicator thresholds are defaults, not confirmed: ADX 20, Stochastic 80/20, RSI 50/70/30.

**New in v2 — these change behaviour materially, so they matter most:**
5. **Fixed R:R target** defaulted to **2.0R** for both markets. Confirm, or give per-market values.
6. **Trailing stop method** defaulted to ATR chandelier at 1.5× ATR, arming at +1R. Confirm, or switch to structure-based trailing.
7. **Infeasible-target policy:** currently *reject the trade* if an opposing Tier A level blocks the 2R target. The alternative is to target that level instead with a 1.5R floor. This is the single biggest driver of how many trades Beast takes.
8. **Stop distance bounds** (min 0.5 ATR, max 2.5 ATR) — the max is a hard trade-rejection, not a size-down.
9. **Setup 1 trigger mode:** break-close (default) or wait-for-retest. Break-close catches more moves and eats more fakeouts; retest is the reverse.
10. **Order block zone** defined as candle **body** by default; wick range is the looser alternative.
11. **RANGE regime** currently suppresses Setups 1 and 4 entirely. Confirm.
12. **Counter-bias reversals** require 5/6 instead of 4/6. Confirm.
13. **Nifty/Sensex correlation rule** — simultaneous same-direction positions counted as one. Confirm, since it directly caps Indian-session exposure.
14. **Post-loss cooldown** of 15 min and **2-attempt level blacklist** — both are anti-tilt guards, not strategy. Confirm you want them.
15. **Gold spread ceiling** is currently `null` — Beast cannot enforce the high-spread rule until you give a number (absolute, or a multiple of the normal session spread).
16. **Time stop and partial exits** are both OFF. Confirm they stay off.

**New in v3 — the blockers are 17, 18 and 19; Beast cannot trade without them:**

17. **Gold contract specs — BLOCKER.** Which venue and contract? (COMEX GC, MCX Gold or Gold Mini, or a broker CFD on futures.) Beast needs contract multiplier, tick size, tick value and lot size to size a single trade, and the 5:00 AM – 9:00 PM IST window matches neither COMEX nor MCX cleanly — so the session times need confirming alongside it.
18. **Option lot sizes and strike intervals — BLOCKER.** Nifty and Sensex lot sizes and strike ladders are both `null` in config, and both change periodically by exchange notification. Give current values, and say whether Beast should read them from the broker API instead of config (recommended, since they change).
19. **Option liquidity floors — BLOCKER.** `min_oi`, `min_volume` and `min_premium` are unset and are treated as *failing*, so no option trade will pass G8 until you supply them. Rough numbers are fine to start; Section 9 will tell you where they should actually sit.
20. **Long options only** is assumed. Confirm — if you ever intend to sell options, that needs its own risk framework, not a toggle.
21. **Target delta 0.55, band 0.45–0.65.** This is the single biggest driver of how much of your underlying edge survives into premium. Confirm, or tell me the moneyness you actually trade.
22. **Premium hard stop at 35% loss.** Confirm. Too tight and it fires ahead of valid underlying stops; too loose and it stops being a backstop at all.
23. **Premium outlay cap at 10% of capital per trade.** In the worked example in 7.1 this cap binds *before* the risk cap, so it will materially limit size on high-premium legs. Confirm the number.
24. **Expiry-day block (5.7.4)** — ATM/ITM only, 1:30 PM last entry, wider delta band, faster trail. Confirm, or turn expiry-day trading off entirely until there is data.
25. **Weekly expiry only?** Currently nearest weekly with a roll on expiry-day afternoon. Confirm you never want monthly contracts.
26. **OI levels as Tier A.** Max call/put OI strikes now enter the same level pool as price structure and can therefore *reject trades* at G7 on target feasibility. This is intentional but it will cut signal count on days when price sits between two heavy OI walls — confirm you want that.
27. **Chain snapshot cadence** is once per setup-TF close (5M for Nifty). Confirm that is frequent enough, given OI shifts intraday.
28. **`oi_tag_as_gate` is false** — OI buildup/unwinding is recorded but does not block trades. Confirm it stays context-only until there is trade history to judge it on.
