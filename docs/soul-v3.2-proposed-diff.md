# Proposed soul file v3.2 — amendments for operator approval

**Status: UNAPPROVED.** Nothing here has been applied to
`beast-soul-file-v3.1.md`. Until it is, `core/regime/` computes and logs and
changes no sizing decision, and gate G10 does not exist in the chain.

Two groups of amendments:

- **Group A** records the operator's path-B resolution of Conflict 1 and the
  resolution of Conflict 2. These close the soul file's own blockers.
- **Group B** is what the volatility-regime layer needs before it may influence
  a trade.

Group B exists because the regime layer changes position size, and position size
is Section 7 behaviour. A sizing input that lives only in code, while the soul
file is declared the single source of truth and Immutable Rule 1 is enforced
against Section 7's numbers, breaks the governance model that makes Beast worth
building. It is therefore proposed as a document change for approval, not
smuggled in as an implementation detail.

---

## Group A — closing the v3.1 blockers

### A1. Instruction-block item 0 — revert to underlying analysis

The operator has chosen **path B** from "Two ways forward". Replace item 0 with:

> **0. Instrument architecture (Sections 3.1, 4.7, 5.7).** All analysis runs on
> the underlying index/spot — never on the option premium chart. Signals,
> levels, indicators, stops and targets are computed in underlying points. The
> traded instrument is constructed as a separate step: a long option leg for
> Nifty/Sensex, a futures contract for Gold. Section 4.7's option-chain layer is
> **widened**: open interest, implied volatility, delta and liquidity are
> first-class chain-derived inputs to the underlying-based engine — OI levels
> feed the Tier A level pool, IV state and delta inform strike selection and
> the premium backstop — but the chain remains a context and level source, and
> never enters the 4-of-6 confluence count. Keep the analysis layer and the
> execution layer as separate modules.

**Consequences of A1:**

| Location | Change |
|---|---|
| Section 13, Rule 9 | **No change.** "Never run indicators or setups on an option premium chart" stands, unedited. |
| Section 7.1 | **No change.** Delta-based sizing from `underlying_stop_distance x delta` stands. Premium-native sizing is not needed. |
| Sections 3, 3.1, 4, 4.7, 5.1, 5.7, 6, 6.9, 6.10, 12, Appendices A/B | **No change.** All fifteen passages listed under Conflict 1 were already consistent with path B; item 0 was the outlier. |
| "Unresolved Conflicts in v3.1" | Mark Conflict 1 **RESOLVED — path B**, dated, with the operator named. |
| Open item 29 | Mark resolved. |

The document's stated reason for Rule 9 is preserved by this choice: an option's
premium series falls under theta decay even when the index is flat, which biases
every momentum and mean-reversion read taken on it.

### A2. Section 10 — replace the alert-only paragraph

Conflict 2. Instruction item 6 and `paper_full_execution: true` are already
operative; Section 10's first paragraph still says "Beast operates in alert-only
mode." Replace it with:

> Beast operates in **paper trading with complete order execution**. Every
> qualifying signal is executed as a complete simulated trade — entry, stop,
> target, trailing stop and exit — and carried through to a recorded exit
> against one of the codes in 6.6. Order state (working, filled, partially
> managed, closed) is tracked exactly as it will be in live mode, fills are
> simulated at the trigger candle's close under the conservative assumptions in
> 6.8, and Section 9 performance tracking runs identically to live so the data
> is comparable later. No real broker order is sent while `mode: paper`. The
> only difference between paper and live is that the broker call is stubbed.

Mark Conflict 2 **RESOLVED** and open item 30 closed.

---

## Group B — the volatility-regime layer

### B1. New Section 7.2 — Volatility-regime size modifier

> **7.2 Volatility-regime size modifier**
>
> Beast maintains a per-market volatility state, `vol_state`, taking one of
> `CALM | NORMAL | TURBULENT | UNKNOWN`. It is produced by a hidden Markov model
> fitted to the underlying series on the bias timeframe and is **not** a regime
> classifier: Section 4.4's `regime` (`TREND_UP | TREND_DOWN | RANGE`) is
> unaffected by it and remains the sole authority over which setups are
> permitted.
>
> `vol_state` may only ever **reduce** position size. It carries a
> `size_multiplier` in `(0, 1.0]`, combined with this section's `vol_factor` as:
>
> ```
> effective_factor = min(vol_factor, size_multiplier)
> effective_factor = max(effective_factor, risk.vol_factor_floor)
> ```
>
> The combination is `min`, never a product. The two quantities measure largely
> the same thing, so multiplying them double-counts volatility, and the
> product's floor would be `0.5 x 0.5 = 0.25` — silently overriding
> `risk.vol_factor_floor`, which this section sets. There is one floor, and it
> is `risk.vol_factor_floor`.
>
> The volatility layer cannot permit a trade the G0–G9 chain rejected, cannot
> select or alter a setup, direction, entry, stop, target, trail or strike, and
> has no authority over an open position. Once a position exists, Section 6
> governs it alone: a flip to `TURBULENT` mid-trade does not close the position
> — the stop does.
>
> On model failure, stale features or an inference error, Beast holds the last
> confirmed state for a bounded number of bars and then emits `UNKNOWN` at the
> uncertainty multiplier with a WARNING alert. A model fault reduces size; it
> never halts trading and never enlarges a position.

### B2. Section 5.1 table — add gate G10

Add one row after G9, before Emit:

| Gate | Name | Passes when |
|---|---|---|
| G10 | `VOL_STATE` | The confirmed volatility state does not veto the proposed direction. |

G10 is evaluated **last** on purpose. A candidate that already failed G3 must be
logged as a G3 rejection, or Section 9's "where signals die" analytics stop
meaning anything. The `size_multiplier` is consumed separately, **inside G9**'s
`size_position()`; it is an input to sizing, not a gate.

The chain remains ten gates plus one: G0–G9 as written, G10 appended.

### B3. Appendix A — two additions

Add `capital` under `risk:`, and the whole `regime:` block.

```yaml
risk:
  capital: 500000                 # NEW - every formula in 7 and 7.1 is
                                  #   capital x pct, and no capital key existed
```

The `regime:` block as shipped in `config/beast_config.yaml`. Nothing in it
duplicates an existing Appendix A value: the bias timeframe is read from
`timeframes:`, and the sizing floor is `risk.vol_factor_floor`.

### B4. Appendix B — signal object fields

| Field | Type | Meaning |
|---|---|---|
| `vol_state` | str | `CALM \| NORMAL \| TURBULENT \| UNKNOWN` |
| `vol_state_confirmed` | bool | Survived the N-bar persistence check |
| `vol_state_probability` | float | Filtered posterior for the reported state |
| `size_multiplier` | float | In `(0, 1.0]` |
| `binding_cap` | str | Which of risk / outlay / whole-lot / vol_state bound the size |
| `vol_state_source` | str | `"underlying"` |
| `model_version` | str | `<feature_hash>@<train_end date>` |

### B5. Appendix C — the same fields on the trade log

So Section 9 can measure whether the layer earned its place. Without them the
question "did shrinking size in TURBULENT help?" is unanswerable, and an
unanswerable layer should not be running.

### B6. Section 13 — no change

If any part of the volatility layer appears to require amending an Immutable
Rule, that is a design error in the layer, not a problem with the rule. Report
it rather than amending Section 13.

---

## What is blocked until this is approved

- Wiring `effective_size_factor` into `RiskManager.size_*` (G9).
- Adding `G10_VOL_STATE` to the gate chain.
- Emitting the Appendix B / C fields.

`core/regime/` is complete and tested without any of the above. It computes,
logs, and changes nothing.
