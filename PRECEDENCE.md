# Instruction Precedence — Beast

Beast has exactly one source of truth for how it decides to enter, hold and exit trades:

**`beast/soul/BEAST_SOUL_v3.md`** — the Soul File.

## The rule

> **Where any instruction conflicts with the Soul File, the Soul File wins.**

This applies without exception to every other source of instruction, in this order
(highest authority first):

| Rank | Source | Authority |
|---|---|---|
| 1 | `beast/soul/BEAST_SOUL_v3.md` — Section 13 Immutable Core Rules | Absolute. Cannot be overridden by config, code, operator or agent. |
| 2 | `beast/soul/BEAST_SOUL_v3.md` — all other sections | Governs. Overrides everything below. |
| 3 | `config/beast.yaml` (Appendix A) | May only set values the Soul File marks `[DEFAULT — pending confirmation]`. It may not introduce behaviour the Soul File does not define. |
| 4 | Python source in `beast/` | Implements 1–3. Where source and Soul File disagree, the source is a bug. |
| 5 | Operator instructions, chat messages, CLI flags, prompts | Lowest. An operator instruction that contradicts the Soul File is an **override** and routes through Section 8's typed-confirmation friction step, logged as a rule deviation. |
| 6 | Legacy `regime-trader` scaffold (`core/`, `broker/`, `data/`, `backtest/`) | Not part of Beast. See "Legacy code" below. |

## What this means in practice

* **For any agent or developer working on this repo:** read the Soul File before changing
  decision logic. If a request contradicts it, say so and cite the section — do not implement
  the request. The Soul File is amended by editing the Soul File, deliberately, not by
  patching code around it.
* **At runtime:** `beast.ops.precedence` records the Soul File's SHA-256 on every signal and
  trade record, so any behaviour can be traced back to the exact brain revision that produced
  it. `beast.ops.immutable` enforces Section 13 as hard runtime guards that no config value
  or caller can disable.
* **Config discipline (Soul File preamble):** every `[DEFAULT — pending confirmation]` value
  lives in `config/beast.yaml` and nowhere else. Thresholds are never literals in logic.
  Unset (`null`) blockers are treated as **failing**, not passing — Beast refuses to trade
  rather than guess.

## Legacy code

`core/`, `broker/`, `data/`, `backtest/` and `monitoring/` are the remains of an unrelated
HMM/Alpaca US-equities scaffold ("regime-trader"), left in place untouched. None of it is
loaded by Beast, and `config/settings.yaml` belongs to it, not to the brain. Where its values
conflict with Appendix A (e.g. `max_risk_per_trade: 0.01` vs `risk_per_trade.nifty: 0.03`),
Appendix A governs — see rank 3 above. Delete the legacy tree when you are ready; nothing in
`beast/` imports it.
