"""Process supervision for an unattended host.

Four pieces, deliberately small and dependency-light so they keep working
when the trading process does not:

* :mod:`ops.heartbeat` - Beast writes ``heartbeat.json`` every cycle.
* :mod:`ops.watchdog` - a separate process that restarts Beast on a stale
  heartbeat, and only then. It never touches an order.
* :mod:`ops.killswitch` - the operator's SSH-side stop: halt entries, or
  flatten through the section 8 friction step.
* :mod:`ops.preflight` - refuses to start a box that is not fit to trade.

Exit codes ``main.py`` returns, so systemd and the watchdog can tell the cases
apart (D-68):

======  ====================================  ========
code    meaning                               restart?
======  ====================================  ========
0       clean shutdown                        no
1       config or startup failure             no - page
2       deliberate halt (error streak,
        circuit breaker)                      no - page
3       KILL flag                             no
>=10    unexpected crash                      yes
======  ====================================  ========
"""

EXIT_CLEAN = 0
EXIT_STARTUP_FAILURE = 1
EXIT_DELIBERATE_HALT = 2
EXIT_KILL = 3
EXIT_CRASH = 10

NO_RESTART_CODES = frozenset({EXIT_CLEAN, EXIT_STARTUP_FAILURE, EXIT_DELIBERATE_HALT, EXIT_KILL})
