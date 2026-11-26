"""Analytics engine: KPIs, Pareto, RFM, cohorts/LTV and time-of-day profiles.

Every function takes a *transactions frame* (see :mod:`bizpulse.analytics.data`) and is
pure: no I/O and no wall-clock time. The as-of date always comes from the data.
"""
