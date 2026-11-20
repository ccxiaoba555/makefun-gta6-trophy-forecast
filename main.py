"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every case file <id>.in from the input directory and writes one
forecast <id>.out per case: for each trophy, five quantiles (5/25/50/75/95%)
of its earn rate on Exophase at the case's target, scored by pinball loss
on the logit scale.

Method, per trophy (all in logit space, logit(p) = ln(p / (100 - p))):

1.  Collapse the history into distinct snapshots: consecutive reads with
    the same rate and players-tracked carry no new information (Exophase
    refreshes a game's counts on its own schedule, hours apart, not with
    every read), so only the last read of each constant run is kept.
2.  Fit on the last 30 hours of snapshots (the current regime: early
    launch-night reads are dominated by Exophase's first refresh, a
    one-off dilution when tracked players jumped 638 -> 3,868):
      a. a weighted least-squares line of logit(rate) on ln(1 + hours
         since release), weights ~ n * p * (1 - p) (the binomial
         precision of a logit estimate), slope shrunk toward a prior
         set by the trophy's kind (story chapters rise fastest,
         platinum / 100% completion / grinds slowest);
      b. a saturating curve rate(t) = S * (1 - exp(-t / tau)), tau
         chosen by a grid search, which is how a trophy's earn rate
         actually behaves: fast early growth among the launch rush,
         then flattening as the remaining tracked players are casuals
         and late buyers while new tracked players dilute the rate.
    The median is the mean of the two forecasts in logit space: the
    line extrapolates the recent trend, the curve enforces saturation,
    and averaging hedges the shape neither can prove.
3.  Spread: a random-walk term (0.5 * sqrt(horizon in days), the house
    baseline's width) combined with the fit's residual uncertainty and a
    refresh term. Exophase refreshes are discrete jumps, so for short
    horizons the lower tail is widened: the answer is the page as of its
    last refresh, which can sit well below the trend when a refresh
    lands inside the window (launch night's refresh cut most rates to
    about a fifth). The widening decays with horizon; over the final's
    4.5 days many refreshes happen and the trend absorbs them.

Deterministic: no randomness, no network, standard library only.
Runs in well under the 8-second cap (55 trophies, a few hundred reads).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# GTA VI released 2026-11-19 (UTC); the clock every trend is measured on.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# How far back the trend is fitted (hours): the current regime only.
WINDOW_HOURS = 30.0
# Prior slope (logit per unit ln(1 + hours)) uncertainty for shrinkage.
PRIOR_TAU = 0.6
# Median move allowed away from the last read, in logits.
DELTA_LO = -2.5
DELTA_HI = 3.5
# Spread floors / caps, in logits.
SPREAD_MIN = 0.35
SPREAD_MAX = 2.0
# Saturating-curve grid for tau (hours).
TAU_GRID = [6, 9, 13, 19, 28, 40, 58, 84, 120, 170, 240, 340, 480, 680, 960]


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    if l >= 40:
        return 100.0
    if l <= -40:
        return 0.0
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    """Round to 4 decimals, half up (the house baseline's rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def norm_ppf(p):
    """Acklam's inverse standard normal CDF (stdlib only)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def prior_slope(trophy):
    """Prior logit-slope for a trophy, from what it asks the player to do.

    Story chapters are finished by a large share of the launch crowd
    within days; the platinum and 100% completion need everything;
    grinds (250 stunts, 100 locations, 50 challenges) move slowest.
    """
    desc = (trophy.get("description") or "").lower()
    tier = trophy.get("tier", "")
    if tier == "platinum":
        return 0.25
    if "100% completion" in desc:
        return 0.45
    if "chapter" in desc:
        return 1.30
    if "final mission" in desc:
        return 0.90
    if any(k in desc for k in ("250 stunts", "100 locations", "50 challenges",
                               "collectible card", "meditate at every", "win 20")):
        return 0.40
    if any(k in desc for k in ("bank vault", "earn $", "spend $",
                               "purchase all", "register 10")):
        return 0.55
    if "complete" in desc and any(k in desc for k in ("mission", "strand", "heist",
                                                       "takedown", "operation", "run", "play")):
        return 1.00
    if tier == "gold":
        return 0.50
    if tier == "silver":
        return 0.60
    return 0.80


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, sxx, xbar, resid_sd, se_b)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0, sxx, mx, 0.0, 9.9
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    resid_sd = math.sqrt(max(var, 0.0))
    se_b = math.sqrt(var / sxx) if var > 0 else 0.0
    return a, b, sxx, mx, resid_sd, se_b


def fit_saturating(hours, rates, ws):
    """Fit rate(t) = S * (1 - exp(-t / tau)); tau by grid, S by weighted LS.

    Returns (S, tau) or None when the window is too thin.
    """
    best = None
    for tau in TAU_GRID:
        f = [1.0 - math.exp(-max(t, 0.0) / tau) for t in hours]
        den = sum(w * ff * ff for w, ff in zip(ws, f))
        if den <= 1e-12:
            continue
        s = sum(w * r * ff for w, r, ff in zip(ws, rates, f)) / den
        sse = sum(w * (r - s * ff) ** 2 for w, r, ff in zip(ws, rates, f))
        if best is None or sse < best[0]:
            best = (sse, s, tau)
    if best is None:
        return None
    return best[1], best[2]


def distinct_snapshots(history, index):
    """Collapse consecutive identical (rate, players) reads, keeping the
    last time of each constant run. Returns [(time, rate, players)]."""
    pts = []
    prev_key = None
    for read in history:
        rate = read["rates"][index]
        if rate is None:
            continue
        key = (rate, read["players_tracked"])
        when = parse_ts(read["at"])
        if prev_key is not None and prev_key == key:
            pts[-1] = (when, rate, read["players_tracked"])
        else:
            pts.append((when, rate, read["players_tracked"]))
        prev_key = key
    return pts


def forecast_trophy(case, trophy, index, z_values):
    history = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_ts(case["target"])
    target_hours = max((target - RELEASE).total_seconds() / 3600.0, 0.0)
    x_target = math.log1p(target_hours)
    times = [parse_ts(r["at"]) for r in history]
    t_end = times[-1]
    x_last = math.log1p(max((t_end - RELEASE).total_seconds() / 3600.0, 0.0))

    # Last non-null rate, for the anchor (and the fallback median).
    last_rate = None
    for read in reversed(history):
        if read["rates"][index] is not None:
            last_rate = read["rates"][index]
            break
    if last_rate is None:
        last_rate = 1.0
    y_last = logit(last_rate)

    pts = distinct_snapshots(history, index)
    window = [p for p in pts if (t_end - p[0]).total_seconds() <= WINDOW_HOURS * 3600]
    use = window if len(window) >= 3 else pts

    xs, ys, ws, hours, rates = [], [], [], [], []
    for when, rate, n in use:
        h = max((when - RELEASE).total_seconds() / 3600.0, 0.0)
        p = clamp_pct(rate) / 100.0
        xs.append(math.log1p(h))
        ys.append(logit(rate))
        hours.append(h)
        rates.append(p)
        ws.append(max(n * p * (1.0 - p), 1.0))
    if ws:
        med = sorted(ws)[len(ws) // 2]
        ws = [min(w, 5.0 * med) for w in ws]

    b_prior = prior_slope(trophy)
    s_fit = 0.45
    # Fewer than 3 distinct snapshots: no trend is observable (the page
    # only moves when Exophase refreshes), so hold the last rate rather
    # than letting the prior invent movement.
    m_lin = y_last
    if len(xs) >= 3 and (max(xs) - min(xs)) > 0.05:
        _a, b, sxx, mx, resid_sd, se_b = wls(xs, ys, ws)
        prec = 1.0 / (se_b * se_b + 1e-9)
        prec_p = 1.0 / (PRIOR_TAU * PRIOR_TAU)
        b_shrunk = (b * prec + b_prior * prec_p) / (prec + prec_p)
        m_lin = y_last + b_shrunk * (x_target - x_last)
        leverage = 1.0 + ((x_target - mx) ** 2 / sxx if sxx > 0 else 0.0)
        s_fit = min(1.2, max(0.15, resid_sd * math.sqrt(leverage)))

    sat = fit_saturating(hours, rates, ws) if len(hours) >= 3 else None
    if sat is not None:
        # No evidence of movement in the window (flat reads): the curve
        # fit would invent growth from a flat line, so hold the last rate.
        moved = (max(ys) - min(ys)) if ys else 0.0
        if moved < 0.15:
            m_sat = y_last
        else:
            s_level, tau = sat
            frac = 1.0 - math.exp(-target_hours / tau)
            m_sat = logit(100.0 * min(0.9999, max(0.0001, s_level * frac)))
    else:
        m_sat = y_last

    # Median: the mean of the trend line and the saturating curve, in logits.
    m = 0.5 * (m_lin + m_sat)
    m = min(y_last + DELTA_HI, max(y_last + DELTA_LO, m))
    m = min(logit(99.99), max(logit(0.01), m))

    # Spread: random walk + fit residual + refresh jumps (decays with horizon).
    s_rw = 0.5 * math.sqrt(horizon_days)
    s_ref = 0.30 + 0.55 * math.exp(-horizon_days / 1.5)
    s = math.sqrt(s_rw * s_rw + s_fit * s_fit + s_ref * s_ref)
    s = min(SPREAD_MAX, max(SPREAD_MIN, s))
    # Short horizons: a refresh inside the window can only pull the
    # answer down, so widen the lower tail more than the upper one.
    gamma = 0.45 * math.exp(-horizon_days / 1.0)
    if last_rate >= 99.9:
        # A trophy everyone holds stays held; keep the band tight.
        s = min(s, 0.12)
        gamma = 0.0

    qs = []
    for z in z_values:
        widen = (1.0 + gamma) if z < 0 else (1.0 - gamma * 0.5)
        qs.append(round4(clamp_pct(inv_logit(m + z * s * widen))))
    for k in range(1, len(qs)):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return {"id": trophy["id"], "q": qs}


def forecast_case(case):
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    z_values = [norm_ppf(level) for level in levels]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rows.append(forecast_trophy(case, trophy, i, z_values))
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
