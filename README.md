# makefun-gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ccxiaoba555/makefun-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ccxiaoba555/makefun-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase a week after launch (2026-11-26 00:00 UTC), as five quantiles per trophy, from the house's recording of the trophy page up to the deadline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every `<id>.in` case in the input directory and writes `<id>.out` with `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## How it forecasts

All modelling is in logit space. Per trophy:

1. **Distinct snapshots** — consecutive reads with the same rate and players-tracked carry no new information (Exophase refreshes a game's counts on its own schedule, hours apart), so only the last read of each constant run is kept.
2. **Two median models on the last 30 hours**, averaged in logit space:
   - a weighted least-squares line of logit(rate) on `ln(1 + hours since release)`, weights ~ `n·p·(1−p)`, slope shrunk toward a prior set by the trophy's kind (story chapters rise fastest; platinum / 100% completion / grinds slowest);
   - a saturating curve `rate(t) = S·(1 − exp(−t/τ))`, which is how a trophy's earn rate actually behaves: fast early growth among the launch rush, then flattening as the remaining tracked players are casuals and late buyers while new tracked players dilute the rate.
3. **Calibrated spread** — a random-walk term (`0.5·√horizon in days`) plus fit-residual and refresh uncertainty. For short horizons the lower tail is widened: a refresh landing inside the window can only pull the answer down (launch night's refresh cut most rates to about a fifth when tracked players jumped 638 → 3,868).

GTA VI released 2026-11-19; there is no official trophy list beyond what Exophase publishes, and every forecast here is a statistical prediction from the recorded rates, not an official figure.

---

<sub>Entered on [make.fun](https://make.fun/?ref=102db11r&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
