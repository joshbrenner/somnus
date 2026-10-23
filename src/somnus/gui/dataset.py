"""Building a fine-tuning set: what it still needs, and which clips to show next.

The user labels short clips, played one after another in the scorer. The
first round spreads them evenly over the model's Wake, NREM and REM calls and
over the recordings; later rounds go after whatever the labels still lack.
Every count here is made on the user's own labels, never on the model's.
None of it touches Qt.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from somnus.gui import core

CLIP_EPOCHS = 15                   # 1 min
FIRST_ROUND_CLIPS = 15             # 15 min
NEXT_ROUND_CLIPS = 5               # 5 min
MIN_STATE_EPOCHS = 20
MIN_REM_BOUTS = 3
MIN_NOT_CORRECTED = 0.25
REM_KEYS = {"REM", "REM bouts"}


@dataclass
class Need:
    """One requirement of the set, and how far the labels go toward it."""
    key: str
    have: float
    target: float
    share: bool = False

    @property
    def met(self) -> bool:
        return self.have >= self.target

    @property
    def counted_only(self) -> bool:
        """Reported, but with nothing to reach."""
        return not self.target

    def text(self) -> str:
        if self.share:
            return f"{self.key}: {self.have:.0%} (need {self.target:.0%})"
        return f"{self.key}: {int(self.have)} of {int(self.target)}"


def needs(project: core.Project, names: list[str]) -> list[Need]:
    """What the manual labels across these recordings hold, against what fine-tuning needs."""
    counts = dict.fromkeys(core.STATES, 0)
    bouts = dict.fromkeys(core.STATES, 0)
    covered = corrected = total = 0
    for name in names:
        st = core.LabelStore(project, name)
        m = st.df[st.manual_mask()].sort_values("epoch")
        if not len(m):
            continue
        covered += 1
        ep, s = m["epoch"].to_numpy(int), m["state"].to_numpy()
        # one unlabelled (often mixed) epoch inside a bout does not split it
        adj = np.diff(ep) <= 2
        for k in core.STATES:
            x = s == k
            counts[k] += int(x.sum())
            bouts[k] += int(x[0]) + int(np.sum(x[1:] & ~(x[:-1] & adj)))
        corrected += int((m["source"] == core.SRC_CORRECTED).sum())
        total += len(m)
    return ([Need("Recordings labelled", covered, len(names))]
            + [Need(k, counts[k], MIN_STATE_EPOCHS) for k in core.STATES]
            + [Need(f"{k} bouts", bouts[k], MIN_REM_BOUTS if k == "REM" else 0)
               for k in core.STATES]
            + [Need("Confirmed or imported", 1 - corrected / total if total else 0.0,
                    MIN_NOT_CORRECTED, share=True)])


class _Recording:
    """One recording's model output, and which clip starts are still free."""

    def __init__(self, project: core.Project, name: str):
        self.name = name
        self.labels, _, self.proba = core.load_model_output(project, name)
        self.n = len(self.labels)
        st = core.LabelStore(project, name)
        reviewed = st.df.loc[st.protected_mask(), "epoch"].to_numpy(int)
        self.used = np.zeros(self.n, dtype=bool)
        self.used[reviewed[(reviewed >= 0) & (reviewed < self.n)]] = True
        self.theta_delta = None
        r = project.get(name)
        cache = core.feature_cache(project, r) if r is not None else None
        if cache and os.path.exists(cache):
            f = pd.read_csv(cache, usecols=["epoch", "theta_delta_log_z"])
            self.theta_delta = (f.set_index("epoch")["theta_delta_log_z"]
                                .reindex(range(self.n)).to_numpy(float))

    def take(self, centre: int) -> int | None:
        """Claim the clip around this epoch, if none of it was shown or labelled before."""
        start = int(min(max(centre - CLIP_EPOCHS // 2 + 1, 0), self.n - CLIP_EPOCHS))
        if start < 0 or self.used[start:start + CLIP_EPOCHS].any():
            return None
        self.used[start:start + CLIP_EPOCHS] = True
        return start

    def rem_score(self) -> np.ndarray:
        """How REM-like each epoch looks: the mean percentile of the model's REM
        probability and of theta/delta, over 2 epochs, outside likely wake."""
        def pct(x):
            x = pd.Series(x).rolling(2, min_periods=1).mean()
            return x.rank(pct=True).to_numpy()
        score = pct(self.proba[:, core.STATES.index("REM")])
        if self.theta_delta is not None:
            score = (score + pct(np.nan_to_num(self.theta_delta, nan=-np.inf))) / 2.0
        score[self.proba[:, core.STATES.index("Wake")] >= 0.5] = -np.inf
        return score


def _round_robin(recs: list[_Recording], order: dict[str, list[int]], n: int,
                 rng: np.random.Generator) -> list[tuple[str, int, int]]:
    """Take up to n clips, one recording at a time, each from its own ranked
    list of centre epochs."""
    clips, pos = [], {r.name: 0 for r in recs}
    i = int(rng.integers(len(recs))) if recs else 0
    misses = 0
    while len(clips) < n and misses < len(recs):
        r = recs[i % len(recs)]
        i += 1
        cand = order[r.name]
        while pos[r.name] < len(cand):
            start = r.take(cand[pos[r.name]])
            pos[r.name] += 1
            if start is not None:
                clips.append((r.name, start, CLIP_EPOCHS))
                misses = 0
                break
        else:
            misses += 1
    return clips


def _on_call(recs, state, n, rng):
    """n clips centred on random epochs the model calls this state."""
    order = {r.name: list(rng.permutation(np.flatnonzero(r.labels == state)))
             for r in recs}
    return _round_robin(recs, order, n, rng)


def _by_call(recs, n, rng):
    """n clips split evenly over the model's Wake, NREM and REM calls."""
    clips, quota = [], [n // 3 + (i < n % 3) for i in range(3)]
    for k, state in enumerate(core.STATES):
        got = _on_call(recs, state, quota[k], rng)
        clips += got
        if k + 1 < len(quota):
            quota[k + 1] += quota[k] - len(got)
    if len(clips) < n:
        order = {r.name: list(rng.permutation(r.n)) for r in recs}
        clips += _round_robin(recs, order, n - len(clips), rng)
    return clips


def _rem_search(recs, n, rng):
    """n clips centred on the most REM-like epochs not yet seen."""
    order = {}
    for r in recs:
        score = r.rem_score()
        ok = np.isfinite(score)
        order[r.name] = list(np.flatnonzero(ok)[np.argsort(-score[ok], kind="stable")])
    return _round_robin(recs, order, n, rng)


def _shares(short: dict[str, float], n: int) -> dict[str, int]:
    """Split n clips between states in proportion to what each still lacks."""
    total = sum(short.values())
    exact = {k: n * v / total for k, v in short.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: out[k] - exact[k])[:n - sum(out.values())]:
        out[k] += 1
    return out


def plan_round(project: core.Project, names: list[str], rem_search: bool = False,
               rng: np.random.Generator | None = None) -> list[tuple[str, int, int]]:
    """The clips to show next, as (recording, first epoch, epochs), grouped by
    recording and in time order.

    With no labels yet this is the first round; `rem_search` asks for REM
    candidates only. Otherwise the round goes after whatever is still unmet,
    with clips shared between the states in proportion to what each lacks,
    counting every manual label already there, imported ones included.
    """
    rng = rng or np.random.default_rng()
    recs = [_Recording(project, n) for n in names
            if core.load_model_output(project, n) is not None]
    state = needs(project, names)
    unmet = {x.key for x in state if not x.met}
    if rem_search:
        clips = _rem_search(recs, NEXT_ROUND_CLIPS, rng)
    elif not any(x.have for x in state if x.key in core.STATES):
        clips = _by_call(recs, FIRST_ROUND_CLIPS, rng)
    else:
        clips = []
        for r in recs:
            if not core.LabelStore(project, r.name).n_manual():
                clips += _by_call([r], 3, rng)
        short = {x.key: max(0.0, x.target - x.have) for x in state
                 if x.key in core.STATES}
        if unmet & REM_KEYS:
            short["REM"] = max(short["REM"], CLIP_EPOCHS)
        n = int(np.clip(np.ceil(sum(short.values()) / CLIP_EPOCHS),
                        NEXT_ROUND_CLIPS, FIRST_ROUND_CLIPS)) - len(clips)
        if n > 0 and sum(short.values()):
            for k, m in _shares(short, n).items():
                clips += (_rem_search(recs, m, rng) if k == "REM"
                          else _on_call(recs, k, m, rng))
        elif n > 0:
            clips += _by_call(recs, n, rng)
    rank = {n: i for i, n in enumerate(names)}
    return sorted(clips, key=lambda c: (rank[c[0]], c[1]))
