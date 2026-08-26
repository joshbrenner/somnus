"""Non-GUI core for the Somnus app: project state, labels with provenance, scoring.

Kept free of Qt so it can be tested and scripted without a display, and so the
GUI layer stays thin.

READ-ONLY ON SOURCE DATA. Every write goes under the project directory. Source
EDFs, scored CSVs, videos and tracking files are opened for reading only and are
never modified, moved or written next to.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from somnus.predict import DEFAULT_ARTIFACT

STATES = ["Wake", "NREM", "REM"]
EPOCH_SEC = 4.0

# How a label came to be. Fine-tuning may only ever train on manually sourced rows:
# training on the model's own output is a feedback loop that raises apparent
# confidence while adding no information.
SRC_MODEL = "model"        # produced by the classifier
SRC_CONFIRMED = "manual_confirmed"   # the user looked and agreed
SRC_CORRECTED = "manual_corrected"   # the user changed it
SRC_IMPORTED = "manual_imported"     # from a pre-existing manual scoring file
SRC_EXCLUDED = "manual_excluded"     # the user marked it Artifact/unscorable
# Sources that may be used as fine-tuning targets. SRC_EXCLUDED is deliberately
# absent: the user said the epoch is not a clean state, so it must not train the
# model, but it is still recorded so the decision survives a re-score.
MANUAL_SOURCES = (SRC_CONFIRMED, SRC_CORRECTED, SRC_IMPORTED)
# Sources a model write must not clobber (includes exclusions).
PROTECTED_SOURCES = MANUAL_SOURCES + (SRC_EXCLUDED,)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------- project
@dataclass
class Recording:
    name: str
    edf: str
    scored: str | None = None          # pre-existing manual labels, if any
    coords: str | None = None          # tracking positions
    timestamps: str | None = None      # true per-frame times for the tracking
    video: str | None = None
    selected: bool = False             # ticked in the Project tab
    notes: str = ""

    @property
    def has_velocity(self) -> bool:
        """Velocity needs positions AND trustworthy frame times.

        Cameras can drop frames, so assuming a constant fps misplaces positions
        by minutes. Without timestamps the feature is withheld rather than
        guessed.
        """
        return bool(self.coords) and bool(self.timestamps)


@dataclass
class Project:
    path: str
    name: str = "somnus_project"
    model: str = ""                    # active model artifact
    recordings: list[Recording] = field(default_factory=list)
    created: str = field(default_factory=_now)

    # ---- persistence
    @property
    def file(self) -> str:
        return os.path.join(self.path, "project.json")

    @property
    def labels_dir(self) -> str:
        return os.path.join(self.path, "labels")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.path, "cache")

    @property
    def models_dir(self) -> str:
        return os.path.join(self.path, "models")

    def ensure_dirs(self) -> None:
        for d in (self.path, self.labels_dir, self.cache_dir, self.models_dir):
            os.makedirs(d, exist_ok=True)

    def save(self) -> None:
        self.ensure_dirs()
        d = asdict(self)
        d.pop("path", None)            # implied by location
        with open(self.file, "w") as fh:
            json.dump(d, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(os.path.join(path, "project.json")) as fh:
            d = json.load(fh)
        recs = [Recording(**r) for r in d.pop("recordings", [])]
        return cls(path=path, recordings=recs, **d)

    @classmethod
    def create(cls, path: str, name: str | None = None,
               model: str = "") -> "Project":
        p = cls(path=path, name=name or os.path.basename(path.rstrip("/")),
                model=model or DEFAULT_ARTIFACT)
        p.ensure_dirs()
        p.save()
        return p

    def get(self, name: str) -> Recording | None:
        return next((r for r in self.recordings if r.name == name), None)


def discover_recordings(folder: str) -> list[Recording]:
    """Find EDFs in a folder and pair them with sibling label/tracking/video files.

    Purely descriptive: nothing is opened for writing, nothing is copied. Matching
    is by filename prefix, so a recording's video and tracking files are found
    whatever suffix the camera or tracker appends to the EDF's base name.
    """
    import glob as _glob
    out = []
    for root, _dirs, files in os.walk(folder):
        names = sorted(files)
        for f in names:
            if not f.lower().endswith(".edf"):
                continue
            base = f[:-4]
            edf = os.path.join(root, f)

            def first(pattern: str, exclude: tuple[str, ...] = ()) -> str | None:
                hits = [p for p in sorted(_glob.glob(os.path.join(root, pattern)))
                        if not any(x in os.path.basename(p) for x in exclude)]
                return hits[0] if hits else None

            scored = None
            for s in ("_scored.csv", "_scored_man.csv"):
                cand = os.path.join(root, base + s)
                if os.path.exists(cand):
                    scored = cand
                    break

            # Tracking: the `*coordinates.pkl` convention first, then a
            # DeepLabCut export. Only files named as DLC writes them are
            # considered, so a sibling `_scored.csv` is never mistaken for
            # tracking. DLC's `_full.pickle` is its pre-assembly dump, not a
            # coordinate table, so it is excluded.
            coords = (first(base + "*coordinates.pkl")
                      or first(base + "*DLC*.h5")
                      or first(base + "*DLC*.csv", exclude=("_full", "_meta")))
            # Timestamps are looked for ONLY in the folder the user pointed at.
            # Deliberately no widened/recursive search: picking up a stale or
            # mismatched timestamps file from elsewhere would pair positions with
            # the wrong times, and where frames have been dropped that
            # misaligns tracking from the EEG by minutes -- silently. Reporting
            # "not found" is the safe failure, so velocity is withheld instead.
            ts = first(base + "*timestamps.npy")
            video = first(base + "*.mp4", exclude=("_annotated", "_mask",
                                                  "_overlay", "_highlights"))
            out.append(Recording(name=base, edf=edf, scored=scored,
                                 coords=coords, timestamps=ts, video=video))
    return out


# ---------------------------------------------------------------------- labels
# `model_state` records what the classifier predicted for this epoch, kept even
# after the user overrides `state`. Without it there is no way to tell a real
# correction from a label that was edited and then put back: the row would stay
# flagged as manually corrected, show as reviewed, and be fed to fine-tuning even
# though it is identical to the model's own output -- which is the feedback loop
# the provenance tracking exists to prevent.
LABEL_COLS = ["epoch", "t_start", "state", "source", "confidence",
              "model", "model_state", "updated"]


class LabelStore:
    """Per-recording label table with provenance, stored under the project.

    One row per 4 s epoch:
        epoch, t_start, state, source, confidence, model, updated

    Invariant enforced here: **a model write never overwrites a manual row.**
    That is the whole point of tracking `source` -- it keeps re-scoring a
    recording from silently discarding someone's manual corrections, and it lets
    fine-tuning select manual rows only.
    """

    def __init__(self, project: Project, recording: str):
        self.project = project
        self.recording = recording
        self.path = os.path.join(project.labels_dir, f"{recording}_labels.csv")
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        if os.path.exists(self.path):
            df = pd.read_csv(self.path)
            for c in LABEL_COLS:
                if c not in df.columns:
                    df[c] = np.nan
            return df[LABEL_COLS]
        return pd.DataFrame(columns=LABEL_COLS)

    def save(self) -> None:
        self.project.ensure_dirs()
        self.df.sort_values("epoch").to_csv(self.path, index=False)

    # ---- queries
    def manual_mask(self) -> np.ndarray:
        """Rows usable as fine-tuning targets (excludes Artifact-marked epochs)."""
        return self.df["source"].isin(MANUAL_SOURCES).to_numpy()

    def protected_mask(self) -> np.ndarray:
        """Rows a model write must not overwrite (manual labels AND exclusions)."""
        return self.df["source"].isin(PROTECTED_SOURCES).to_numpy()

    def n_confirmed(self) -> int:
        return int((self.df['source'] == SRC_CONFIRMED).sum())

    def n_corrected(self) -> int:
        return int((self.df['source'] == SRC_CORRECTED).sum())

    def n_imported(self) -> int:
        return int((self.df['source'] == SRC_IMPORTED).sum())

    def n_excluded(self) -> int:
        return int((self.df["source"] == SRC_EXCLUDED).sum())

    def revert_to_model(self, epoch: int) -> bool:
        """Drop the manual flag when the label matches the model again.

        Used when an epoch was edited and then changed back: leaving it marked as
        manually corrected would keep it green in the hypnogram and, worse, feed the
        model its own prediction as a training target.
        """
        row = self.df[self.df["epoch"] == epoch]
        if not len(row):
            return False
        ms = row["model_state"].iloc[0]
        if not isinstance(ms, str) or row["state"].iloc[0] != ms:
            return False
        idx = row.index[0]
        self.df.loc[idx, "source"] = SRC_MODEL
        self.df.loc[idx, "updated"] = _now()
        return True

    def set_excluded(self, epoch: int) -> None:
        """Mark an epoch as manually excluded (Artifact / not a clean state)."""
        row = self.df[self.df["epoch"] == epoch]
        t = float(row["t_start"].iloc[0]) if len(row) else epoch * EPOCH_SEC
        model = row["model"].iloc[0] if len(row) else ""
        mstate = row["model_state"].iloc[0] if len(row) else None
        keep = self.df[self.df["epoch"] != epoch]
        new = pd.DataFrame([[epoch, t, "Artifact", SRC_EXCLUDED, np.nan,
                             model, mstate, _now()]], columns=LABEL_COLS)
        self.df = pd.concat([keep, new], ignore_index=True)

    def n_manual(self) -> int:
        return int(self.manual_mask().sum())

    def counts(self) -> dict:
        return self.df["state"].value_counts().to_dict()

    # ---- writes
    def set_model_labels(self, epochs: np.ndarray, t_start: np.ndarray,
                         states: np.ndarray, confidence: np.ndarray,
                         model: str) -> int:
        """Write model predictions, leaving every manually sourced row untouched.

        Returns the number of epochs actually written.
        """
        new = pd.DataFrame({
            "epoch": epochs, "t_start": t_start, "state": states,
            "source": SRC_MODEL, "confidence": confidence,
            "model": os.path.basename(model), "model_state": states,
            "updated": _now(),
        })
        if self.df.empty:
            self.df = new
            return len(new)
        protected = set(self.df.loc[self.protected_mask(), "epoch"].tolist())
        keep = self.df[self.df["epoch"].isin(protected)].copy()
        # refresh the model baseline on protected rows without touching `state`,
        # so "is this back to what the model said?" stays answerable after a
        # re-score with different settings
        base = new.set_index("epoch")["state"]
        keep["model_state"] = keep["epoch"].map(base).fillna(keep["model_state"])
        new = new[~new["epoch"].isin(protected)]
        self.df = (pd.concat([keep, new], ignore_index=True)
                   .drop_duplicates("epoch", keep="last"))
        return len(new)

    def set_manual_label(self, epoch: int, state: str,
                         corrected: bool | None = None) -> None:
        """Record the user's decision for one epoch.

        `corrected=None` infers it: same as the current label -> confirmed,
        different -> corrected.
        """
        row = self.df[self.df["epoch"] == epoch]
        prev = row["state"].iloc[0] if len(row) else None
        if corrected is None:
            corrected = (prev is not None) and (state != prev)
        src = SRC_CORRECTED if corrected else SRC_CONFIRMED
        t = float(row["t_start"].iloc[0]) if len(row) else epoch * EPOCH_SEC
        conf = float(row["confidence"].iloc[0]) if len(row) else np.nan
        model = row["model"].iloc[0] if len(row) else ""
        mstate = row["model_state"].iloc[0] if len(row) else None
        # Build the replacement as its own frame and concat. Do NOT use
        # `self.df.loc[len(self.df)] = ...` after filtering: the surviving index
        # is non-contiguous, so that positional guess silently OVERWRITES an
        # existing row.
        keep = self.df[self.df["epoch"] != epoch]
        new = pd.DataFrame([[epoch, t, state, src, conf, model, mstate, _now()]],
                           columns=LABEL_COLS)
        self.df = pd.concat([keep, new], ignore_index=True)

    def import_manual(self, scored_csv: str, n_epochs: int) -> int:
        """Import a pre-existing one-hot _scored.csv as manual labels.

        The source file is read only. Its 0.5 s bins are collapsed onto 4 s
        epochs, and an epoch is only accepted if every bin inside it agrees and
        none is artifact/unknown -- a mixed epoch is not a label.
        """
        s = pd.read_csv(scored_csv)
        step = float(np.median(np.diff(s["Time_sec"].to_numpy()))) or 0.5
        per = max(1, int(round(EPOCH_SEC / step)))
        rows = []
        for e in range(n_epochs):
            blk = s.iloc[e * per:(e + 1) * per]
            if len(blk) < per:
                break
            if blk.get("Artifact", pd.Series(0, index=blk.index)).sum() > 0:
                continue
            if "Unclear" in blk and blk["Unclear"].sum() > 0:
                continue
            hit = [st for st in STATES if st in blk and blk[st].sum() == per]
            if len(hit) != 1:
                continue
            rows.append((e, e * EPOCH_SEC, hit[0], SRC_IMPORTED, np.nan,
                         os.path.basename(scored_csv), None, _now()))
        if not rows:
            return 0
        imp = pd.DataFrame(rows, columns=LABEL_COLS)
        self.df = (pd.concat([self.df, imp], ignore_index=True)
                   .drop_duplicates("epoch", keep="last"))
        return len(imp)


# --------------------------------------------------------------------- scoring
def featurize(recording: Recording, cache: str | None = None) -> pd.DataFrame:
    """Feature table for one recording, cached under the project."""
    if cache and os.path.exists(cache):
        return pd.read_csv(cache)
    from somnus.data import datasets as B
    entry = {"recording": recording.name, "edf": recording.edf,
             "dataset": "user", "group": "user",
             "subject": recording.name.split("_")[0],
             "scored": recording.scored, "pkl": recording.coords}
    df = B.featurize(entry)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def score(feat: pd.DataFrame, model_path: str, decode: bool = True,
          stickiness: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Predict states. Returns (labels, per-epoch probabilities)."""
    from somnus.predict import load_model, predict
    art = load_model(model_path)
    return predict(art, feat, decode=decode, stickiness=stickiness)


def uncertainty(proba: np.ndarray, labels: np.ndarray | None = None,
                raw: np.ndarray | None = None) -> np.ndarray:
    """Per-epoch review priority in [0, 1]; higher means 'look at me first'.

    Combines two complementary signals:
      * margin -- how close the top two class probabilities are. A 0.51/0.49
        epoch is genuinely ambiguous to the model.
      * decode override -- the smoothed label disagrees with the raw per-epoch
        argmax, i.e. the HMM overruled the evidence. These are exactly where
        smoothing may have erased a real short bout, so they are worth checking
        even when the model looks confident.
    """
    p = np.sort(proba, axis=1)
    margin = 1.0 - (p[:, -1] - p[:, -2])          # 1 = tied, 0 = certain
    u = margin
    if labels is not None and raw is not None:
        u = np.maximum(u, (labels != raw).astype(float))
    return np.clip(u, 0.0, 1.0)


def review_queue(proba: np.ndarray, labels: np.ndarray, raw: np.ndarray,
                 manual: np.ndarray | None = None,
                 top: int | None = None) -> np.ndarray:
    """Epoch indices ordered by review priority, already-reviewed ones dropped."""
    u = uncertainty(proba, labels, raw)
    if manual is not None:
        u = np.where(manual, -1.0, u)
    order = np.argsort(-u)
    order = order[u[order] >= 0]
    return order if top is None else order[:top]


# ----------------------------------------------------- hand-off to the scorer UI
def smooth_trace(x: np.ndarray, window: int = 15) -> np.ndarray:
    """Centered rolling median, for displaying a readable confidence trace.

    A per-epoch confidence trace on a multi-hour recording is mostly vertical
    hash; a ~1 min median keeps the shape while making it legible. Display only --
    never fed back into scoring.
    """
    if window <= 1 or len(x) == 0:
        return np.asarray(x, dtype=float)
    return (pd.Series(np.asarray(x, dtype=float))
            .rolling(int(window), center=True, min_periods=1).median().to_numpy())


MAX_SCORING_VERSIONS = 5


def archive_scoring(project: "Project", recording: Recording,
                    keep: int = MAX_SCORING_VERSIONS) -> str | None:
    """Snapshot the current scoring CSV into labels/history/ before it is replaced.

    Manual scoring is the expensive, irreplaceable part of this workflow, and the
    scorer overwrites its CSV in place on save. Keeping the last `keep` snapshots
    means a mis-paint, a wrong-brush drag, or a re-score that clobbers work can be
    recovered. Oldest snapshots are pruned so the folder cannot grow without
    bound. Returns the snapshot path, or None if there was nothing to archive.
    """
    import glob as _glob
    import shutil
    cur = os.path.join(project.labels_dir, f"{recording.name}_scored.csv")
    if not os.path.exists(cur):
        return None
    hist = os.path.join(project.labels_dir, "history")
    os.makedirs(hist, exist_ok=True)
    stamp = _now().replace(":", "").replace("-", "").replace("+0000", "")
    # Timestamps are second-resolution, so two saves within the same second would
    # collide; walk a counter until the name is free.
    dest = os.path.join(hist, f"{recording.name}_scored_{stamp}.csv")
    k = 1
    while os.path.exists(dest):
        dest = os.path.join(hist, f"{recording.name}_scored_{stamp}_{k:02d}.csv")
        k += 1
    shutil.copy2(cur, dest)

    old = sorted(_glob.glob(os.path.join(hist, f"{recording.name}_scored_*.csv")))
    for path in old[:-keep]:
        try:
            os.remove(path)
        except OSError:
            pass
    return dest


def scoring_versions(project: "Project", recording: Recording) -> list[str]:
    """Snapshots for this recording, newest last."""
    import glob as _glob
    hist = os.path.join(project.labels_dir, "history")
    return sorted(_glob.glob(os.path.join(hist,
                                          f"{recording.name}_scored_*.csv")))


def write_viewer_bundle(project: "Project", recording: Recording,
                        labels: np.ndarray, proba: np.ndarray,
                        raw: np.ndarray, store: "LabelStore",
                        bin_sec: float = 0.5) -> tuple[str, str]:
    """Write the two files the scorer UI reads, inside the project.

    Returns (scored_csv, meta_csv).

    * scored_csv -- the one-hot layout the UI already expects, expanded from 4 s
      epochs to `bin_sec` bins. Written into the project, NOT beside the EDF:
      the UI's save_csv() overwrites whatever path it is given, and pointing it at
      the source folder would modify original data.
    * meta_csv -- per-epoch `uncertainty` and `hmm_smoothed`, which the UI uses
      for the certainty readout, the jump-to-most-uncertain control and the
      distinct color for smoothed epochs. Optional: without it the UI still runs.
    """
    project.ensure_dirs()
    n_ep = len(labels)
    per = max(1, int(round(EPOCH_SEC / bin_sec)))

    # merge in any manual labels already recorded, so the UI opens on the latest
    lab = np.array(labels, dtype=object)
    if len(store.df):
        h = store.df[store.manual_mask()]
        for e, s in zip(h["epoch"].to_numpy(), h["state"].to_numpy()):
            if 0 <= int(e) < n_ep:
                lab[int(e)] = s

    rep = np.repeat(lab, per)
    t = np.arange(len(rep)) * bin_sec
    # carry existing confirmations back into the scorer so they render as
    # already-affirmed instead of looking unreviewed on every reopen
    conf_ep = set()
    if len(store.df):
        conf_ep = set(store.df.loc[store.df["source"] == SRC_CONFIRMED,
                                   "epoch"].astype(int).tolist())
    conf_flag = np.repeat(
        np.array([1 if e in conf_ep else 0 for e in range(n_ep)]), per)

    scored = pd.DataFrame({
        "Time_sec": t,
        "Wake": (rep == "Wake").astype(int),
        "NREM": (rep == "NREM").astype(int),
        "REM": (rep == "REM").astype(int),
        "Artifact": 0,
        "Unclear": 0,
        "Confirmed": conf_flag,
    })
    # snapshot whatever the scorer last saved before replacing it
    archive_scoring(project, recording)
    scored_path = os.path.join(project.labels_dir, f"{recording.name}_scored.csv")
    scored.to_csv(scored_path, index=False)

    manual = np.zeros(n_ep, dtype=int)
    if len(store.df):
        h = store.df.loc[store.manual_mask(), "epoch"].to_numpy()
        h = h[(h >= 0) & (h < n_ep)]
        manual[h.astype(int)] = 1
    meta = pd.DataFrame({
        "epoch": np.arange(n_ep),
        "t_start": np.arange(n_ep) * EPOCH_SEC,
        # margin only: an epoch changed by the decode is NOT treated as
        # uncertain, it gets its own color instead, so the jump
        # control targets genuine model ambiguity rather than smoothing edits
        "uncertainty": uncertainty(proba),
        "confidence": proba.max(axis=1),
        "hmm_smoothed": (np.asarray(labels) != np.asarray(raw)).astype(int),
        "reviewed": manual,
    })
    # per-state probabilities, so the scorer can show the model's belief averaged
    # over whatever window is on screen
    for j, s in enumerate(STATES):
        meta[f"p_{s}"] = proba[:, j]
    meta_path = os.path.join(project.labels_dir, f"{recording.name}_review_meta.csv")
    meta.to_csv(meta_path, index=False)
    return scored_path, meta_path


def read_viewer_labels(project: "Project", recording: Recording,
                       store: "LabelStore", n_epochs: int,
                       bin_sec: float = 0.5) -> dict:
    """Pull edits made in the scorer UI back into the label store.

    Returns {'corrected': n, 'confirmed': n, 'excluded': n, 'reverted': n}.

    **Only epochs whose label CHANGED are recorded as manual.** The CSV handed to
    the scorer is pre-filled with the model's own predictions, so an untouched
    epoch comes back identical. Treating those as "manually confirmed" would convert
    the model's entire output into training targets -- a feedback loop that
    inflates apparent confidence while adding no information, and precisely what
    the `source` column exists to prevent. A genuine "I looked and agreed" is
    indistinguishable from "I never visited this epoch" once it is a CSV, so the
    conservative reading is the only safe one. (Confirmations can still be
    recorded explicitly elsewhere; they are simply not inferred from a round-trip.)

    Epochs painted Artifact or Unclear are recorded as `manual_excluded`: the user
    is saying "this is not a clean state", so the epoch must be kept OUT of
    fine-tuning even if it previously carried a valid label.
    """
    path = os.path.join(project.labels_dir, f"{recording.name}_scored.csv")
    if not os.path.exists(path):
        return {"corrected": 0, "confirmed": 0, "excluded": 0, "reverted": 0}
    s = pd.read_csv(path)
    per = max(1, int(round(EPOCH_SEC / bin_sec)))
    n_cor = n_exc = n_rev = n_con = 0
    for e in range(min(n_epochs, len(s) // per)):
        blk = s.iloc[e * per:(e + 1) * per]
        if len(blk) < per:
            break
        cur = store.df[store.df["epoch"] == e]
        cur_state = cur["state"].iloc[0] if len(cur) else None
        cur_src = cur["source"].iloc[0] if len(cur) else None

        marked = any(col in blk and blk[col].sum() > 0
                     for col in ("Artifact", "Unclear"))
        if marked:
            if cur_src != SRC_EXCLUDED:
                store.set_excluded(e)
                n_exc += 1
            continue

        hit = [st for st in STATES if st in blk and blk[st].sum() == per]
        if len(hit) != 1:
            continue                       # mixed epoch is not a label
        # The Confirm brush is the ONLY affirmative signal available: a label left
        # untouched is byte-identical to one that was checked and agreed with, so
        # without this flag an unedited epoch can never count as reviewed.
        confirmed = ("Confirmed" in blk
                     and blk["Confirmed"].sum() == per)

        if hit[0] == cur_state:
            if confirmed:
                if cur_src != SRC_CONFIRMED:
                    store.set_manual_label(e, hit[0], corrected=False)
                    n_con += 1
                continue
            # Not confirmed and unchanged. If the row is flagged manual but now
            # equals the model's own prediction, the user edited it and then put
            # it back -- drop the flag, else it stays green in the hypnogram and
            # gets fed to fine-tuning as the model's own output.
            if cur_src in MANUAL_SOURCES and store.revert_to_model(e):
                n_rev += 1
            continue

        store.set_manual_label(e, hit[0], corrected=True)
        # A "correction" back to exactly what the model said is not a correction,
        # UNLESS the user explicitly confirmed it with the Confirm brush.
        if confirmed:
            store.set_manual_label(e, hit[0], corrected=False)
            n_con += 1
        elif store.revert_to_model(e):
            n_rev += 1
        else:
            n_cor += 1
    return {"corrected": n_cor, "confirmed": n_con,
            "excluded": n_exc, "reverted": n_rev}


# ------------------------------------------------------------------- QC checks
def qc_flags(labels: np.ndarray, epoch_sec: float = EPOCH_SEC) -> list[dict]:
    """Physiologically suspicious patterns worth a manual review.

    These are hints, not errors: a flag means 'unusual', and unusual is exactly
    what a disease model may legitimately produce. They are never auto-corrected.
    """
    flags = []
    if len(labels) == 0:
        return flags
    change = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    bounds = np.r_[change, len(labels)]
    for i in range(len(change)):
        s, e = bounds[i], bounds[i + 1]
        st, dur = labels[s], (e - s) * epoch_sec
        if st == "REM":
            prev = labels[s - 1] if s > 0 else None
            if prev is not None and prev != "NREM":
                flags.append(dict(epoch=int(s), kind="REM_without_NREM",
                                  detail=f"REM bout preceded by {prev}"))
            if dur < 8:
                flags.append(dict(epoch=int(s), kind="very_short_REM",
                                  detail=f"{dur:.0f}s REM bout"))
        elif dur < epoch_sec * 2:
            flags.append(dict(epoch=int(s), kind="very_short_bout",
                              detail=f"{dur:.0f}s {st} bout"))
    return flags


# --------------------------------------------------------- architecture export
def architecture(labels: np.ndarray, epoch_sec: float = EPOCH_SEC) -> dict:
    """Sleep-architecture summary: the numbers that go in a paper."""
    n = len(labels)
    out: dict = {"n_epochs": int(n),
                 "recording_hours": round(n * epoch_sec / 3600.0, 3)}
    if n == 0:
        return out
    change = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    bounds = np.r_[change, n]
    durs: dict[str, list[float]] = {s: [] for s in STATES}
    for i in range(len(change)):
        st = labels[change[i]]
        if st in durs:
            durs[st].append((bounds[i + 1] - change[i]) * epoch_sec)
    for s in STATES:
        frac = float((labels == s).mean())
        out[f"pct_{s}"] = round(100 * frac, 2)
        out[f"minutes_{s}"] = round(float((labels == s).sum()) * epoch_sec / 60, 2)
        out[f"bouts_{s}"] = len(durs[s])
        out[f"mean_bout_s_{s}"] = round(float(np.mean(durs[s])), 1) if durs[s] else 0.0
        out[f"median_bout_s_{s}"] = round(float(np.median(durs[s])), 1) if durs[s] else 0.0
    # transitions
    idx = {s: i for i, s in enumerate(STATES)}
    T = np.zeros((3, 3), dtype=int)
    for a, b in zip(labels[:-1], labels[1:]):
        if a in idx and b in idx and a != b:
            T[idx[a], idx[b]] += 1
    for a in STATES:
        for b in STATES:
            if a != b:
                out[f"trans_{a}_to_{b}"] = int(T[idx[a], idx[b]])
    for s in ("NREM", "REM"):
        hit = np.flatnonzero(labels == s)
        out[f"latency_min_{s}"] = round(float(hit[0]) * epoch_sec / 60, 2) \
            if len(hit) else None
    return out
