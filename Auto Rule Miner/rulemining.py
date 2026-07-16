"""
Auto Mining Demo (自动挖掘 Demo)

A runnable reference implementation for:
- Numeric threshold mining (head/tail search with lift constraints)
- Optional RandomForest path rule mining (rf_rule_mining)
- Greedy OR-union rule selection (greedy_or_union / greedy_union_selection)
- Train/OOT/Total reporting with optional profit metrics
- Optional time sorting (important when upstream features depend on ordering)

Notes
-----
1) All rule expressions are pandas.query-compatible strings.
2) The demo intentionally keeps dependencies light. RF mining requires scikit-learn.
3) Profit is optional: if profit_col is None, profit-related outputs are suppressed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Any

import math
import re
from statistics import NormalDist
import numpy as np
import pandas as pd


# -----------------------------
# Progress helpers
# -----------------------------
def _progress(it, *, total=None, desc: Optional[str] = None, show: bool = True):
    """Wrap an iterable with a progress bar (tqdm) when available.

    If tqdm is not installed or show=False, returns the iterable unchanged.
    """
    if not show:
        return it
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm(it, total=total, desc=desc)
    except Exception:
        return it


# -----------------------------
# Time sorting helpers
# -----------------------------
def _ensure_datetime(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce")


def sort_by_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Return a copy sorted by time_col asc (stable)."""
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in df.columns")
    out = df.copy()
    out[time_col] = _ensure_datetime(out[time_col])
    out = out.sort_values(time_col, ascending=True, kind="mergesort").reset_index(drop=True)
    return out


# -----------------------------
# Rule construction & evaluation
# -----------------------------
def _safe_col(col: str) -> str:
    """Quote a column name for pandas.query if needed."""
    if col.isidentifier():
        return col
    return f"`{col}`"



# -----------------------------
# Threshold formatting helpers
# -----------------------------
# Default threshold rounding for rule strings. This only affects rule *text* / de-duplication;
# all metrics are always re-evaluated on the raw data via masks.
DEFAULT_THR_NDIGITS = 4

def _fmt_thr(thr: float, ndigits: int = DEFAULT_THR_NDIGITS) -> str:
    """Format numeric threshold for pandas.query. Round to `ndigits` decimals."""
    try:
        x = float(thr)
    except Exception:
        return repr(thr)
    if not math.isfinite(x):
        return repr(x)
    return format(round(x, ndigits), f".{ndigits}f")


def make_condition(var: str, op: str, thr: float) -> str:
    """Build a pandas.query-compatible condition. Example: `x` <= 1.23"""
    if op not in ("<=", ">"):
        raise ValueError(f"Unsupported op: {op}")
    thr_str = _fmt_thr(thr)
    return f"{_safe_col(var)} {op} {thr_str}"


def mask_from_condition(df: pd.DataFrame, condition: str) -> pd.Series:
    """
    Return a boolean mask for `condition` on df (safe for 'True'/'False').

    Uses df.query(engine='python') to keep compatibility with backtick-quoted column names.
    """
    cond = (condition or "").strip()
    if cond.lower() in ("true", "(true)"):
        return pd.Series(True, index=df.index)
    if cond.lower() in ("false", "(false)"):
        return pd.Series(False, index=df.index)
    try:
        idx = df.query(cond, engine="python").index
    except Exception as e:
        raise ValueError(f"Failed to evaluate condition: {cond}") from e
    return pd.Series(df.index.isin(idx), index=df.index)


def _extract_vars_from_condition(condition: str) -> List[str]:
    """Best-effort variable extraction from a pandas.query condition string."""
    s = (condition or "")
    if not s:
        return []
    vars_bt = re.findall(r"`([^`]+)`\s*(?:<=|>=|<|>|==|!=)", s)
    vars_id = re.findall(r"(?<![\w`])([A-Za-z_][A-Za-z0-9_]*)\s*(?:<=|>=|<|>|==|!=)", s)
    out: List[str] = []
    for v in vars_bt + vars_id:
        if v in {"and", "or", "not", "True", "False"}:
            continue
        if v not in out:
            out.append(v)
    return out

def _safe_div(a: float, b: float) -> float:
    """a/b with 0-safe."""
    return float(a) / float(b) if float(b) != 0.0 else 0.0

def _z_value(ci_level: float = 0.95, one_sided: bool = True) -> float:
    """Return z for a Normal CI.

    - one_sided=True: z = Phi^{-1}(ci_level)  (e.g., 0.95 -> 1.645)
    - one_sided=False: z = Phi^{-1}((1+ci_level)/2) (e.g., 0.95 -> 1.96)
    """
    prob = float(ci_level) if one_sided else (1.0 + float(ci_level)) / 2.0
    prob = min(max(prob, 1e-6), 1 - 1e-6)
    return float(NormalDist().inv_cdf(prob))


def _cnt_lift_lb_ci(
    *,
    k1: int,
    n1: int,
    k0: int,
    n0: int,
    ci_level: float = 0.95,
    one_sided: bool = True,
    cc: float = 0.5,
) -> Tuple[float, float]:
    """Approximate CI lower-bound for count-lift using delta method on log(RR).

    lift = (k1/n1) / (k0/n0). We compute CI on log(lift) and return:
      - lb: exp(log(lift) - z * SE)
      - se_log: SE(log(lift))

    Notes:
    - cc is a continuity correction / smoothing to avoid k==0 or k==n pathologies.
      With cc=0.5, it corresponds to (k+0.5)/(n+1).
    """
    if n1 <= 0 or n0 <= 0:
        return (float("nan"), float("nan"))

    # smoothing (always apply; it is mild and avoids edge-case blow-ups)
    k1a = float(k1) + float(cc)
    n1a = float(n1) + 2.0 * float(cc)
    k0a = float(k0) + float(cc)
    n0a = float(n0) + 2.0 * float(cc)

    p1 = k1a / n1a
    p0 = k0a / n0a
    if p1 <= 0.0 or p0 <= 0.0:
        return (float("nan"), float("nan"))

    lift = p1 / p0
    se_log = math.sqrt((1.0 - p1) / (n1a * p1) + (1.0 - p0) / (n0a * p0))
    z = _z_value(ci_level=ci_level, one_sided=one_sided)
    lb = math.exp(math.log(lift) - z * se_log)
    return (float(lb), float(se_log))



# -----------------------------
# Metrics & reporting
# -----------------------------
@dataclass
class SegmentStats:
    n: int
    bad: int
    bad_rate: float
    amt: float
    bad_amt: float
    bad_amt_rate: float
    profit: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "n": int(self.n),
            "bad": int(self.bad),
            "bad_rate": float(self.bad_rate),
            "amt": float(self.amt),
            "bad_amt": float(self.bad_amt),
            "bad_amt_rate": float(self.bad_amt_rate),
        }
        if self.profit is not None:
            d["profit"] = float(self.profit)
        return d


def segment_stats(
    df: pd.DataFrame,
    target_col: str,
    amount_col: str,
    profit_col: Optional[str] = None,
    profit_scale: Optional[float] = None,
) -> SegmentStats:
    """Compute count/amount metrics for a segment."""
    n = int(len(df))
    if n == 0:
        return SegmentStats(n=0, bad=0, bad_rate=0.0, amt=0.0, bad_amt=0.0, bad_amt_rate=0.0,
                            profit=(0.0 if profit_col else None))

    y = df[target_col].astype(int)
    bad = int(y.sum())
    bad_rate = bad / n if n else 0.0

    amt = float(df[amount_col].sum()) if amount_col in df.columns else float(n)
    bad_amt = float(df.loc[y == 1, amount_col].sum()) if amount_col in df.columns else float(bad)
    bad_amt_rate = bad_amt / amt if amt != 0 else 0.0

    profit_val = None
    if profit_col and profit_col in df.columns:
        p = df[profit_col].sum()
        p = float(p)
        if profit_scale:
            p *= float(profit_scale)
        profit_val = p

    return SegmentStats(n=n, bad=bad, bad_rate=bad_rate, amt=amt, bad_amt=bad_amt, bad_amt_rate=bad_amt_rate, profit=profit_val)


def report_condition(
    train: pd.DataFrame,
    oot: pd.DataFrame,
    condition: str,
    *,
    target_col: str,
    amount_col: str,
    profit_col: Optional[str] = None,
    profit_scale: Optional[float] = None,
    title: str = "",
    print_profit: bool = True,
    do_print: bool = True,
) -> pd.DataFrame:
    """
    Print and return a report for condition on train/oot/total.

    Output rows: base / fit / non_fit for each dataset.
    """
    total = pd.concat([train, oot], axis=0, ignore_index=True)

    def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        m = mask_from_condition(df, condition)
        return df.loc[m], df.loc[~m]

    def _one(name: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        df_fit, df_non = _split(df)
        base = segment_stats(df, target_col, amount_col, profit_col if print_profit else None, profit_scale).as_dict()
        fit = segment_stats(df_fit, target_col, amount_col, profit_col if print_profit else None, profit_scale).as_dict()
        non = segment_stats(df_non, target_col, amount_col, profit_col if print_profit else None, profit_scale).as_dict()

        base_rate = base["bad_rate"]
        base_mrate = base["bad_amt_rate"]

        def _enh(row: Dict[str, Any], seg: str, fit_row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            out = {"dataset": name, "segment": seg, **row}
            if seg == "fit" and fit_row is not None:
                out["cnt_rate"] = _safe_div(fit_row["n"], base["n"])
                out["money_rate"] = _safe_div(fit_row["amt"], base["amt"])
                out["cnt_lift"] = _safe_div(fit_row["bad_rate"], base_rate)
                out["money_lift"] = _safe_div(fit_row["bad_amt_rate"], base_mrate)
            else:
                out["cnt_rate"] = np.nan
                out["money_rate"] = np.nan
                out["cnt_lift"] = np.nan
                out["money_lift"] = np.nan
            return out

        return [
            _enh(base, "base"),
            _enh(fit, "fit", fit),
            _enh(non, "non_fit"),
        ]

    rows = []
    for name, df in [("total", total), ("train", train), ("oot", oot)]:
        rows.extend(_one(name, df))

    df_rep = pd.DataFrame(rows)

    cols = [
        "dataset", "segment",
        "n", "bad", "bad_rate",
        "amt", "bad_amt", "bad_amt_rate",
        "cnt_rate", "money_rate", "cnt_lift", "money_lift",
    ]
    if print_profit and profit_col and profit_col in df_rep.columns:
        cols.append("profit")
    df_rep = df_rep[cols]

    if do_print:
        if title:
            print(title)
        print(f"Condition: {condition}")
        print(df_rep.to_string(index=False))
        print("-" * 80)
    return df_rep


# -----------------------------
# Threshold mining (numeric)
# -----------------------------
@dataclass
class ThresholdCandidate:
    var: str
    op: str
    thr: float
    obs: int
    obs_rate: float
    bad: int
    bad_rate: float
    lift: float
    selected: bool

    def condition(self) -> str:
        return make_condition(self.var, self.op, self.thr)


def _candidate_thresholds_from_quantiles(
    s: pd.Series,
    *,
    min_rate_act: float,
    sub_div_bin: float,
    n_grid: int = 30,
) -> Dict[str, np.ndarray]:
    """
    Generate threshold candidates from head & tail ranges using quantiles.

    Returns dict with keys: "head" and "tail", each contains candidate thresholds.
    """
    s2 = s.dropna()
    if len(s2) == 0:
        return {"head": np.array([]), "tail": np.array([])}

    # head range: [min_rate_act, sub_div_bin]
    q_head = np.linspace(min_rate_act, max(min_rate_act, sub_div_bin), n_grid)
    q_tail = np.linspace(min(1.0 - min_rate_act, 1.0 - sub_div_bin), 1.0 - min_rate_act, n_grid)

    thr_head = np.unique(np.quantile(s2, q_head))
    thr_tail = np.unique(np.quantile(s2, q_tail))
    return {"head": thr_head, "tail": thr_tail}


def mine_best_threshold(
    train: pd.DataFrame,
    var: str,
    *,
    target_col: str,
    min_num: int,
    min_rate: float,
    hit_num: int,
    min_lift: float,
    sub_div_bin: float,
) -> Optional[ThresholdCandidate]:
    """
    For one numeric var, search best threshold among head/tail candidates on TRAIN only.

    For head candidates: rule is (x <= thr)
    For tail candidates: rule is (x > thr)
    """
    if var not in train.columns:
        return None
    s = train[var]
    if s.dropna().nunique() <= 1:
        return None

    total_n = len(train)
    min_rate_act = max(min_rate, min_num / total_n if total_n else min_rate)

    cand = _candidate_thresholds_from_quantiles(s, min_rate_act=min_rate_act, sub_div_bin=sub_div_bin)
    base_bad_rate = float(train[target_col].mean()) if total_n else 0.0
    if base_bad_rate <= 0:
        return None

    best: Optional[ThresholdCandidate] = None

    def _eval(op: str, thr: float) -> Optional[ThresholdCandidate]:
        cond = make_condition(var, op, float(thr))
        m = mask_from_condition(train, cond)
        obs = int(m.sum())
        if obs < hit_num:
            return None
        bad = int(train.loc[m, target_col].sum())
        br = bad / obs if obs else 0.0
        lf = br / base_bad_rate if base_bad_rate > 0 else 0.0
        selected = (obs >= hit_num) and (lf >= min_lift)
        return ThresholdCandidate(
            var=var, op=op, thr=float(thr),
            obs=obs, obs_rate=obs / total_n if total_n else 0.0,
            bad=bad, bad_rate=br, lift=lf, selected=selected
        )

    # head: <=
    for thr in cand["head"]:
        c = _eval("<=", float(thr))
        if c is None:
            continue
        if best is None or c.lift > best.lift:
            best = c

    # tail: >
    for thr in cand["tail"]:
        c = _eval(">", float(thr))
        if c is None:
            continue
        if best is None or c.lift > best.lift:
            best = c

    return best


def mine_thresholds(
    train: pd.DataFrame,
    numeric_vars: Iterable[str],
    *,
    target_col: str = "tar",
    min_num: int = 80,
    min_rate: float = 0.01,
    hit_num: int = 80,
    min_lift: float = 1.15,
    sub_div_bin: float = 0.2,
    show_progress: bool = True,
    progress_desc: str = "Threshold mining",
) -> pd.DataFrame:
    """Mine best threshold rule per variable (TRAIN only)."""
    rows = []
    vars_list = list(numeric_vars)
    for v in _progress(vars_list, total=len(vars_list), desc=progress_desc, show=show_progress):
        c = mine_best_threshold(
            train=train,
            var=v,
            target_col=target_col,
            min_num=min_num,
            min_rate=min_rate,
            hit_num=hit_num,
            min_lift=min_lift,
            sub_div_bin=sub_div_bin,
        )
        if c is None:
            continue
        rows.append({
            "source": "threshold",
            "var": c.var,
            "op": c.op,
            "thr": c.thr,
            "obs": c.obs,
            "obs_rate": c.obs_rate,
            "bad": c.bad,
            "bad_rate": c.bad_rate,
            "lift": c.lift,
            "label2": "Y" if c.selected else "N",
            # label3 defaults to label2 (matches your strategic.py comment)
            "label3": "Y" if c.selected else "N",
            "condition": c.condition(),
        })
    if not rows:
        return pd.DataFrame(columns=["source","var","op","thr","obs","obs_rate","bad","bad_rate","lift","label2","label3","condition"])
    out = pd.DataFrame(rows)
    out = out.sort_values("lift", ascending=False).reset_index(drop=True)
    return out


# -----------------------------
# RF rule mining (optional)
# -----------------------------
def rf_rule_mining(
    train: pd.DataFrame,
    feature_cols: List[str],
    *,
    target_col: str = "tar",
    max_depth: int = 3,
    n_estimators: int = 200,
    min_samples_leaf: int = 80,
    random_state: int = 42,
    top_k_per_tree: int = 5,
    min_hit: int = 80,
    min_lift: float = 1.15,
    max_rules: int = 500,
    show_progress: bool = True,
    progress_desc: str = "RF mining",
) -> pd.DataFrame:
    """
    Mine conjunction rules from RandomForest leaf paths and pre-filter by lift on TRAIN.

    Returns DataFrame columns compatible with mine_thresholds(), with source='rf'.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        raise ImportError("scikit-learn is required for rf_rule_mining()") from e

    df = train.copy()

    X = df[feature_cols].copy()
    for c in feature_cols:
        if not np.issubdtype(X[c].dtype, np.number):
            raise ValueError(f"rf_rule_mining expects numeric feature columns. '{c}' dtype={X[c].dtype}")
        med = float(X[c].median()) if X[c].notna().any() else 0.0
        X[c] = X[c].fillna(med)

    y = df[target_col].astype(int).values
    base_bad_rate = float(np.mean(y)) if len(y) else 0.0
    if base_bad_rate <= 0:
        return pd.DataFrame(columns=["source","var","op","thr","obs","obs_rate","bad","bad_rate","lift","label2","label3","condition"])

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)

    def _path_condition(tree, leaf_id: int) -> str:
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold

        parent = np.full(children_left.shape[0], -1, dtype=int)
        is_left = np.zeros(children_left.shape[0], dtype=bool)
        for nid in range(children_left.shape[0]):
            cl, cr = children_left[nid], children_right[nid]
            if cl != -1:
                parent[cl] = nid
                is_left[cl] = True
            if cr != -1:
                parent[cr] = nid
                is_left[cr] = False

        lb: Dict[int, float] = {}
        ub: Dict[int, float] = {}
        node = leaf_id
        while parent[node] != -1:
            p = parent[node]
            fid = int(feature[p])
            thr = float(threshold[p])
            if is_left[node]:
                ub[fid] = min(ub.get(fid, float("inf")), thr)
            else:
                lb[fid] = max(lb.get(fid, float("-inf")), thr)
            node = p

        terms = []
        for fid in sorted(set(lb.keys()) | set(ub.keys())):
            col = feature_cols[fid]
            if fid in lb and math.isfinite(lb[fid]):
                terms.append(f"({_safe_col(col)} > {_fmt_thr(lb[fid])})")
            if fid in ub and math.isfinite(ub[fid]):
                terms.append(f"({_safe_col(col)} <= {_fmt_thr(ub[fid])})")
        return " & ".join(terms) if terms else "True"

    rows: List[Dict[str, Any]] = []
    for est in _progress(rf.estimators_, total=len(rf.estimators_), desc=progress_desc, show=show_progress):
        tr = est.tree_
        is_leaf = (tr.children_left == -1) & (tr.children_right == -1)
        leaf_ids = np.where(is_leaf)[0]

        leaf_infos = []
        for leaf_id in leaf_ids:
            n_node = int(tr.n_node_samples[leaf_id])
            if n_node < min_hit:
                continue
            val = tr.value[leaf_id][0]
            bad_cnt = float(val[1]) if len(val) > 1 else 0.0
            br = bad_cnt / max(1.0, float(np.sum(val)))
            lf = br / base_bad_rate if base_bad_rate > 0 else 0.0
            if lf >= min_lift:
                leaf_infos.append((int(leaf_id), float(lf)))

        if not leaf_infos:
            continue

        leaf_infos.sort(key=lambda x: x[1], reverse=True)
        leaf_infos = leaf_infos[:max(1, top_k_per_tree)]

        for leaf_id, _ in leaf_infos:
            cond = _path_condition(tr, leaf_id)
            m = mask_from_condition(df, cond)
            obs = int(m.sum())
            if obs < min_hit:
                continue
            bad = int(df.loc[m, target_col].sum())
            br = bad / obs if obs else 0.0
            lf = br / base_bad_rate if base_bad_rate > 0 else 0.0
            if lf < min_lift:
                continue
            rows.append({
                "source": "rf",
                "var": None,
                "op": None,
                "thr": None,
                "obs": obs,
                "obs_rate": obs / len(df) if len(df) else 0.0,
                "bad": bad,
                "bad_rate": br,
                "lift": lf,
                "label2": "Y",
                "label3": "Y",
                "condition": cond,
            })

    if not rows:
        return pd.DataFrame(columns=["source","var","op","thr","obs","obs_rate","bad","bad_rate","lift","label2","label3","condition"])

    out = pd.DataFrame(rows).drop_duplicates(subset=["condition"]).sort_values("lift", ascending=False)
    if len(out) > max_rules:
        out = out.head(max_rules).copy()
    return out.reset_index(drop=True)


# -----------------------------
# DT rule mining (optional)
# -----------------------------
def dt_rule_mining(
    train: pd.DataFrame,
    feature_cols: List[str],
    *,
    target_col: str = "tar",
    max_depth: int = 2,
    max_iter: int = 64,
    min_samples_leaf: int = 80,
    random_state: int = 42,
    min_hit: int = 80,
    min_lift: float = 1.15,
    max_samples: float = 0.5,
    max_rules: int = 500,
    remove_top_feature_each_iter: bool = True,
    show_progress: bool = True,
    progress_desc: str = "DT mining",
) -> pd.DataFrame:
    """
    Mine short conjunction rules from shallow DecisionTree leaf paths and pre-filter by lift on TRAIN.

    This is a candidate generator (short rules) and complements threshold/RF mining.
    Threshold text is rounded via _fmt_thr (default 4 decimals) for stable de-duplication.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier
    except Exception as e:
        raise ImportError("scikit-learn is required for dt_rule_mining()") from e

    df = train.copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["source","var","op","thr","obs","obs_rate","bad","bad_rate","lift","label2","label3","condition"])

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    y = df[target_col].astype(int).values
    base_br = segment_stats(df, target_col, None).bad_rate  # baseline count bad rate

    def _eval_cond(cond: str) -> Tuple[int, int, float, float]:
        mask = mask_from_condition(df, cond)
        obs = int(mask.sum())
        if obs <= 0:
            return 0, 0, 0.0, 0.0
        bad = int(df.loc[mask, target_col].sum())
        br = bad / obs if obs else 0.0
        lf = _safe_div(br, base_br)
        return obs, bad, br, lf

    def _leaf_condition(tree, leaf_id: int, feat_cols: List[str]) -> str:
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold

        parent = {0: -1}
        is_left = {}
        stack = [0]
        while stack:
            nid = stack.pop()
            cl, cr = children_left[nid], children_right[nid]
            if cl != -1:
                parent[cl] = nid
                is_left[cl] = True
                stack.append(cl)
            if cr != -1:
                parent[cr] = nid
                is_left[cr] = False
                stack.append(cr)

        lb: Dict[int, float] = {}
        ub: Dict[int, float] = {}
        node = leaf_id
        while parent.get(node, -1) != -1:
            p = parent[node]
            fid = int(feature[p])
            thr = float(threshold[p])
            if is_left.get(node, False):
                ub[fid] = min(ub.get(fid, float("inf")), thr)
            else:
                lb[fid] = max(lb.get(fid, float("-inf")), thr)
            node = p

        terms = []
        for fid in sorted(set(lb.keys()) | set(ub.keys())):
            if fid < 0 or fid >= len(feat_cols):
                continue
            col = feat_cols[fid]
            if fid in lb and math.isfinite(lb[fid]):
                terms.append(f"({_safe_col(col)} > {_fmt_thr(lb[fid])})")
            if fid in ub and math.isfinite(ub[fid]):
                terms.append(f"({_safe_col(col)} <= {_fmt_thr(ub[fid])})")
        return " & ".join(terms) if terms else "True"

    rows: List[Dict[str, Any]] = []
    feat_pool = list(feature_cols)

    for it in _progress(range(int(max_iter)), total=int(max_iter), desc=progress_desc, show=show_progress):
        if len(rows) >= int(max_rules):
            break
        if not feat_pool:
            break

        clf = DecisionTreeClassifier(
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state + it),
        )
        clf.fit(X[feat_pool], y)

        tree = clf.tree_
        leaf_ids = np.where((tree.children_left == -1) & (tree.children_right == -1))[0].tolist()

        for leaf_id in leaf_ids:
            if len(rows) >= int(max_rules):
                break
            cond = _leaf_condition(tree, int(leaf_id), feat_pool)
            if not cond or cond.strip().lower() == "true":
                continue
            obs, bad, br, lf = _eval_cond(cond)
            if obs < int(min_hit):
                continue
            if (obs / len(df)) > float(max_samples):
                continue
            if lf < float(min_lift):
                continue

            rows.append({
                "source": "dt",
                "var": None,
                "op": None,
                "thr": None,
                "obs": obs,
                "obs_rate": obs / len(df) if len(df) else 0.0,
                "bad": bad,
                "bad_rate": br,
                "lift": lf,
                "label2": "Y",
                "label3": "Y",
                "condition": cond,
            })

        if remove_top_feature_each_iter and len(feat_pool) > 1:
            imps = getattr(clf, "feature_importances_", None)
            if imps is not None and len(imps) == len(feat_pool):
                top_i = int(np.argmax(imps))
                feat_pool.pop(top_i)

    if not rows:
        return pd.DataFrame(columns=["source","var","op","thr","obs","obs_rate","bad","bad_rate","lift","label2","label3","condition"])

    out = pd.DataFrame(rows).drop_duplicates(subset=["condition"]).sort_values("lift", ascending=False)
    if len(out) > int(max_rules):
        out = out.head(int(max_rules)).copy()
    return out.reset_index(drop=True)

# -----------------------------
# Greedy OR-union selection
# -----------------------------
@dataclass
class UnionPick:
    picked_conditions: List[str]
    report_total: pd.DataFrame


def greedy_or_union(
    train: pd.DataFrame,
    oot: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    target_col: str,
    amount_col: str,
    base_lift: float = 1.10,
    delta_lift: Optional[float] = None,
    delta_money_rate: float = 0.01,
    union_metric_mode: str = "money",
    delta_cnt_rate: Optional[float] = None,
    base_cnt_lift: Optional[float] = None,
    delta_cnt_lift: Optional[float] = None,
    profit_col: Optional[str] = None,
    profit_scale: Optional[float] = None,
    print_profit: bool = False,
    verbose: bool = True,
    # Whether to print the final union report (in addition to returning it).
    # NOTE: This is intentionally separate from `verbose`.
    print_final_summary: bool = True,
    max_picks: int = 50,
    selection_mode: str = "lift",
    var_priority: Optional[List[str]] = None,
    var_priority_mode: str = "min",
    show_progress: bool = True,
    progress_desc: str = "Greedy selection",
) -> UnionPick:
    """
    Greedy OR-union selection.

    Modes
    -----
    - selection_mode="lift" (recommended for speed):
        Sort candidates by their precomputed TRAIN lift (candidates['lift']) descending,
        then scan once and add a rule if it passes:
          * union money_lift >= base_lift on BOTH (train and oot)
          * constraints are enforced per union_metric_mode ("money" / "count" / "both"):
              - coverage (incremental):
                  * money_rate >= delta_money_rate (amount-based)
                  * cnt_rate   >= delta_cnt_rate_eff (count-based; defaults to delta_money_rate if delta_cnt_rate is None)
              - lift (incremental, optional):
                  * money_lift >= delta_lift
                  * cnt_lift   >= delta_cnt_lift_eff
              - lift (overall):
                  * money_lift >= base_lift
                  * cnt_lift   >= base_cnt_lift_eff
            All constraints are checked on BOTH (train and oot).
        Complexity is ~O(N).

    - selection_mode="single_var_first":
        Same as "lift", but scan all single-variable rules first (source=="threshold" OR var/op/thr not null),
        then scan multi-variable rules. This tends to produce more interpretable rule sets.

    - selection_mode="union_score":
        The exhaustive greedy that, at each step, searches all remaining rules and picks the one
        that maximizes total money_lift subject to the same constraints. Complexity is ~O(N^2).

    - selection_mode="var_order":
        Order candidates by a user-specified variable priority list (var_priority), then scan once (like "lift").
        This is useful when you want the greedy to preferentially pick rules involving certain key variables first.
        For multi-variable rules, the rule-level priority is determined by var_priority_mode:
          * "min": use the best (earliest) variable rank among variables in the rule (default, recommended)
          * "max": use the worst (latest) variable rank among variables in the rule

    - selection_mode="var_order_single_var_first":
        Same as "var_order", but still scans all single-variable rules first, then multi-variable rules.

    Notes
    -----
    candidates must contain at least:
      - 'condition'
      - optionally 'label2' (we keep label2=='Y')
      - optionally 'lift' (used for ordering in 'lift'/'single_var_first' modes)
      - optionally 'source' (used for single_var_first ordering)

      - optionally 'var','op','thr' (preferred for identifying single-variable rules)
      - otherwise, variables are parsed from 'condition' (pandas query expression; backticks are supported)
      - var_priority/var_priority_mode are used only for 'var_order*' modes
    """
    if len(candidates) == 0:
        return UnionPick(picked_conditions=[], report_total=pd.DataFrame())

    cands = candidates.copy()
    if "label2" in cands.columns:
        cands = cands[cands["label2"] == "Y"].copy()
    if len(cands) == 0:
        return UnionPick(picked_conditions=[], report_total=pd.DataFrame())

    selection_mode = (selection_mode or "lift").lower().strip()
    if selection_mode not in {"lift", "single_var_first", "union_score", "var_order", "var_order_single_var_first"}:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")


    if delta_lift is None:
        delta_lift = base_lift

    # Which metric(s) to enforce during greedy OR-union selection:
    # - "money": enforce amount-based coverage/lift (default; backward compatible)
    # - "count": enforce count-based coverage/lift
    # - "both": enforce BOTH amount-based and count-based constraints (strictest)
    union_metric_mode = (union_metric_mode or "money").lower().strip()
    if union_metric_mode not in {"money", "count", "both"}:
        raise ValueError(f"Unsupported union_metric_mode: {union_metric_mode}")

    # Effective thresholds for COUNT constraints (fallback to money thresholds if not provided)
    delta_cnt_rate_eff = float(delta_money_rate) if delta_cnt_rate is None else float(delta_cnt_rate)
    base_cnt_lift_eff = float(base_lift) if base_cnt_lift is None else float(base_cnt_lift)
    delta_cnt_lift_eff: Optional[float]
    if delta_cnt_lift is None:
        delta_cnt_lift_eff = float(delta_lift) if delta_lift is not None else None
    else:
        delta_cnt_lift_eff = float(delta_cnt_lift)


    total = pd.concat([train, oot], axis=0, ignore_index=True)

    base_amt_train = float(train[amount_col].sum())
    base_amt_oot = float(oot[amount_col].sum())
    base_amt_total = float(total[amount_col].sum())
    base_n_train = int(len(train))
    base_n_oot = int(len(oot))
    base_n_total = int(len(total))

    picked: List[str] = []
    best_mask_train = pd.Series(False, index=train.index)
    best_mask_oot = pd.Series(False, index=oot.index)
    best_mask_total = pd.concat(
        [best_mask_train.reset_index(drop=True), best_mask_oot.reset_index(drop=True)],
        ignore_index=True,
    )

    def _cnt_lift(df: pd.DataFrame, mask: pd.Series) -> float:
        base = segment_stats(df, target_col, amount_col).bad_rate
        fit_rate = segment_stats(df.loc[mask], target_col, amount_col).bad_rate
        return _safe_div(fit_rate, base)

    def _cnt_rate(base_n: int, mask: pd.Series) -> float:
        return _safe_div(int(mask.sum()), base_n)

    def _money_lift(df: pd.DataFrame, mask: pd.Series) -> float:
        base = segment_stats(df, target_col, amount_col).bad_amt_rate
        fit_rate = segment_stats(df.loc[mask], target_col, amount_col).bad_amt_rate
        return _safe_div(fit_rate, base)

    def _money_rate(base_amt: float, df: pd.DataFrame, mask: pd.Series) -> float:
        amt = float(df.loc[mask, amount_col].sum())
        return _safe_div(amt, base_amt)

    # Precompute rule masks on train/oot (and derived total) to avoid repeated df.query in loops
    conds = cands["condition"].tolist()
    masks_train = [mask_from_condition(train, c) for c in conds]
    masks_oot = [mask_from_condition(oot, c) for c in conds]
    masks_total = [
        pd.concat(
            [masks_train[i].reset_index(drop=True), masks_oot[i].reset_index(drop=True)],
            ignore_index=True,
        )
        for i in range(len(conds))
    ]

    def _is_single_var(i: int) -> bool:
        if "source" in cands.columns and str(cands["source"].iloc[i]) == "threshold":
            return True
        # fall back: presence of var/op/thr indicates a single variable threshold rule
        if all(col in cands.columns for col in ["var", "op", "thr"]):
            return pd.notna(cands["var"].iloc[i]) and pd.notna(cands["op"].iloc[i]) and pd.notna(cands["thr"].iloc[i])
        return False

    if verbose:
        report_condition(
            train, oot, "False",
            target_col=target_col, amount_col=amount_col,
            profit_col=profit_col, profit_scale=profit_scale,
            title="[BASELINE]",
            print_profit=(print_profit and profit_col is not None),
            do_print=True,
        )

    # ---- Fast scan-by-lift modes ----
    if selection_mode in {"lift", "single_var_first", "var_order", "var_order_single_var_first"}:
        idx = list(range(len(conds)))
        lift_vals = None
        if "lift" in cands.columns:
            # candidates['lift'] is computed on TRAIN; keep NaNs at the end
            lift_vals = pd.to_numeric(cands["lift"], errors="coerce").fillna(-np.inf).values

        # default: sort by lift desc
        if lift_vals is not None:
            idx.sort(key=lambda i: lift_vals[i], reverse=True)

        # variable-order priority sorting (before lift) when requested
        if selection_mode in {"var_order", "var_order_single_var_first"}:
            order_list = list(var_priority) if var_priority else []
            order_map = {v: k for k, v in enumerate(order_list)}
            big = 10**9

            def _rank_vars(vs: List[str]) -> int:
                if not order_map:
                    return big
                ranks = [order_map.get(v, big) for v in vs]
                if not ranks:
                    return big
                mode = (var_priority_mode or "min").lower().strip()
                return max(ranks) if mode == "max" else min(ranks)

            def _cand_vars(i: int) -> List[str]:
                if "source" in cands.columns and str(cands["source"].iloc[i]) == "threshold" and pd.notna(cands["var"].iloc[i]):
                    return [str(cands["var"].iloc[i])]
                cond = str(cands["condition"].iloc[i]) if "condition" in cands.columns else ""
                return _extract_vars_from_condition(cond)

            idx.sort(key=lambda i: (_rank_vars(_cand_vars(i)), -(lift_vals[i] if lift_vals is not None else 0.0)))

        if selection_mode in {"single_var_first", "var_order_single_var_first"}:
            idx_single = [i for i in idx if _is_single_var(i)]
            idx_multi = [i for i in idx if not _is_single_var(i)]
            idx = idx_single + idx_multi
        for i in _progress(idx, total=len(idx), desc=progress_desc, show=show_progress):
            if len(picked) >= max_picks:
                break
            cond = conds[i]
            if cond in picked:
                continue

            new_train = best_mask_train | masks_train[i]
            new_oot = best_mask_oot | masks_oot[i]
            new_total = pd.concat(
                [new_train.reset_index(drop=True), new_oot.reset_index(drop=True)],
                ignore_index=True,
            )

            inc_train = new_train & (~best_mask_train)
            inc_oot = new_oot & (~best_mask_oot)

                        # --- incremental coverage constraints ---
            if union_metric_mode in {"money", "both"}:
                if _money_rate(base_amt_train, train, inc_train) < float(delta_money_rate):
                    continue
                if _money_rate(base_amt_oot, oot, inc_oot) < float(delta_money_rate):
                    continue
            if union_metric_mode in {"count", "both"}:
                if _cnt_rate(base_n_train, inc_train) < float(delta_cnt_rate_eff):
                    continue
                if _cnt_rate(base_n_oot, inc_oot) < float(delta_cnt_rate_eff):
                    continue

            # --- incremental lift constraints (optional) ---
            if union_metric_mode in {"money", "both"} and delta_lift is not None:
                ml_inc_train = _money_lift(train, inc_train)
                ml_inc_oot = _money_lift(oot, inc_oot)
                if ml_inc_train < float(delta_lift) or ml_inc_oot < float(delta_lift):
                    continue
            if union_metric_mode in {"count", "both"} and delta_cnt_lift_eff is not None:
                cl_inc_train = _cnt_lift(train, inc_train)
                cl_inc_oot = _cnt_lift(oot, inc_oot)
                if cl_inc_train < float(delta_cnt_lift_eff) or cl_inc_oot < float(delta_cnt_lift_eff):
                    continue

            # --- overall lift constraints ---
            if union_metric_mode in {"money", "both"}:
                ml_train = _money_lift(train, new_train)
                ml_oot = _money_lift(oot, new_oot)
                if ml_train < float(base_lift) or ml_oot < float(base_lift):
                    continue
            if union_metric_mode in {"count", "both"}:
                cl_train = _cnt_lift(train, new_train)
                cl_oot = _cnt_lift(oot, new_oot)
                if cl_train < float(base_cnt_lift_eff) or cl_oot < float(base_cnt_lift_eff):
                    continue

            # accept
            picked.append(cond)
            best_mask_train = new_train
            best_mask_oot = new_oot
            best_mask_total = new_total

            if verbose:
                report_condition(
                    train, oot,
                    "(" + ") | (".join(picked) + ")",
                    target_col=target_col, amount_col=amount_col,
                    profit_col=profit_col, profit_scale=profit_scale,
                    title=f"[PICK {len(picked)}] add: {picked[-1]}",
                    print_profit=(print_profit and profit_col is not None),
                    do_print=True,
                )

    # ---- Exhaustive union_score mode ----
    else:
        for _ in _progress(range(max_picks), total=max_picks, desc=progress_desc, show=show_progress):
            best_i = None
            best_score = -1.0
            best_new_total = None
            best_new_oot = None
            best_new_train = None

            for i, cond in enumerate(conds):
                if cond in picked:
                    continue
                m_train_i = masks_train[i]
                m_oot_i = masks_oot[i]

                new_train = best_mask_train | m_train_i
                new_oot = best_mask_oot | m_oot_i
                new_total = pd.concat(
                    [new_train.reset_index(drop=True), new_oot.reset_index(drop=True)],
                    ignore_index=True,
                )

                inc_train = new_train & (~best_mask_train)
                inc_oot = new_oot & (~best_mask_oot)

                if _money_rate(base_amt_train, train, inc_train) < delta_money_rate:
                    continue
                if _money_rate(base_amt_oot, oot, inc_oot) < delta_money_rate:
                    continue

                if delta_lift is not None:
                    ml_inc_train = _money_lift(train, inc_train)
                    ml_inc_oot = _money_lift(oot, inc_oot)
                    if ml_inc_train < float(delta_lift) or ml_inc_oot < float(delta_lift):
                        continue

                ml_train = _money_lift(train, new_train)
                ml_oot = _money_lift(oot, new_oot)
                if ml_train < base_lift or ml_oot < base_lift:
                    continue

                ml_total = _money_lift(total, new_total)
                score = ml_total  # optimize on total
                if score > best_score:
                    best_score = score
                    best_i = i
                    best_new_total = new_total
                    best_new_oot = new_oot
                    best_new_train = new_train

            if best_i is None:
                break

            picked.append(conds[best_i])
            best_mask_total = best_new_total  # type: ignore
            best_mask_oot = best_new_oot      # type: ignore
            best_mask_train = best_new_train  # type: ignore

            if verbose:
                report_condition(
                    train, oot,
                    "(" + ") | (".join(picked) + ")",
                    target_col=target_col, amount_col=amount_col,
                    profit_col=profit_col, profit_scale=profit_scale,
                    title=f"[PICK {len(picked)}] add: {picked[-1]}",
                    print_profit=(print_profit and profit_col is not None),
                    do_print=True,
                )

    # final report
    final_cond = "False" if len(picked) == 0 else "(" + ") | (".join(picked) + ")"
    rep_total = report_condition(
        train, oot, final_cond,
        target_col=target_col, amount_col=amount_col,
        profit_col=profit_col, profit_scale=profit_scale,
        title="[GREEDY FINAL]",
        print_profit=(print_profit and profit_col is not None),
        do_print=print_final_summary,
    )
    return UnionPick(picked_conditions=picked, report_total=rep_total)


# Backwards-compatible alias (common naming in notebooks)
greedy_union_selection = greedy_or_union


# -----------------------------
# Excel reporting
# -----------------------------
def _rule_metrics_one(
    df: pd.DataFrame,
    mask: pd.Series,
    *,
    target_col: str,
    amount_col: str,
    profit_col: Optional[str] = None,
    profit_scale: Optional[float] = None,
    include_lift_ci: bool = False,
    lift_ci_level: float = 0.95,
    lift_ci_one_sided: bool = True,
    lift_ci_cc: float = 0.5,
) -> Dict[str, float]:
    base = segment_stats(df, target_col, amount_col, profit_col, profit_scale).as_dict()
    fit = segment_stats(df.loc[mask], target_col, amount_col, profit_col, profit_scale).as_dict()
    out: Dict[str, float] = {}
    out["base_n"] = base["n"]
    out["base_bad_rate"] = base["bad_rate"]
    out["base_amt"] = base["amt"]
    out["base_bad_amt_rate"] = base["bad_amt_rate"]
    if profit_col and "profit" in base:
        out["base_profit"] = base["profit"]

    out["fit_n"] = fit["n"]
    out["fit_bad_rate"] = fit["bad_rate"]
    out["fit_amt"] = fit["amt"]
    out["fit_bad_amt_rate"] = fit["bad_amt_rate"]
    if profit_col and "profit" in fit:
        out["fit_profit"] = fit["profit"]

    out["cnt_rate"] = _safe_div(out["fit_n"], out["base_n"])
    out["money_rate"] = _safe_div(out["fit_amt"], out["base_amt"])
    out["cnt_lift"] = _safe_div(out["fit_bad_rate"], out["base_bad_rate"])
    out["money_lift"] = _safe_div(out["fit_bad_amt_rate"], out["base_bad_amt_rate"])
    if include_lift_ci:
        lb_key = f"cnt_lift_lb_{int(round(lift_ci_level * 100))}"
        lb, se_log = _cnt_lift_lb_ci(
            k1=int(fit.get("bad", 0)),
            n1=int(fit.get("n", 0)),
            k0=int(base.get("bad", 0)),
            n0=int(base.get("n", 0)),
            ci_level=float(lift_ci_level),
            one_sided=bool(lift_ci_one_sided),
            cc=float(lift_ci_cc),
        )
        out["cnt_lift_se_log"] = se_log
        out[lb_key] = lb

    return out




# -----------------------------
# Excel field glossary (CN)
# -----------------------------
def _cn_dataset(prefix: str) -> str:
    return {"total": "全量", "train": "训练集", "oot": "OOT"}.get(prefix, prefix)

def _cn_segment(seg: str) -> str:
    return {"base": "基准总体", "fit": "命中组", "inc": "新增命中组", "remain": "剩余组"}.get(seg, seg)

def _cn_metric(metric: str) -> Tuple[str, str]:
    """
    Returns (cn_name, cn_desc_fragment) for metric part.
    """
    mapping = {
        "n": ("样本量", "样本数量"),
        "bad_rate": ("坏率", "坏样本占比（bad/obs）"),
        "amt": ("金额", "金额汇总（amount_col 求和）"),
        "bad_amt_rate": ("坏金额率", "坏金额/总金额"),
        "cnt_rate": ("样本占比", "命中样本量/基准样本量"),
        "money_rate": ("金额占比", "命中金额/基准金额"),
        "cnt_lift": ("坏率Lift", "命中坏率/基准坏率"),
        "money_lift": ("坏金额Lift", "命中坏金额率/基准坏金额率"),
        "bad_rate_delta": ("坏率差", "（组坏率 - 基准坏率）"),
        "bad_amt_rate_delta": ("坏金额率差", "（组坏金额率 - 基准坏金额率）"),
        "cnt_lift_se_log": ("log(Lift)标准误", "对 log(Lift) 做 delta method 估计的标准误"),
        "cnt_lift_lb_95": ("Lift下界", "Lift 的 95% 置信下界（单侧/双侧由配置决定）"),
        "step_delta_remain_bad_rate": ("剩余坏率改善量", "（上一步剩余坏率 - 本步剩余坏率）"),
        "step_delta_remain_bad_amt_rate": ("剩余坏金额率改善量", "（上一步剩余坏金额率 - 本步剩余坏金额率）"),
        "step_delta_remain_bad_rate_rel": ("剩余坏率改善比例", "（上一步剩余坏率 - 本步剩余坏率）/上一步剩余坏率"),
        "step_delta_remain_bad_amt_rate_rel": ("剩余坏金额率改善比例", "（上一步剩余坏金额率 - 本步剩余坏金额率）/上一步剩余坏金额率"),
        "step_delta_remain_cnt_lift": ("剩余坏率Lift改善量", "（上一步剩余坏率Lift - 本步剩余坏率Lift）"),
        "step_delta_remain_money_lift": ("剩余坏金额Lift改善量", "（上一步剩余坏金额Lift - 本步剩余坏金额Lift）"),
    }
    if metric in mapping:
        return mapping[metric]
    return (metric, "")

def _explain_column(col: str) -> Tuple[str, str]:
    """
    Map an English column name to (Chinese name, Chinese explanation).
    This is best-effort: unknown columns will be returned with a generic placeholder.
    """
    # Special columns
    special = {
        "lift_order": ("排序序号", "按 lift（或 sort_by 指定口径）排序后的序号"),
        "source": ("规则来源", "规则生成来源：threshold（单变量阈值）、rf（随机森林路径）、dt（决策树路径）等"),
        "rule_lift": ("候选Lift", "候选挖掘阶段计算的 lift（默认使用 train 口径，用于初步排序/筛选）"),
        "rule_obs": ("候选命中量", "候选挖掘阶段命中样本量（obs）"),
        "condition": ("规则条件", "pandas.query 兼容的规则表达式（用于命中样本）"),
        "step": ("步骤序号", "累计规则/已选规则中的第几步（按加入顺序）"),
        "added_condition": ("新增规则条件", "本步新加入的规则条件（OR-Union 中的一个子条件）"),
        "lift": ("Lift", "报表排序用的 lift 列（由 sort_by 映射而来）"),
        "lift_lb_95": ("Lift下界", "与 lift 同口径的 95% 置信下界（若启用 CI）"),
        "lift_pass": ("Lift是否通过", "是否满足 lift 门槛（CI 启用时用下界，否则用点估计）"),
        "train_cnt_lift_pass": ("训练集Lift通过", "训练集坏率Lift 是否通过门槛"),
        "oot_cnt_lift_pass": ("OOT Lift通过", "OOT 坏率Lift 是否通过门槛"),
        "total_cnt_lift_pass": ("全量Lift通过", "全量坏率Lift 是否通过门槛"),
        "key": ("键", "meta sheet 的键名（参数名）"),
        "value": ("值", "meta sheet 的值"),
    }
    if col in special:
        return special[col]

    # Pattern: <ds>_<seg>_<metric>
    m = re.match(r"^(total|train|oot)_(base|fit|inc|remain)_(.+)$", col)
    if m:
        ds, seg, metric = m.group(1), m.group(2), m.group(3)
        cn_m, cn_desc = _cn_metric(metric)
        cn_name = f"{_cn_dataset(ds)}{_cn_segment(seg)}{cn_m}"
        desc = f"{_cn_dataset(ds)}口径，{_cn_segment(seg)}的{cn_desc or cn_m}"
        return cn_name, desc

    # Pattern: <ds>_<metric>  (fit group derived ratios / lifts, or CI metrics)
    m2 = re.match(r"^(total|train|oot)_(.+)$", col)
    if m2:
        ds, metric = m2.group(1), m2.group(2)
        cn_m, cn_desc = _cn_metric(metric)
        if metric.startswith("remain_"):
            # already handled by m, but keep safe
            pass
        cn_name = f"{_cn_dataset(ds)}{cn_m}"
        # Heuristic: most of these are fit-group derived metrics (cnt_rate, cnt_lift, money_rate, ...)
        desc = f"{_cn_dataset(ds)}口径，{cn_desc or cn_m}"
        return cn_name, desc

    # Fallback
    return col, "（未在字段释义中定义，可按需要补充）"

def build_excel_field_glossary(
    dfs: Dict[str, pd.DataFrame],
    meta_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a CN glossary for all English column headers that appear in exported excel sheets.
    Optionally append meta keys (values of meta['key']) for parameter explanation purposes.
    """
    rows: List[Dict[str, Any]] = []
    col_to_sheets: Dict[str, List[str]] = {}
    for sname, df in dfs.items():
        if df is None:
            continue
        for c in df.columns:
            col_to_sheets.setdefault(c, []).append(sname)

    for col, sheets in sorted(col_to_sheets.items(), key=lambda x: x[0]):
        cn, desc = _explain_column(col)
        rows.append(
            {
                "type": "column",
                "field_en": col,
                "field_cn": cn,
                "description_cn": desc,
                "sheets": ",".join(sorted(set(sheets))),
            }
        )

    # Optional: meta keys dictionary (parameters) - useful for readers of the meta sheet
    if meta_keys:
        meta_dict = {
            "sort_by": ("排序口径", "报表排序使用的字段名（如 rule_lift / train_cnt_lift / total_cnt_lift 等）"),
            "top_n": ("TopN", "单规则报表最多保留的规则条数"),
            "include_profit": ("是否包含利润", "是否输出利润相关字段（需要提供 profit_col）"),
            "min_lift": ("最小Lift", "规则通过门槛：坏率Lift（或其置信下界）需 ≥ min_lift"),
            "include_lift_ci": ("是否启用Lift置信区间", "是否对 log(Lift) 计算标准误并构造置信下界过滤"),
            "lift_ci_level": ("置信水平", "Lift 置信区间的置信水平（如 0.95）"),
            "lift_ci_one_sided": ("是否单侧CI", "True：只计算下界（更偏准入过滤）；False：双侧区间"),
            "lift_ci_cc": ("连续性校正", "在命中坏样本等极端小样本时的平滑项（避免除零/过度波动）"),
            "selection_mode": ("选规模式", "规则集选择策略（如 lift / single_var_first / union_score 等）"),
        }
        for k in meta_keys:
            cn, desc = meta_dict.get(k, (k, "（未在参数释义中定义，可按需要补充）"))
            rows.append(
                {
                    "type": "meta_key",
                    "field_en": k,
                    "field_cn": cn,
                    "description_cn": desc,
                    "sheets": "meta",
                }
            )

    return pd.DataFrame(rows, columns=["type", "field_en", "field_cn", "description_cn", "sheets"])
def export_rule_report_excel(
    train: pd.DataFrame,
    oot: pd.DataFrame,
    rules: pd.DataFrame,
    *,
    target_col: str,
    amount_col: str,
    filepath: str,
    profit_col: Optional[str] = None,
    profit_scale: Optional[float] = None,
    sort_by: str = "rule_lift",
    top_n: Optional[int] = 200,
    include_profit: bool = False,
    min_lift: float = 1.15,
    include_lift_ci: bool = True,
    lift_ci_level: float = 0.95,
    lift_ci_one_sided: bool = True,
    lift_ci_cc: float = 0.5,
    picked_conditions: Optional[List[str]] = None,
    selection_mode: Optional[str] = None,
) -> str:
    """
    Output an Excel report suitable for detailed strategy review.

    Sheets
    ------
    - single_rules:
        Each rule's hit metrics on total/train/oot. Sorted by lift order (default: rule_lift).
    - cumulative_rules:
        Prefix-OR union metrics when accumulating rules in the same lift order as single_rules.
- residual_cumulative_rules:
        Residual portfolio metrics after EXCLUDING the cumulative hit set at each step (i.e., quality of remaining assets).
    - picked_rules (optional):
        Metrics for the finally picked rules (in pick order).
    - cumulative_picked (optional):
        Prefix-OR union metrics for picked rules.
- residual_cumulative_picked (optional):
        Residual portfolio metrics after EXCLUDING cumulative picked hits at each step.

    Notes
    -----
    - "rule_lift" comes from rules['lift'] (TRAIN lift used during candidate mining).
    - Profit columns are included only when include_profit=True AND profit_col provided.
    """
    total = pd.concat([train, oot], axis=0, ignore_index=True)

    rules2 = rules.copy()
    if "condition" not in rules2.columns:
        raise ValueError("rules must contain a 'condition' column")
    if "source" not in rules2.columns:
        rules2["source"] = ""

    # Keep only label2==Y by default (if present)
    if "label2" in rules2.columns:
        rules2 = rules2[rules2["label2"] == "Y"].copy()

    if top_n is not None and len(rules2) > top_n:
        rules2 = rules2.head(top_n).copy()

    if len(rules2) == 0:
        raise ValueError("No rules available for reporting (after label2/top_n filtering).")

    conds = rules2["condition"].tolist()
    masks_train = [mask_from_condition(train, c) for c in conds]
    masks_oot = [mask_from_condition(oot, c) for c in conds]
    masks_total = [
        pd.concat([masks_train[i].reset_index(drop=True), masks_oot[i].reset_index(drop=True)], ignore_index=True)
        for i in range(len(conds))
    ]

    prof = profit_col if (include_profit and profit_col is not None) else None

    # Precompute base segment stats (used by incremental and residual step deltas)
    base_stats = {
        "total": segment_stats(total, target_col, amount_col, prof, profit_scale).as_dict(),
        "train": segment_stats(train, target_col, amount_col, prof, profit_scale).as_dict(),
        "oot": segment_stats(oot, target_col, amount_col, prof, profit_scale).as_dict(),
    }

    def _append_inc_metrics(row: Dict[str, float], df: pd.DataFrame, inc_mask: pd.Series, ds: str) -> None:
        """Append incremental (newly covered) segment metrics for one step."""
        base = base_stats[ds]
        inc = segment_stats(df.loc[inc_mask], target_col, amount_col, prof, profit_scale).as_dict()

        row[f"{ds}_inc_n"] = inc["n"]
        row[f"{ds}_inc_bad_rate"] = inc["bad_rate"]
        row[f"{ds}_inc_amt"] = inc["amt"]
        row[f"{ds}_inc_bad_amt_rate"] = inc["bad_amt_rate"]
        if prof and "profit" in inc:
            row[f"{ds}_inc_profit"] = inc["profit"]

        row[f"{ds}_inc_cnt_rate"] = _safe_div(inc["n"], base["n"])
        row[f"{ds}_inc_money_rate"] = _safe_div(inc["amt"], base["amt"])
        row[f"{ds}_inc_cnt_lift"] = _safe_div(inc["bad_rate"], base["bad_rate"])
        row[f"{ds}_inc_money_lift"] = _safe_div(inc["bad_amt_rate"], base["bad_amt_rate"])

        # deltas vs base (absolute)
        row[f"{ds}_inc_bad_rate_delta"] = inc["bad_rate"] - base["bad_rate"]
        row[f"{ds}_inc_bad_amt_rate_delta"] = inc["bad_amt_rate"] - base["bad_amt_rate"]


    # --- single rules ---
    rows_single = []
    for i, cond in enumerate(conds):
        row = {
            "source": rules2["source"].iloc[i],
            "rule_lift": float(pd.to_numeric(rules2["lift"].iloc[i], errors="coerce")) if "lift" in rules2.columns else np.nan,
            "rule_obs": int(rules2["obs"].iloc[i]) if "obs" in rules2.columns and pd.notna(rules2["obs"].iloc[i]) else np.nan,
            "condition": cond,
        }
        mt = _rule_metrics_one(total, masks_total[i], target_col=target_col, amount_col=amount_col, profit_col=prof, profit_scale=profit_scale, include_lift_ci=include_lift_ci, lift_ci_level=lift_ci_level, lift_ci_one_sided=lift_ci_one_sided, lift_ci_cc=lift_ci_cc)
        mtr = _rule_metrics_one(train, masks_train[i], target_col=target_col, amount_col=amount_col, profit_col=prof, profit_scale=profit_scale, include_lift_ci=include_lift_ci, lift_ci_level=lift_ci_level, lift_ci_one_sided=lift_ci_one_sided, lift_ci_cc=lift_ci_cc)
        mo = _rule_metrics_one(oot, masks_oot[i], target_col=target_col, amount_col=amount_col, profit_col=prof, profit_scale=profit_scale, include_lift_ci=include_lift_ci, lift_ci_level=lift_ci_level, lift_ci_one_sided=lift_ci_one_sided, lift_ci_cc=lift_ci_cc)
        for k, v in mt.items():
            row[f"total_{k}"] = v
        for k, v in mtr.items():
            row[f"train_{k}"] = v
        for k, v in mo.items():
            row[f"oot_{k}"] = v
        rows_single.append(row)

    df_single = pd.DataFrame(rows_single)

    # Sorting preference
    if sort_by not in df_single.columns:
        # Backward compatible fallbacks
        for c in ["rule_lift", "train_cnt_lift", "total_cnt_lift", "train_money_lift", "total_money_lift"]:
            if c in df_single.columns:
                sort_by = c
                break
    df_single = df_single.sort_values(sort_by, ascending=False).reset_index(drop=True)
    df_single.insert(0, "lift_order", np.arange(1, len(df_single) + 1))

    # --- lift CI (delta method on log-lift) ---
    # Add compact columns for reviewers: lift / lift_lb_95 / lift_pass (based on TRAIN count-lift).
    if include_lift_ci:
        lb_suffix = f"cnt_lift_lb_{int(round(lift_ci_level * 100))}"
        train_lb_col = f"train_{lb_suffix}"
        if "train_cnt_lift" in df_single.columns:
            df_single["lift"] = df_single["train_cnt_lift"]
        elif "rule_lift" in df_single.columns:
            df_single["lift"] = df_single["rule_lift"]
        else:
            df_single["lift"] = np.nan

        if train_lb_col in df_single.columns:
            df_single["lift_lb_95"] = df_single[train_lb_col]
            df_single["lift_pass"] = df_single["lift_lb_95"].astype(float) >= float(min_lift)
        else:
            df_single["lift_lb_95"] = np.nan
            df_single["lift_pass"] = False

        # Keep per-dataset pass flags as well (useful when scanning TRAIN/OOT stability).
        for prefix in ("train", "oot", "total"):
            col = f"{prefix}_{lb_suffix}"
            if col in df_single.columns:
                df_single[f"{prefix}_cnt_lift_pass"] = df_single[col].astype(float) >= float(min_lift)


    # --- cumulative union in that order ---
    order = df_single["condition"].tolist()
    cond_to_idx = {c: i for i, c in enumerate(conds)}

    cum_train = pd.Series(False, index=train.index)
    cum_oot = pd.Series(False, index=oot.index)
    cum_total = pd.Series(False, index=total.index)

    cum_rows = []
    residual_rows = []
    prev_remain = {k: dict(v) for k, v in base_stats.items()}

    for step, cond in enumerate(order, 1):
        i = cond_to_idx[cond]

        # incremental coverage (newly hit by this rule)
        prev_cum_train = cum_train
        prev_cum_oot = cum_oot
        prev_cum_total = cum_total

        inc_train = masks_train[i] & (~prev_cum_train)
        inc_oot = masks_oot[i] & (~prev_cum_oot)
        inc_total = masks_total[i] & (~prev_cum_total)

        # update cumulative union
        cum_train = prev_cum_train | masks_train[i]
        cum_oot = prev_cum_oot | masks_oot[i]
        cum_total = prev_cum_total | masks_total[i]

        row = {"step": step, "added_condition": cond}
        mt = _rule_metrics_one(total, cum_total, target_col=target_col, amount_col=amount_col, profit_col=prof, profit_scale=profit_scale)
        mtr = _rule_metrics_one(train, cum_train, target_col=target_col, amount_col=amount_col, profit_col=prof, profit_scale=profit_scale)
        mo = _rule_metrics_one(oot, cum_oot, target_col=target_col, amount_col=amount_col, profit_col=prof, profit_scale=profit_scale)
        for k, v in mt.items():
            row[f"total_{k}"] = v
        for k, v in mtr.items():
            row[f"train_{k}"] = v
        for k, v in mo.items():
            row[f"oot_{k}"] = v

        # marginal / incremental metrics (new coverage at this step)
        _append_inc_metrics(row, total, inc_total, "total")
        _append_inc_metrics(row, train, inc_train, "train")
        _append_inc_metrics(row, oot, inc_oot, "oot")

        cum_rows.append(row)

        # residual (remaining after excluding cumulative hits)
        rrow = {"step": step, "added_condition": cond}

        def _remain(df: pd.DataFrame, cmask: pd.Series, prefix: str) -> None:
            base = base_stats[prefix]
            rem = segment_stats(df.loc[~cmask], target_col, amount_col, prof, profit_scale).as_dict()
            prev = prev_remain[prefix]

            rrow[f"{prefix}_remain_n"] = rem["n"]
            rrow[f"{prefix}_remain_bad_rate"] = rem["bad_rate"]
            rrow[f"{prefix}_remain_amt"] = rem["amt"]
            rrow[f"{prefix}_remain_bad_amt_rate"] = rem["bad_amt_rate"]
            if prof and "profit" in rem:
                rrow[f"{prefix}_remain_profit"] = rem["profit"]

            rrow[f"{prefix}_remain_cnt_rate"] = _safe_div(rem["n"], base["n"])
            rrow[f"{prefix}_remain_money_rate"] = _safe_div(rem["amt"], base["amt"])
            rrow[f"{prefix}_remain_cnt_lift"] = _safe_div(rem["bad_rate"], base["bad_rate"])
            rrow[f"{prefix}_remain_money_lift"] = _safe_div(rem["bad_amt_rate"], base["bad_amt_rate"])
            rrow[f"{prefix}_remain_bad_rate_delta"] = rem["bad_rate"] - base["bad_rate"]
            rrow[f"{prefix}_remain_bad_amt_rate_delta"] = rem["bad_amt_rate"] - base["bad_amt_rate"]

            # step deltas vs previous residual (improvement if positive)
            rrow[f"{prefix}_step_delta_remain_bad_rate"] = prev["bad_rate"] - rem["bad_rate"]
            rrow[f"{prefix}_step_delta_remain_bad_amt_rate"] = prev["bad_amt_rate"] - rem["bad_amt_rate"]
            rrow[f"{prefix}_step_delta_remain_bad_rate_rel"] = _safe_div(prev["bad_rate"] - rem["bad_rate"], prev["bad_rate"])
            rrow[f"{prefix}_step_delta_remain_bad_amt_rate_rel"] = _safe_div(prev["bad_amt_rate"] - rem["bad_amt_rate"], prev["bad_amt_rate"])

            prev_cnt_lift = _safe_div(prev["bad_rate"], base["bad_rate"])
            prev_money_lift = _safe_div(prev["bad_amt_rate"], base["bad_amt_rate"])
            rrow[f"{prefix}_step_delta_remain_cnt_lift"] = prev_cnt_lift - rrow[f"{prefix}_remain_cnt_lift"]
            rrow[f"{prefix}_step_delta_remain_money_lift"] = prev_money_lift - rrow[f"{prefix}_remain_money_lift"]

            # update prev residual
            prev_remain[prefix] = rem

        _remain(total, cum_total, "total")
        _remain(train, cum_train, "train")
        _remain(oot, cum_oot, "oot")
        residual_rows.append(rrow)


    df_cum = pd.DataFrame(cum_rows)
    df_residual = pd.DataFrame(residual_rows)

    # --- picked rules (optional) ---
    df_picked = None
    df_cum_picked = None
    if picked_conditions:
        pick_rows = []
        cum_p_tr = pd.Series(False, index=train.index)
        cum_p_oo = pd.Series(False, index=oot.index)
        cum_p_to = pd.Series(False, index=total.index)

        cum_pick_rows = []
        residual_picked_rows = []
        prev_remain_picked = {k: dict(v) for k, v in base_stats.items()}

        for step, cond in enumerate(picked_conditions, 1):
            mtr = mask_from_condition(train, cond)
            moo = mask_from_condition(oot, cond)
            mto = pd.concat([mtr.reset_index(drop=True), moo.reset_index(drop=True)], ignore_index=True)

            # single picked rule metrics
            row = {"step": step, "condition": cond}
            mt = _rule_metrics_one(total, mto, target_col=target_col, amount_col=amount_col, profit_col=prof, profit_scale=profit_scale)
            tr = _rule_metrics_one(train, mtr, target_col=target_col, amount_col=amount_col, profit_col=prof, profit_scale=profit_scale)
            oo = _rule_metrics_one(oot, moo, target_col=target_col, amount_col=amount_col, profit_col=prof, profit_scale=profit_scale)
            for k, v in mt.items():
                row[f"total_{k}"] = v
            for k, v in tr.items():
                row[f"train_{k}"] = v
            for k, v in oo.items():
                row[f"oot_{k}"] = v
            pick_rows.append(row)

            # incremental coverage for this step (excluding already-hit)
            prev_p_tr = cum_p_tr
            prev_p_oo = cum_p_oo
            prev_p_to = cum_p_to

            inc_tr = mtr & (~prev_p_tr)
            inc_oo = moo & (~prev_p_oo)
            inc_to = mto & (~prev_p_to)

            # update cumulative picked union
            cum_p_tr = prev_p_tr | mtr
            cum_p_oo = prev_p_oo | moo
            cum_p_to = prev_p_to | mto

            crow = {"step": step, "added_condition": cond}
            mtc = _rule_metrics_one(total, cum_p_to, target_col=target_col, amount_col=amount_col, profit_col=prof, profit_scale=profit_scale)
            trc = _rule_metrics_one(train, cum_p_tr, target_col=target_col, amount_col=amount_col, profit_col=prof, profit_scale=profit_scale)
            ooc = _rule_metrics_one(oot, cum_p_oo, target_col=target_col, amount_col=amount_col, profit_col=prof, profit_scale=profit_scale)
            for k, v in mtc.items():
                crow[f"total_{k}"] = v
            for k, v in trc.items():
                crow[f"train_{k}"] = v
            for k, v in ooc.items():
                crow[f"oot_{k}"] = v

            # marginal / incremental metrics (new coverage at this step)
            _append_inc_metrics(crow, total, inc_to, "total")
            _append_inc_metrics(crow, train, inc_tr, "train")
            _append_inc_metrics(crow, oot, inc_oo, "oot")

            cum_pick_rows.append(crow)

            # residual (remaining after excluding cumulative picked hits)
            pr = {"step": step, "added_condition": cond}

            def _remain(df: pd.DataFrame, cmask: pd.Series, prefix: str) -> None:
                base = base_stats[prefix]
                rem = segment_stats(df.loc[~cmask], target_col, amount_col, prof, profit_scale).as_dict()
                prev = prev_remain_picked[prefix]

                pr[f"{prefix}_remain_n"] = rem["n"]
                pr[f"{prefix}_remain_bad_rate"] = rem["bad_rate"]
                pr[f"{prefix}_remain_amt"] = rem["amt"]
                pr[f"{prefix}_remain_bad_amt_rate"] = rem["bad_amt_rate"]
                if prof and "profit" in rem:
                    pr[f"{prefix}_remain_profit"] = rem["profit"]

                pr[f"{prefix}_remain_cnt_rate"] = _safe_div(rem["n"], base["n"])
                pr[f"{prefix}_remain_money_rate"] = _safe_div(rem["amt"], base["amt"])
                pr[f"{prefix}_remain_cnt_lift"] = _safe_div(rem["bad_rate"], base["bad_rate"])
                pr[f"{prefix}_remain_money_lift"] = _safe_div(rem["bad_amt_rate"], base["bad_amt_rate"])
                pr[f"{prefix}_remain_bad_rate_delta"] = rem["bad_rate"] - base["bad_rate"]
                pr[f"{prefix}_remain_bad_amt_rate_delta"] = rem["bad_amt_rate"] - base["bad_amt_rate"]

                # step deltas vs previous residual (improvement if positive)
                pr[f"{prefix}_step_delta_remain_bad_rate"] = prev["bad_rate"] - rem["bad_rate"]
                pr[f"{prefix}_step_delta_remain_bad_amt_rate"] = prev["bad_amt_rate"] - rem["bad_amt_rate"]
                pr[f"{prefix}_step_delta_remain_bad_rate_rel"] = _safe_div(prev["bad_rate"] - rem["bad_rate"], prev["bad_rate"])
                pr[f"{prefix}_step_delta_remain_bad_amt_rate_rel"] = _safe_div(prev["bad_amt_rate"] - rem["bad_amt_rate"], prev["bad_amt_rate"])

                prev_cnt_lift = _safe_div(prev["bad_rate"], base["bad_rate"])
                prev_money_lift = _safe_div(prev["bad_amt_rate"], base["bad_amt_rate"])
                pr[f"{prefix}_step_delta_remain_cnt_lift"] = prev_cnt_lift - pr[f"{prefix}_remain_cnt_lift"]
                pr[f"{prefix}_step_delta_remain_money_lift"] = prev_money_lift - pr[f"{prefix}_remain_money_lift"]

                prev_remain_picked[prefix] = rem

            _remain(total, cum_p_to, "total")
            _remain(train, cum_p_tr, "train")
            _remain(oot, cum_p_oo, "oot")
            residual_picked_rows.append(pr)


        df_picked = pd.DataFrame(pick_rows)
        df_cum_picked = pd.DataFrame(cum_pick_rows)
        df_residual_picked = pd.DataFrame(residual_picked_rows)

    # --- write excel ---
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_single.to_excel(writer, index=False, sheet_name="single_rules")
        df_cum.to_excel(writer, index=False, sheet_name="cumulative_rules")
        df_residual.to_excel(writer, index=False, sheet_name="residual_cumulative_rules")

        meta_rows = [
            {"key": "sort_by", "value": sort_by},
            {"key": "top_n", "value": top_n},
            {"key": "include_profit", "value": bool(prof)},
            {"key": "min_lift", "value": min_lift},
            {"key": "include_lift_ci", "value": include_lift_ci},
            {"key": "lift_ci_level", "value": lift_ci_level},
            {"key": "lift_ci_one_sided", "value": lift_ci_one_sided},
            {"key": "lift_ci_cc", "value": lift_ci_cc},
        ]
        if selection_mode:
            meta_rows.append({"key": "selection_mode", "value": selection_mode})
        meta = pd.DataFrame(meta_rows)
        meta.to_excel(writer, index=False, sheet_name="meta")

        if df_picked is not None:
            df_picked.to_excel(writer, index=False, sheet_name="picked_rules")
        if df_cum_picked is not None:
            df_cum_picked.to_excel(writer, index=False, sheet_name="cumulative_picked")
            df_residual_picked.to_excel(writer, index=False, sheet_name="residual_cumulative_picked")

        # --- field glossary (CN) ---
        dfs_for_glossary: Dict[str, pd.DataFrame] = {
            "single_rules": df_single,
            "cumulative_rules": df_cum,
            "residual_cumulative_rules": df_residual,
            "meta": meta,
        }
        if df_picked is not None:
            dfs_for_glossary["picked_rules"] = df_picked
        if df_cum_picked is not None and df_residual_picked is not None:
            dfs_for_glossary["cumulative_picked"] = df_cum_picked
            dfs_for_glossary["residual_cumulative_picked"] = df_residual_picked

        try:
            meta_keys = meta["key"].astype(str).tolist() if "key" in meta.columns else None
            df_glossary = build_excel_field_glossary(dfs_for_glossary, meta_keys=meta_keys)
            df_glossary.to_excel(writer, index=False, sheet_name="字段释义")
        except Exception:
            # Glossary is a convenience sheet; do not fail the main report if anything unexpected happens.
            pass


        wb = writer.book
        for sname in wb.sheetnames:
            ws = wb[sname]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            # basic width tuning for readability
            for col_cells in ws.columns:
                col_letter = col_cells[0].column_letter
                max_len = 0
                for cell in col_cells[:200]:  # cap scan for speed
                    try:
                        v = str(cell.value) if cell.value is not None else ""
                        max_len = max(max_len, len(v))
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(60, max(10, max_len + 2))

    return filepath


# -----------------------------
# End-to-end entry
# -----------------------------
@dataclass
class AutoMiningResult:
    candidates: pd.DataFrame
    picked_conditions: List[str]
    final_report: pd.DataFrame
    excel_report_path: Optional[str] = None


def run_auto_mining(
    train: pd.DataFrame,
    oot: pd.DataFrame,
    numeric_vars: List[str],
    *,
    time_col: Optional[str] = None,
    target_col: str = "tar",
    amount_col: str = "normal_sure_repay_amount",
    profit_col: Optional[str] = None,
    profit_scale: Optional[float] = None,

    # threshold mining params
    hit_num: int = 80,
    sub_div_bin: float = 0.2,
    min_num: int = 80,
    min_rate: float = 0.01,
    min_lift: float = 1.15,

    # RF mining (optional) - these rules are pre-filtered by lift BEFORE greedy selection
    use_rf_rules: bool = False,
    rf_feature_cols: Optional[List[str]] = None,
    rf_max_depth: int = 3,
    rf_n_estimators: int = 200,
    rf_min_samples_leaf: int = 80,
    rf_top_k_per_tree: int = 5,
    rf_max_rules: int = 500,

    # DT mining (optional) - shallow DecisionTree leaf paths as short conjunction rules
    use_dt_rules: bool = False,
    dt_feature_cols: Optional[List[str]] = None,
    dt_max_depth: int = 2,
    dt_max_iter: int = 64,
    dt_min_samples_leaf: int = 80,
    dt_random_state: int = 42,
    dt_max_samples: float = 0.5,
    dt_max_rules: int = 500,
    dt_remove_top_feature_each_iter: bool = True,

    # variable-priority ordering (optional, used when selection_mode starts with "var_order")
    var_priority: Optional[List[str]] = None,
    var_priority_mode: str = "min",

    # greedy union params
    base_lift: float = 1.10,
    delta_lift: Optional[float] = None,
    delta_money_rate: float = 0.01,
    union_metric_mode: str = "money",
    delta_cnt_rate: Optional[float] = None,
    base_cnt_lift: Optional[float] = None,
    delta_cnt_lift: Optional[float] = None,
    max_picks: int = 50,
    selection_mode: str = "lift",

    # printing / reporting
    verbose: bool = True,
    print_profit: bool = False,
    print_final_summary: bool = True,

    # excel report (optional)
    excel_report_path: Optional[str] = None,
    excel_sort_by: str = "train_cnt_lift",
    excel_top_n: Optional[int] = 200,
    excel_include_profit: bool = False,
    show_progress: bool = True,
) -> AutoMiningResult:
    """
    Full pipeline:
    1) optional time sorting
    2) threshold mining on TRAIN
    3) optional RF rule mining on TRAIN (pre-filter by lift to reduce search space)
    4) greedy OR-union selection with coverage and lift constraints
    5) optional Excel report (single rules + cumulative by lift order)
    """
    train_s = sort_by_time(train, time_col) if time_col else train.copy()
    oot_s = sort_by_time(oot, time_col) if time_col else oot.copy()

    # 1) threshold candidates
    c_thresh = mine_thresholds(
        train=train_s,
        numeric_vars=numeric_vars,
        target_col=target_col,
        min_num=min_num,
        min_rate=min_rate,
        hit_num=hit_num,
        min_lift=min_lift,
        sub_div_bin=sub_div_bin,
    )

    # 2) optional RF candidates (pre-filtered by lift & hit_num)
    c_rf = pd.DataFrame()
    if use_rf_rules:
        feats = rf_feature_cols if rf_feature_cols is not None else list(numeric_vars)
        c_rf = rf_rule_mining(
            train=train_s,
            feature_cols=feats,
            target_col=target_col,
            max_depth=rf_max_depth,
            n_estimators=rf_n_estimators,
            min_samples_leaf=rf_min_samples_leaf,
            top_k_per_tree=rf_top_k_per_tree,
            min_hit=hit_num,
            min_lift=min_lift,
            max_rules=rf_max_rules,
        )

    # 2b) optional DT candidates (pre-filtered by lift & hit_num)
    c_dt = pd.DataFrame()
    if use_dt_rules:
        feats_dt = dt_feature_cols if dt_feature_cols is not None else list(numeric_vars)
        c_dt = dt_rule_mining(
        train=train_s,
        feature_cols=feats_dt,
        target_col=target_col,
        max_depth=dt_max_depth,
        max_iter=dt_max_iter,
        min_samples_leaf=dt_min_samples_leaf,
        random_state=dt_random_state,
        min_hit=hit_num,
        min_lift=min_lift,
        max_samples=dt_max_samples,
        max_rules=dt_max_rules,
        remove_top_feature_each_iter=dt_remove_top_feature_each_iter,
        )

    cands = pd.concat([c_thresh, c_rf, c_dt], axis=0, ignore_index=True)
    if len(cands):
        cands = cands.sort_values("lift", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"[CANDIDATES] threshold={len(c_thresh)} rf={len(c_rf)} dt={len(c_dt)} total={len(cands)} (train pre-filter by lift/min_hit)")

    # 3) greedy union (search space already reduced by label2 + RF prefilter)
    union = greedy_or_union(
        train=train_s,
        oot=oot_s,
        candidates=cands,
        target_col=target_col,
        amount_col=amount_col,
        base_lift=base_lift,
        delta_lift=delta_lift,
        delta_money_rate=delta_money_rate,
        union_metric_mode=union_metric_mode,
        delta_cnt_rate=delta_cnt_rate,
        base_cnt_lift=base_cnt_lift,
        delta_cnt_lift=delta_cnt_lift,
        profit_col=profit_col,
        profit_scale=profit_scale,
        print_profit=(print_profit and profit_col is not None),
        verbose=verbose,
        print_final_summary=print_final_summary,
        max_picks=max_picks,
        selection_mode=selection_mode,
        var_priority=var_priority,
        var_priority_mode=var_priority_mode,
    )

    final_cond = "False" if len(union.picked_conditions) == 0 else "(" + ") | (".join(union.picked_conditions) + ")"
    if print_final_summary:
        # High-level pipeline summary (counts)
        try:
            n_vars = len(numeric_vars) if numeric_vars is not None else 0
        except Exception:
            n_vars = 0
        try:
            n_cands = int(len(cands))
            n_thresh = int(len(c_thresh))
            n_rf = int(len(c_rf))
            n_dt = int(len(c_dt))
        except Exception:
            n_cands = n_thresh = n_rf = n_dt = 0
        print("=" * 80)
        print("[PIPELINE SUMMARY]")
        print(f"Variables analyzed (numeric_vars): {n_vars}")
        print(f"Candidates generated: total={n_cands} (threshold={n_thresh}, rf={n_rf}, dt={n_dt})")
        print(f"Picked conditions: {len(union.picked_conditions)}")
        print("=" * 80)

    final_rep = report_condition(
        train=train_s,
        oot=oot_s,
        condition=final_cond,
        target_col=target_col,
        amount_col=amount_col,
        profit_col=profit_col,
        profit_scale=profit_scale,
        title="[FINAL SUMMARY]",
        print_profit=(print_profit and profit_col is not None),
        do_print=verbose,
    )

    excel_path_out = None
    if excel_report_path:
        excel_path_out = export_rule_report_excel(
            train=train_s,
            oot=oot_s,
            rules=cands,
            target_col=target_col,
            amount_col=amount_col,
            filepath=excel_report_path,
            profit_col=profit_col,
            profit_scale=profit_scale,
            sort_by=excel_sort_by,
            top_n=excel_top_n,
            include_profit=(excel_include_profit and profit_col is not None),
            min_lift=min_lift,
            picked_conditions=union.picked_conditions,
            selection_mode=selection_mode,
        )
        if verbose:
            print(f"[EXCEL] report saved to: {excel_path_out}")

    return AutoMiningResult(
        candidates=cands,
        picked_conditions=union.picked_conditions,
        final_report=final_rep,
        excel_report_path=excel_path_out,
    ) 
