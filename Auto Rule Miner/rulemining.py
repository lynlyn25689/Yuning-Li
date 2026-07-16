"""
RuleMiner: Automated Rule Discovery with Greedy Selection

KEY HIGHLIGHTS:
1. Iterative Decision Tree mining with feature dropout (generates diverse rules)
2. Greedy OR-Union selection with Lift confidence intervals
3. Train/OOT validation with incremental coverage constraints
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from statistics import NormalDist
import math
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# Part 1: Core Utilities
# =============================================================================

def mask_from_condition(df: pd.DataFrame, condition: str) -> pd.Series:
    """Convert pandas.query condition to boolean mask."""
    if not condition or condition.lower() in ("false", "(false)"):
        return pd.Series(False, index=df.index)
    if condition.lower() in ("true", "(true)"):
        return pd.Series(True, index=df.index)
    return pd.Series(df.index.isin(df.query(condition, engine="python").index), index=df.index)


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def _lift_ci_lower_bound(k1: int, n1: int, k0: int, n0: int, ci_level: float = 0.95) -> float:
    """Lift confidence lower bound using delta method (prevents small-sample overfitting)."""
    if n1 <= 0 or n0 <= 0:
        return float("nan")
    
    # Continuity correction
    cc = 0.5
    k1a, n1a = k1 + cc, n1 + 2 * cc
    k0a, n0a = k0 + cc, n0 + 2 * cc
    
    p1, p0 = k1a / n1a, k0a / n0a
    if p1 <= 0 or p0 <= 0:
        return float("nan")
    
    lift = p1 / p0
    se_log = math.sqrt((1 - p1) / (n1a * p1) + (1 - p0) / (n0a * p0))
    z = NormalDist().inv_cdf(ci_level)
    
    return math.exp(math.log(lift) - z * se_log)


@dataclass
class SegmentMetrics:
    n: int
    bad: int
    bad_rate: float
    amount: float
    bad_amount: float
    bad_amount_rate: float
    
    def lift(self, base_rate: float) -> float:
        return _safe_div(self.bad_rate, base_rate)


def compute_metrics(df: pd.DataFrame, target_col: str, amount_col: str) -> SegmentMetrics:
    """Compute segment metrics."""
    n = len(df)
    if n == 0:
        return SegmentMetrics(0, 0, 0.0, 0.0, 0.0, 0.0)
    
    bad = int(df[target_col].sum())
    amount = float(df[amount_col].sum())
    bad_amount = float(df.loc[df[target_col] == 1, amount_col].sum())
    
    return SegmentMetrics(
        n=n,
        bad=bad,
        bad_rate=bad / n,
        amount=amount,
        bad_amount=bad_amount,
        bad_amount_rate=bad_amount / amount if amount > 0 else 0.0,
    )


# =============================================================================
# Part 2: Rule Generation (Iterative DT with Feature Dropout)
# =============================================================================

def generate_dt_rules(
    train: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    max_depth: int = 2,
    max_iter: int = 20,
    min_samples_leaf: int = 80,
    min_lift: float = 1.15,
    min_hit: int = 80,
    max_coverage: float = 0.5,
    remove_top_feature: bool = True,
) -> pd.DataFrame:
    """
    Generate candidate rules using iterative DecisionTree with feature dropout.
    
    KEY INSIGHT: By removing the most important feature after each iteration,
    we force the model to explore different feature combinations, producing
    diverse and interpretable rules.
    
    Example:
        Iteration 1: Tree uses 'income' as root → generates rules with income
        Remove 'income' from feature pool
        Iteration 2: Tree uses 'debt_ratio' as root → generates rules with debt_ratio
        Remove 'debt_ratio' from feature pool
        Iteration 3: Tree uses 'age' as root → generates rules with age
        ...
    """
    df = train.copy()
    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    y = df[target_col].astype(int).values
    
    base_metrics = compute_metrics(df, target_col, "amount")  # amount_col not needed for count
    
    rules = []
    feat_pool = list(feature_cols)
    
    for iteration in range(max_iter):
        if len(feat_pool) < 2:  # Need at least 2 features for depth=2
            break
        
        # --- Train a shallow decision tree on current feature pool ---
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42 + iteration,
        )
        clf.fit(X[feat_pool], y)
        
        # --- Extract rules from leaf paths ---
        tree = clf.tree_
        leaf_ids = np.where((tree.children_left == -1) & (tree.children_right == -1))[0]
        
        for leaf_id in leaf_ids:
            # Build condition from tree path
            condition = _extract_path_condition(tree, leaf_id, feat_pool)
            if not condition or condition == "True":
                continue
            
            # Evaluate rule on training data
            mask = mask_from_condition(df, condition)
            obs = mask.sum()
            
            if obs < min_hit or obs / len(df) > max_coverage:
                continue
            
            metrics = compute_metrics(df.loc[mask], target_col, "amount")
            lift = metrics.lift(base_metrics.bad_rate)
            
            # Use confidence interval for robustness
            lift_ci = _lift_ci_lower_bound(
                metrics.bad, metrics.n,
                base_metrics.bad, base_metrics.n,
            )
            
            if lift_ci >= min_lift:
                rules.append({
                    "condition": condition,
                    "lift": lift,
                    "lift_ci_lb": lift_ci,
                    "obs": obs,
                    "bad_rate": metrics.bad_rate,
                    "features_used": len(_extract_vars(condition)),
                    "iteration": iteration,
                })
        
        # --- Remove the most important feature ---
        if remove_top_feature and len(feat_pool) > 1:
            importances = clf.feature_importances_
            top_idx = np.argmax(importances)
            removed = feat_pool.pop(top_idx)
            print(f"  Iteration {iteration+1}: Removed '{removed}' (importance={importances[top_idx]:.3f})")
    
    if not rules:
        return pd.DataFrame(columns=["condition", "lift", "lift_ci_lb", "obs", "bad_rate"])
    
    return pd.DataFrame(rules).drop_duplicates("condition").sort_values("lift_ci_lb", ascending=False)


def _extract_path_condition(tree, leaf_id: int, feature_cols: List[str]) -> str:
    """Extract AND-condition from a decision tree leaf path."""
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    
    # Build parent map
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
    
    # Traverse from leaf to root
    lb, ub = {}, {}
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
    
    # Build condition string
    terms = []
    for fid in sorted(set(lb.keys()) | set(ub.keys())):
        col = feature_cols[fid]
        if fid in lb and math.isfinite(lb[fid]):
            terms.append(f"({col} > {lb[fid]:.4f})")
        if fid in ub and math.isfinite(ub[fid]):
            terms.append(f"({col} <= {ub[fid]:.4f})")
    
    return " & ".join(terms) if terms else "True"


def _extract_vars(condition: str) -> List[str]:
    """Extract variable names from a condition string."""
    import re
    return re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)", condition)


# =============================================================================
# Part 3: Greedy Rule Selection (OR-Union)
# =============================================================================

@dataclass
class SelectionResult:
    selected_rules: List[str]
    step_metrics: pd.DataFrame
    final_metrics: SegmentMetrics


def greedy_select_rules(
    train: pd.DataFrame,
    oot: pd.DataFrame,
    candidates: pd.DataFrame,
    target_col: str = "target",
    amount_col: str = "amount",
    base_lift: float = 1.10,
    delta_lift: float = 1.10,
    delta_coverage: float = 0.01,
    max_rules: int = 10,
    use_lift_ci: bool = True,
    ci_level: float = 0.95,
) -> SelectionResult:
    """
    Greedy OR-Union selection with incremental constraints.
    
    KEY INSIGHTS:
    1. Each rule must improve portfolio lift on BOTH train and OOT
    2. Incremental coverage ensures each rule adds new samples
    3. Lift confidence intervals prevent small-sample overfitting
    """
    if len(candidates) == 0:
        return SelectionResult([], pd.DataFrame(), SegmentMetrics(0, 0, 0.0, 0.0, 0.0, 0.0))
    
    # Baseline metrics
    base_train = compute_metrics(train, target_col, amount_col)
    base_oot = compute_metrics(oot, target_col, amount_col)
    base_total = compute_metrics(pd.concat([train, oot]), target_col, amount_col)
    
    # Precompute masks
    conditions = candidates["condition"].tolist()
    train_masks = [mask_from_condition(train, c) for c in conditions]
    oot_masks = [mask_from_condition(oot, c) for c in conditions]
    
    # Greedy selection
    selected_rules = []
    train_union = pd.Series(False, index=train.index)
    oot_union = pd.Series(False, index=oot.index)
    step_records = []
    
    for step in range(max_rules):
        best_idx, best_score = None, -1.0
        
        for idx, cond in enumerate(conditions):
            if cond in selected_rules:
                continue
            
            new_train = train_union | train_masks[idx]
            new_oot = oot_union | oot_masks[idx]
            
            # Incremental segments
            inc_train = new_train & (~train_union)
            inc_oot = new_oot & (~oot_union)
            
            # --- Constraint 1: Incremental coverage ---
            inc_rate_train = inc_train.sum() / base_train.n
            inc_rate_oot = inc_oot.sum() / base_oot.n
            if inc_rate_train < delta_coverage or inc_rate_oot < delta_coverage:
                continue
            
            # --- Constraint 2: Incremental lift (with CI) ---
            inc_metrics_train = compute_metrics(train.loc[inc_train], target_col, amount_col)
            inc_metrics_oot = compute_metrics(oot.loc[inc_oot], target_col, amount_col)
            
            inc_lift_train = inc_metrics_train.lift(base_train.bad_rate)
            inc_lift_oot = inc_metrics_oot.lift(base_oot.bad_rate)
            
            if use_lift_ci:
                inc_lift_train = _lift_ci_lower_bound(
                    inc_metrics_train.bad, inc_metrics_train.n,
                    base_train.bad, base_train.n, ci_level
                )
                inc_lift_oot = _lift_ci_lower_bound(
                    inc_metrics_oot.bad, inc_metrics_oot.n,
                    base_oot.bad, base_oot.n, ci_level
                )
            
            if inc_lift_train < delta_lift or inc_lift_oot < delta_lift:
                continue
            
            # --- Constraint 3: Overall union lift ---
            union_train = compute_metrics(train.loc[new_train], target_col, amount_col)
            union_oot = compute_metrics(oot.loc[new_oot], target_col, amount_col)
            
            if (union_train.lift(base_train.bad_rate) < base_lift or 
                union_oot.lift(base_oot.bad_rate) < base_lift):
                continue
            
            # Score: total lift
            total_new = pd.concat([train.loc[new_train], oot.loc[new_oot]])
            union_total = compute_metrics(total_new, target_col, amount_col)
            score = union_total.lift(base_total.bad_rate)
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx is None:
            break
        
        # Accept rule
        cond = conditions[best_idx]
        selected_rules.append(cond)
        train_union = train_union | train_masks[best_idx]
        oot_union = oot_union | oot_masks[best_idx]
        
        # Record
        union_train = compute_metrics(train.loc[train_union], target_col, amount_col)
        union_oot = compute_metrics(oot.loc[oot_union], target_col, amount_col)
        step_records.append({
            "step": step + 1,
            "rule": cond,
            "train_lift": union_train.lift(base_train.bad_rate),
            "oot_lift": union_oot.lift(base_oot.bad_rate),
            "train_coverage": train_union.sum() / base_train.n,
            "oot_coverage": oot_union.sum() / base_oot.n,
        })
    
    final_total = pd.concat([train.loc[train_union], oot.loc[oot_union]])
    final_metrics = compute_metrics(final_total, target_col, amount_col)
    
    return SelectionResult(selected_rules, pd.DataFrame(step_records), final_metrics)


# =============================================================================
# Part 4: End-to-End Pipeline
# =============================================================================

def run_rule_mining(
    train: pd.DataFrame,
    oot: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    amount_col: str = "amount",
    # Rule generation params
    dt_max_iter: int = 20,
    dt_max_depth: int = 2,
    dt_min_samples_leaf: int = 80,
    dt_min_lift: float = 1.15,
    # Selection params
    base_lift: float = 1.10,
    delta_lift: float = 1.10,
    delta_coverage: float = 0.01,
    max_rules: int = 10,
    use_lift_ci: bool = True,
    verbose: bool = True,
) -> SelectionResult:
    """
    End-to-end rule mining pipeline.
    
    Pipeline:
    1. Generate candidate rules using iterative DT with feature dropout 
    2. Greedy select optimal subset with train/OOT validation
    """
    if verbose:
        print("=" * 70)
        print("STEP 1: Generating Candidate Rules (Iterative DT + Feature Dropout)")
        print("=" * 70)
    
    candidates = generate_dt_rules(
        train=train,
        feature_cols=feature_cols,
        target_col=target_col,
        max_iter=dt_max_iter,
        max_depth=dt_max_depth,
        min_samples_leaf=dt_min_samples_leaf,
        min_lift=dt_min_lift,
    )
    
    if verbose:
        print(f"\nGenerated {len(candidates)} candidate rules")
        if len(candidates) > 0:
            print("\nTop 5 candidates (by lift CI lower bound):")
            print(candidates[["condition", "lift_ci_lb", "obs", "bad_rate"]].head(5).to_string(index=False))
    
    if len(candidates) == 0:
        return SelectionResult([], pd.DataFrame(), SegmentMetrics(0, 0, 0.0, 0.0, 0.0, 0.0))
    
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: Greedy OR-Union Selection")
        print("=" * 70)
    
    result = greedy_select_rules(
        train=train,
        oot=oot,
        candidates=candidates,
        target_col=target_col,
        amount_col=amount_col,
        base_lift=base_lift,
        delta_lift=delta_lift,
        delta_coverage=delta_coverage,
        max_rules=max_rules,
        use_lift_ci=use_lift_ci,
    )
    
    if verbose:
        print(f"\nSelected {len(result.selected_rules)} rules:")
        for i, rule in enumerate(result.selected_rules, 1):
            print(f"  {i}. {rule}")
        print(f"\nFinal Lift: {result.final_metrics.lift(compute_metrics(pd.concat([train, oot]), target_col, amount_col).bad_rate):.2f}x")
    
    return result


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Full demonstration with synthetic data."""
    np.random.seed(42)
    n = 5000
    
    # Generate data with known patterns
    df = pd.DataFrame({
        "income": np.random.normal(50000, 20000, n),
        "debt_ratio": np.random.normal(0.3, 0.15, n),
        "age": np.random.normal(35, 10, n),
        "credit_score": np.random.normal(700, 50, n),
        "amount": np.random.uniform(100, 1000, n),
        "target": np.random.binomial(1, 0.08, n),
    })
    
    # Inject known patterns
    df.loc[df["income"] < 30000, "target"] = np.random.binomial(1, 0.25, sum(df["income"] < 30000))
    df.loc[df["debt_ratio"] > 0.5, "target"] = np.random.binomial(1, 0.22, sum(df["debt_ratio"] > 0.5))
    df.loc[df["credit_score"] < 620, "target"] = np.random.binomial(1, 0.20, sum(df["credit_score"] < 620))
    
    split = int(0.7 * n)
    train, oot = df.iloc[:split], df.iloc[split:]
    
    result = run_rule_mining(
        train=train,
        oot=oot,
        feature_cols=["income", "debt_ratio", "age", "credit_score"],
        target_col="target",
        amount_col="amount",
        dt_max_iter=15,
        dt_max_depth=2,
        dt_min_lift=1.15,
        base_lift=1.10,
        delta_lift=1.10,
        delta_coverage=0.01,
        max_rules=5,
        use_lift_ci=True,
        verbose=True,
    )


if __name__ == "__main__":
    demo()
