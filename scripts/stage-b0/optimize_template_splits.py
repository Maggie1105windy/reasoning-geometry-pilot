#!/usr/bin/env python3
"""Deterministic constrained optimizer for fine and mid template held-out splits."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from common_b02b import (
    OUT, PROVISIONAL, RELATIONS, SEED, SPLITS, cell_index, distribution,
    manifest_b02b, split_group_string, split_overlap, stable_int, write_json,
)


def encode(df: pd.DataFrame, group_col: str):
    group_values = sorted({item for value in df[group_col] for item in split_group_string(value)})
    group_index = {value: i for i, value in enumerate(group_values)}
    row_groups = [np.asarray([group_index[x] for x in split_group_string(value)], dtype=np.int16) for value in df[group_col]]
    max_len = max(len(x) for x in row_groups)
    padded = np.full((len(df), max_len), -1, dtype=np.int16)
    for i, groups in enumerate(row_groups):
        padded[i, : len(groups)] = groups
    cells = cell_index(df)
    incident = [[] for _ in group_values]
    support = np.zeros((len(group_values), 40), dtype=np.int32)
    for row_idx, groups in enumerate(row_groups):
        for group in groups:
            incident[int(group)].append(row_idx)
            support[int(group), int(cells[row_idx])] += 1
    return group_values, row_groups, padded, cells, incident, support


def row_assignments(owner: np.ndarray, padded: np.ndarray) -> np.ndarray:
    first = owner[padded[:, 0]]
    valid = first >= 0
    for j in range(1, padded.shape[1]):
        group = padded[:, j]
        present = group >= 0
        values = np.full(len(group), -9, dtype=np.int8)
        values[present] = owner[group[present]]
        valid &= (~present) | (values == first)
    return np.where(valid, first, -1).astype(np.int8)


def count_cells(assignments: np.ndarray, cells: np.ndarray) -> np.ndarray:
    counts = np.zeros((3, 40), dtype=np.int32)
    valid = assignments >= 0
    np.add.at(counts, (assignments[valid], cells[valid]), 1)
    return counts


def compatible_mask(owner: np.ndarray, padded: np.ndarray, split: int) -> tuple[np.ndarray, np.ndarray]:
    compatible = np.ones(padded.shape[0], dtype=bool)
    has_unassigned = np.zeros(padded.shape[0], dtype=bool)
    for j in range(padded.shape[1]):
        group = padded[:, j]
        present = group >= 0
        values = np.full(len(group), -9, dtype=np.int8)
        values[present] = owner[group[present]]
        compatible &= (~present) | (values == -1) | (values == split)
        has_unassigned |= present & (values == -1)
    return compatible, has_unassigned


def seed_partition(
    padded: np.ndarray,
    cells: np.ndarray,
    support: np.ndarray,
    eval_quota: int,
    restart: int,
) -> np.ndarray | None:
    owner = np.full(support.shape[0], -1, dtype=np.int8)
    quotas = np.asarray([5, eval_quota, eval_quota], dtype=np.int32)
    for _ in range(support.shape[0] * 3 + 100):
        assignments = row_assignments(owner, padded)
        counts = count_cells(assignments, cells)
        deficit_positions = [(s, c) for s in range(3) for c in range(40) if counts[s, c] < quotas[s]]
        if not deficit_positions:
            return owner
        choices = []
        masks = {}
        for split, cell in deficit_positions:
            compatible, unassigned = compatible_mask(owner, padded, split)
            rows = np.flatnonzero(compatible & unassigned & (cells == cell))
            masks[(split, cell)] = rows
            choices.append((len(rows), -cell // 8, stable_int("deficit", restart, split, cell), split, cell))
        _, _, _, split, cell = min(choices)
        rows = masks[(split, cell)]
        if len(rows) == 0:
            return None
        scored = []
        for row_idx in rows:
            groups = padded[row_idx]
            groups = groups[groups >= 0]
            new_groups = groups[owner[groups] == -1]
            reuse = int(sum(support[int(g), cell] for g in new_groups))
            global_use = int(sum(support[int(g)].sum() for g in new_groups))
            scored.append((
                len(new_groups), -reuse, -global_use,
                stable_int("row-choice", restart, split, cell, int(row_idx)), int(row_idx),
            ))
        row_idx = min(scored)[-1]
        groups = padded[row_idx]
        groups = groups[groups >= 0]
        if np.any((owner[groups] >= 0) & (owner[groups] != split)):
            raise AssertionError("compatibility bug")
        owner[groups[owner[groups] == -1]] = split
    return None


def assign_remaining(
    owner: np.ndarray,
    padded: np.ndarray,
    cells: np.ndarray,
    incident: list[list[int]],
    restart: int,
) -> np.ndarray:
    owner = owner.copy()
    unassigned = [g for g, value in enumerate(owner) if value < 0]
    unassigned.sort(key=lambda g: (-len(incident[g]), stable_int("remaining", restart, g)))
    for group in unassigned:
        scores = []
        rows = np.asarray(incident[group], dtype=int)
        for split in range(3):
            coherence = 0.0
            conflicts = 0
            for row_idx in rows:
                groups = padded[row_idx]
                groups = groups[groups >= 0]
                others = owner[groups]
                assigned = others[others >= 0]
                if len(assigned) and np.any(assigned != split):
                    conflicts += 1
                else:
                    coherence += 1.0 / max(1, len(groups))
            train_bonus = 0.05 * len(rows) if split == 0 else 0.0
            scores.append((-conflicts, coherence + train_bonus, -stable_int("remain-split", restart, group, split), split))
        owner[group] = max(scores)[-1]
    return owner


def improve(
    owner: np.ndarray,
    padded: np.ndarray,
    cells: np.ndarray,
    min_eval: int,
    restart: int,
) -> np.ndarray:
    def objective(candidate):
        assignments = row_assignments(candidate, padded)
        counts = count_cells(assignments, cells)
        sizes = np.bincount(assignments[assignments >= 0], minlength=3)
        return (
            int(min(counts[1].min(), counts[2].min())),
            int((counts > 0).sum()),
            int(counts[0].min()),
            int(sizes.sum()),
            -int(abs(sizes[1] - sizes[2])),
        )

    current = owner.copy()
    current_score = objective(current)
    order = sorted(range(len(owner)), key=lambda g: stable_int("local", restart, g))
    for _ in range(3):
        changed = False
        for group in order:
            original = int(current[group])
            best_owner, best_score = original, current_score
            for split in range(3):
                if split == original:
                    continue
                trial = current.copy()
                trial[group] = split
                score = objective(trial)
                if score[0] >= min_eval and score > best_score:
                    best_owner, best_score = split, score
            if best_owner != original:
                current[group] = best_owner
                current_score = best_score
                changed = True
        if not changed:
            break
    return current


def optimize(df: pd.DataFrame, group_col: str, label: str, restarts: int = 32):
    groups, row_groups, padded, cells, incident, support = encode(df, group_col)
    best = None
    successful_quota = None
    for quota in range(10, 4, -1):
        for restart in range(restarts):
            seeded = seed_partition(padded, cells, support, quota, restart + stable_int(label) % 10000)
            if seeded is None:
                continue
            owner = assign_remaining(seeded, padded, cells, incident, restart)
            assignments = row_assignments(owner, padded)
            counts = count_cells(assignments, cells)
            sizes = np.bincount(assignments[assignments >= 0], minlength=3)
            score = (
                int(min(counts[1].min(), counts[2].min())),
                int((counts > 0).sum()),
                int(counts[0].min()),
                int(sizes.sum()),
                -int(abs(sizes[1] - sizes[2])),
            )
            if best is None or score > best[0]:
                best = (score, owner.copy(), assignments.copy(), counts.copy())
                successful_quota = quota
        if best is not None and best[0][0] >= quota:
            break
    if best is None:
        raise RuntimeError(f"No feasible three-way partition for {label}")
    score, owner, assignments, counts = best
    selected = df.loc[assignments >= 0].copy()
    selected["split"] = [SPLITS[int(x)] for x in assignments[assignments >= 0]]
    group_owners = {groups[i]: SPLITS[int(owner[i])] for i in range(len(groups))}
    dist = distribution(selected)
    info = {
        "label": label,
        "group_column": group_col,
        "template_groups": len(groups),
        "assigned_group_counts": {s: int(sum(v == s for v in group_owners.values())) for s in SPLITS},
        "rows": int(len(selected)),
        "sizes": {s: int(selected["split"].eq(s).sum()) for s in SPLITS},
        "covered_cells": {s: int((dist.loc[dist["split"].eq(s), "count"] > 0).sum()) for s in SPLITS},
        "min_cells": {s: int(dist.loc[dist["split"].eq(s), "count"].min()) for s in SPLITS},
        "template_overlap": split_overlap(selected, group_col),
        "story_overlap": {
            "train_val": len(set(selected.loc[selected.split.eq("train"), "story_group_id"]) & set(selected.loc[selected.split.eq("val"), "story_group_id"])),
            "train_test": len(set(selected.loc[selected.split.eq("train"), "story_group_id"]) & set(selected.loc[selected.split.eq("test"), "story_group_id"])),
            "val_test": len(set(selected.loc[selected.split.eq("val"), "story_group_id"]) & set(selected.loc[selected.split.eq("test"), "story_group_id"])),
        },
        "successful_seed_quota": int(successful_quota),
        "objective": list(score),
        "all_40_cells": bool(all((dist.loc[dist["split"].eq(s), "count"] > 0).all() for s in SPLITS)),
    }
    return selected, info, group_owners, dist


def optimize_milp(df: pd.DataFrame, group_col: str, label: str):
    groups, row_groups, padded, cells, incident, support = encode(df, group_col)
    n_groups = len(groups)
    n_rows = len(df)
    x_offset = 0
    y_offset = 3 * n_groups
    t_index = y_offset + 3 * n_rows
    n_vars = t_index + 1

    def x_idx(split, group):
        return x_offset + split * n_groups + group

    def y_idx(split, row):
        return y_offset + split * n_rows + row

    row_i, col_i, values, lower, upper = [], [], [], [], []
    constraint = 0

    def add(entries, lb, ub):
        nonlocal constraint
        for col, value in entries:
            row_i.append(constraint)
            col_i.append(col)
            values.append(value)
        lower.append(lb)
        upper.append(ub)
        constraint += 1

    # Each template group belongs to exactly one split.
    for group in range(n_groups):
        add([(x_idx(split, group), 1.0) for split in range(3)], 1.0, 1.0)

    # y(row, split) is 1 iff every required path group belongs to that split.
    for row, required in enumerate(row_groups):
        k = len(required)
        for split in range(3):
            y = y_idx(split, row)
            for group in required:
                add([(y, 1.0), (x_idx(split, int(group)), -1.0)], -np.inf, 0.0)
            add(
                [(y, 1.0), *[(x_idx(split, int(group)), -1.0) for group in required]],
                1.0 - k, np.inf,
            )

    # Train must cover every cell; t is the common val/test minimum cell count.
    for cell in range(40):
        members = np.flatnonzero(cells == cell)
        add([(y_idx(0, int(row)), 1.0) for row in members], 5.0, np.inf)
        for split in (1, 2):
            add(
                [(y_idx(split, int(row)), 1.0) for row in members] + [(t_index, -1.0)],
                0.0, np.inf,
            )

    matrix = coo_matrix((values, (row_i, col_i)), shape=(constraint, n_vars)).tocsr()
    c = np.zeros(n_vars, dtype=float)
    c[y_offset:t_index] = -1.0
    c[t_index] = -100000.0
    integrality = np.zeros(n_vars, dtype=np.int8)
    integrality[:y_offset] = 1
    lb = np.zeros(n_vars, dtype=float)
    ub = np.ones(n_vars, dtype=float)
    ub[t_index] = 10000.0
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 300.0, "mip_rel_gap": 0.0, "disp": True},
    )
    if result.x is None:
        raise RuntimeError(f"MILP found no solution for {label}: {result.message}")
    owner = np.asarray([
        int(np.argmax([result.x[x_idx(split, group)] for split in range(3)]))
        for group in range(n_groups)
    ], dtype=np.int8)
    assignments = row_assignments(owner, padded)
    counts = count_cells(assignments, cells)
    selected = df.loc[assignments >= 0].copy()
    selected["split"] = [SPLITS[int(x)] for x in assignments[assignments >= 0]]
    group_owners = {groups[i]: SPLITS[int(owner[i])] for i in range(n_groups)}
    dist = distribution(selected)
    info = {
        "label": label,
        "group_column": group_col,
        "template_groups": n_groups,
        "assigned_group_counts": {s: int(sum(v == s for v in group_owners.values())) for s in SPLITS},
        "rows": int(len(selected)),
        "sizes": {s: int(selected["split"].eq(s).sum()) for s in SPLITS},
        "covered_cells": {s: int((dist.loc[dist["split"].eq(s), "count"] > 0).sum()) for s in SPLITS},
        "min_cells": {s: int(dist.loc[dist["split"].eq(s), "count"].min()) for s in SPLITS},
        "template_overlap": split_overlap(selected, group_col),
        "story_overlap": {
            "train_val": len(set(selected.loc[selected.split.eq("train"), "story_group_id"]) & set(selected.loc[selected.split.eq("val"), "story_group_id"])),
            "train_test": len(set(selected.loc[selected.split.eq("train"), "story_group_id"]) & set(selected.loc[selected.split.eq("test"), "story_group_id"])),
            "val_test": len(set(selected.loc[selected.split.eq("val"), "story_group_id"]) & set(selected.loc[selected.split.eq("test"), "story_group_id"])),
        },
        "milp_status": int(result.status),
        "milp_success": bool(result.success),
        "milp_message": str(result.message),
        "milp_objective": float(result.fun),
        "common_eval_min_from_t": float(result.x[t_index]),
        "all_40_cells": bool(all((dist.loc[dist["split"].eq(s), "count"] > 0).all() for s in SPLITS)),
    }
    return selected, info, group_owners, dist


def optimize_seed_milp(df: pd.DataFrame, group_col: str, label: str):
    groups, row_groups, padded, cells, incident, support = encode(df, group_col)
    n_groups = len(groups)
    attempts = []
    solution = None
    for quota in (5,):
        for pool_limit in (60, 120):
            pool = []
            for cell in range(40):
                members = np.flatnonzero(cells == cell).tolist()
                members.sort(key=lambda row: (
                    len(row_groups[row]),
                    -sum(int(support[int(g), cell]) for g in row_groups[row]),
                    stable_int("pool", label, quota, pool_limit, cell, row),
                ))
                pool.extend(members[:pool_limit])
            pool = sorted(set(pool))
            z_keys = [(split, row) for split in range(3) for row in pool]
            z_index = {key: 3 * n_groups + i for i, key in enumerate(z_keys)}
            n_vars = 3 * n_groups + len(z_keys)
            row_i, col_i, values, lower, upper = [], [], [], [], []
            constraint = 0

            def x_idx(split, group):
                return split * n_groups + group

            def add(entries, lb, ub):
                nonlocal constraint
                for col, value in entries:
                    row_i.append(constraint); col_i.append(col); values.append(value)
                lower.append(lb); upper.append(ub); constraint += 1

            # A group may be claimed by at most one split; unused groups stay free.
            for group in range(n_groups):
                add([(x_idx(split, group), 1.0) for split in range(3)], 0.0, 1.0)
            for split, row in z_keys:
                z = z_index[(split, row)]
                for group in row_groups[row]:
                    add([(z, 1.0), (x_idx(split, int(group)), -1.0)], -np.inf, 0.0)
            for split in range(3):
                required = 1 if split == 0 else quota
                for cell in range(40):
                    members = [row for row in pool if cells[row] == cell]
                    add([(z_index[(split, row)], 1.0) for row in members], required, np.inf)
            matrix = coo_matrix((values, (row_i, col_i)), shape=(constraint, n_vars)).tocsr()
            c = np.zeros(n_vars, dtype=float)
            result = milp(
                c=c,
                integrality=np.ones(n_vars, dtype=np.int8),
                bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
                constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
                options={"time_limit": 180.0, "mip_rel_gap": 0.0, "disp": True},
            )
            attempts.append({
                "quota": quota, "pool_limit": pool_limit, "pool_rows": len(pool),
                "status": int(result.status), "success": bool(result.success), "message": str(result.message),
            })
            if result.x is None:
                continue
            owner = np.full(n_groups, -1, dtype=np.int8)
            for group in range(n_groups):
                for split in range(3):
                    if result.x[x_idx(split, group)] > 0.5:
                        owner[group] = split
            owner = assign_remaining(owner, padded, cells, incident, stable_int(label, quota, pool_limit) % 10000)
            assignments = row_assignments(owner, padded)
            counts = count_cells(assignments, cells)
            if counts[0].min() < 1 or counts[1].min() < quota or counts[2].min() < quota:
                continue
            solution = (quota, pool_limit, result, owner, assignments, counts)
            break
        if solution is not None:
            break
    if solution is None:
        raise RuntimeError(f"No seed-MILP solution for {label}: {attempts}")
    quota, pool_limit, result, owner, assignments, counts = solution
    selected = df.loc[assignments >= 0].copy()
    selected["split"] = [SPLITS[int(x)] for x in assignments[assignments >= 0]]
    group_owners = {groups[i]: SPLITS[int(owner[i])] for i in range(n_groups)}
    dist = distribution(selected)
    info = {
        "label": label,
        "group_column": group_col,
        "template_groups": n_groups,
        "assigned_group_counts": {s: int(sum(v == s for v in group_owners.values())) for s in SPLITS},
        "rows": int(len(selected)),
        "sizes": {s: int(selected["split"].eq(s).sum()) for s in SPLITS},
        "covered_cells": {s: int((dist.loc[dist["split"].eq(s), "count"] > 0).sum()) for s in SPLITS},
        "min_cells": {s: int(dist.loc[dist["split"].eq(s), "count"].min()) for s in SPLITS},
        "template_overlap": split_overlap(selected, group_col),
        "story_overlap": {
            "train_val": len(set(selected.loc[selected.split.eq("train"), "story_group_id"]) & set(selected.loc[selected.split.eq("val"), "story_group_id"])),
            "train_test": len(set(selected.loc[selected.split.eq("train"), "story_group_id"]) & set(selected.loc[selected.split.eq("test"), "story_group_id"])),
            "val_test": len(set(selected.loc[selected.split.eq("val"), "story_group_id"]) & set(selected.loc[selected.split.eq("test"), "story_group_id"])),
        },
        "seed_quota": quota,
        "pool_limit": pool_limit,
        "milp_status": int(result.status),
        "milp_success": bool(result.success),
        "milp_message": str(result.message),
        "milp_objective": float(result.fun),
        "attempts": attempts,
        "all_40_cells": bool(all((dist.loc[dist["split"].eq(s), "count"] > 0).all() for s in SPLITS)),
    }
    return selected, info, group_owners, dist


def optimize_local_search(df: pd.DataFrame, group_col: str, label: str, restarts: int = 8, sweeps: int = 8):
    groups, row_groups, padded, cells, incident, support = encode(df, group_col)
    rng = np.random.default_rng(SEED + stable_int(label) % 100000)

    def evaluate(owner):
        assignments = row_assignments(owner, padded)
        counts = count_cells(assignments, cells)
        sizes = np.bincount(assignments[assignments >= 0], minlength=3)
        coverage = (counts > 0).sum(axis=1)
        eval_min = int(min(counts[1].min(), counts[2].min()))
        score = (
            min(eval_min, 5),
            int(coverage[0]),
            int(coverage.sum()),
            eval_min,
            int(np.minimum(counts[1:], 10).sum()),
            int(np.minimum(counts[0], 10).sum()),
            int(sizes.sum()),
            -int(abs(sizes[1] - sizes[2])),
        )
        return score, assignments, counts

    best = None
    for restart in range(restarts):
        owner = rng.choice(3, size=len(groups), p=[0.34, 0.33, 0.33]).astype(np.int8)
        score, assignments, counts = evaluate(owner)
        for sweep in range(sweeps):
            changed = False
            order = rng.permutation(len(groups))
            for group in order:
                original = int(owner[group])
                candidates = []
                for split in range(3):
                    if split == original:
                        trial_score = score
                    else:
                        owner[group] = split
                        trial_score = evaluate(owner)[0]
                    candidates.append((trial_score, -stable_int("coord", restart, sweep, int(group), split), split))
                owner[group] = original
                chosen = max(candidates)
                if chosen[0] > score:
                    owner[group] = chosen[-1]
                    score, assignments, counts = evaluate(owner)
                    changed = True
            if best is None or score > best[0]:
                best = (score, owner.copy(), assignments.copy(), counts.copy(), restart, sweep)
            if score[0] >= 5 and score[1] == 40:
                break
            if not changed:
                # Deterministic seeded perturbation escapes capped-count plateaus.
                for group in rng.choice(len(groups), size=min(7, len(groups)), replace=False):
                    owner[group] = rng.integers(0, 3)
                score, assignments, counts = evaluate(owner)
        if best is not None and best[0][0] >= 5 and best[0][1] == 40:
            break
    if best is None:
        raise RuntimeError(f"Local search produced no partition for {label}")
    # Escape single-group local optima with path-level moves: assigning every
    # required group of one query path together preserves the held-out contract.
    repair_owner = best[1].copy()
    repair_score, repair_assignments, repair_counts = evaluate(repair_owner)
    for repair_step in range(160):
        if repair_counts.min() >= 5:
            break
        deficits = sorted(
            [(int(repair_counts[s, c]), s, c) for s in range(3) for c in range(40) if repair_counts[s, c] < 5]
        )
        candidate_moves = []
        seen = set()
        for _, split, cell in deficits[:6]:
            cell_rows = np.flatnonzero(cells == cell).tolist()
            cell_rows.sort(key=lambda row: (
                len(row_groups[int(row)]),
                stable_int("repair-row", label, repair_step, split, cell, int(row)),
            ))
            for row in cell_rows[:120]:
                path_groups = tuple(sorted(int(x) for x in row_groups[int(row)]))
                key = (split, path_groups)
                if key in seen:
                    continue
                seen.add(key)
                trial = repair_owner.copy()
                trial[list(path_groups)] = split
                trial_score, _, _ = evaluate(trial)
                if trial_score > repair_score:
                    candidate_moves.append((
                        trial_score,
                        -len(path_groups),
                        -stable_int("path-repair", label, repair_step, split, cell, path_groups),
                        trial,
                    ))
        if not candidate_moves:
            break
        _, _, _, repair_owner = max(candidate_moves, key=lambda x: x[:3])
        repair_score, repair_assignments, repair_counts = evaluate(repair_owner)
    if repair_score > best[0]:
        best = (
            repair_score, repair_owner.copy(), repair_assignments.copy(), repair_counts.copy(),
            best[4], best[5],
        )
    score, owner, assignments, counts, best_restart, best_sweep = best
    selected = df.loc[assignments >= 0].copy()
    selected["split"] = [SPLITS[int(x)] for x in assignments[assignments >= 0]]
    group_owners = {groups[i]: SPLITS[int(owner[i])] for i in range(len(groups))}
    dist = distribution(selected)
    info = {
        "label": label,
        "group_column": group_col,
        "template_groups": len(groups),
        "assigned_group_counts": {s: int(sum(v == s for v in group_owners.values())) for s in SPLITS},
        "rows": int(len(selected)),
        "sizes": {s: int(selected["split"].eq(s).sum()) for s in SPLITS},
        "covered_cells": {s: int((dist.loc[dist["split"].eq(s), "count"] > 0).sum()) for s in SPLITS},
        "min_cells": {s: int(dist.loc[dist["split"].eq(s), "count"].min()) for s in SPLITS},
        "template_overlap": split_overlap(selected, group_col),
        "story_overlap": {
            "train_val": len(set(selected.loc[selected.split.eq("train"), "story_group_id"]) & set(selected.loc[selected.split.eq("val"), "story_group_id"])),
            "train_test": len(set(selected.loc[selected.split.eq("train"), "story_group_id"]) & set(selected.loc[selected.split.eq("test"), "story_group_id"])),
            "val_test": len(set(selected.loc[selected.split.eq("val"), "story_group_id"]) & set(selected.loc[selected.split.eq("test"), "story_group_id"])),
        },
        "search_method": "deterministic multi-start coordinate search over global group ownership",
        "objective": list(score),
        "best_restart": int(best_restart),
        "best_sweep": int(best_sweep),
        "path_repair_steps_applied": int(repair_step + 1),
        "all_40_cells": bool(all((dist.loc[dist["split"].eq(s), "count"] > 0).all() for s in SPLITS)),
    }
    return selected, info, group_owners, dist


def save_candidate(prefix: str, df: pd.DataFrame, info: dict, owners: dict, dist: pd.DataFrame) -> None:
    group_col = info["group_column"]
    for split in SPLITS:
        manifest_b02b(df.loc[df["split"].eq(split)], group_col).to_csv(
            PROVISIONAL / f"{prefix}_{split}.csv", index=False
        )
    dist.to_csv(PROVISIONAL / f"{prefix}_distribution.csv", index=False)
    write_json(PROVISIONAL / f"{prefix}_group_assignment.json", owners)
    write_json(OUT / f"{prefix}_feasibility.json", info)


def main() -> None:
    df = pd.read_parquet(PROVISIONAL / "template_candidates.parquet")
    surface = optimize_seed_milp(df, "query_path_surface_skeletons", "surface_skeleton")
    save_candidate("surface_template_heldout", *surface)
    mid = optimize_seed_milp(df, "query_path_mid_families", "mid_level_family")
    save_candidate("mid_template_heldout", *mid)
    print(json.dumps({"surface": surface[1], "mid": mid[1]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
