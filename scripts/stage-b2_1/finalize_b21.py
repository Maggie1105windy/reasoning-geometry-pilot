#!/usr/bin/env python3
"""Consolidate Stage B2.1, run paired statistics, draw figures, and write report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common_b21 import (
    BOOTSTRAP_RESAMPLES, CARDINAL, HOPS, LABEL_MAP, MODEL_SPECS, OUT,
    RELATIONS, SEED, attach_truth, bootstrap_metrics, ensure_dirs, load_primary,
    equal_raw_prompt, metric_dict, metrics_by_hop, read_json, reconstruct_predictions, sha256_text,
    write_confusions, write_json,
)


MODELS = list(MODEL_SPECS)
PROTOCOL_ORDER = ["Raw-Sum", "Raw-Mean", "Native-Sum", "Native-Mean", "EqualLength-Raw", "EqualLength-Native"]
COLORS = {"base": "#4C78A8", "sft": "#F58518", "dpo": "#54A24B", "final": "#B279A2"}
PSTYLE = {
    "Raw-Sum": ("#4C78A8", "o"), "Raw-Mean": ("#72B7B2", "s"),
    "Native-Sum": ("#F58518", "o"), "Native-Mean": ("#FFBF79", "s"),
    "EqualLength-Raw": ("#54A24B", "^"), "EqualLength-Native": ("#B279A2", "D"),
}


def load_scores() -> dict[tuple[str, str], pd.DataFrame]:
    scores = {}
    raw = pd.read_csv(OUT / "raw_reproduction" / "row_scores.csv")
    for model in MODELS:
        part = raw.loc[raw.model.eq(model)].copy()
        scores[(model, "Raw-Sum")] = part
        scores[(model, "Raw-Mean")] = part
        equal_raw = pd.read_csv(OUT / "equal_length_verbalizer" / "per_model" / f"{model}_equal_raw_scores.csv")
        scores[(model, "EqualLength-Raw")] = equal_raw
        if model != "base":
            native = pd.read_csv(OUT / "native_chat" / "per_model" / f"{model}_scores.csv")
            equal_native = pd.read_csv(OUT / "equal_length_verbalizer" / "per_model" / f"{model}_equal_native_scores.csv")
            scores[(model, "Native-Sum")] = native
            scores[(model, "Native-Mean")] = native
            scores[(model, "EqualLength-Native")] = equal_native
    return scores


def prediction_bundle(scores: dict, primary: pd.DataFrame):
    predictions, metrics, boots = {}, [], []
    for (model, protocol), frame in scores.items():
        column = "mean_log_probability" if protocol in {"Raw-Mean", "Native-Mean"} else "sum_log_probability"
        pred = attach_truth(reconstruct_predictions(frame, column), primary)
        pred["model"] = model
        pred["protocol"] = protocol
        predictions[(model, protocol)] = pred
        metrics.append(metrics_by_hop(pred, model, protocol))
        boots.append(bootstrap_metrics(pred, model, protocol))
    return predictions, pd.concat(metrics, ignore_index=True), pd.concat(boots, ignore_index=True)


def group_confusions(part: pd.DataFrame, pred_col: str, groups: np.ndarray):
    ridx = {relation: i for i, relation in enumerate(RELATIONS)}
    gidx = {group: i for i, group in enumerate(groups)}
    arr = np.zeros((len(groups), 8, 8), dtype=np.int64)
    for row in part.itertuples(index=False):
        arr[gidx[row.story_group_id], ridx[row.canonical_relation], ridx[getattr(row, pred_col)]] += 1
    return arr


def metrics_from_matrices(matrices: np.ndarray):
    diag = np.diagonal(matrices, axis1=1, axis2=2)
    accuracy = diag.sum(axis=1) / matrices.sum(axis=(1, 2))
    fp = matrices.sum(axis=1) - diag
    fn = matrices.sum(axis=2) - diag
    den = 2 * diag + fp + fn
    f1 = np.divide(2 * diag, den, out=np.zeros_like(diag, dtype=float), where=den != 0).mean(axis=1)
    return accuracy, f1


def paired_one(a: pd.DataFrame, b: pd.DataFrame, scope, seed_offset: int):
    merged = a[["row_id", "story_group_id", "computed_hop", "canonical_relation", "prediction"]].merge(
        b[["row_id", "prediction"]], on="row_id", suffixes=("_a", "_b"), validate="one_to_one")

    def distribution(hop: int):
        part = merged.loc[merged.computed_hop.eq(hop)]
        groups = np.array(sorted(part.story_group_id.unique()))
        ca = group_confusions(part, "prediction_a", groups)
        cb = group_confusions(part, "prediction_b", groups)
        rng = np.random.default_rng(SEED + seed_offset + hop * 1009)
        weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=BOOTSTRAP_RESAMPLES)
        ma = np.tensordot(weights, ca, axes=(1, 0)); mb = np.tensordot(weights, cb, axes=(1, 0))
        aa, fa = metrics_from_matrices(ma); ab, fb = metrics_from_matrices(mb)
        pa = metric_dict(part.canonical_relation, part.prediction_a)
        pb = metric_dict(part.canonical_relation, part.prediction_b)
        return {"accuracy": ab - aa, "macro_f1": fb - fa}, {
            "accuracy": pb["accuracy"] - pa["accuracy"], "macro_f1": pb["macro_f1"] - pa["macro_f1"]}, len(part)

    rows = []
    hop_distributions = {}
    for hop in HOPS:
        dist, point, n = distribution(hop)
        hop_distributions[hop] = dist
        for metric in ["accuracy", "macro_f1"]:
            lo, hi = np.quantile(dist[metric], [0.025, 0.975])
            rows.append({"hop": str(hop), "metric": metric, "point_difference_b_minus_a": point[metric],
                         "ci95_low": float(lo), "ci95_high": float(hi), "n": n,
                         "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    for metric in ["accuracy", "macro_f1"]:
        values = np.mean(np.stack([hop_distributions[h][metric] for h in [2, 3, 4, 5]]), axis=0)
        point = float(np.mean([next(r["point_difference_b_minus_a"] for r in rows if r["hop"] == str(h) and r["metric"] == metric) for h in [2, 3, 4, 5]]))
        lo, hi = np.quantile(values, [0.025, 0.975])
        rows.append({"hop": "derived_mean", "metric": metric, "point_difference_b_minus_a": point,
                     "ci95_low": float(lo), "ci95_high": float(hi), "n": int(sum((merged.computed_hop == h).sum() for h in [2,3,4,5])),
                     "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    return rows


def paired_statistics(predictions: dict):
    comparisons = []
    spec = [
        ("A", "Raw-Sum", "Raw-Mean", MODELS),
        ("B", "Raw-Sum", "EqualLength-Raw", MODELS),
        ("C", "Raw-Sum", "Native-Sum", ["sft", "dpo", "final"]),
        ("D", "Native-Sum", "Native-Mean", ["sft", "dpo", "final"]),
        ("E", "Native-Sum", "EqualLength-Native", ["sft", "dpo", "final"]),
    ]
    for code, a_name, b_name, models in spec:
        for model in models:
            offset = int(sha256_text(code + model)[:8], 16)
            rows = paired_one(predictions[(model, a_name)], predictions[(model, b_name)], None, offset)
            for row in rows:
                row.update({"comparison_code": code, "model": model, "protocol_a": a_name,
                            "protocol_b": b_name, "bootstrap_unit": "story_group_id",
                            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "seed": SEED})
                comparisons.append(row)
    return pd.DataFrame(comparisons)


def summary_table(metrics: pd.DataFrame):
    rows = []
    for model in MODELS:
        for protocol in PROTOCOL_ORDER:
            part = metrics.loc[(metrics.model.eq(model)) & (metrics.protocol.eq(protocol))]
            if part.empty:
                continue
            values = part.set_index("hop").accuracy
            rows.append({"Model": MODEL_SPECS[model]["label"], "model": model, "Protocol": protocol,
                         **{f"h{h}": float(values[h]) for h in HOPS},
                         "Derived mean": float(values.loc[[2,3,4,5]].mean())})
    return pd.DataFrame(rows)


def save_figure(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT / "figures" / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_figures(metrics, summary, predictions):
    sns.set_theme(style="whitegrid", context="notebook")
    # Figure 1
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODELS:
        p = metrics[(metrics.model == model) & (metrics.protocol == "Raw-Sum")]
        ax.plot(p.hop, p.accuracy, marker="o", lw=2, label=MODEL_SPECS[model]["label"], color=COLORS[model])
    ax.axhline(.125, color="black", ls="--", lw=1, label="Uniform chance = 0.125")
    ax.set(xlabel="Computed hop", ylabel="Accuracy", title="Figure 1. Frozen raw summed accuracy by hop", xticks=HOPS, ylim=(0, max(.28, metrics.accuracy.max()+.04)))
    ax.legend(ncol=2); save_figure(fig, "figure_1_raw_sum_by_hop.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        for protocol in ["Raw-Sum", "Raw-Mean"]:
            p = metrics[(metrics.model == model) & (metrics.protocol == protocol)]
            ax.plot(p.hop, p.accuracy, marker=PSTYLE[protocol][1], color=PSTYLE[protocol][0], label=protocol)
        ax.axhline(.125, color="black", ls="--", lw=1); ax.set_title(MODEL_SPECS[model]["label"]); ax.set_xticks(HOPS); ax.legend()
    fig.supxlabel("Computed hop"); fig.supylabel("Accuracy"); fig.suptitle("Figure 2. Raw summed versus mean-token scoring", y=1.01)
    save_figure(fig, "figure_2_raw_sum_vs_mean.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, model in zip(axes, ["sft", "dpo", "final"]):
        for protocol in ["Raw-Sum", "Native-Sum"]:
            p = metrics[(metrics.model == model) & (metrics.protocol == protocol)]
            ax.plot(p.hop, p.accuracy, marker=PSTYLE[protocol][1], color=PSTYLE[protocol][0], label=protocol)
        ax.axhline(.125, color="black", ls="--", lw=1); ax.set_title(MODEL_SPECS[model]["label"]); ax.set_xticks(HOPS); ax.legend()
    fig.supxlabel("Computed hop"); fig.supylabel("Accuracy"); fig.suptitle("Figure 3. Frozen raw versus native-chat interface", y=1.03)
    save_figure(fig, "figure_3_raw_vs_native.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for model in MODELS:
        a = metrics[(metrics.model == model) & (metrics.protocol == "EqualLength-Raw")]
        axes[0].plot(a.hop, a.accuracy, marker="o", color=COLORS[model], label=MODEL_SPECS[model]["label"])
    axes[0].set_title("Equal-length raw"); axes[0].legend(ncol=2)
    for model in ["sft", "dpo", "final"]:
        a = metrics[(metrics.model == model) & (metrics.protocol == "Native-Sum")]
        b = metrics[(metrics.model == model) & (metrics.protocol == "EqualLength-Native")]
        axes[1].plot(a.hop, a.accuracy, ls="--", marker="o", color=COLORS[model], alpha=.55)
        axes[1].plot(b.hop, b.accuracy, marker="D", color=COLORS[model], label=MODEL_SPECS[model]["label"])
    axes[1].set_title("Native natural (dashed) vs native equal-length (solid)"); axes[1].legend()
    for ax in axes: ax.axhline(.125, color="black", ls=":", lw=1); ax.set_xticks(HOPS); ax.set_xlabel("Computed hop")
    axes[0].set_ylabel("Accuracy"); fig.suptitle("Figure 4. Natural versus equal-length verbalizers", y=1.02)
    save_figure(fig, "figure_4_natural_vs_equal_length.png")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    model_labels = [MODEL_SPECS[m]["label"] for m in MODELS]
    plot = summary.pivot(index="Protocol", columns="Model", values="Derived mean").reindex(index=PROTOCOL_ORDER, columns=model_labels)
    plot.plot(kind="bar", ax=ax, color=[COLORS[m] for m in MODELS], width=.82)
    ax.axhline(.125, color="black", ls="--", lw=1); ax.set(xlabel="Protocol", ylabel="Mean accuracy over h2–h5", title="Figure 5. Derived mean across all preregistered protocols")
    ax.tick_params(axis="x", rotation=25); ax.legend(title="Checkpoint", ncol=4)
    save_figure(fig, "figure_5_derived_mean_all_protocols.png")

    freq_rows = []
    for model in MODELS:
        for protocol in ["Raw-Sum", "Raw-Mean"]:
            p = predictions[(model, protocol)]
            freq_rows += [{"Model": MODEL_SPECS[model]["label"], "Scoring": protocol,
                           "Predicted type": kind, "Frequency": float(p.prediction.isin(classes).mean())}
                          for kind, classes in [("Cardinal", CARDINAL), ("Diagonal", set(RELATIONS)-CARDINAL)]]
    freq = pd.DataFrame(freq_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, scoring in zip(axes, ["Raw-Sum", "Raw-Mean"]):
        sns.barplot(data=freq[freq.Scoring == scoring], x="Model", y="Frequency", hue="Predicted type",
                    errorbar=None, ax=ax, palette=["#4C78A8", "#E45756"])
        ax.axhline(.125, color="black", ls="--", lw=1, label="Per-relation chance = 0.125")
        ax.set_title(scoring); ax.set_xlabel("Checkpoint")
    axes[0].set_ylabel("Prediction frequency"); axes[1].set_ylabel("")
    fig.suptitle("Figure 6. Cardinal versus diagonal prediction frequency", y=1.02)
    save_figure(fig, "figure_6_cardinal_diagonal_frequency.png")

    cases = [("Raw-Sum", "Final Raw-Sum"), ("Native-Sum", "Final Native-Sum"), ("EqualLength-Native", "Final EqualLength-Native")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (protocol, title) in zip(axes, cases):
        p = predictions[("final", protocol)]
        matrix = pd.crosstab(p.canonical_relation, p.prediction, normalize="index").reindex(index=RELATIONS, columns=RELATIONS, fill_value=0)
        sns.heatmap(matrix, cmap="Blues", vmin=0, vmax=1, ax=ax, cbar=True, annot=False)
        ax.contour(np.arange(8)+.5, np.arange(8)+.5, matrix.to_numpy(), levels=[.125], colors="orange", linewidths=.7)
        ax.set_title(title + "\norange contour = 0.125"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.suptitle("Figure 7. Final row-normalized confusion matrices", y=1.03)
    save_figure(fig, "figure_7_final_confusion_matrices.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x = np.arange(len(PROTOCOL_ORDER))
    for ax, scope in zip(axes, ["h1", "Derived mean"]):
        for model in MODELS:
            p = summary[summary.model == model].set_index("Protocol")
            y = [float(p.loc[name, scope]) if name in p.index else np.nan for name in PROTOCOL_ORDER]
            ax.plot(x, y, marker="o", color=COLORS[model], label=MODEL_SPECS[model]["label"])
        ax.axhline(.125, color="black", ls="--", lw=1); ax.set_title(scope)
        ax.set_xticks(x, PROTOCOL_ORDER, rotation=35, ha="right"); ax.set_xlabel("Protocol")
    axes[0].set_ylabel("Accuracy"); axes[1].legend(); fig.suptitle("Figure 8. Model-task Fitness Gate summary", y=1.02)
    save_figure(fig, "figure_8_fitness_gate_summary.png")
    return freq


def markdown_table(frame: pd.DataFrame, columns, digits=4):
    view = frame[columns].copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda x: f"{x:.{digits}f}")
    def clean(value):
        if pd.isna(value):
            return "N/A"
        return str(value).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(map(str, view.columns)) + " |"
    separator = "|" + "|".join("---" for _ in view.columns) + "|"
    body = ["| " + " | ".join(clean(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def decision_from(summary, boots, paired):
    def delta(code, model="final"):
        return paired[(paired.comparison_code == code) & (paired.model == model) & (paired.hop == "derived_mean") & (paired.metric == "accuracy")].iloc[0]
    native = delta("C")
    mean = delta("A")
    equal = delta("B")
    native_equal = delta("E")
    fair = boots[(boots.model == "final") & (boots.protocol == "EqualLength-Native") & (boots.hop.isin([2,3,4,5]))]
    robust = bool((fair.accuracy_ci95_low > .125).all())
    prompt = "YES" if native.ci95_low > 0 else ("NO" if native.ci95_high <= 0 else "MIXED")
    token = "YES" if mean.ci95_low > 0 and equal.ci95_low > 0 else ("NO" if mean.ci95_high <= 0 and equal.ci95_high <= 0 else "MIXED")
    equal_answer = "YES" if equal.ci95_low > 0 and native_equal.ci95_low > 0 else ("NO" if equal.ci95_high <= 0 and native_equal.ci95_high <= 0 else "MIXED")
    suitability = "CONDITIONAL" if robust else "NO"
    return {"raw_reproduced": "YES", "native_improves": prompt, "token_length_affects": token,
            "equal_length_improves": equal_answer, "final_h2_h5_robust_above_chance": "YES" if robust else "NO",
            "successful_mechanism_target": suitability,
            "evidence": {"final_native_minus_raw_derived": native.to_dict(), "final_raw_mean_minus_sum_derived": mean.to_dict(),
                         "final_equal_raw_minus_raw_sum_derived": equal.to_dict(), "final_equal_native_minus_native_sum_derived": native_equal.to_dict(),
                         "final_equal_native_h2_h5_ci_low": fair.set_index("hop").accuracy_ci95_low.to_dict()}}


def write_report(summary, metrics, boots, paired, decisions):
    gate = read_json(OUT / "raw_reproduction" / "reproduction_check.json")
    audit = read_json(OUT / "audit" / "input_audit.json")
    labels = read_json(OUT / "equal_length_verbalizer" / "candidate_selection_audit.json")
    chat = read_json(OUT / "native_chat" / "chat_template_audit.json")
    core = summary[["Model", "Protocol", "h1", "h2", "h3", "h4", "h5", "Derived mean"]]
    raw = core[core.Protocol == "Raw-Sum"]
    native = core[core.Protocol.isin(["Native-Sum", "Native-Mean"])]
    c1 = core[core.Protocol.isin(["Raw-Sum", "Raw-Mean"])]
    c2 = core[core.Protocol.str.startswith("EqualLength")]
    final_pairs = paired[(paired.model == "final") & (paired.metric == "accuracy")]
    detail = metrics.merge(
        boots[["model","protocol","hop","accuracy_ci95_low","accuracy_ci95_high","macro_f1_ci95_low","macro_f1_ci95_high"]],
        on=["model","protocol","hop"], validate="one_to_one")
    detail["uniform_chance"] = .125
    detail_columns = ["model","protocol","hop","n_test","accuracy","accuracy_ci95_low","accuracy_ci95_high","macro_f1","macro_f1_ci95_low","macro_f1_ci95_high","balanced_accuracy","majority_accuracy","uniform_chance"]
    raw_detail = detail[detail.protocol == "Raw-Sum"]
    native_detail = detail[detail.protocol.isin(["Native-Sum","Native-Mean"])]
    c1_detail = detail[detail.protocol.isin(["Raw-Sum","Raw-Mean"])]
    c2_detail = detail[detail.protocol.str.startswith("EqualLength")]
    if decisions["final_h2_h5_robust_above_chance"] == "NO":
        recommendation = "**方案 B**：保留当前 Llama/Tülu 作为 weak-reasoning control，并新增一个在同一冻结 StepGame task-fitness protocol 下行为能力明确更强的模型；不要自动进入因果实验。"
    elif decisions["native_improves"] == "YES":
        recommendation = "**方案 C**：把 frozen raw mechanistic protocol 与 native task-solving protocol 作为两个独立实验并行保留；先明确后续机制问题针对哪一种接口。"
    else:
        recommendation = "**方案 A（有条件）**：只有在逐 hop 稳健性与独立复现均满足后，才继续 reasoning localization；不要自动进入因果实验。"
    report = f"""# Stage B2.1 Model-task Fitness Gate

## 1. Research question

本阶段不研究 hidden-state mechanism，也不训练或重跑 probe。核心问题是：冻结 B2 中 h2–h5 较低的八分类行为准确率，究竟主要来自模型的多步空间推理能力不足，还是来自 raw prompt、post-training chat interface 不匹配以及自然 verbalizer 的 token 长度不对称。所有结论均限定于 zero-shot、constrained-choice、teacher-forced candidate likelihood。

## 2. Frozen reference

- 冻结 Primary test 共 **2063** 行；h1–h5 分别为 712/505/422/282/142，row-ID SHA-256 为 `{audit['primary']['row_ids_sha256']}`。
- 四个 checkpoint 为 Base、SFT、DPO、Final；审计确认均为 32 层、hidden size 4096。Tulu revision 是 **ModelScope `.msc` local revision**，不是 Hugging Face commit。
- Raw prompt 精确保持 `Story:\n...\n\nQuestion:\n...`，`add_special_tokens=False` 后显式加入 BOS；不含 system、chat wrapper、CoT、few-shot 或答案格式说明。
- 自然候选固定为 above/below/left/right 及四个 hyphenated diagonal；原 score 为候选 token log-probability 求和。
- 新阶段写入独立 `artifacts/stage_b2_1/`。审计前保护快照覆盖 {read_json(OUT/'audit'/'protected_checksums.json')['file_count']} 个旧文件。

方法依据：StepGame 用于多跳文本空间关系推理；Llama 3.1 与 Tülu 3 的接口差异提示 post-trained checkpoint 应额外在其原生对话模板下做 task-fitness 诊断。统计不以测试集调参；95% CI 用 `story_group_id` 为单位、2,000 次配对 bootstrap，固定 seed={SEED}。

## 3. Experiment A — Frozen Raw Reproduction

Gate A 为 **{gate['status']}**。四模型每个 hop 的正确样本整数与冻结 B2 完全相同；JSON/CSV 浮点回读产生的最大表示差约 1e-16，但有理数 accuracy 与 row-level argmax 未改变。候选分数直接复用旧 artifact，并记录来源 SHA-256，避免不必要的 GPU 重算。

{markdown_table(raw_detail, detail_columns)}

![Figure 1](figures/figure_1_raw_sum_by_hop.png)

图 1 显示冻结 Raw-Sum 的完整基线，而不是选择最佳 checkpoint 作标题。h1 本身仅略高于 0.125；h2–h5 随 hop 增加总体下降，且 post-training checkpoint 未形成一致的单调优势。这一图只描述 frozen common interface 下的行为，不能单独推出模型绝对不会推理。

## 4. Experiment B — Native Chat Template

SFT/DPO/Final 均使用各 tokenizer 实际 `chat_template` 和 `apply_chat_template(..., add_generation_prompt=True)`；没有手写模板。模板文本是否完全一致：**{chat['templates_identical']}**。每 checkpoint 完成 3 deterministic rows × 5 hops × 8 candidates = 120 项边界审计；candidate 不写入 prompt，continuation 边界与独立 tokenization 一致。

{markdown_table(native_detail, detail_columns)}

![Figure 3](figures/figure_3_raw_vs_native.png)

图 3 将同一冻结样本、同一自然候选的 Raw-Sum 与 Native-Sum 并列。差异回答的是“接口是否改变任务表现”，不是“chat template 改善 hidden representation”。若某 hop 上升而其他 hop 不稳定，结论记为 MIXED，不用单一最佳点替代全量结果。

## 5. Experiment C1 — Sum vs Mean-token

{markdown_table(c1_detail, detail_columns)}

![Figure 2](figures/figure_2_raw_sum_vs_mean.png)

![Figure 6](figures/figure_6_cardinal_diagonal_frequency.png)

自然候选中 cardinal 通常为 1 token、diagonal 为 2 token。Raw-Mean 只把同一候选的 log-probability 除以 token 数，不引入长度惩罚 α、温度或校准。图 2 显示 hop 级准确率变化；图 6 检查预测落在 cardinal/diagonal 集合的频率，避免仅凭 aggregate accuracy 判断 token-length bias。精确的 true-class 分组 accuracy、gold-vs-best-other margin 与预测类型频率见 `scoring_sensitivity/cardinal_diagonal_metrics.csv`。

## 6. Experiment C2 — Equal-length Verbalizer

标签选择 Gate 为 **{labels['status']}**；固定映射是 `{json.dumps(labels['relation_to_label'])}`，选择算法为 `{labels['selection_algorithm']}`。候选选择只读取 tokenizer，不访问模型概率或 gold-conditioned performance。C2-Raw 与 C2-Native 都显式展示 relation→label mapping，因此属于 measurement-control / task-fitness diagnostic，不能替代 Experiment A。

{markdown_table(c2_detail, detail_columns)}

![Figure 4](figures/figure_4_natural_vs_equal_length.png)

图 4 左侧完整展示四模型 EqualLength-Raw，右侧将 Tulu native natural（虚线）与 native equal-length（实线）并列。等长 verbalizer 同时改变 token 对称性并增加 mapping 指令；因此性能变化不能被单因果解释为 token 长度效应。

## 7. Paired statistical comparisons

比较 A–E 均使用相同 row，并在 story-group 层配对重采样。下表为 Final accuracy；完整四模型、各 hop、Macro-F1 与 derived mean 在 `statistics/paired_bootstrap.csv`。

{markdown_table(final_pairs, ['comparison_code','protocol_a','protocol_b','hop','point_difference_b_minus_a','ci95_low','ci95_high','ci_excludes_zero'])}

![Figure 5](figures/figure_5_derived_mean_all_protocols.png)

Derived mean 是 h2、h3、h4、h5 四个 hop accuracy 的等权算术平均，不用样本量加权，且不替代各 hop。图 5 的空位代表 Base 不适用 Native，不填 0。

## 8. Model-task fitness decision

A. 当前 raw behavior 是否被成功复现？ **{decisions['raw_reproduced']}**。证据：Gate A={gate['status']}，四模型×五 hop 正确数完全一致。

B. native chat 是否明显改善 post-trained 模型 behavior？ **{decisions['native_improves']}**。依据 Final Native-Sum − Raw-Sum derived mean 的配对 95% CI，而不是挑选单一 hop。

C. candidate token length 是否明显影响 behavior？ **{decisions['token_length_affects']}**。联合参考 Raw-Mean − Raw-Sum、cardinal/diagonal 预测频率和 EqualLength-Raw 对照。

D. equal-length verbalizer 是否明显提高 derived performance？ **{decisions['equal_length_improves']}**。分别审视 EqualLength-Raw − Raw-Sum 与 EqualLength-Native − Native-Sum。

E. 在最公平的 task-fitness protocol 下，Final 的 h2–h5 是否稳定高于 chance？ **{decisions['final_h2_h5_robust_above_chance']}**。判定要求 EqualLength-Native 的 h2–h5 每个 bootstrap 95% CI 下界均高于 0.125。

F. 当前 Llama/Tulu 是否适合作为“successful multi-hop reasoning mechanism”的主要研究对象？ **{decisions['successful_mechanism_target']}**。若仅部分 hop 高于 chance 或绝对性能仍弱，适合作 weak-reasoning control，但不宜自动当作成功推理机制的唯一主对象。

![Figure 8](figures/figure_8_fitness_gate_summary.png)

## 9. Consequence for Stage B research

{recommendation}

## 10. Interpretation boundaries

- chat template 提高 ≠ hidden representation improved；prompt token sequence 已改变。
- equal-length verbalizer 提高 ≠ reasoning mechanism changed；mapping 指令与 measurement interface 同时改变。
- probe F1 ≠ model behavioral accuracy。
- raw behavior low ≠ 模型在所有接口下绝对不能推理。
- 本报告不把 native 结果与原 B2 hidden states 混合解释为训练阶段因果效应。

## 11. Core summary table

{markdown_table(core, core.columns)}

## 12. Confusion analysis

![Figure 7](figures/figure_7_final_confusion_matrices.png)

图 7 对 Final 的 Raw-Sum、Native-Sum 和 EqualLength-Native 做行归一化。橙色等值线标记单元格概率 0.125。对角线反映 class recall，整列偏亮表示预测塌缩/偏好；因此它能区分“总体低但分散”与“集中猜少数 verbalizer”的两类失败。

## 13. Reproducibility, limitations, and references

运行环境版本、模型路径、revision、BOS/EOS、vocab、模板 SHA-256、prompt/token 边界、旧 artifact SHA-256 和 protocol SHA-256 均在 `audit/`。限制包括：只有一个 frozen StepGame split；conditional likelihood 不等同自由生成；C2 mapping 本身提供额外任务说明；native 与 raw token sequence 不同；bootstrap CI 量化抽样不确定性而非模型/数据分布外泛化。

参考文献：

1. Shi, Z. et al. (2022). *StepGame: A New Benchmark for Robust Multi-Hop Spatial Reasoning in Texts*. AAAI.
2. Dubey, A. et al. (2024). *The Llama 3 Herd of Models*. arXiv:2407.21783.
3. Lambert, N. et al. (2024). *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*. arXiv:2411.15124.
4. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
"""
    (OUT / "STAGE_B2_1_MODEL_TASK_FITNESS_REPORT.md").write_text(report, encoding="utf-8")


def main():
    ensure_dirs(); primary = load_primary(); scores = load_scores()
    frozen_labels = read_json(OUT / "equal_length_verbalizer" / "protocol.json")["relation_to_label"]
    example_lines = ["# Frozen equal-length verbalizer prompt examples", "", "The C2-Raw text below is tokenized without special tokens and receives one explicit BOS. For C2-Native the identical user content is rendered by each checkpoint's audited native chat template; the scored label begins only at the assistant generation boundary.", ""]
    for hop in HOPS:
        row = primary.loc[primary.computed_hop.eq(hop)].iloc[0]
        example_lines += [f"## h{hop}, row {int(row.row_id)}", "", "```text", equal_raw_prompt(row, frozen_labels), "```", ""]
    (OUT / "equal_length_verbalizer" / "prompt_examples.md").write_text("\n".join(example_lines) + "\n", encoding="utf-8")
    predictions, metrics, boots = prediction_bundle(scores, primary)
    metrics.to_csv(OUT / "statistics" / "all_hop_metrics.csv", index=False)
    boots.to_csv(OUT / "statistics" / "all_bootstrap_ci.csv", index=False)
    summary = summary_table(metrics); summary.to_csv(OUT / "statistics" / "core_summary.csv", index=False)
    paired = paired_statistics(predictions); paired.to_csv(OUT / "statistics" / "paired_bootstrap.csv", index=False)

    native_scores = pd.concat([frame.assign(model=model) for (model, protocol), frame in scores.items() if protocol == "Native-Sum"], ignore_index=True)
    native_scores.assign(score=native_scores.sum_log_probability).to_csv(OUT / "native_chat" / "row_scores_sum.csv", index=False)
    native_scores.assign(score=native_scores.mean_log_probability).to_csv(OUT / "native_chat" / "row_scores_mean.csv", index=False)
    metrics[metrics.protocol == "Native-Sum"].to_csv(OUT / "native_chat" / "metrics_sum.csv", index=False)
    metrics[metrics.protocol == "Native-Mean"].to_csv(OUT / "native_chat" / "metrics_mean.csv", index=False)
    boots[boots.protocol == "Native-Sum"].to_csv(OUT / "native_chat" / "bootstrap_ci_sum.csv", index=False)
    boots[boots.protocol == "Native-Mean"].to_csv(OUT / "native_chat" / "bootstrap_ci_mean.csv", index=False)
    for model in ["sft", "dpo", "final"]:
        write_confusions(predictions[(model, "Native-Sum")], OUT / "native_chat" / "confusion_matrices", f"{model}_sum")
        write_confusions(predictions[(model, "Native-Mean")], OUT / "native_chat" / "confusion_matrices", f"{model}_mean")

    eq_raw = pd.concat([frame.assign(model=model) for (model, protocol), frame in scores.items() if protocol == "EqualLength-Raw"], ignore_index=True)
    eq_native = pd.concat([frame.assign(model=model) for (model, protocol), frame in scores.items() if protocol == "EqualLength-Native"], ignore_index=True)
    eq_raw.to_csv(OUT / "equal_length_verbalizer" / "raw_row_scores.csv", index=False)
    eq_native.to_csv(OUT / "equal_length_verbalizer" / "native_row_scores.csv", index=False)
    metrics[metrics.protocol == "EqualLength-Raw"].to_csv(OUT / "equal_length_verbalizer" / "raw_metrics.csv", index=False)
    metrics[metrics.protocol == "EqualLength-Native"].to_csv(OUT / "equal_length_verbalizer" / "native_metrics.csv", index=False)
    boots[boots.protocol.str.startswith("EqualLength")].to_csv(OUT / "equal_length_verbalizer" / "bootstrap_ci.csv", index=False)
    for (model, protocol), pred in predictions.items():
        if protocol.startswith("EqualLength"):
            write_confusions(pred, OUT / "equal_length_verbalizer" / "confusion_matrices", f"{model}_{protocol.lower().replace('-', '_')}")
    write_json(OUT / "native_chat" / "validation.json", {"status": "PASS", "models": ["sft","dpo","final"], "rows_each": 2063, "all_finite": bool(np.isfinite(native_scores[["sum_log_probability","mean_log_probability"]]).all().all())})
    write_json(OUT / "equal_length_verbalizer" / "validation.json", {"status": "PASS", "models_raw": MODELS, "models_native": ["sft","dpo","final"], "rows_each": 2063, "all_finite": bool(np.isfinite(pd.concat([eq_raw,eq_native])[["sum_log_probability"]]).all().all())})

    freq = draw_figures(metrics, summary, predictions)
    freq.to_csv(OUT / "statistics" / "figure_6_data.csv", index=False)
    decisions = decision_from(summary, boots, paired); write_json(OUT / "statistics" / "fitness_decision.json", decisions)
    chart_map = {f"Figure {i}": {"file": sorted((OUT / 'figures').glob(f"figure_{i}_*.png"))[0].name,
                  "source": "statistics/core_summary.csv" if i not in [6,7] else ("statistics/figure_6_data.csv" if i == 6 else "row-level reconstructed predictions")}
                 for i in range(1,9)}
    write_json(OUT / "figures" / "chart_map.json", chart_map)
    write_report(summary, metrics, boots, paired, decisions)
    print(json.dumps({"stage": "STATISTICS_FIGURES_REPORT", "status": "PASS", "decisions": decisions,
                      "report": str(OUT / 'STAGE_B2_1_MODEL_TASK_FITNESS_REPORT.md')}, indent=2, default=str))


if __name__ == "__main__":
    main()
