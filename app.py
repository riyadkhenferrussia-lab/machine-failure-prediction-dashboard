"""Machine Failure Prediction and Predictive Maintenance Dashboard.

    streamlit run app.py

The dashboard is an offline analysis of the public, synthetic AI4I 2020 dataset. It is not
connected to any machine, does not monitor anything in real time, and every result is a
screening prediction, never a confirmed fault.

All logic lives in ``src/`` (and is unit-tested); this file only arranges widgets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import plots  # noqa: E402
from src.dashboard import (  # noqa: E402
    PAGES,
    build_examples,
    comparison_frame,
    eda_findings,
    escape_markdown,
    filter_records,
    format_bytes,
    indicators_frame,
    overview_numbers,
    read_uploaded_csv,
    results_to_csv,
    rules_frame,
    sample_batch_frame,
    threshold_metrics,
)
from src.data import load_and_validate, load_dataset  # noqa: E402
from src.evaluate import build_figures  # noqa: E402
from src.predict import LoadedModel, assess_batch, assess_reading, ensure_model, load_holdout_predictions  # noqa: E402
from src.recommend import DISCLAIMER, RiskLevel  # noqa: E402
from src.utils import read_json, to_jsonable  # noqa: E402
from src.validation import InputValidationError  # noqa: E402

st.set_page_config(
    page_title="Machine Failure Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCOPE_NOTE = (
    "**Scope.** An offline analysis of a public, *synthetic* benchmark dataset. A model scores each machine "
    "record for failure risk and a transparent rule layer turns the score into an inspection recommendation. "
    "Results are **predictions for a person to check, not confirmed faults**. Nothing here is connected to real "
    "machines or monitors anything in real time."
)
RUN_HINT = "Run `python -m src.train` and then `python -m src.evaluate` in the project folder, then reload this page."


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the saved model (it is retrained quickly if missing or incompatible) ...")
def get_model() -> LoadedModel:
    return ensure_model()


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_data(show_spinner=False)
def get_data_report() -> dict:
    _, report = load_and_validate()
    return report.to_dict()


@st.cache_data(show_spinner=False)
def get_holdout() -> pd.DataFrame | None:
    return load_holdout_predictions()


@st.cache_data(show_spinner=False)
def get_metrics() -> dict | None:
    return read_json(cfg.METRICS_PATH) if cfg.METRICS_PATH.exists() else None


@st.cache_data(show_spinner=False)
def get_importance() -> pd.DataFrame | None:
    return pd.read_csv(cfg.IMPORTANCE_PATH) if cfg.IMPORTANCE_PATH.exists() else None


@st.cache_data(show_spinner=False)
def get_eda_summary() -> dict | None:
    return read_json(cfg.EDA_SUMMARY_PATH) if cfg.EDA_SUMMARY_PATH.exists() else None


@st.cache_data(show_spinner=False)
def get_table(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if Path(path).exists() else None


@st.cache_data(show_spinner="Rendering evaluation figures ...")
def evaluation_pngs() -> dict[str, bytes]:
    figures = build_figures(get_model(), get_metrics(), get_holdout(), get_importance(), get_dataset())
    return {name: plots.figure_png(figure) for name, figure in figures.items()}


@st.cache_data(show_spinner=False)
def data_png(name: str, x: str = "", y: str = "") -> bytes:
    """PNG bytes of an exploratory figure (built without pyplot, cached per argument)."""
    data = get_dataset()
    if name == "failure_modes":
        stats = get_data_report()["stats"]
        rnf_only = stats["label_inconsistencies"]["rnf_flag_without_failure"]
        figure = plots.plot_failure_modes(stats["failure_mode_counts"], stats["n_rows"], stats["n_failures"], rnf_only)
    elif name == "failure_rates":
        figure = plots.plot_failure_rates(data)
    elif name == "distribution":
        figure = plots.plot_distribution(data, x)
    elif name == "scatter":
        figure = plots.plot_scatter(data, x, y)
    elif name == "mechanisms":
        figure = plots.plot_failure_mechanisms(data)
    elif name == "correlation":
        figure = plots.plot_correlation(data)
    else:
        raise ValueError(f"unknown figure {name!r}")
    return plots.figure_png(figure)


def show_png(png: bytes) -> None:
    st.image(png)


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def bullets(lines) -> None:
    st.markdown("\n".join(f"- {escape_markdown(line)}" for line in lines))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(model: LoadedModel, metrics: dict) -> str:
    with st.sidebar:
        st.title("Failure prediction")
        page = st.radio("Section", list(PAGES), key="page")
        st.divider()
        st.caption(f"Model: {model.label}")
        st.caption(f"Operating threshold: {model.threshold:.3f}")
        st.caption(f"Trained: {metrics['meta']['created_utc'][:10]}")
        st.divider()
        st.caption(
            "Synthetic benchmark data (AI4I 2020, UCI, CC BY 4.0). Results are screening predictions, "
            "not confirmed faults. No live machine data."
        )
    return page


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def page_overview(model: LoadedModel, metrics: dict) -> None:
    n = overview_numbers(metrics)
    final = metrics["final"]
    test = final["test"]
    ci = n["ci95"]

    st.title("Machine Failure Prediction and Predictive Maintenance Dashboard")
    st.caption(
        "Portfolio project on the AI4I 2020 Predictive Maintenance dataset (UCI, synthetic data). "
        "Offline analysis: no live machine data."
    )
    st.info(SCOPE_NOTE)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{n['records']:,}", help="Rows in the dataset: one snapshot of one machine each.")
    c2.metric("Recorded failures", f"{n['failures']:,}", help=f"{n['failure_rate']:.2%} of the records carry Machine failure = 1.")
    c3.metric(
        "Hold-out recall",
        pct(n["recall"]),
        help=f"{test['tp']} of {n['holdout_failures']} recorded failures in the hold-out set were flagged. "
        f"95% bootstrap interval {pct(ci['recall'][0])} to {pct(ci['recall'][1])}.",
    )
    c4.metric(
        "Hold-out precision",
        pct(n["precision"]),
        help=f"{test['tp']} of {test['tp'] + test['fp']} flagged machines had a recorded failure. "
        f"95% bootstrap interval {pct(ci['precision'][0])} to {pct(ci['precision'][1])}.",
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("How it works")
        st.markdown(
            "1. **Data.** 10,000 records with five sensor readings and a product variant; validated on load.\n"
            "2. **Model.** A random forest on the raw readings plus three engineered features (temperature "
            "difference, mechanical power, wear x torque), tuned by cross-validation on 80% of the records and "
            "evaluated once on the remaining 20%.\n"
            "3. **Recommendation.** The model score and documented engineering limits are combined into a "
            "Low / Elevated / High level with the reasoning shown.\n"
            "4. **Assessment.** Single machines or a CSV batch, with input validation and a clear statement "
            "that the outcome is a prediction."
        )
    with right:
        st.subheader("Final model")
        st.markdown(
            f"**{escape_markdown(n['model'])}**  \n"
            f"Operating threshold **{n['threshold']:.3f}**  \n"
            f"Trained {n['created_utc'][:10]} with scikit-learn {n['environment']['scikit_learn']}, "
            f"Python {n['environment']['python']}"
        )

    st.subheader("Results at a glance")
    modes = {m["mode"]: m for m in final["per_mode_recall"]}
    flagged = test["tp"] + test["fp"]
    glance = [
        f"At threshold {n['threshold']:.3f} the model flagged {flagged} of {n['holdout_records']:,} hold-out machines; "
        f"{test['tp']} of them had a recorded failure (precision {pct(test['precision'])}) and {test['fn']} "
        f"recorded failures were not flagged (recall {pct(test['recall'])}).",
        f"Failures that follow the documented rules are found almost always (power {modes['pwf']['detected']}/{modes['pwf']['n_failures']}, "
        f"overstrain {modes['osf']['detected']}/{modes['osf']['n_failures']}, heat dissipation {modes['hdf']['detected']}/{modes['hdf']['n_failures']}); "
        f"tool-wear failures are mostly missed ({modes['twf']['detected']}/{modes['twf']['n_failures']}) because the data generate them partly at random.",
        f"Accuracy is not informative here: always answering 'no failure' scores "
        f"{pct(metrics['models']['baseline_majority']['test']['accuracy'])}.",
        f"Uncertainty is wide: with {n['holdout_failures']} hold-out failures the 95% interval for recall is "
        f"{pct(ci['recall'][0])} to {pct(ci['recall'][1])}.",
    ]
    bullets(glance)

    st.subheader("Data and quality checks")
    report = get_data_report()
    stats = report["stats"]
    show_png(data_png("failure_modes"))
    with st.expander("About the data, its provenance and label quirks"):
        st.markdown(
            "**AI4I 2020 Predictive Maintenance Dataset** (S. Matzka, UCI Machine Learning Repository, "
            f"DOI {cfg.DATASET_DOI}, licence CC BY 4.0). The authors generated it to resemble milling-machine "
            "maintenance data, so it is **synthetic**: failure labels follow documented rules plus random components. "
            "The model largely re-learns those rules, so the scores below describe how well the pipeline recovers "
            "them, not how it would perform on real equipment."
        )
        bullets(
            [
                f"{stats['n_rows']:,} rows, {stats['n_failures']} failures ({stats['failure_rate']:.2%}); "
                "the documented failure criteria for power, heat dissipation and overstrain reproduce those labels exactly.",
                *[issue["message"] for issue in report["issues"]],
                "The CSV in data/raw was copied from a public GitHub mirror and checked against the published statistics and "
                "documented rules; compare it with the UCI download using python -m src.data --verify-official.",
            ]
        )
    with st.expander("Read before using the results"):
        bullets(
            [
                "Synthetic data: performance figures say how well the pipeline recovers generated rules, not how it would do on real machines.",
                f"Few failures: the hold-out set has {n['holdout_failures']}, so every metric carries a wide interval.",
                "The score is a ranking signal, not a calibrated probability.",
                "The operating threshold assumes a missed failure costs more than an unnecessary inspection; that cost ratio is a design choice.",
                "Inputs outside the training range are extrapolations and are flagged as such.",
                "Rows are independent snapshots: there is no sensor history, so trends and degradation over time are not modelled.",
                "A flag means 'inspect', never 'this machine is broken'.",
            ]
        )


# --------------------------------------------------------------------------- #
# Data exploration
# --------------------------------------------------------------------------- #
EXPLORER_FEATURES = cfg.NUMERIC_SENSOR_FEATURES + cfg.ENGINEERED_FEATURES


def feature_label(key: str) -> str:
    return cfg.FEATURE_LABELS.get(key, key)


def page_data() -> None:
    st.header("Data exploration")
    st.caption("AI4I 2020, all 10,000 records. Failures are drawn in red, other records in grey.")
    frame = get_dataset()
    summary = get_eda_summary()
    tab_failures, tab_dist, tab_rel, tab_mech, tab_records = st.tabs(
        ["Failures", "Distributions", "Relationships", "Failure mechanisms", "Records"]
    )

    with tab_failures:
        if summary:
            bullets(eda_findings(summary))
        show_png(data_png("failure_rates"))

    with tab_dist:
        feature = st.selectbox("Feature", EXPLORER_FEATURES, format_func=feature_label, key="dist_feature")
        show_png(data_png("distribution", x=feature))
        st.caption("Each class is scaled to equal area so the shapes can be compared despite the 3.4% failure rate.")

    with tab_rel:
        left, right = st.columns(2)
        x = left.selectbox("X axis", EXPLORER_FEATURES, index=2, format_func=feature_label, key="scatter_x")
        y = right.selectbox("Y axis", EXPLORER_FEATURES, index=3, format_func=feature_label, key="scatter_y")
        if x == y:
            st.info("Pick two different features to see their relationship.")
        else:
            show_png(data_png("scatter", x=x, y=y))
        st.subheader("Correlation matrix")
        show_png(data_png("correlation"))

    with tab_mech:
        st.markdown(
            "The dataset documents how failures arise. Power outside 3,500-9,000 W, a small temperature difference "
            "at low speed, and wear x torque above a per-variant limit each cause a failure; tool wear and random "
            "failures are assigned partly at random. The thin lines below are those documented limits drawn over the real records."
        )
        show_png(data_png("mechanisms"))

    with tab_records:
        left, middle, right = st.columns([1, 1, 2])
        types = left.multiselect("Product variant (Type)", cfg.TYPE_VALUES, default=cfg.TYPE_VALUES, key="rec_types")
        outcome = middle.radio("Outcome", ["All records", "Failures only", "No failure only"], key="rec_outcome")
        max_wear = int(frame["tool_wear_min"].max())
        wear = right.slider("Tool wear [min]", 0, max_wear, (0, max_wear), key="rec_wear")
        if not types:
            st.info("Select at least one product variant.")
        else:
            subset = filter_records(frame, types, outcome, (float(wear[0]), float(wear[1])))
            original_names = {canonical: raw for raw, canonical in cfg.RAW_TO_CANONICAL.items()}
            display = subset.rename(columns=original_names)
            st.write(f"{len(subset):,} of {len(frame):,} records match.")
            st.dataframe(display.head(1000), hide_index=True)
            if len(subset) > 1000:
                st.caption("The table shows the first 1,000 matching rows; the download contains all of them.")
            st.download_button(
                "Download matching records (CSV)",
                data=display.to_csv(index=False).encode("utf-8"),
                file_name="ai4i2020_filtered.csv",
                mime="text/csv",
            )


# --------------------------------------------------------------------------- #
# Model evaluation
# --------------------------------------------------------------------------- #
def page_model(model: LoadedModel, metrics: dict) -> None:
    st.header("Model evaluation")
    meta, final = metrics["meta"], metrics["final"]
    test, ci = final["test"], final["test_ci95"]
    st.caption(
        f"Computed on the hold-out set: {meta['n_test']:,} records ({meta['failures_test']} failures) that were not used "
        "for training, hyper-parameter selection or choosing the threshold."
    )
    pngs = evaluation_pngs()
    holdout = get_holdout()
    tab_summary, tab_curves, tab_drivers, tab_modes, tab_levels = st.tabs(
        ["Summary", "Curves and threshold", "What drives the model", "By failure mode", "Inspection levels"]
    )

    with tab_summary:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Precision", pct(test["precision"]), help=f"95% interval {pct(ci['precision'][0])} to {pct(ci['precision'][1])}")
        c2.metric("Recall", pct(test["recall"]), help=f"95% interval {pct(ci['recall'][0])} to {pct(ci['recall'][1])}")
        c3.metric("F1", f"{test['f1']:.3f}", help=f"95% interval {ci['f1'][0]:.3f} to {ci['f1'][1]:.3f}")
        c4.metric("PR-AUC", f"{test['pr_auc']:.3f}", help=f"95% interval {ci['pr_auc'][0]:.3f} to {ci['pr_auc'][1]:.3f}")
        c5.metric("ROC-AUC", f"{test['roc_auc']:.3f}", help=f"95% interval {ci['roc_auc'][0]:.3f} to {ci['roc_auc'][1]:.3f}")
        st.caption(
            f"Model: {final['label']}, threshold {final['threshold']:.3f}. Intervals are stratified bootstrap "
            f"percentile intervals ({final['bootstrap_replicates']} replicates)."
        )
        left, right = st.columns([2, 3])
        with left:
            show_png(pngs["confusion_matrix"])
        with right:
            st.subheader("All candidate models")
            st.dataframe(comparison_frame(metrics), hide_index=True)
            st.caption(
                "Final model = highest cross-validated PR-AUC on the training set; hold-out results played no part in the "
                f"choice. Accuracy is omitted on purpose: always predicting 'no failure' already scores "
                f"{pct(metrics['models']['baseline_majority']['test']['accuracy'])}."
            )

    with tab_curves:
        show_png(pngs["roc_pr"])
        show_png(pngs["threshold_tradeoff"])
        if holdout is not None:
            st.subheader("Explore the threshold")
            st.caption(
                "Exploration only: moving the slider recomputes the hold-out metrics from the stored scores. "
                f"The operating threshold ({final['threshold']:.3f}) was fixed on training data before the hold-out set was used."
            )
            chosen = st.slider("Decision threshold", 0.01, 0.99, float(round(final["threshold"], 2)), 0.01, key="threshold_slider")
            result = threshold_metrics(holdout, final["key"], float(chosen))
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Machines flagged", f"{result['tp'] + result['fp']:,}")
            m2.metric("Precision", pct(result["precision"]))
            m3.metric("Recall", pct(result["recall"]))
            m4.metric("F1", f"{result['f1']:.3f}")
            m5.metric("F2", f"{result['f2']:.3f}")
            left, right = st.columns([2, 3])
            with left:
                show_png(plots.figure_png(plots.plot_confusion_matrix(result, float(chosen), "Hold-out confusion matrix at the chosen threshold")))
            with right:
                st.markdown(
                    f"- True positives: **{result['tp']}**, false alarms: **{result['fp']}**\n"
                    f"- Missed failures: **{result['fn']}**, true negatives: **{result['tn']:,}**\n"
                    f"- Share of machines flagged: **{pct(result['alert_rate'])}**"
                )

    with tab_drivers:
        show_png(pngs["feature_importance"])
        st.markdown(
            "Torque, speed and the two temperatures matter most when shuffled one at a time; the temperatures work "
            "through their difference, and wear x torque and power (engineered from the raw readings) carry the most "
            "weight inside the forest. This matches the documented failure rules, which is expected for synthetic data."
        )
        show_png(pngs["risk_surfaces"])

    with tab_modes:
        show_png(pngs["per_mode_recall"])
        show_png(pngs["cumulative_gains"])
        show_png(pngs["calibration"])

    with tab_levels:
        st.markdown(
            "The dashboard turns the model score and four engineering indicators into one level per machine. "
            "The table shows how those levels line up with the recorded outcomes on the hold-out set."
        )
        levels = get_table(str(cfg.RECOMMENDATION_LEVELS_PATH))
        if levels is not None:
            shown = levels.rename(
                columns={
                    "risk_level": "Level",
                    "machines": "Machines",
                    "share_of_machines": "Share of machines",
                    "recorded_failures": "Recorded failures",
                    "failure_rate": "Failure rate",
                    "share_of_all_failures": "Share of all failures",
                }
            )
            for column in ("Share of machines", "Failure rate", "Share of all failures"):
                shown[column] = (shown[column] * 100).round(2).astype(str) + "%"
            st.dataframe(shown, hide_index=True)
        st.subheader("The rules")
        st.dataframe(rules_frame(), hide_index=True)
        st.caption(
            "Final level = the higher of the model level and the indicator level. Limits for ALERT are the criteria documented "
            "for this dataset; for real equipment they would have to come from the manufacturer's specification."
        )
        design = get_table(str(cfg.RECOMMENDATION_DESIGN_PATH))
        if design is not None:
            with st.expander("How the watch margins were chosen (training split only)"):
                elevated = design[design["risk_level"] == "Elevated"][["rules", "share_of_machines", "failure_rate"]]
                low = design[design["risk_level"] == "Low"][["rules", "recorded_failures"]].rename(columns={"recorded_failures": "failures_left_in_low"})
                table = elevated.merge(low, on="rules")
                table["share_of_machines"] = (table["share_of_machines"] * 100).round(1).astype(str) + "%"
                table["failure_rate"] = (table["failure_rate"] * 100).round(2).astype(str) + "%"
                table.columns = ["Rule variant", "Elevated: share of machines", "Elevated: failure rate", "Failures left in Low"]
                st.dataframe(table, hide_index=True)
                st.caption(
                    "Out-of-fold model scores on the 8,000 training records. Wider margins put a third of the machines on the "
                    "watch list for little gain in failure rate, so the margins were tightened. The hold-out set played no part."
                )
        if cfg.MODEL_CARD_MD_PATH.exists():
            st.download_button(
                "Download the model card (Markdown)",
                data=cfg.MODEL_CARD_MD_PATH.read_bytes(),
                file_name="MODEL_CARD.md",
                mime="text/markdown",
            )


# --------------------------------------------------------------------------- #
# Machine assessment
# --------------------------------------------------------------------------- #
INPUT_WIDGETS = {
    "air_temp_k": {"step": 0.1, "format": "%.1f", "help": "Air temperature in kelvin (K = degrees C + 273.15)."},
    "process_temp_k": {"step": 0.1, "format": "%.1f", "help": "Process temperature in kelvin (K = degrees C + 273.15)."},
    "rotational_speed_rpm": {"step": 1.0, "format": "%.0f", "help": "Spindle speed in revolutions per minute."},
    "torque_nm": {"step": 0.1, "format": "%.1f", "help": "Torque in newton metres."},
    "tool_wear_min": {"step": 1.0, "format": "%.0f", "help": "Minutes of use of the current tool."},
}
LEVEL_BOX = {RiskLevel.HIGH: st.error, RiskLevel.ELEVATED: st.warning, RiskLevel.LOW: st.success}


def render_assessment(assessment, holdout: pd.DataFrame | None) -> None:
    rec = assessment.recommendation
    LEVEL_BOX[rec.level](
        f"**{rec.level.label} risk: {rec.headline}.** Model score {assessment.score:.3f} against the operating "
        f"threshold {assessment.threshold:.3f}. This is a prediction, not a confirmed fault."
    )
    for warning in assessment.validation_warnings:
        st.warning(escape_markdown(warning))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final level", rec.level.label, help="The higher of the model level and the indicator level.")
    c2.metric("Model level", rec.model_level.label, help="High at or above the operating threshold, Elevated from half of it.")
    c3.metric("Indicator level", rec.indicator_level.label, help="Worst of the four engineering indicators (ALERT = High, WATCH = Elevated).")
    c4.metric("Failure score", f"{assessment.score:.3f}", help="Not a calibrated probability.")
    show_png(plots.figure_png(plots.plot_score_strip(assessment.score, assessment.threshold)))

    st.subheader("Why this level")
    bullets(rec.reasoning)
    st.subheader("Recommended actions")
    bullets(rec.actions)
    st.subheader("Engineering indicators")
    st.dataframe(indicators_frame(assessment), hide_index=True)
    st.subheader("Which inputs move the score")
    show_png(plots.figure_png(plots.plot_sensitivity(assessment.sensitivity)))

    context = assessment.context
    if context:
        st.subheader("Hold-out context")
        share = context["failure_share_at_or_above"]
        st.markdown(
            f"{context['records_at_or_above']:,} of the {context['holdout_records']:,} hold-out machines scored at or above "
            f"{assessment.score:.3f}; {context['recorded_failures_at_or_above']} of them had a recorded failure"
            + (f" ({share:.0%})." if share is not None else ".")
            + " Descriptive only: hold-out records are historical dataset rows, not live data."
        )
    st.download_button(
        "Download this assessment (JSON)",
        data=json.dumps(to_jsonable(assessment.to_dict()), indent=2).encode("utf-8"),
        file_name="machine_assessment.json",
        mime="application/json",
    )
    st.caption(DISCLAIMER)


def tab_single(model: LoadedModel, holdout: pd.DataFrame | None) -> None:
    examples = build_examples(model, holdout)
    by_key = {example.key: example for example in examples}
    chosen = st.selectbox("Start from", list(by_key), format_func=lambda key: by_key[key].label, key="example_choice")
    example = by_key[chosen]
    st.caption(example.description)

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Readings")
        reading = {
            "type": st.selectbox(
                "Product variant (Type)",
                cfg.TYPE_VALUES,
                index=cfg.TYPE_VALUES.index(example.reading["type"]),
                key=f"{chosen}_type",
                help="L, M and H are the low, medium and high quality variants.",
            )
        }
        for column in cfg.NUMERIC_SENSOR_FEATURES:
            spec = INPUT_WIDGETS[column]
            low, high = model.training_ranges[column]
            reading[column] = st.number_input(
                feature_label(column),
                min_value=0.0,
                max_value=float(cfg.HARD_LIMITS[column][1]),
                value=float(example.reading[column]),
                step=spec["step"],
                format=spec["format"],
                key=f"{chosen}_{column}",
                help=f"{spec['help']} Training range: {low:g} to {high:g}.",
            )
    with right:
        try:
            assessment = assess_reading(model, reading, holdout)
        except InputValidationError as exc:
            st.error("These readings cannot be assessed:")
            bullets(exc.messages)
        else:
            render_assessment(assessment, holdout)


def tab_batch(model: LoadedModel, holdout: pd.DataFrame | None) -> None:
    st.markdown(
        "Upload a CSV with one row per machine. Required columns: **Type** (L, M or H), **Air temperature [K]**, "
        "**Process temperature [K]**, **Rotational speed [rpm]**, **Torque [Nm]** and **Tool wear [min]**. "
        "Optional identifier columns (UDI, Product ID) are kept in the results. Failure labels are not needed and never used."
    )
    with st.expander("Accepted column names and limits"):
        st.markdown(
            f"Each column also accepts its snake_case name (for example `air_temp_k`). Comma, semicolon or tab separated; "
            f"up to {format_bytes(cfg.MAX_UPLOAD_BYTES)} and {cfg.MAX_BATCH_ROWS:,} rows. Temperatures must be in kelvin. "
            "The file is processed in memory for this session and is not stored by the app."
        )
    try:
        sample = sample_batch_frame(model, holdout)
    except ValueError:
        sample = None
    if sample is not None:
        st.download_button(
            "Download a sample file (real hold-out records)",
            data=results_to_csv(sample),
            file_name="sample_machines.csv",
            mime="text/csv",
        )

    upload = st.file_uploader("CSV file", type=["csv"], key="batch_upload")
    if upload is None:
        st.info("Upload a file to assess many machines at once.")
        return
    try:
        table = read_uploaded_csv(upload.getvalue())
        batch = assess_batch(model, table)
    except InputValidationError as exc:
        st.error("The file cannot be assessed:")
        bullets(exc.messages)
        return

    validation = batch.validation
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows read", f"{validation.n_input_rows:,}")
    c2.metric("Assessed", f"{validation.n_valid:,}")
    c3.metric("Rejected", f"{validation.n_invalid:,}", help="Rows with a missing, non-numeric or implausible value.")
    c4.metric("High", f"{batch.counts['High']:,}")
    c5.metric("Elevated", f"{batch.counts['Elevated']:,}")
    for warning in validation.warnings:
        st.warning(escape_markdown(warning))
    if not validation.problems.empty:
        st.error(f"{validation.n_invalid:,} row(s) were not assessed. Fix these values and upload again:")
        st.dataframe(validation.problems.head(500), hide_index=True)
    if batch.results.empty:
        return

    st.subheader("Results, highest priority first")
    results = batch.results.drop(columns=["risk_rank"]).copy()
    results["failure_score"] = results["failure_score"].round(4)
    st.dataframe(results.head(1000), hide_index=True)
    if len(results) > 1000:
        st.caption("The table shows the first 1,000 rows; the download contains all of them.")
    st.download_button(
        "Download all results (CSV)",
        data=results_to_csv(results),
        file_name="machine_assessment_results.csv",
        mime="text/csv",
    )
    st.caption(DISCLAIMER)


def page_assessment(model: LoadedModel) -> None:
    st.header("Machine assessment")
    st.warning(
        "A result is a screening prediction for decision support, not a confirmed fault. The model was trained on "
        "synthetic data and is not validated for real equipment."
    )
    holdout = get_holdout()
    single, batch = st.tabs(["Single machine", "Batch (CSV)"])
    with single:
        tab_single(model, holdout)
        with st.expander("How the recommendation is derived"):
            st.markdown(
                "Two independent sources are combined. The **model level** compares the failure score with the operating "
                "threshold chosen on training data. The **indicator level** applies the documented failure criteria of the "
                "dataset. The final level is the higher of the two, and every step is listed above."
            )
            st.dataframe(rules_frame(), hide_index=True)
    with batch:
        tab_batch(model, holdout)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        model = get_model()
        metrics = get_metrics()
        get_dataset()
    except FileNotFoundError as exc:
        st.error(f"A required file is missing: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001 - show a readable message instead of a stack trace
        st.error(f"The dashboard could not start: {type(exc).__name__}: {exc}")
        st.stop()
    if metrics is None or get_holdout() is None or get_importance() is None:
        st.error("The evaluation reports are missing. " + RUN_HINT)
        st.stop()

    page = render_sidebar(model, metrics)
    if page == "Overview":
        page_overview(model, metrics)
    elif page == "Data exploration":
        page_data()
    elif page == "Model evaluation":
        page_model(model, metrics)
    else:
        page_assessment(model)


main()
