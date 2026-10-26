
"""Results dashboard for PromptBench."""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def prepare_results(results: list[dict]) -> pd.DataFrame:
    """Validate experiment results and prepare a displayable table."""

    if not results:
        return pd.DataFrame(
            columns=[
                "technique",
                "response",
                "latency_seconds",
                "input_tokens",
                "output_tokens",
                "evaluation_score",
            ]
        )

    df = pd.DataFrame(results)

    required_columns = {"technique", "response"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required result fields: {', '.join(sorted(missing))}"
        )

    optional_columns = [
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "evaluation_score",
    ]

    for column in optional_columns:
        if column not in df.columns:
            df[column] = None

    return df


RUBRIC_COLUMNS = [
    "relevance",
    "completeness",
    "clarity",
    "instruction_following",
]


def flatten_results(results: list[dict]) -> list[dict]:
    """Copy rubric scores and the validation verdict into top-level fields."""

    rows = []
    for result in results:
        row = dict(result)
        judge_scores = row.get("judge_scores") or {}
        for name in RUBRIC_COLUMNS:
            if name in judge_scores:
                row[name] = judge_scores[name]
        validation = row.get("validation") or {}
        if "passed" in validation:
            row["rules_passed"] = validation["passed"]
        rows.append(row)
    return rows


def display_results(
    results: list[dict],
    header: str = "3. Experiment results",
) -> None:
    """Display experiment results and basic comparisons."""

    st.header(header)

    if not results:
        st.info(
            "No experiment results yet. Run an experiment to see "
            "responses, performance metrics, and comparisons here."
        )
        return

    try:
        df = prepare_results(flatten_results(results))
    except ValueError as error:
        st.error(str(error))
        return

    st.subheader("Response comparison")

    display_columns = [
        "technique",
        "response",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "evaluation_score",
    ]
    extra_columns = [
        column
        for column in ["status", *RUBRIC_COLUMNS, "rules_passed"]
        if column in df.columns
    ]

    st.dataframe(
        df[display_columns + extra_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Performance metrics")

    col1, col2 = st.columns(2)

    latency = pd.to_numeric(
        df["latency_seconds"], errors="coerce"
    )
    scores = pd.to_numeric(
        df["evaluation_score"], errors="coerce"
    )

    with col1:
        valid_latency = latency.dropna()

        if not valid_latency.empty:
            st.metric(
                "Average response time",
                f"{valid_latency.mean():.2f} seconds",
            )
        else:
            st.metric("Average response time", "Not available")

    with col2:
        valid_scores = scores.dropna()

        if not valid_scores.empty:
            st.metric(
                "Average evaluation score",
                f"{valid_scores.mean():.2f}",
            )
        else:
            st.metric("Average evaluation score", "Not available")

    if not scores.dropna().empty:
        chart_data = pd.DataFrame(
            {
                "Technique": df.loc[scores.notna(), "technique"],
                "Score": scores[scores.notna()],
            }
        ).set_index("Technique")

        st.subheader("Evaluation score by technique")

        fig, ax = plt.subplots()
        chart_data.plot(kind="bar", ax=ax, legend=False)
        ax.set_xlabel("Prompting technique")
        ax.set_ylabel("Evaluation score")
        ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Details")

    for row in flatten_results(results):
        label = row["technique"]
        if row.get("evaluation_score") is not None:
            label += f" (score {row['evaluation_score']})"
        if row.get("status", "success") != "success":
            label += f" ({row['status']})"

        with st.expander(label):
            if row.get("error"):
                st.error(row["error"])
            if row.get("prompt"):
                st.markdown("**Prompt**")
                st.code(row["prompt"], language="text")
            st.markdown("**Response**")
            st.write(row.get("response") or "_No response_")
            if row.get("judge_feedback"):
                st.markdown(f"**Judge feedback:** {row['judge_feedback']}")
            validation = row.get("validation") or {}
            for issue in validation.get("issues", []):
                st.warning(issue)
            if validation.get("checks"):
                st.json(validation["checks"], expanded=False)
