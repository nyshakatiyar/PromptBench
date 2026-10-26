
import streamlit as st
from backend import run_full_experiment
from config import ConfigError, create_client, get_settings
from results_dashboard import display_results
from prompt_generator import FEW_SHOT_EXAMPLES, generate_prompts, TECHNIQUES
from storage import list_experiments, load_experiment


st.set_page_config(
    page_title="PromptBench",
    page_icon="PB",
    layout="wide",
)

st.title("PromptBench")
st.subheader("Test. Compare. Optimize.")

st.write(
    "Explore how different prompting techniques change the "
    "way an AI model responds to the same task."
)

st.divider()

st.header("1. Configure your experiment")

task = st.text_area(
    "Enter your prompt or task",
    placeholder=(
        "Example: Explain machine learning to a beginner "
        "using a practical example."
    ),
    height=140,
)

task_category = st.selectbox(
    "Task category",
    list(FEW_SHOT_EXAMPLES),
)

technique_ids = list(TECHNIQUES)

selected_techniques = st.multiselect(
    "Select prompting techniques",
    options=technique_ids,
    default=technique_ids[:5],
    format_func=lambda technique: TECHNIQUES[technique],
)

st.caption(
    "Choose at least one technique. Each selected technique "
    "will generate a different version of your original task."
)

generate_button = st.button(
    "Generate prompt variants",
    type="primary",
    use_container_width=True,
)

if generate_button:
    if not task.strip():
        st.error("Please enter a prompt or task.")

    elif not selected_techniques:
        st.error("Please select at least one technique.")

    else:
        try:
            variants = generate_prompts(
                task,
                selected_techniques,
                task_category=task_category,
            )

            st.session_state["promptbench_task"] = task
            st.session_state["promptbench_category"] = task_category
            st.session_state["promptbench_variants"] = variants
            st.session_state.pop("promptbench_experiment", None)

        except ValueError as error:
            st.error(str(error))


st.divider()
st.header("2. Generated prompt variants")

variants = st.session_state.get(
    "promptbench_variants",
    {},
)

if variants:
    st.success(f"Generated {len(variants)} prompt variants.")

    for technique, prompt in variants.items():
        with st.expander(
            TECHNIQUES[technique],
            expanded=(technique == "zero_shot"),
        ):
            st.code(prompt, language="text")

            st.button(
                f"Copy-ready prompt: {TECHNIQUES[technique]}",
                key=f"copy_info_{technique}",
                disabled=True,
                help=(
                    "Select and copy the prompt displayed above. "
                    "Clipboard integration can be added later."
                ),
            )

else:
    st.caption(
        "Your generated prompts will appear here after you "
        "click Generate prompt variants."
    )

st.divider()
st.header("3. Run experiment")

settings = get_settings()

if not variants:
    st.caption("Generate prompt variants first, then run them here.")

else:
    if not settings.api_key:
        st.warning(
            "No Gemini API key found. Set GEMINI_API_KEY in your "
            "environment, a .env file, or .streamlit/secrets.toml, "
            "then restart the app."
        )

    reference_answer = st.text_area(
        "Reference answer (optional)",
        help=(
            "Used for rule-based correctness checks: label match for "
            "classification, field match (JSON) for extraction, and "
            "word overlap for other categories."
        ),
        height=80,
    )
    use_judge = st.checkbox(
        f"Score responses with an LLM judge ({settings.judge_model})",
        value=True,
    )

    run_button = st.button(
        f"Run {len(variants)} prompts on {settings.model}",
        type="primary",
        use_container_width=True,
        disabled=not settings.api_key,
    )

    if run_button:
        try:
            client = create_client(settings)
        except ConfigError as error:
            st.error(str(error))
        else:
            with st.status("Running experiment...", expanded=True) as status:
                record = run_full_experiment(
                    st.session_state["promptbench_task"],
                    st.session_state["promptbench_category"],
                    variants,
                    client,
                    model=settings.model,
                    judge_model=settings.judge_model,
                    reference_answer=reference_answer,
                    evaluate=use_judge,
                    data_dir=settings.data_dir,
                    max_retries=settings.max_retries,
                    on_progress=st.write,
                )
                status.update(label="Experiment complete", state="complete")

            st.session_state["promptbench_experiment"] = record

experiment = st.session_state.get("promptbench_experiment")

st.divider()

if experiment:
    failures = [
        row for row in experiment["results"]
        if row.get("status", "success") != "success"
    ]
    if failures:
        st.warning(
            f"{len(failures)} of {len(experiment['results'])} prompts "
            "failed. See the details below."
        )
    if experiment.get("id"):
        st.caption(f"Saved as experiment {experiment['id']}")
    display_results(experiment["results"], header="4. Experiment results")
else:
    display_results([], header="4. Experiment results")

st.divider()
st.header("5. Experiment history")

history = list_experiments(data_dir=settings.data_dir)

if not history:
    st.caption("Saved experiments will appear here.")

else:
    labels = {
        item["id"]: (
            f"{item['created_at'][:19].replace('T', ' ')} UTC | "
            f"{item['metadata'].get('task_category', '')} | "
            f"{(item['metadata'].get('task') or '')[:60]}"
        )
        for item in history
    }
    selected_id = st.selectbox(
        "Saved experiments",
        options=list(labels),
        format_func=labels.get,
    )

    if selected_id:
        saved = load_experiment(selected_id, data_dir=settings.data_dir)
        metadata = saved["metadata"]
        st.write(f"**Task:** {metadata.get('task')}")
        st.caption(
            f"Model: {metadata.get('model')} | "
            f"Judge: {metadata.get('judge_model') or 'none'} | "
            f"ID: {saved['id']}"
        )
        display_results(saved["results"], header="Saved results")

st.divider()

st.header("About PromptBench")

st.write(
    f"Current task category: {task_category}. "
    "PromptBench is designed to compare prompting strategies "
    "using consistent inputs and evaluation criteria."
)
