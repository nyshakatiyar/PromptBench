
import streamlit as st

from prompt_generator import generate_prompts, TECHNIQUES


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
    [
        "General question answering",
        "Text classification",
        "Summarization",
        "Content generation",
        "Information extraction",
    ],
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
            )

            st.session_state["promptbench_task"] = task
            st.session_state["promptbench_category"] = task_category
            st.session_state["promptbench_variants"] = variants

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

    st.info(
        "Prompt generation is working. AI response generation, "
        "evaluation scores, experiment history, and charts will "
        "appear after the experiment runner is integrated."
    )

else:
    st.caption(
        "Your generated prompts will appear here after you "
        "click Generate prompt variants."
    )

st.divider()

st.header("About PromptBench")

st.write(
    f"Current task category: {task_category}. "
    "PromptBench is designed to compare prompting strategies "
    "using consistent inputs and evaluation criteria."
)
