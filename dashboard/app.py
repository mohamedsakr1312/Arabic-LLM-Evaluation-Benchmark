import streamlit as st
import pandas as pd

# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Arabic LLM Benchmark",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# Load Data
# =========================================================

@st.cache_data
def load_data():
    benchmark = pd.read_csv("results/final_benchmark.csv")

    qwen_results = pd.read_csv(
        "results/qwen2.5-1.5b-results.csv"
    )

    mt5_results = pd.read_csv(
        "results/mt5-arabic-results.csv"
    )

    qwen_fact = pd.read_csv(
        "results/qwen-factuality-results.csv"
    )

    mt5_fact = pd.read_csv(
        "results/mt5-factuality-results.csv"
    )

    return (
        benchmark,
        qwen_results,
        mt5_results,
        qwen_fact,
        mt5_fact
    )


benchmark, qwen_results, mt5_results, qwen_fact, mt5_fact = load_data()

# =========================================================
# Header
# =========================================================

st.title("🤖 Arabic LLM Evaluation & Benchmarking")

st.markdown(
    """
    **Benchmarking Arabic summarization models using
    multiple quality and performance metrics.**
    """
)

st.divider()

# =========================================================
# Model Selection
# =========================================================

model_names = benchmark["Model"].tolist()

selected_model = st.selectbox(
    "Select Model",
    model_names
)

selected_row = benchmark[
    benchmark["Model"] == selected_model
].iloc[0]

# =========================================================
# Model Performance
# =========================================================

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "ROUGE-1",
        f"{selected_row['ROUGE-1']:.4f}"
    )

with col2:
    st.metric(
        "ROUGE-L",
        f"{selected_row['ROUGE-L']:.4f}"
    )

with col3:
    st.metric(
        "BERTScore-F1",
        f"{selected_row['BERTScore-F1']:.4f}"
    )

with col4:
    st.metric(
        "Factuality",
        f"{selected_row['Factuality']:.2f}/5"
    )

# =========================================================
# Inference Performance
# =========================================================

st.subheader("⚡ Inference Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Latency",
        f"{selected_row['Avg Latency (sec)']:.2f}s"
    )

with col2:
    st.metric(
        "Throughput",
        f"{selected_row['Throughput (samples/min)']:.2f}/min"
    )

with col3:
    st.metric(
        "Output Length",
        f"{selected_row['Avg Output Length']:.1f} words"
    )

with col4:
    st.metric(
        "Hallucination Rate",
        f"{selected_row['Hallucination Rate']:.1%}"
    )

st.divider()

# =========================================================
# Model Comparison Table
# =========================================================

st.subheader("🔎 Model Comparison")

comparison_columns = [
    "Model",
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "BERTScore-F1",
    "Factuality",
    "Relevance",
    "Hallucination Rate",
    "Avg Latency (sec)",
    "Throughput (samples/min)",
    "Avg Output Length"
]

st.dataframe(
    benchmark[comparison_columns].round(4),
    use_container_width=True,
    hide_index=True
)

st.divider()

# =========================================================
# Quality Metrics
# =========================================================

st.subheader("📈 Quality Metrics")

quality_col1, quality_col2 = st.columns(2)

with quality_col1:

    st.markdown("#### ROUGE Metrics")

    rouge_chart = benchmark.set_index("Model")[
        [
            "ROUGE-1",
            "ROUGE-2",
            "ROUGE-L"
        ]
    ]

    st.bar_chart(rouge_chart)


with quality_col2:

    st.markdown("#### BERTScore")

    bert_chart = benchmark.set_index("Model")[
        ["BERTScore-F1"]
    ]

    st.bar_chart(bert_chart)

# =========================================================
# Inference Performance Charts
# =========================================================

st.subheader("⚡ Inference Performance")

speed_col1, speed_col2 = st.columns(2)

with speed_col1:

    st.markdown("#### ⏱ Average Latency")

    latency_chart = benchmark.set_index("Model")[
        ["Avg Latency (sec)"]
    ]

    st.bar_chart(latency_chart)


with speed_col2:

    st.markdown("#### 🚀 Throughput")

    throughput_chart = benchmark.set_index("Model")[
        ["Throughput (samples/min)"]
    ]

    st.bar_chart(throughput_chart)

# =========================================================
# Output Characteristics
# =========================================================

st.subheader("📝 Output Characteristics")

length_chart = benchmark.set_index("Model")[
    ["Avg Output Length"]
]

st.bar_chart(length_chart)

st.divider()

# =========================================================
# Sample Evaluation
# =========================================================

st.subheader("📝 Sample Evaluation")

# Select corresponding data

if selected_model == "Qwen2.5-1.5B-Instruct":

    results_df = qwen_results
    fact_df = qwen_fact

else:

    results_df = mt5_results
    fact_df = mt5_fact


# Sample selector

sample_index = st.number_input(
    "Select Sample ID",
    min_value=0,
    max_value=len(results_df) - 1,
    value=0,
    step=1
)

sample = results_df.iloc[sample_index]

sample_id = sample["id"]

fact_result = fact_df[
    fact_df["id"] == sample_id
]


# =========================================================
# Source Article
# =========================================================

st.markdown("### 📰 Source Article")

st.write(sample["text"])

# =========================================================
# Generated Summary
# =========================================================

st.markdown("### 🤖 Generated Summary")

st.info(sample["prediction"])

# =========================================================
# Reference Summary
# =========================================================

st.markdown("### 📌 Reference Summary")

st.success(sample["reference_summary"])

# =========================================================
# Factuality Evaluation
# =========================================================

if len(fact_result) > 0:

    fact_result = fact_result.iloc[0]

    st.markdown("### 🔍 Factuality Evaluation")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Factuality",
            f"{fact_result['factuality']:.1f}/5"
        )

    with col2:

        st.metric(
            "Relevance",
            f"{fact_result['relevance']:.1f}/5"
        )

    with col3:

        hallucination = fact_result["hallucination"]

        st.metric(
            "Hallucination",
            "Yes" if hallucination else "No"
        )

    st.markdown("**Judge Explanation:**")

    st.write(
        fact_result["explanation"]
    )