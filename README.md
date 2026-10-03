# 🤖 Arabic LLM Evaluation & Benchmarking

A practical evaluation and benchmarking system for comparing Arabic Large Language Models on text summarization tasks using both **quality** and **inference-performance** metrics.

## 🎯 Project Objective

The goal of this project is to build a reproducible benchmark for Arabic summarization models rather than evaluating a model using a single metric.

The system compares multiple models on the same fixed evaluation dataset and measures:

* Lexical similarity
* Semantic similarity
* Factuality
* Relevance
* Hallucination
* Inference latency
* Throughput
* Output length

The project also provides an interactive **Streamlit dashboard** for exploring the results.

---

## 🏗️ System Architecture

```text
Arabic Dataset
      │
      ▼
Fixed Benchmark Dataset
      │
      ├───────────────┐
      ▼               ▼
 Qwen 2.5 1.5B       mT5 Arabic
      │               │
      └───────┬───────┘
              ▼
       Generated Summaries
              │
              ▼
       ┌──────────────────┐
       │    Evaluation    │
       ├──────────────────┤
       │ ROUGE            │
       │ BERTScore        │
       │ Factuality       │
       │ Relevance        │
       │ Hallucination    │
       │ Latency          │
       │ Throughput       │
       └──────────────────┘
              │
              ▼
        Benchmark Results
              │
              ▼
      Streamlit Dashboard
```

---

## 📊 Dataset

The benchmark was created from the **MArSum Arabic summarization dataset**.

Dataset:

`arbml/MArSum`

A fixed subset of **200 Arabic article-summary pairs** was sampled using a fixed random seed to ensure reproducibility.

```python
benchmark_df = df.sample(
    n=200,
    random_state=42
).reset_index(drop=True)
```

---

## 🤖 Models

### Qwen2.5-1.5B-Instruct

A decoder-only instruction-tuned language model used for Arabic summarization through prompt-based inference.

### mT5-small-Arabic-Summarization

An Arabic summarization model based on the mT5 architecture.

Both models were evaluated on the **same 200 benchmark samples**.

---

## 📏 Evaluation Metrics

### ROUGE

ROUGE measures lexical overlap between generated summaries and reference summaries.

The project evaluates:

* ROUGE-1
* ROUGE-2
* ROUGE-L
* ROUGE-Lsum

### BERTScore

BERTScore measures semantic similarity between the generated summary and the reference summary using contextual embeddings.

This is useful because two summaries can express similar meanings using different wording.

### Factuality

An LLM-as-a-Judge evaluator assesses whether claims in the generated summary are supported by the source article.

The evaluator produces:

* Factuality score: 1–5
* Relevance score: 1–5
* Hallucination classification
* Short explanation

The reference summary is intentionally excluded from the factuality evaluation.

---

## ⚡ Performance Metrics

The benchmark also measures inference performance:

### Latency

Average inference time per sample.

### Throughput

Number of samples processed per minute.

### Output Length

Average generated summary length in words.

---

## 📊 Benchmark Results

| Metric              |      Qwen2.5-1.5B |        mT5 Arabic |
| ------------------- | ----------------: | ----------------: |
| ROUGE-1             |            0.0683 |            0.0270 |
| ROUGE-2             |            0.0016 |            0.0000 |
| ROUGE-L             |            0.0679 |            0.0275 |
| BERTScore-F1        |            0.6594 |            0.6772 |
| Factuality          |            3.96/5 |            3.94/5 |
| Relevance           |            4.00/5 |            4.02/5 |
| Hallucination Rate* |              0.0% |              0.0% |
| Avg Latency         |          4.86 sec |          1.16 sec |
| Throughput          | 12.34 samples/min | 51.52 samples/min |
| Avg Output Length   |       47.61 words |       18.11 words |

* Hallucination rate was calculated from a 50-sample LLM-as-a-Judge evaluation subset and should not be interpreted as proof of zero hallucination.

---

## 🔎 Key Observations

The benchmark demonstrates that different evaluation metrics can produce different conclusions about model behavior.

For example:

* Qwen achieved higher ROUGE scores on this benchmark.
* mT5 achieved a higher BERTScore-F1.
* mT5 had substantially lower inference latency.
* mT5 processed more samples per minute.
* Qwen generated considerably longer summaries.
* Factuality and relevance scores were very similar between the models on the evaluated subset.

These observations are specific to this benchmark and should not be generalized to all Arabic summarization tasks.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive dashboard providing:

* Model selection
* Quality metrics
* Inference-performance metrics
* Model comparison tables
* ROUGE charts
* BERTScore comparison
* Latency comparison
* Throughput comparison
* Output-length comparison
* Sample-level evaluation
* Direct Qwen vs mT5 comparison

Run the dashboard with:

```bash
python -m streamlit run dashboard/app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## 📁 Project Structure

```text
LLM-Evaluation-Benchmark/
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_baseline_inference.ipynb
│   └── 03_factuality_evaluation.ipynb
│
├── data/
│   └── benchmark/
│       └── arabic_summarization_benchmark.csv
│
├── results/
│   ├── qwen2.5-1.5b-results.csv
│   ├── mt5-arabic-results.csv
│   ├── qwen-factuality-results.csv
│   ├── mt5-factuality-results.csv
│   └── final_benchmark.csv
│
├── dashboard/
│   └── app.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Installation

Create and activate the environment:

```bash
conda create -n ai_env python=3.11
conda activate ai_env
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the notebooks in order:

```text
01_dataset_analysis.ipynb
        ↓
02_baseline_inference.ipynb
        ↓
03_factuality_evaluation.ipynb
```

Then launch the dashboard:

```bash
python -m streamlit run dashboard/app.py
```

---

## ⚠️ Limitations

This project has several important limitations:

1. The benchmark contains 200 samples and may not represent all Arabic domains or dialects.

2. ROUGE can underestimate semantic similarity when generated and reference summaries use different wording.

3. The reference summaries in the dataset can be significantly shorter than generated summaries, which can affect lexical metrics.

4. Factuality evaluation uses an LLM-as-a-Judge approach and therefore is not equivalent to human-verified ground truth.

5. The initial factuality evaluation uses a subset of 50 samples.

6. The same Qwen model family was used as both a summarization model and the initial evaluator, which may introduce evaluator bias.

7. Results are benchmark-specific and should not be interpreted as universal model rankings.

---

## 🚀 Future Improvements

Possible extensions include:

* Add more Arabic LLMs
* Add human evaluation
* Add multilingual evaluation
* Add factuality metrics beyond LLM-as-a-Judge
* Evaluate larger benchmark subsets
* Add confidence intervals
* Add statistical significance testing
* Add experiment tracking
* Add automated benchmark pipelines
* Compare different prompting strategies
* Evaluate fine-tuned models against base models

---

## 🧠 Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Evaluate
* ROUGE
* BERTScore
* Pandas
* Streamlit
* Jupyter Notebook

---

## 👨‍💻 Purpose

This project was built as a practical **Generative AI / LLM Engineering** project focusing on model evaluation, benchmarking, inference, and deployment rather than simply generating text from an LLM.
