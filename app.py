"""Q-Sentinel — Streamlit Dashboard.

A cybersecurity-themed web interface for phishing URL detection.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

from src.features import FEATURE_NAMES, extract_features
from src.model import load_model, predict_url

# ---------------------------------------------------------------------------
# Page config & global CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Q-Sentinel",
    page_icon="🛡️",
    layout="wide",
)

_CSS = """
<style>
    /* Dark cyber theme overrides */
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #00e5ff;
    }
    .stMetric label {
        color: #b388ff;
    }
    .safe-badge {
        background: linear-gradient(135deg, #00e5ff22, #00e5ff11);
        border: 1px solid #00e5ff;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #00e5ff;
        font-size: 1.15rem;
        text-align: center;
    }
    .danger-badge {
        background: linear-gradient(135deg, #ff174422, #ff174411);
        border: 1px solid #ff1744;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #ff1744;
        font-size: 1.15rem;
        text-align: center;
    }
    .info-card {
        background: #1a1a2e;
        border: 1px solid #b388ff44;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.5rem;
    }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🔍 URL Scanner", "📊 Model Evaluation", "📁 Batch Analysis", "ℹ️ About"],
)

# ---------------------------------------------------------------------------
# Helper: load model once and cache
# ---------------------------------------------------------------------------
_MODEL_AVAILABLE = (Path("models") / "model.joblib").is_file()


@st.cache_resource
def _get_model():
    return load_model()


# ---------------------------------------------------------------------------
# PAGE: URL Scanner
# ---------------------------------------------------------------------------
if page == "🔍 URL Scanner":
    st.markdown("# 🛡️ Q-Sentinel")
    st.markdown(
        "*Hybrid Quantum–Classical AI for Phishing URL Detection*",
    )
    st.markdown("---")

    if not _MODEL_AVAILABLE:
        st.warning(
            "⚠️ No trained model found.  "
            "Run `python train.py --data <your_dataset.csv>` first."
        )
    else:
        url_input = st.text_input(
            "Enter a URL to analyse",
            placeholder="https://example.com/login?user=123",
        )

        if st.button("🔎 Scan URL", type="primary"):
            if not url_input or not url_input.strip():
                st.error("Please enter a valid URL.")
            else:
                clf, scaler = _get_model()
                result = predict_url(url_input, clf=clf, scaler=scaler)

                # Prediction badge
                if result["label"] == 1:
                    st.markdown(
                        f'<div class="danger-badge">⚠️ {result["label_text"]}<br>'
                        f'Confidence: {result["probability_phishing"]:.1%}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="safe-badge">✅ {result["label_text"]}<br>'
                        f'Confidence: {result["probability_legitimate"]:.1%}</div>',
                        unsafe_allow_html=True,
                    )

                st.caption(
                    "⚠️ Disclaimer: This prediction is a statistical estimate, "
                    "not proof of safety. Always verify URLs independently."
                )

                # Feature inspection
                st.markdown("### 🔬 Feature Inspection")
                feat_df = pd.DataFrame(
                    list(result["features"].items()),
                    columns=["Feature", "Value"],
                )
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PAGE: Model Evaluation
# ---------------------------------------------------------------------------
elif page == "📊 Model Evaluation":
    st.markdown("# 📊 Model Evaluation")
    st.markdown("---")

    metrics_path = Path("results") / "metrics.json"
    report_path = Path("results") / "classification_report.txt"

    if not metrics_path.is_file():
        st.warning(
            "⚠️ No evaluation results found.  "
            "Train a model first with `python train.py --data <dataset.csv>`."
        )
    else:
        with open(metrics_path) as f:
            metrics = json.load(f)

        st.info(
            "ℹ️ These metrics were computed on the held-out test split during "
            "training. They reflect model performance on the dataset you "
            "provided — not on all possible phishing URLs.",
            icon="📌",
        )

        # KPI cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{metrics['precision']:.4f}")
        c2.metric("Recall", f"{metrics['recall']:.4f}")
        c3.metric("F1-Score", f"{metrics['f1']:.4f}")
        c4.metric("False Positive Rate", f"{metrics['false_positive_rate']:.4f}")

        st.markdown(f"**Train samples:** {metrics['num_train']}  |  "
                    f"**Test samples:** {metrics['num_test']}")

        # Confusion matrix heatmap
        st.markdown("### Confusion Matrix")
        cm = metrics["confusion_matrix"]
        labels = ["Legitimate", "Phishing"]
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[[0, "#0e1117"], [1, "#00e5ff"]],
            showscale=True,
        )
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="#e0e0e0",
            width=500,
            height=450,
        )
        st.plotly_chart(fig, use_container_width=False)

        # Classification report
        if report_path.is_file():
            st.markdown("### Classification Report")
            st.code(report_path.read_text(), language="text")

# ---------------------------------------------------------------------------
# PAGE: Batch Analysis
# ---------------------------------------------------------------------------
elif page == "📁 Batch Analysis":
    st.markdown("# 📁 Batch URL Analysis")
    st.markdown("---")

    if not _MODEL_AVAILABLE:
        st.warning(
            "⚠️ No trained model found.  "
            "Run `python train.py --data <dataset.csv>` first."
        )
    else:
        uploaded = st.file_uploader(
            "Upload a CSV with a column named 'url'", type=["csv"]
        )

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
                st.stop()

            if "url" not in batch_df.columns:
                st.error(
                    f"CSV must contain a column named 'url'. "
                    f"Found: {list(batch_df.columns)}"
                )
                st.stop()

            clf, scaler = _get_model()

            results = []
            progress = st.progress(0, text="Scanning URLs...")
            total = len(batch_df)
            for i, url in enumerate(batch_df["url"]):
                r = predict_url(str(url), clf=clf, scaler=scaler)
                results.append({
                    "url": url,
                    "prediction": r["label_text"],
                    "phishing_probability": round(r["probability_phishing"], 4),
                })
                progress.progress((i + 1) / total)

            progress.empty()
            result_df = pd.DataFrame(results)

            # Summary
            phish_count = sum(1 for r in results if r["prediction"] == "Potentially phishing")
            legit_count = total - phish_count

            c1, c2, c3 = st.columns(3)
            c1.metric("Total URLs", total)
            c2.metric("Potentially Phishing", phish_count)
            c3.metric("Likely Legitimate", legit_count)

            st.caption(
                "⚠️ These predictions are statistical estimates, "
                "not proof of safety."
            )

            # Distribution chart
            fig = go.Figure(data=[
                go.Bar(
                    x=["Likely Legitimate", "Potentially Phishing"],
                    y=[legit_count, phish_count],
                    marker_color=["#00e5ff", "#ff1744"],
                )
            ])
            fig.update_layout(
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font_color="#e0e0e0",
                yaxis_title="Count",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Detailed Results")
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            # Download
            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results as CSV",
                csv_out,
                file_name="qsentinel_batch_results.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# PAGE: About
# ---------------------------------------------------------------------------
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About Q-Sentinel")
    st.markdown("---")
    st.markdown(
        """
        **Q-Sentinel** is a hybrid quantum–classical AI system designed to
        detect phishing URLs using machine-learning analysis of URL structure.

        ### Current Version (v1 — Classical Baseline)

        - **Feature extraction**: 13 lexical / structural features parsed from
          the URL string (no network requests).
        - **Model**: scikit-learn `RandomForestClassifier` with 100 estimators.
        - **Preprocessing**: `StandardScaler` fitted only on training data.
        - **Evaluation**: Precision, recall, F1-score, false-positive rate, and
          confusion matrix computed on a held-out stratified test split.

        ### Roadmap

        - 🔬 **v2** — Quantum feature map & variational classifier (Qiskit /
          PennyLane).
        - 🌐 **v3** — Real-time URL reputation enrichment.
        - 📦 **v4** — Browser extension & API.

        ### ⚠️ Disclaimer

        This tool provides *statistical estimates* based on URL lexical
        patterns.  It is **not** a substitute for professional cybersecurity
        software.  Always verify suspicious URLs through trusted sources.

        ---
        *Built for VTU Q-Sentinel Hackathon · 2026*
        """
    )
