import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_DATA_PATH = os.path.join("data", "raw", "heart_disease_raw.csv")
SCREENSHOT_DIR = os.path.join("reports", "screenshots")


def generate_eda_visualizations(data_path: str = RAW_DATA_PATH, output_dir: str = SCREENSHOT_DIR):
    """
    Performs Exploratory Data Analysis and generates high-resolution
    custom visualizations for the report and MLflow tracking.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    # Set a distinct visual theme
    plt.style.use("seaborn-v0_8-whitegrid")
    palette = ["#2b5c8f", "#d95f02"]  # Custom Slate Blue & Burnt Orange

    # 1. Class Balance Plot with Percentage Annotations
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    counts = df["target"].value_counts().sort_index()
    total = len(df)
    bars = ax.bar(
        ["Absence (0)", "Presence (1)"],
        counts.values,
        color=palette,
        edgecolor="black",
        width=0.55,
        alpha=0.9
    )
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(
            f"{height} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )
    ax.set_title("Heart Disease Target Class Distribution (n = 303)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Patients", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.18)
    plt.tight_layout()
    class_balance_path = os.path.join(output_dir, "01_class_balance.png")
    plt.savefig(class_balance_path)
    plt.close()
    logging.info(f"Saved class balance plot to {class_balance_path}")

    # 2. Masked Upper-Triangle Spearman Correlation Heatmap
    fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
    corr = df.corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        vmin=-0.7, vmax=0.7,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8.5},
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.75, "label": "Spearman Correlation Coefficient"},
        ax=ax
    )
    ax.set_title("Spearman Rank Correlation Matrix of Clinical Features", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "02_correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    logging.info(f"Saved correlation heatmap to {heatmap_path}")

    # 3. Continuous Feature Distributions (Histograms + KDE Stratified by Target)
    continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    titles = [
        "Age (Years)",
        "Resting Blood Pressure (mm Hg)",
        "Serum Cholesterol (mg/dl)",
        "Max Heart Rate Achieved (bpm)",
        "ST Depression Induced by Exercise (oldpeak)"
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    axes = axes.flatten()

    for idx, (col, title) in enumerate(zip(continuous_cols, titles)):
        sns.histplot(
            data=df, x=col, hue="target", kde=True,
            palette=palette, element="step", stat="density",
            common_norm=False, alpha=0.35, ax=axes[idx]
        )
        axes[idx].set_title(title, fontsize=11, fontweight="bold")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Density")

    # Use the 6th subplot for Chest Pain Type (cp) vs Target
    cp_ct = pd.crosstab(df["cp"], df["target"], normalize="index") * 100
    cp_ct.plot(kind="bar", stacked=True, color=palette, edgecolor="black", ax=axes[5], alpha=0.9)
    axes[5].set_title("Heart Disease Rate by Chest Pain Type (cp)", fontsize=11, fontweight="bold")
    axes[5].set_xlabel("Chest Pain Type (1: Typical, 2: Atypical, 3: Non-anginal, 4: Asymptomatic)")
    axes[5].set_ylabel("Percentage (%)")
    axes[5].tick_params(axis="x", rotation=0)
    axes[5].legend(["Absence (0)", "Presence (1)"], loc="upper left", fontsize=9)

    plt.suptitle(
        "Clinical Feature Distributions Stratified by Heart Disease Status",
        fontsize=15,
        fontweight="bold",
        y=1.01)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "03_feature_distributions.png")
    plt.savefig(dist_path, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved feature distributions plot to {dist_path}")


if __name__ == "__main__":
    generate_eda_visualizations()
