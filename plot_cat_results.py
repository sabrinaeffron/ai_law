import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("outputs/test_results_gpt-4-turbo.csv")

variant_labels = {
    0: "25-30",
    1: "40-45",
    2: "55-60",
    3: "70-75"
}

df["variant_label"] = df["variant"].map(variant_labels)

df["variant_label"] = pd.Categorical(
    df["variant_label"],
    categories=["25-30", "40-45", "55-60", "70-75"],
    ordered=True
)

g = sns.catplot(
    data=df,
    kind="bar",
    x="variant_label",
    y="score",
    col="category",
    estimator="mean",
    errorbar="se",
    height=4,
    aspect=0.8,
)

g.set_axis_labels("Age Range", "Average Hire Score")
g.set_titles("{col_name}")
for ax in g.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

plt.show()