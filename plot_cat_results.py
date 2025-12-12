import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


models = [
    "gpt-5",
    "o3-mini",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "mistral-small-2503",
    "Mistral-Large-2411",
]

model_files = {
    model: f"outputs/test_results_{model}.csv" for model in models
}

dfs = []
for model_name, path in model_files.items():
    df_model = pd.read_csv(path)
    df_model["model"] = model_name
    dfs.append(df_model)

variant_labels = {
    0: "25-30",
    1: "40-45",
    2: "55-60",
    3: "70-75"
}

df = pd.concat(dfs, ignore_index=True)

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