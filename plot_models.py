import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

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

df = pd.concat(dfs, ignore_index=True)

print("Loaded columns:", df.columns.tolist())
print("Models in combined df:", df["model"].unique())

variant_labels = {
    0: "25-30",
    1: "40-45",
    2: "55-60",
    3: "70-75"
}

df["variant_label"] = df["variant"].map(variant_labels)

g = sns.catplot(
    data=df,
    kind="bar",
    x="variant_label",
    y='score',
    hue="model",
    col="category",
    estimator="mean",
    errorbar="se",
    height=4,
    aspect=0.9
)

g.set_axis_labels("Age range", "Average hire score")
g.set_titles("{col_name}")

for ax in g.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

plt.tight_layout()
plt.show()
