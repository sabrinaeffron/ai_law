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

plt.figure(figsize=(8,5))
sns.barplot(
    data=df,
    x="variant_label",
    y="score",
    estimator="mean",
    errorbar="se"
)

plt.xlabel("Age Range")
plt.ylabel("Average Score")
plt.title("Average Score by Age Range")
plt.ylim(0,100)
plt.tight_layout()
plt.show()