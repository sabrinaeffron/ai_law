from scipy.stats import f_oneway
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

df = pd.concat(dfs, ignore_index=True)

variant_labels = {
    0: "25-30",
    1: "40-45",
    2: "55-60",
    3: "70-75"
}

df["variant_label"] = df["variant"].map(variant_labels)

anova_results = []

for (model, category), g in df.groupby(["model", "category"]):
    groups = [
        g[g["variant_label"] == v]["score"]
        for v in ["25-30", "40-45", "55-60", "70-75"]
    ]

    F, p = f_oneway(*groups)
    anova_results.append({
        "model": model,
        "category": category,
        "F": F,
        "p_value": p
    })

anova_df = pd.DataFrame(anova_results)
print(anova_df)

table = anova_df.copy()

# Optional: mark significance
table["significant"] = table["p_value"] < 0.05

# Optional: nicer formatting
table["F"] = table["F"].round(2)
table["p_value"] = table["p_value"].apply(
    lambda x: f"{x:.2e}" if pd.notnull(x) else ""
)

latex_table = table.to_latex(
    index=False,
    escape=False,
    caption="One-way ANOVA results testing the effect of age range on predicted hire scores, conducted separately for each model and job category.",
    label="tab:anova_age_effects",
)

print(latex_table)
