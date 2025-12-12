from scipy.stats import chi2_contingency
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

results = []

for (model, category), g in df.groupby(["model", "category"]):
    table = pd.crosstab(g["variant_label"], g["hire"])
    chi2, p, dof, expected = chi2_contingency(table)
    results.append({
        "model": model,
        "category": category,
        "chi2": chi2,
        "p_value": p
    })

chi_df = pd.DataFrame(results)
chi_df["significant"] = chi_df["p_value"] < 0.05
print(chi_df)

table = chi_df.copy()
table["chi2"] = table["chi2"].round(2)
table["p_value"] = table["p_value"].apply(lambda x: f"{x:.1e}")
latex_table = table.to_latex(
    index=False,
    escape=False,
    caption=(
        "Chi-square tests of independence assessing whether the probability "
        "of a positive hiring decision differs across age ranges, "
        "conducted separately for each model and job category."
    ),
    label="tab:chi_square_hire",
)

print(latex_table)
