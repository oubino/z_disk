import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sheets = [0,1,2,3,4,5]

file_name = "table_g1_3_12_25"

# Load in excel table
df = pd.read_excel(
    f"manuscript_figures/{file_name}.xlsx",
    sheet_name=sheets
)

# for each sheet
dfs = []
vals = []
for sheet in sheets:
    dfx = df[sheet]

    dist_cols = [col for col in dfx.columns if col.startswith("characteristic_distance_")]
    dist_cols.remove("characteristic_distance_broadening")
    
    all_cols = ["Protein", "Direction"] + dist_cols

    dfx = dfx[all_cols]

    for col in dist_cols:
        dfx[[f"{col}_value", f"{col}_error"]] = (
            dfx[col].str.split("\+\-", expand=True)
        )

    new_cols = [f"{col}_value" for col in dist_cols] + \
           [f"{col}_error" for col in dist_cols]

    dfx[new_cols] = dfx[new_cols].apply(pd.to_numeric)

    dfs.append(dfx)

    vals.append(dfx["Protein"][0])
    vals.append(dfx["Direction"][0])
    vals.append(dfx[new_cols].median())

print(vals)

# visualise spread
df = pd.concat(dfs, axis=0, ignore_index=True)
df["Protein & direction"] = df["Protein"].astype(str) + " " + df["Direction"].astype(str)

df = df.drop(columns=["Protein", "Direction"])

df = df.melt(
    id_vars=["Protein & direction"], 
    value_vars=["characteristic_distance_1_value", "characteristic_distance_2_value", "characteristic_distance_3_value"],
    var_name="measurement",
    value_name="Distance / nm"
)

plt.figure(figsize=(6, 4))

df["measurement"] = df["measurement"].replace({
    "characteristic_distance_1_value": "Characteristic Distance 1",
    "characteristic_distance_2_value": "Characteristic Distance 2",
    "characteristic_distance_3_value": "Characteristic Distance 3",
})


sns.stripplot(
    data=df,
    x="Protein & direction",
    y="Distance / nm",
    hue="measurement",
    alpha=1,
    s=4,
    legend=True,
    dodge=False,
    order=["ACTN2 axial", "Z1Z2 axial", "ZASP6 axial", "ACTN2 transverse", "Z1Z2 transverse", "ZASP6 transverse"]
)

# horizontal lines
plt.axhline(y=16.25, color="black", linestyle="--", linewidth=1)
plt.axhline(y=16.88, color="black", linestyle="--", linewidth=1)
plt.axhline(y=18.22, color="black", linestyle="--", linewidth=1)
plt.axhline(y=10.32, color="black", linestyle="--", linewidth=1)
plt.axhline(y=30.00, color="black", linestyle="--", linewidth=1)
plt.axhline(y=51.41, color="black", linestyle="--", linewidth=1)
plt.axhline(y=19.14, color="black", linestyle="--", linewidth=1)
plt.axhline(y=27.07, color="black", linestyle="--", linewidth=1)
plt.axhline(y=38.28, color="black", linestyle="--", linewidth=1)
plt.axhline(y=28.17, color="black", linestyle="--", linewidth=1)

#sns.pointplot(
#    data=df, 
#    x="Protein_Direction", 
#    y="Distance / nm", 
#    hue="measurement",
#    estimator="median",
#    linestyle="none", 
#    errorbar=None,
#    marker="_", 
#    markersize=20, 
#    markeredgewidth=2,
#)

plt.xticks(rotation=45, ha="right")

plt.savefig(f"manuscript_figures/figure_H1/charac_dists_{file_name}.svg",bbox_inches="tight", transparent=True)