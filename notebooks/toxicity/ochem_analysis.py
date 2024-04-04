# %%
# Basket name: Patents+OCHEM+Enamine+Bradley+Begström (training)
# This basket belongs to user: published, created by dan2097

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from rdkit.Chem import AllChem, MolFromSmiles, Descriptors
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tzip
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np

random.seed(42)

# %%
HOME = "/home/ec2-user/np-clinical-trials"

# %%
coconut_smiles_list = (
    pd.read_csv(f"{HOME}/data/COCONUT_DB.smi", delim_whitespace=True, header=None)[0]
    .unique()
    .tolist()
)
# coconut_smiles_list = random.sample(coconut_smiles_list, 1000)

# %%
synthetic_smiles_list = (
    pd.read_csv(f"{HOME}/data/DDS-50.smi", delim_whitespace=True, header=None)[0]
    .unique()
    .tolist()
)
# synthetic_smiles_list = random.sample(synthetic_smiles_list, 1000)

# %%
OCHEM_subset_smiles_list = (
    pd.read_csv(f"{HOME}/data/OCHEM_data.csv", sep=";")["SMILES"].unique().tolist()
)
# OCHEM_subset_smiles_list = random.sample(OCHEM_subset_smiles_list, 1000)

# %%
results_dictionary = {
    "OCHEM": {"SMILES": OCHEM_subset_smiles_list},
    "Coconut": {"SMILES": coconut_smiles_list},
    "Synthetic": {"SMILES": synthetic_smiles_list},
}


def conversion_with_rdkit(input_smiles):
    try:
        return MolFromSmiles(input_smiles)
    except:
        return None


fpgen = AllChem.GetRDKitFPGenerator()
for current_subset in results_dictionary:
    molecules_list = []
    with Pool(16) as p:
        molecules_list = list(
            tqdm(
                p.imap(
                    conversion_with_rdkit, results_dictionary[current_subset]["SMILES"]
                ),
                total=len(results_dictionary[current_subset]["SMILES"]),
            )
        )

    results_dictionary[current_subset]["Molecules"] = molecules_list

# %%
fpgen = AllChem.GetRDKitFPGenerator()


# %%
def calculate_fingerprints(input_molecule):
    try:
        return Descriptors.CalcMolDescriptors(input_molecule)
    except:
        return None


for current_subset in results_dictionary:
    fps_list, valid_smiles = [], []
    with Pool(16) as p:
        fps_list = list(
            tqdm(
                p.imap(
                    calculate_fingerprints,
                    results_dictionary[current_subset]["Molecules"],
                ),
                total=len(results_dictionary[current_subset]["Molecules"]),
            )
        )
    valid_smiles = []
    for current_smiles, current_fingerprint in tzip(
        results_dictionary[current_subset]["SMILES"], fps_list
    ):
        if current_fingerprint is not None:
            valid_smiles.append(current_smiles)
        else:
            valid_smiles.append(None)

    fps_df = pd.DataFrame(fps_list)
    fps_df["SMILES"] = valid_smiles
    fps_df["source"] = current_subset
    results_dictionary[current_subset]["features"] = fps_df

# %%
full_fingerprints_df = pd.concat(
    [
        results_dictionary["OCHEM"]["features"],
        results_dictionary["Coconut"]["features"],
        results_dictionary["Synthetic"]["features"],
    ]
)

# %%
full_fingerprints_df.to_csv(f"{HOME}/data/full_fingerprints_df.csv", index=False)

# %%
ids_columns = full_fingerprints_df[["source", "SMILES"]]
features_df = full_fingerprints_df.drop(columns=["SMILES", "source"])

# %%
scaler = StandardScaler()
scaled_features = scaler.fit_transform(
    features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
)
scaled_features = pd.DataFrame(
    scaled_features, columns=features_df.columns, index=ids_columns.index
)

# %%
PCA_object = PCA(n_components=2)
PCA_features = PCA_object.fit_transform(scaled_features)

# %%
usable_df = pd.DataFrame(PCA_features, columns=["PC1", "PC2"]).reset_index(drop=True)
usable_df["source"] = ids_columns["source"].reset_index(drop=True)

# %%
usable_df

# %%
# Create a jointplot with Seaborn
sns.jointplot(
    data=usable_df,
    x="PC1",
    y="PC2",
    hue="source",
    alpha=0.3,
)
plt.xlabel("PC1")
plt.ylabel("PC2")


# Optionally, you can save the plot to a file
plt.savefig("PCA_plot.png")
plt.show()

# %%
Z = linkage(
    scaled_features, method="ward"
)  # Perform hierarchical/agglomerative clustering

# %%
clusters = fcluster(Z, t=30, criterion="maxclust")

# %%
cluster_colors = {
    1: "red",
    2: "green",
    3: "blue",
    4: "yellow",
    5: "purple",
    6: "orange",
    7: "pink",
    8: "brown",
    9: "black",
    0: "grey",
    10: "cyan",
    11: "magenta",
    12: "gold",
    13: "silver",
    14: "olive",
    15: "maroon",
    16: "navy",
    17: "teal",
    18: "aqua",
    19: "lime",
    20: "fuchsia",
    21: "silver",
    22: "olive",
    23: "maroon",
    24: "navy",
    25: "teal",
    26: "aqua",
    27: "lime",
    28: "fuchsia",
    29: "silver",
    30: "olive",
}
origin_colors = {"OCHEM": "red", "Coconut": "blue", "Synthetic": "green"}

# %%
"""
# kr-collapse
# Prepare row_colors for clusters
ids_columns["cluster"] = clusters

row_colors_clusters = ids_columns["cluster"].map(cluster_colors)
row_colors_origin = ids_columns["source"].map(origin_colors)

# Combine row_colors for both clusters and patients
row_colors_combined = pd.DataFrame(
    {
        "Cluster": row_colors_clusters,
        "Dataset": row_colors_origin,
    }
)

# Create the clustermap
ax = sns.clustermap(
    scaled_features,
    cmap="vlag",
    col_cluster=False,
    figsize=(10, 10),
    row_colors=row_colors_combined,
    method="ward",
    metric="euclidean",
)

ax.ax_heatmap.set_yticklabels([])
# hide x-axis labels
plt.setp(ax.ax_heatmap.get_xticklabels(), visible=False)

# Show the plot
plt.show()
"""

# %%
"""
ids_subset_list, features_subset_list = [], []
# Pick random sample from each cluster
for current_cluster in ids_columns["cluster"].unique():
    current_ids_subset = ids_columns.loc[ids_columns["cluster"] == current_cluster]
    current_features_subset = features_df.loc[current_ids_subset.index]
    ids_subset_list.append(current_ids_subset.sample(1))
    features_subset_list.append(current_features_subset.sample(1))

curated_ids_df = pd.concat(ids_subset_list, axis = 0)
curated_features_df = pd.concat(features_subset_list, axis = 0)
"""
