"""Utils to run NP-likeness analysis on clinical trials data"""

import os
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import requests
import numpy as np
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem, Crippen, Descriptors, Lipinski, RDConfig
from rdkit.Contrib.NP_Score import npscorer
from tqdm import tqdm

# Needed to get the Synthetic accessibility score loaded from RDKit (https://github.com/rdkit/rdkit/issues/2279)
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402


def smiles_to_inchikey(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(mol).split("-")[0]
    except Exception as e:
        return None


def get_duplicates(all_smiles):
    # Generate Morgan fingerprints for all smiles

    # parse smiles and validate they are valid
    mol_list = []
    valid_smiles = []

    for smiles in all_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as e:
            continue

        if mol is None:
            continue

        mol_list.append(mol)
        valid_smiles.append(smiles)

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=2, nBits=2048)
        for mol in mol_list
    ]

    # Calculate pairwise similarity Tanimoto
    similarity_matrix = []
    for i in range(len(fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        similarity_matrix.append(similarities)

    # Identify duplicates
    duplicates = []
    for i in tqdm(range(len(similarity_matrix))):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] > 0.95:
                duplicates.append((i, j))

    return valid_smiles, duplicates


def get_npclassifier(
    smiles,
    np_classifier_cache={},  # noqa: B006
):
    if smiles in np_classifier_cache:
        return np_classifier_cache[smiles]

    try:
        response = requests.get(
            f"https://npclassifier.ucsd.edu/classify?smiles={smiles}"
        )
        data = response.json()
    except:
        return {}

    metadata = {}

    if "class_results" in data and len(data["class_results"]) > 0:
        metadata["class"] = data["class_results"][0]

    if "superclass_results" in data and len(data["superclass_results"]) > 0:
        metadata["superclass"] = data["superclass_results"][0]

    if "pathway_results" in data and len(data["pathway_results"]) > 0:
        metadata["pathway"] = data["pathway_results"][0]

    if "isglycoside" in data:
        metadata["is_glycoside"] = data["isglycoside"]

    np_classifier_cache[smiles] = metadata

    return metadata


def calculate_percentage_nps_synthetics_hybrids(
    scores: list, dataset_name: str, plot=False
) -> None:
    """Calculate the percentage of each class and plot the distribution."""

    # Calculate the percentage of each class
    hybrid_percentage = round(
        len([score for score in scores if score >= 0 and score <= 0.6])
        / len(scores)
        * 100,
        2,
    )

    np_like_percentage = round(
        len([score for score in scores if score > 0.6]) / len(scores) * 100, 2
    )

    synthetics_percentage = round(
        len([score for score in scores if score < 0]) / len(scores) * 100, 2
    )

    if plot:
        sns.distplot(scores, bins=300)

        # add x and y labels

        plt.xlabel("NP Likeness")
        plt.ylabel("Density")
        plt.title(f"Distribution of NP Likeness Scores {len(scores)} {dataset_name}")

        # y limit
        plt.ylim(0, 0.6)

        # add vertical line on -1 and +1

        plt.axvline(x=0.6, color="green")
        plt.axvline(x=0, color="orange")

        # Add text to the plot right of the vertical line at x = 1
        plt.text(
            2.05,
            0.4,
            "NP like (>0.6)",
            horizontalalignment="left",
            size="medium",
            color="green",
            weight="semibold",
        )

        plt.text(
            -3.05,
            0.4,
            "Synthetic (< 0)",
            horizontalalignment="left",
            size="medium",
            color="orange",
            weight="semibold",
        )

        plt.text(
            -0.25,
            0.5,
            f"{hybrid_percentage}%",
            horizontalalignment="left",
            size="medium",
            color="black",
            weight="semibold",
        )

        plt.text(
            2.05,
            0.5,
            f"{np_like_percentage}%",
            horizontalalignment="left",
            size="medium",
            color="green",
            weight="semibold",
        )

        plt.text(
            -3.05,
            0.5,
            f"{synthetics_percentage}%",
            horizontalalignment="left",
            size="medium",
            color="orange",
            weight="semibold",
        )

        # save the plot
        # plt.savefig(f"np_likeness_{filename}.png", dpi=500)

        assert sum([np_like_percentage, synthetics_percentage, hybrid_percentage]) > 99

        assert sum([np_like_percentage, synthetics_percentage, hybrid_percentage]) < 101

        plt.show()

    return {
        "hybrid_percentage": hybrid_percentage,
        "np_like_percentage": np_like_percentage,
        "synthetics_percentage": synthetics_percentage,
    }


def remove_duplicates(all_smiles):
    print(f"{len(set(all_smiles))} smiles before removing duplicates")
    valid_smiles, duplicates = get_duplicates(list(all_smiles))

    print(
        f"{len(duplicates)} duplicates found {round(len(duplicates) / len(valid_smiles), 2) * 100}% of the total)"
    )

    smiles_to_remove = []
    duplicate_pairs = []

    # Remove one of the duplicates
    for i, j in duplicates:
        smiles_to_remove.append(valid_smiles[j])
        duplicate_pairs.append((valid_smiles[i], valid_smiles[j]))

    # remove duplicates
    for smiles in smiles_to_remove:
        if smiles in valid_smiles:
            valid_smiles.remove(smiles)

    return valid_smiles, duplicate_pairs


def get_first_part_inchikey(inchikey):
    return inchikey.split("-")[0]


def get_np_likeness_score(
    smiles,
    np_model,
    np_likeness_cache={},  # noqa: B006
):
    if smiles in np_likeness_cache:
        return np_likeness_cache[smiles]

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        return None

    if not mol:
        return None

    score = npscorer.scoreMol(mol, np_model)

    if not score:
        return None

    np_likeness_cache[smiles] = score

    return score


def get_np_scores(
    all_smiles,
    skip_duplicate=True,
    skip_np_classifier=False,
    np_likeness_cache={},  # noqa: B006
    sas_cache={},  # noqa: B006
    lipinski_cache={},  # noqa: B006
    np_classifier_cache={},  # noqa: B006
    plot=False,
    logging=False,
):
    if skip_duplicate:
        all_smiles, duplicate_pairs = remove_duplicates(all_smiles)

    # Calculate NP-likeness scores
    np_model = npscorer.readNPModel()

    skipped = 0

    scores = []
    is_lipinski_np = []
    is_lipinski_synthetics = []
    qed_scores_np = []
    qed_scores_synthetics = []

    inchikeys_dict = defaultdict(set)

    super_class_dict = defaultdict(set)
    class_dict = defaultdict(set)
    pathway_dict = defaultdict(set)

    synthetic_accessibility_np = []
    synthetic_accessibility_synthetics = []

    for smiles in tqdm(all_smiles):
        # Process salts to get the largest molecule
        if "." in smiles:
            smiles = sorted(smiles.split("."), key=len)[-1]

        if " " in smiles:
            smiles = sorted(smiles.split(" "), key=len)[-1]

        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as e:
            skipped += 1
            continue

        if not mol:
            print(f"Could not parse smiles: {smiles}")
            continue

        # if number of atoms less than 4, skip
        if mol.GetNumAtoms() < 4:
            continue

        if skip_np_classifier is False:
            if smiles in np_classifier_cache:
                np_classifier_metadata = np_classifier_cache[smiles]
            else:
                try:
                    np_classifier_metadata = get_npclassifier(smiles)
                except Exception as e:
                    print(e)
                    continue

                np_classifier_cache[smiles] = np_classifier_metadata

        if smiles in np_likeness_cache:
            score = np_likeness_cache[smiles]

        else:
            score = npscorer.scoreMol(mol, np_model)
            np_likeness_cache[smiles] = score

        if smiles in sas_cache:
            sas_score = sas_cache[smiles]
        else:
            sas_score = sascorer.calculateScore(mol)
            sas_cache[smiles] = sas_score

        if not sas_score:
            skipped += 1
            continue

        scores.append(score)

        if smiles in lipinski_cache:
            is_lipinski = lipinski_cache[smiles]

        else:
            is_lipinski = lipinski_pass(mol)
            lipinski_cache[smiles] = is_lipinski

        if score > 0.6:
            is_lipinski_np.append(is_lipinski)
            synthetic_accessibility_np.append(sas_score)
            qed_scores_np.append(QED.qed(mol))

            inchikeys_dict["np-like"].add(
                get_first_part_inchikey(Chem.inchi.MolToInchiKey(mol))
            )

            if skip_np_classifier is False and np_classifier_metadata:
                if "class" in np_classifier_metadata:
                    class_dict[np_classifier_metadata["class"]].add(
                        get_first_part_inchikey(Chem.inchi.MolToInchiKey(mol))
                    )
                if "superclass" in np_classifier_metadata:
                    super_class_dict[np_classifier_metadata["superclass"]].add(
                        get_first_part_inchikey(Chem.inchi.MolToInchiKey(mol))
                    )
                if "pathway" in np_classifier_metadata:
                    pathway_dict[np_classifier_metadata["pathway"]].add(
                        get_first_part_inchikey(Chem.inchi.MolToInchiKey(mol))
                    )

        elif score < 0:
            is_lipinski_synthetics.append(is_lipinski)
            synthetic_accessibility_synthetics.append(sas_score)
            qed_scores_synthetics.append(QED.qed(mol))
            inchikeys_dict["synthetics"].add(
                get_first_part_inchikey(Chem.inchi.MolToInchiKey(mol))
            )

        else:
            inchikeys_dict["hybrid"].add(
                get_first_part_inchikey(Chem.inchi.MolToInchiKey(mol))
            )

            if skip_np_classifier is False and np_classifier_metadata:
                if "class" in np_classifier_metadata:
                    class_dict[np_classifier_metadata["class"]].add(
                        get_first_part_inchikey(Chem.inchi.MolToInchiKey(mol))
                    )
                if "superclass" in np_classifier_metadata:
                    super_class_dict[np_classifier_metadata["superclass"]].add(
                        get_first_part_inchikey(Chem.inchi.MolToInchiKey(mol))
                    )
                if "pathway" in np_classifier_metadata:
                    pathway_dict[np_classifier_metadata["pathway"]].add(
                        get_first_part_inchikey(Chem.inchi.MolToInchiKey(mol))
                    )

    print(f"{skipped} smiles skipped for less than 4 atoms")

    """Summarize descriptive statistics"""
    if logging:
        print(f"Average score: {sum(scores) / len(scores)}")
        print(f"Total smiles analyzed: {len(all_smiles)}")

        # Do the same for synthetic molecules
        lipinski_passes = Counter(is_lipinski_synthetics)[True]
        print(
            f"Proportion of synthetic molecules that pass Lipinski's rules: {round(lipinski_passes / len(is_lipinski_synthetics), 2) * 100}%"
        )

        # Do the same for NP-like molecules
        lipinski_passes = Counter(is_lipinski_np)[True]
        print(
            f"Proportion of NP-like molecules that pass Lipinski's rules: {round(lipinski_passes / len(is_lipinski_np), 2) * 100}%"
        )

        # print average QED score
        print(
            f"Average QED score (synthetics): {round(sum(qed_scores_synthetics) / len(qed_scores_synthetics), 2)}"
        )
        print(
            f"Average QED score (NPS): {round(sum(qed_scores_np) / len(qed_scores_np), 2)}"
        )

    if plot:
        # Make two side by side plots
        plt.figure(figsize=(10, 3))

        plt.subplot(1, 2, 1)
        sns.distplot(synthetic_accessibility_synthetics, label="Synthetic molecules")
        sns.distplot(synthetic_accessibility_np, label="NP-like molecules")
        # title
        plt.title(
            "Synthetic accessibility scores for NP-like molecules and synthetic molecules",
            fontsize=8,
        )

        plt.subplot(1, 2, 2)
        sns.distplot(qed_scores_synthetics, label="Synthetic molecules")
        sns.distplot(qed_scores_np, label="NP-like molecules")

        # title
        plt.title(
            "QED scores for NP-like molecules and synthetic molecules", fontsize=8
        )
        plt.legend()
        plt.show()

    if skip_duplicate and duplicate_pairs:
        # calculate the difference between the duplicated pairs
        duplicate_pair_scores = []

        for smiles1, smiles2 in duplicate_pairs:
            score1 = get_np_likeness_score(smiles1, np_model)
            score2 = get_np_likeness_score(smiles2, np_model)

            if score1 and score2:
                duplicate_pair_scores.append(score1 - score2)

        print(
            f"""
            NP-likeness stats between duplicate pairs.\n
            Average: {round(sum(duplicate_pair_scores) / len(duplicate_pair_scores), 3)}\n
            Median: {sorted(duplicate_pair_scores)[len(duplicate_pair_scores) // 2]}\n
            Max: {round(max(duplicate_pair_scores), 3)}\n
            Min: {round(min(duplicate_pair_scores), 3)}\n
            Std: {round(np.std(duplicate_pair_scores), 3)}\n
            """
        )

        if plot:
            # Plot distribution
            sns.distplot(
                duplicate_pair_scores,
                bins=100,
            )
            plt.title(
                f"Distribution of difference between duplicate pairs {len(duplicate_pairs)}"
            )
            # x label
            plt.xlabel("NP-likeness difference between duplicate pairs")
            plt.ylabel("Density")

            plt.show()

    return (
        scores,
        inchikeys_dict,
        {
            "pathway": pathway_dict,
            "class": class_dict,
            "superclass": super_class_dict,
        },
    )


def lipinski_trial(mol):
    """
    Returns which of Lipinski's rules a molecule has failed, or an empty list

    Lipinski's rules are:
    Hydrogen bond donors <= 5
    Hydrogen bond acceptors <= 10
    Molecular weight < 500 daltons
    logP < 5
    """
    passed = []
    failed = []

    num_hdonors = Lipinski.NumHDonors(mol)
    num_hacceptors = Lipinski.NumHAcceptors(mol)
    mol_weight = Descriptors.MolWt(mol)
    mol_logp = Crippen.MolLogP(mol)

    failed = []

    if num_hdonors > 5:
        failed.append("Over 5 H-bond donors, found %s" % num_hdonors)
    else:
        passed.append("Found %s H-bond donors" % num_hdonors)

    if num_hacceptors > 10:
        failed.append("Over 10 H-bond acceptors, found %s" % num_hacceptors)
    else:
        passed.append("Found %s H-bond acceptors" % num_hacceptors)

    if mol_weight >= 500:
        failed.append("Molecular weight over 500, calculated %s" % mol_weight)
    else:
        passed.append("Molecular weight: %s" % mol_weight)

    if mol_logp >= 5:
        failed.append("Log partition coefficient over 5, calculated %s" % mol_logp)
    else:
        passed.append("Log partition coefficient: %s" % mol_logp)

    return passed, failed


def lipinski_pass(mol):
    """
    Wraps around lipinski trial, but returns a simple pass/fail True/False
    """
    passed, failed = lipinski_trial(mol)
    if failed:
        return False
    else:
        return True
