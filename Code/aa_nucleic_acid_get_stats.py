import os

import chemparse
import pandas as pd


def get_heavy_atoms(formula):
    atoms = chemparse.parse_formula(formula)
    atoms.pop("H", None)  # Remove hydrogen atoms if present
    return sum(atoms.values())  # Sum of remaining atoms


if __name__ == "__main__":
    out_dir = "../Outputs"
    data_dir = "../Data"
    # Load data
    kegg_df = pd.read_csv(os.path.join(data_dir, "chiral_molweight_formula_labels.csv"))
    # Load compound data
    cpd_df = pd.read_csv(os.path.join(data_dir, "AA_NucleicAcids.txt", delimiter=" "))
    # Load compound appearance data
    cpd_appearence_df = pd.read_csv(os.path.join(data_dir, "cpd_appearence_data.csv"))

    # Merge data
    df = pd.merge(kegg_df, cpd_df, left_on=["C"], right_on=["Cpd"])
    df = df.drop(columns=['S', 'Formula', 'Cpd', "Chiral_Centers", "Mol_Weight"])
    df = pd.merge(df, cpd_appearence_df, left_on=["C"], right_on=["Compound"])
    df = df.drop(columns=["Pathways", "Size_When_Added"])

    df["Heavy Atoms"] = df["Formula"].apply(get_heavy_atoms)
    df.to_csv(os.path.join(out_dir, "AANucleicAcidStats.csv"))
