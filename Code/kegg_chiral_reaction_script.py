import re

import pandas as pd


def load_files(compound_file, reaction_file):
    """
    Load the compound and reaction files.

    Parameters:
    compound_file (str): The path to the compound file.
    reaction_file (str): The path to the reaction file.

    Returns:
    pandas.DataFrame: DataFrame containing compound data.
    pandas.DataFrame: DataFrame containing reaction data.
    """
    kegg_data_C = pd.read_csv(compound_file)
    kegg_data_R = pd.read_csv(reaction_file)
    return kegg_data_C, kegg_data_R


def create_compound_dict(kegg_data_C):
    """
    Create a dictionary from the KEGG compound data for quick lookup of compound chiral centers.

    Parameters:
    kegg_data_C (pandas.DataFrame): DataFrame containing KEGG compound data with 'compound_id' and 'n_chiral_centers' columns.

    Returns:
    dict: A dictionary with compound IDs as keys and the number of chiral centers as values.
    """
    return kegg_data_C.set_index('compound_id')['n_chiral_centers'].to_dict()


def parse_reaction_for_stoichiometry(reaction_str):
    """
    Parse a reaction string to extract reactants and products with their stoichiometry.

    Parameters:
    reaction_str (str): The reaction string in the format 'reactants <=> products'.

    Returns:
    tuple: A tuple containing:
        - reactants (list): A list of tuples with reactant compound IDs and their stoichiometry.
        - products (list): A list of tuples with product compound IDs and their stoichiometry.
        - reactant_stoichiometries (list): A list of stoichiometries for reactants as strings.
        - product_stoichiometries (list): A list of stoichiometries for products as strings.
    """

    def parse_compounds(compound_str):
        """
        Parse a compound string to extract individual compounds and their stoichiometry.

        Parameters:
        compound_str (str): The compound string containing compounds separated by '+'.

        Returns:
        tuple: A tuple containing:
            - parsed_compounds (list): A list of tuples with compound IDs and their stoichiometry.
            - stoichiometries (list): A list of stoichiometries as strings.
        """
        compounds = compound_str.split('+')
        parsed_compounds = []
        stoichiometries = []
        for compound in compounds:
            compound = compound.strip()
            # Match numeric stoichiometry and compound ID
            match = re.match(r'(\d*)\s*(C\d{5})', compound)
            if match:
                stoichiometry = int(match.group(1)) if match.group(1) else 1
                compound_id = match.group(2)
                parsed_compounds.append((compound_id, stoichiometry))
                stoichiometries.append(str(stoichiometry))
            else:
                # Handle non-numeric stoichiometry like "n", "n+1"
                parsed_compounds.append((compound, "NA"))
                stoichiometries.append("NA")
        return parsed_compounds, stoichiometries

    reactants_str, products_str = reaction_str.split('<=>')
    reactants, reactant_stoichiometries = parse_compounds(reactants_str.strip())
    products, product_stoichiometries = parse_compounds(products_str.strip())

    return reactants, products, reactant_stoichiometries, product_stoichiometries


def label_reaction(reactants_chiral, products_chiral):
    """
    Label the reaction type based on the number of chiral centers in reactants and products.

    Parameters:
    reactants_chiral (int): The number of chiral centers in the reactants.
    products_chiral (int): The number of chiral centers in the products.

    Returns:
    str: A string indicating the type of reaction:
        - "achiral conserving" if both reactants and products have zero chiral centers.
        - "chiral conserving" if the number of chiral centers in reactants and products are equal.
        - "chiral modulating" if the number of chiral centers in reactants and products are different.
    """
    if reactants_chiral == 0 and products_chiral == 0:
        return "achiral conserving"
    elif reactants_chiral == products_chiral:
        return "chiral conserving"
    else:
        return "chiral modulating"


def calculate_chiral_centers_with_error_handling(compounds, compound_chiral_dict):
    """
    Calculate the total number of chiral centers for a list of compounds, with error handling for missing compounds.

    Parameters:
    compounds (list): A list of tuples where each tuple contains a compound ID and its stoichiometry.
    compound_chiral_dict (dict): A dictionary with compound IDs as keys and the number of chiral centers as values.

    Returns:
    tuple: A tuple containing:
        - total_chiral_centers (int): The total number of chiral centers.
        - missing_compounds (list): A list of compound IDs that are missing from the dictionary.
    """
    total_chiral_centers = 0
    missing_compounds = []
    for compound_id, stoichiometry in compounds:
        if compound_id.startswith('C') and compound_id in compound_chiral_dict:
            chiral_centers = compound_chiral_dict[compound_id]
            if stoichiometry != "NA":
                total_chiral_centers += chiral_centers * stoichiometry
        else:
            missing_compounds.append(compound_id)
    return total_chiral_centers, missing_compounds


def calculate_fraction_chiral(reactants, products, compound_chiral_dict):
    """
    Calculate the fraction of chiral compounds in the given reactants and products.

    Parameters:
    reactants (list): A list of tuples where each tuple contains a compound ID and its stoichiometry.
    products (list): A list of tuples where each tuple contains a compound ID and its stoichiometry.
    compound_chiral_dict (dict): A dictionary with compound IDs as keys and the number of chiral centers as values.

    Returns:
    float: The fraction of chiral compounds.
    """
    total_chiral_compounds = 0
    total_compounds = 0

    for compound_id, stoichiometry in reactants + products:
        chiral_centers = compound_chiral_dict.get(compound_id, 0)
        if chiral_centers > 0:
            total_chiral_compounds += 1 * stoichiometry
        total_compounds += 1 * stoichiometry

    if total_compounds == 0:
        return 0  # Handle division by zero gracefully

    return total_chiral_compounds / total_compounds


def process_reactions(kegg_data_R, compound_chiral_dict):
    """
    Process the reactions to label them based on chiral centers and calculate additional data.

    Parameters:
    kegg_data_R (pandas.DataFrame): DataFrame containing reaction data.
    compound_chiral_dict (dict): A dictionary with compound IDs as keys and the number of chiral centers as values.

    Returns:
    pandas.DataFrame: DataFrame with added columns for reaction labels, fraction of chiral compounds, stoichiometry, and notes.
    """
    reaction_labels = []
    additional_data = []
    stoichiometry_column = []
    notes_column = []

    for idx, row in kegg_data_R.iterrows():
        reaction_str = row['reaction']

        try:
            # Parse the reaction to extract stoichiometric values
            reactants, products, reactant_stoichiometries, product_stoichiometries = parse_reaction_for_stoichiometry(
                reaction_str)

            # Join stoichiometries of reactants and products into a comma-separated string
            stoichiometry_value = ', '.join(reactant_stoichiometries + product_stoichiometries)

            # Calculate chiral centers for reactants and products, handle missing compounds
            reactants_chiral, reactants_missing = calculate_chiral_centers_with_error_handling(reactants,
                                                                                               compound_chiral_dict)
            products_chiral, products_missing = calculate_chiral_centers_with_error_handling(products,
                                                                                             compound_chiral_dict)

            # If there is any issue with missing compounds or non-numeric stoichiometry, mark the rxn_label as "NA"
            if reactants_missing or products_missing:
                reaction_label = "NA"
                # chiral_order_param = "NA"  # Commenting out chiral order param
                # homochirality_order_param = "NA"  # Commenting out homochirality param
                fraction_chiral = "NA"
                note = f"Missing compounds: {reactants_missing + products_missing}"
            else:
                # Calculate labels and parameters normally
                reaction_label = label_reaction(reactants_chiral, products_chiral)
                # chiral_order_param = calculate_chiral_order_parameter(reactants_chiral, products_chiral)  # Commented out
                # homochirality_order_param = calculate_homochirality_order_parameter(reactants_chiral, products_chiral)  # Commented out
                fraction_chiral = calculate_fraction_chiral(reactants, products, compound_chiral_dict)
                note = ""

            # Append results
            reaction_labels.append([reaction_label])
            additional_data.append([fraction_chiral])  # Only adding fraction_chiral to additional data
            stoichiometry_column.append(stoichiometry_value)
            notes_column.append(note)

        except Exception as e:
            reaction_labels.append(["NA"])
            additional_data.append(["NA"])
            stoichiometry_column.append("Error")
            notes_column.append(f"Error: {str(e)}")

    # Add the calculated labels, stoichiometry column, and notes to the dataframe
    kegg_data_R['rxn_labels'] = [r[0] for r in reaction_labels]
    kegg_data_R['fraction_chiral'] = [d[0] for d in additional_data]
    kegg_data_R['Stoichiometry'] = stoichiometry_column
    kegg_data_R['note'] = notes_column

    return kegg_data_R


def save_to_csv(kegg_data_R, output_file):
    """
    Save the processed reaction data to a CSV file.

    Parameters:
    kegg_data_R (pandas.DataFrame): DataFrame containing the processed reaction data.
    output_file (str): The path to the output CSV file.
    """
    kegg_data_R.to_csv(output_file, index=False)


# Example usage
if __name__ == "__main__":
    compound_file = input("Enter the path to the compound file (kegg_data_C.csv): ")
    reaction_file = input("Enter the path to the reaction file (kegg_data_R.csv): ")
    output_file = input("Enter the path for the output file: ")

    kegg_data_C, kegg_data_R = load_files(compound_file, reaction_file)
    compound_chiral_dict = create_compound_dict(kegg_data_C)
    processed_data = process_reactions(kegg_data_R, compound_chiral_dict)
    save_to_csv(processed_data, output_file)
    print(f"Processed data saved to {output_file}")
