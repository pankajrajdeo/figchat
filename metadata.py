import pandas as pd
import json
import re

def parse_tsv_data(file_path):
    """
    Parses the dataset index TSV file into a structured format for easy interpretation.

    Args:
        file_path (str): Path to the TSV file.

    Returns:
        dict: A dictionary containing the parsed datasets and a notes section.
    """
    data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    datasets = []
    
    for _, row in data.iterrows():
        # Parse individual fields with delimiters "::" and ","
        def parse_field(field):
            if pd.isna(field):
                return {}
            field_parts = field.split("::")
            if len(field_parts) > 1:
                key, values = field_parts[0].strip(), field_parts[1].strip()
                return {key: [v.strip() for v in values.split(",")]}
            return {field.strip(): []}
        
        # Differential Expression parsing
        def parse_deg_field(field):
            if pd.isna(field) or not field.strip():
                return []

            # Optional leading pipe (and maybe space) before disease
            pattern = r"(?:^\||\|)?\s*(.*?)\{(.*?)\}\[(.*?)\]"
            matches = re.findall(pattern, field)

            parsed_entries = []
            for disease, study, celltypes_str in matches:
                disease = disease.strip()
                study = study.strip()
                celltypes = [ct.strip() for ct in celltypes_str.split("|") if ct.strip()]

                parsed_entries.append({
                    "disease": disease,
                    "study": study,
                    "cell_types": celltypes
                })

            return parsed_entries

        # Build structured dataset entry
        datasets.append({
            "**Dataset Metadata**": {
                "**Dataset Name**": row['h5ad'],
                "**Description**": row['Dataset Description'],
                "**Species**": row['Species'],
                "**Research Team**": row['Research Team'],
                "**Publication**": row['Publication'],
                "**Source**": row['Source']
            },
            "**Indexes and Annotations**": {
                "**Donor Index** (Unique donor identifier)": row['donor_index'],
                "**Cell Type Index** (Cell types in dataset)": parse_field(row['cell_type_index']),
                "**Covariate Index** (Covariates and annotations, e.g., diseases)": parse_field(row['covariate_index']),
                "**Age Index** (Age-related annotations)": parse_field(row['age_index']),
                "**Sex Index** (Sex-related annotations)": parse_field(row['sex_index']),
                "**Study Index** (Study names associated with dataset)": parse_field(row['study_index']),
                "**Suspension Index** (Suspension type annotations)": parse_field(row['suspension_index']),
                "**Assay Index** (Assay methods used)": parse_field(row['assay_index']),
                "**Tissue Index** (Tissue types represented)": parse_field(row['tissue_index']),
                "**Ethnicity Index** (Self-reported ethnicity)": parse_field(row['ethnicity_index']),
                "**Sampling Index** (Sampling methods)": parse_field(row['sampling_index']),
            },
            "**Dataset-Specific Fields**": {
                "**Disease Stats**": row['disease_stats'],
                "**Restrict Index** ": row['restrict_index'],
                "**Display Index**": row['display_index']
            },
            "**Differential Expression** (Disease-Study-CellType mappings)": parse_deg_field(row['Disease-Study-CellType-DEGs']),
            "**Directory Path** (File location of the dataset)": row['directory_path']
        })

    # Add notes section at the bottom
    notes = {
        "**Notes** (Explanation of fields and structure)": [
            "1. **Dataset Metadata**: General information about the dataset, including its name, description, species, research team, publication link, and data source.",
            "2. **Indexes and Annotations**: Metadata fields that organize and describe aspects of the dataset such as cell types, covariates, age, sex, studies, assays, tissues, and sampling methods.",
            "3. **Dataset-Specific Fields**: Includes disease statistics, restrict index for filtering, and display index for visualization preferences.",
            "4. **Differential Expression**: Information linking diseases, associated studies, and cell types involved in differential expression analysis.",
            "5. **Directory Path**: The file location where the dataset is stored."
        ]
    }

    return {
        "datasets": datasets,
        "notes": notes
    }

# Example usage:
dataset_info = parse_tsv_data("/path/dataset_index_advanced_paths.tsv")
formatted_output = json.dumps(dataset_info, indent=4)
print(formatted_output)