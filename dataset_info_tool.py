# dataset_info_tool.py
import re
import json
import pandas as pd

def dataset_info_tool() -> str:
    """
    Provides structured and detailed metadata information about all the h5ad datasets we are working with.
    """
    from visualization_tool import PRELOADED_DATASET_INDEX

    if PRELOADED_DATASET_INDEX is None:
        return "Error: Dataset index is not preloaded into memory."

    try:
        data = PRELOADED_DATASET_INDEX
        datasets = []

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

        # Iterate over the rows of the preloaded dataset index DataFrame
        for _, row in data.iterrows():
            datasets.append({
                "**Dataset Metadata**": {
                    "**Dataset Name**": row['h5ad'],
                    "**Description**": row['Dataset Description'],
                    "**Species** (Organism from which data is collected)": row['Species'],
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
                    "**Restrict Index**": row['restrict_index'],
                    "**Display Index**": row['display_index']
                },
                "**Differential Expression** (Disease-Study-CellType mappings)": parse_deg_field(row['Disease-Study-CellType-DEGs'])
            })

        # Add notes section
        notes = {
            "**Notes** (Explanation of fields and structure)": [
                "1. **Dataset Metadata**: General information about the dataset, including its name, description, species, research team, publication link, and data source.",
                "2. **Indexes and Annotations**: Metadata fields that organize and describe aspects of the dataset such as cell types, covariates, age, sex, studies, assays, tissues, and sampling methods.",
                "3. **Dataset-Specific Fields**: Includes disease statistics, restrict index for filtering, and display index for visualization preferences.",
                "4. **Differential Expression**: Information linking diseases, associated studies, and cell types involved in differential expression analysis.",
            ]
        }

        return json.dumps({"datasets": datasets, "notes": notes}, indent=4)
    except Exception as e:
        return f"Error retrieving dataset information: {repr(e)}"
