# dataset_info_tool.py
def dataset_info_tool() -> str:
    """
    Provides structured and detailed metadata information about all the h5ad datasets we are working with.
    """
    # Assuming PRELOADED_DATASET_INDEX is provided externally (e.g., set by the main script)
    from visualization_tool import PRELOADED_DATASET_INDEX

    if PRELOADED_DATASET_INDEX is None:
        return "Error: Dataset index is not preloaded into memory."

    try:
        lines = []
        # Iterate over the rows of the preloaded dataset index DataFrame
        for _, row in PRELOADED_DATASET_INDEX.iterrows():
            lines.append(
                f"Dataset Name: {row['h5ad']}\n"
                f"Description: {row['Dataset Description']}\n"
                f"Species: {row['Species']}\n"
                f"Research Team: {row['Research Team']}\n"
                f"Publication: {row['Publication']}\n"
                f"Source: {row['Source']}\n"
                f"Donor Index: {row.get('donor_index', 'N/A')}\n"
                f"Cell Type Index: {row.get('celltype_index', 'N/A')}\n"
                f"Covariate Index: {row.get('covariate_index', 'N/A')}\n"
                f"Age Index: {row.get('age_index', 'N/A')}\n"
                f"Sex Index: {row.get('sex_index', 'N/A')}\n"
                f"Study Index: {row.get('study_index', 'N/A')}\n"
                f"Suspension Index: {row.get('suspension_index', 'N/A')}\n"
                f"Assay Index: {row.get('assay_index', 'N/A')}\n"
                f"Tissue Index: {row.get('tissue_index', 'N/A')}\n"
                f"Ethnicity Index: {row.get('ethnicity_index', 'N/A')}\n"
                f"Sampling Index: {row.get('sampling_index', 'N/A')}\n"
                f"Disease Stats: {row.get('disease_stats', 'N/A')}\n"
                f"Restrict Index: {row.get('restrict_index', 'N/A')}\n"
                f"Display Index: {row.get('display_index', 'N/A')}\n"
                f"Primary Covariates: {row.get('disease', 'N/A')}, {row.get('study', 'N/A')}, "
                f"{row.get('assay', 'N/A')}, {row.get('suspension_type', 'N/A')}, {row.get('tissue', 'N/A')}\n"
                f"Confounding Covariates: {row.get('covariate_index', 'N/A')}\n"
                f"Differential Expression: {row.get('Disease-Study-CellType-DEGs', 'N/A')}\n"
                "-----------------------------"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error retrieving dataset information: {repr(e)}"