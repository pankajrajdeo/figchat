# utils.py
import os
import pandas as pd
import json
import re

def parse_tsv_data(file_path):
    """
    Parses the dataset index TSV file into a structured format for easy interpretation,
    using hardcoded AnnData structure outputs.

    Args:
        file_path (str): Path to the TSV file.

    Returns:
        dict: A dictionary containing the parsed datasets and a notes section.
    """
    # Predefined AnnData structures for each dataset
    anndata_structures = {
        "HLCA_full_superadata_v3_norm_log_deg.h5ad": {
            "shape": (50520, 56295),
            "X_dtype": "float32",
            "obs_fields": ['orig.ident', 'Metacell', 'suspension_type', 'donor_id', 'assay_ontology_term_id',
                           'cell_type_ontology_term_id', 'development_stage_ontology_term_id', 'disease_ontology_term_id',
                           'self_reported_ethnicity_ontology_term_id', 'tissue_ontology_term_id', 'organism_ontology_term_id',
                           'sex_ontology_term_id', "3'_or_5'", 'age_range', 'ann_finest_level', 'ann_level_1', 
                           'ann_level_2', 'ann_level_3', 'ann_level_4', 'ann_level_5', 'core_or_extension', 'dataset',
                           'fresh_or_frozen', 'lung_condition', 'original_ann_level_1', 'original_ann_level_2', 
                           'original_ann_level_3', 'original_ann_level_4', 'original_ann_level_5', 'original_ann_nonharmonized',
                           'reannotation_type', 'sample', 'scanvi_label', 'sequencing_platform', 'study', 
                           'tissue_sampling_method', 'cell_type', 'assay', 'disease', 'organism', 'sex', 'tissue', 
                           'self_reported_ethnicity', 'development_stage', 'ann', 'ann_sample', 'size', 'is_primary_data', 
                           'datasets', 'transf_ann_level_1_label', 'transf_ann_level_2_label', 'transf_ann_level_3_label', 
                           'transf_ann_level_4_label', 'transf_ann_level_5_label', 'Metacell_ID'],
            "var_fields": [],
            "obsm_keys": ['X_umap'],
            "uns_keys": ['disease_comparison_summary', 'disease_stats', 'interaction_data', 'log1p', 'marker_stats', 'rank_genes_groups'],
            "layers_keys": []
        },
        "HCA_fetal_lung_normalized_log_deg.h5ad": {
            "shape": (71752, 26286),
            "X_dtype": "float32",
            "obs_fields": ['batch', 'dissection', 'chemistry', 'percent_mito', 'n_counts', 'n_genes', 'doublet_scores', 
                           'bh_pval', 'leiden', 'phase', 'S_score', 'G2M_score', 'new_celltype', 'big_cluster', 
                           'broad_celltype', 'assay_ontology_term_id', 'cell_type_ontology_term_id', 
                           'development_stage_ontology_term_id', 'disease_ontology_term_id', 'self_reported_ethnicity_ontology_term_id',
                           'is_primary_data', 'organism_ontology_term_id', 'sex_ontology_term_id', 'tissue_ontology_term_id', 
                           'donor_id', 'suspension_type', 'tissue_type', 'cell_type', 'assay', 'disease', 'organism', 
                           'sex', 'tissue', 'self_reported_ethnicity', 'development_stage', 'observation_joinid'],
            "var_fields": ['n_cells', 'gene_symbols', 'feature_is_filtered', 'feature_name', 'feature_reference',
                           'feature_biotype', 'feature_length', 'feature_type'],
            "obsm_keys": ['X_pca', 'X_umap', 'X_umap_original'],
            "uns_keys": ['batch_colors', 'big_cluster_colors', 'broadcelltype_colors', 'chemistry_colors', 'citation', 
                         'default_embedding', 'disease_comparison_summary', 'disease_stats', 'interaction_data', 'log1p', 
                         'marker_stats', 'newcelltype_colors', 'phase_colors', 'rank_genes_groups', 'rank_genes_groups_global', 
                         'schema_reference', 'schema_version', 'title'],
            "layers_keys": [],
            "raw": {
                "exists": True,
                "shape": (71752, 26286),
                "var_fields": ['n_cells', 'gene_symbols', 'feature_is_filtered', 'feature_name', 'feature_reference',
                               'feature_biotype', 'feature_length', 'feature_type']
            }
        },
        "BPD_infant_Sun_normalized_log_deg.h5ad": {
            "shape": (271381, 36601),
            "X_dtype": "float64",
            "obs_fields": ['sample_id', 'donor_id', 'disease', 'species', 'age', 'sex', 'pna', 'ga', 'pma', 'perc_pma',
                           'race', 'rac', 'wit', 'cit', 'tissue_type', 'tissue_ontology_term_id', 'suspension_type',
                           'assay_ontology_term_id', 'organism_ontology_term_id', 'sex_ontology_term_id', 
                           'disease_ontology_term_id', 'development_stage_ontology_term_id', 
                           'self_reported_ethnicity_ontology_term_id', 'n_genes_by_counts', 'total_counts', 
                           'total_counts_mt', 'pct_counts_mt', 'n_genes', 'n_counts', 'cell type', 
                           'CellRef-level3', 'CellRef-score', 'cell_type_ontology_term_id', 'lineage_level1', 
                           'lineage_level2', 'celltype_level3_fullname', 'disease_mapped', 'lineage-disease', 
                           'donor-disease', 'celltype-disease'],
            "var_fields": ['mt', 'n_cells_by_counts', 'mean_counts', 'pct_dropout_by_counts', 'total_counts'],
            "obsm_keys": ['X_refUMAP', 'X_umap'],
            "uns_keys": ['disease_comparison_summary', 'disease_stats', 'interaction_data', 'marker_stats', 'rank_genes_groups'],
            "layers_keys": []
        },
        "BPD_fetal_normalized_log_deg.h5ad": {
            "shape": (43607, 2000),
            "X_dtype": "float64",
            "obs_fields": ['orig.ident', 'nCount_RNA', 'nFeature_RNA', 'percent.mt', 'dataset', 'age', 'condition', 
                           'broad_condition', 'percent.rb', 'nCount_SCT', 'nFeature_SCT', 'louvain', 'celltype', 
                           'sub.cluster', 'SCT_snn_res.1.4', 'celltype_lineage', 'donor_id', 'tissue', 'sex'],
            "var_fields": ['features'],
            "obsm_keys": ['X_harmony', 'X_pca', 'X_umap'],
            "uns_keys": ['disease_comparison_summary', 'disease_stats', 'interaction_data', 'log1p', 'marker_stats', 'neighbors', 'rank_genes_groups'],
            "layers_keys": [],
            "raw": {
                "exists": True,
                "shape": (43607, 2000),
                "var_fields": ['features']
            }
        }
    }

    data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
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

    for _, row in data.iterrows():
        h5ad_name = row['h5ad']
        anndata_structure = anndata_structures.get(h5ad_name, {})

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
            "**AnnData Structure**": anndata_structure
        })

    # Add notes section at the bottom
    notes = {
        "**Notes** (Explanation of fields and structure)": [
            "1. **Dataset Metadata**: General information about the dataset, including its name, description, species, research team, publication link, and data source.",
            "2. **Indexes and Annotations**: Metadata fields that organize and describe aspects of the dataset such as cell types, covariates, age, sex, studies, assays, tissues, and sampling methods.",
            "3. **Dataset-Specific Fields**: Includes disease statistics, restrict index for filtering, and display index for visualization preferences.",
            "4. **Differential Expression**: Information linking diseases, associated studies, and cell types involved in differential expression analysis.",
            "5. **AnnData Structure**: Structure of the AnnData object for the dataset."
        ]
    }

    return {
        "datasets": datasets,
        "notes": notes
    }
