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

        # Safely parse restrict_index and display_index
        restrict_index = row['restrict_index'] if pd.notna(row['restrict_index']) else ""
        display_index = row['display_index'] if pd.notna(row['display_index']) else ""

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
                "**Restrict Index**": [val.strip() for val in restrict_index.split(',') if val.strip()],
                "**Display Index**": [val.strip() for val in display_index.split(',') if val.strip()]
            },
            "**Differential Expression** (Disease-Study-CellType mappings)": parse_deg_field(row['Disease-Study-CellType-DEGs']),
            "**Directory Path** (File location of the dataset)": row['directory_path'],
            "**AnnData Structure**": anndata_structure
        })


    # Add notes section at the bottom
    notes = {
        "**Notes** (Explanation of fields and structure)": [
            "1. **Dataset Metadata**: General information about the dataset, including its name, description, species, research team, publication link, and data source.",
            "2. **Indexes and Annotations**: Metadata fields that organize and describe aspects of the dataset such as cell types, covariates, age, sex, studies, assays, tissues, and sampling methods.",
            "3. **Dataset-Specific Fields**: Includes disease statistics, restrict index for filtering, and display index for visualization preferences.",
            "4. **Differential Expression**: Information linking diseases, associated studies, and cell types involved in differential expression analysis.",
            "5. **Directory Path**: The file location where the dataset is stored.",
            "6. **AnnData Structure**: Structure of the AnnData object for the dataset."
        ]
    }

    return {
        "datasets": datasets,
        "notes": notes
    }

##########################################################################################################
from typing import Optional, List, Literal, Union, Dict
from pydantic import BaseModel, Field, root_validator

###############################################################################
# 1. Base / Common Classes
###############################################################################
class BasePlotConfig(BaseModel):
    """
    Minimal common fields for any plot configuration.
    Derived classes for each plot type will inherit from this.
 
    Note:
    - 'plot_type' is always provided.
    - 'adata_file' is always needed to locate the .h5ad file.
    """
 
    adata_file: str = Field(..., description="Path to the .h5ad file.")
    plot_type: Literal[
        "stats",
        "heatmap",
        "radar",
        "cell_frequency",
        "volcano",
        "dotplot",
        "violin",
        "venn",
        "upset_genes",
        "umap",
        "network",
        "all"
    ]

    # Added restriction and variable index fields:
    restrict_variable1: Optional[List[str]] = Field(
        default=None,
        description="Optional list of values to restrict on variable1."
    )
    variable1_index: Optional[str] = Field(
        default=None,
        description="Metadata column name corresponding to restrict_variable1."
    )
    restrict_variable2: Optional[List[str]] = Field(
        default=None,
        description="Optional list of values to restrict on variable2."
    )
    variable2_index: Optional[str] = Field(
        default=None,
        description="Metadata column name corresponding to restrict_variable2."
    )
    restrict_variable3: Optional[List[str]] = Field(
        default=None,
        description="Optional list of values to restrict on variable3."
    )
    variable3_index: Optional[str] = Field(
        default=None,
        description="Metadata column name corresponding to restrict_variable3."
    )
    restrict_variable4: Optional[List[str]] = Field(
        default=None,
        description="Optional list of values to restrict on variable4."
    )
    variable4_index: Optional[str] = Field(
        default=None,
        description="Metadata column name corresponding to restrict_variable4."
    )

###############################################################################
# 2. Plot-Specific Classes
###############################################################################

class StatsPlotConfig(BasePlotConfig):
    """
    'stats' or 'all':
    - Creates a TSV file of DEGs or marker genes (based on 'direction').
    - Also expects 'cell_type' and 'disease' to filter relevant DEGs.
    """
    plot_type: Literal["stats", "all"]
    
    direction: Literal["regulated", "up", "down", "markers"] = Field(
        "regulated",
        description="Specifies gene regulation direction. Default is 'regulated'."
    )
    cell_type: str = Field(
        ...,
        description="Specific cell type to focus on for the plot."
    )
    disease: str = Field(
        ...,
        description="Condition to filter by."
    )
    n_genes: Optional[int] = Field(
    100,
    description="Optional. Number of top genes to include if gene_symbols is empty. Defaults to 100."
    )

class HeatmapPlotConfig(BaseModel):
    """
    Schema for generating heatmap visualizations with all logic handled in Pydantic.
    """
    plot_type: Literal["heatmap", "all"] = Field(
        "heatmap",
        description="Specifies the type of plot to generate. Only 'heatmap' is supported."
    )
    adata_file: str = Field(
        ...,
        description="Path to the .h5ad AnnData file to be analyzed."
    )    
    gene_symbols: List[Union[str, Dict]] = Field(
        default_factory=list,
        description="List of gene symbols (str) or dicts with a 'Gene' key."
    )
    cell_type: Optional[str] = Field(
        None,
        description="Subset data to a specific cell type. Must match entries in 'cell_type_index'."
    )
    disease: Optional[str] = Field(
        None,
        description="Disease name to filter data. Derived from covariates."
    )
    covariates: List[str] = Field(
        default_factory=list,
        description="List of covariate values for filtering. Includes 'normal' by default."
    )
    direction: Literal["regulated", "up", "down", "markers"] = Field(
        "regulated",
        description="Specifies gene regulation direction. Default is 'regulated'."
    )
    covariate: Optional[str] = Field(
        None,
        description="Single covariate value for filtering (deprecated; use 'covariates')."
    )
    cell_type_index: str = Field(
        ...,
        description="Column name in adata.obs representing cell type."
    )
    covariate_index: str = Field(
        "disease",
        description="Column name in adata.obs representing the primary covariate."
    )
    show_individual_cells: bool = Field(
        True,
        description="If True, displays individual cells on the heatmap. Default is True."
    )
    median_scale_expression: bool = Field(
        False,
        description="If True, normalizes rows by subtracting the median expression across genes."
    )
    cluster_rows: bool = Field(
        True,
        description="Enable hierarchical clustering on rows (genes). Default is True."
    )
    cluster_columns: bool = Field(
        True,
        description="Enable hierarchical clustering on columns (samples/groups). Default is True."
    )
    heatmap_technology: Literal["seaborn", "imshow"] = Field(
        "seaborn",
        description="Heatmap rendering library. Default is 'seaborn'."
    )
    samples_to_visualize: Literal["cell-type", "all"] = Field(
        "cell-type",
        description="Subset data by cell type ('cell-type') or use all cells ('all')."
    )
    display_variables: List[str] = Field(
        ...,
        description="Mandatory list of grouping fields for aggregation/visualization derived from the 'Display Index' field of the dataset metadata."
    )

    @root_validator(pre=True)
    def validate_fields(cls, values):
        diseases = values.get("disease")
        covariates_existing = values.get("covariates", [])

        # If disease not provided but covariates exist, try to infer disease
        if not diseases and covariates_existing:
            inferred = [cov for cov in covariates_existing if cov.lower() != "normal"]
            if inferred:
                values["disease"] = inferred[0]
                diseases = values["disease"]

        if diseases:
            # Ensure diseases is treated as a list for processing
            if isinstance(diseases, str):
                diseases_list = [diseases]
            else:
                diseases_list = diseases

            # Prepend "normal" to covariates if not already present
            if "normal" not in covariates_existing:
                values["covariates"] = ["normal"] + diseases_list
            else:
                # Combine existing covariates and diseases, avoiding duplicates
                combined = list(dict.fromkeys(covariates_existing + diseases_list))
                values["covariates"] = combined
        else:
            # Default to ["normal"] if no disease and no covariates provided
            if not covariates_existing:
                values["covariates"] = ["normal"]

        # Validate that display_variables is provided
        if not values.get("display_variables"):
            raise ValueError("display_variables must be provided and cannot be empty.")

        return values

class RadarPlotConfig(BasePlotConfig):
    """
    'radar' or 'all':
    - Renders a radar plot of average cell-type frequency by condition.

    figure_generation.py calls:
      plot_aggregated_cell_frequencies_radar(covariate_index, donor_index, cell_type_index, ...)
    """
    plot_type: Literal["radar", "all"]
    
    covariate_index: str = Field(
        ...,
        description="Field in adata.obs representing the grouping variable (e.g., 'disease')."
    )
    donor_index: str = Field(
        ...,
        description="Field in adata.obs representing the donor identifier."
    )
    cell_type_index: str = Field(
        ...,
        description="Field in adata.obs representing the cell type."
    )

class CellFrequencyPlotConfig(BaseModel):
    """
    Configuration for generating a cell frequency plot.

    The first element of 'covariates' is used as the control group.
    """
    plot_type: Literal["cell_frequency", "all"] = Field(
        ...,
        description="Type of plot to generate. 'cell_frequency' generates the cell frequency plot; 'all' includes additional plots."
    )
    adata_file: str = Field(
        ...,
        description="Path to the .h5ad AnnData file to be analyzed."
    )
    covariates: List[str] = Field(
        ...,
        description="List of diseases or conditions. The first item is treated as the control group."
    )
    cell_type_index: str = Field(
        ...,
        description="The obs field used for cell-type annotation."
    )
    covariate_index: str = Field(
        ...,
        description="The obs field representing disease/condition."
    )
    donor_index: str = Field(
        ...,
        description="The obs field representing the donor identifier."
    )
    sex_index: str = Field(
        ...,
        description="The obs field representing sex or gender annotations."
    )

class VolcanoPlotConfig(BasePlotConfig):
    """
    'volcano' or 'all':
    - Creates a volcano plot from precomputed DE results in gene_symbols.
    - Must have 'Gene', 'Log Fold Change', and 'FDR-Corrected P-Value' columns in gene_symbols.

    Code references:
      plot_volcano(cell_type, disease, gene_symbols, top_n=8).
    Optional: top_n can be changed from default=8 if needed.
    """
    plot_type: Literal["volcano", "all"]
    
    cell_type: str = Field(
        ...,
        description="Cell type used in the DE comparison for labeling."
    )
    disease: str = Field(
        ...,
        description="Condition used in the DE comparison for labeling."
    )
    top_n: Optional[int] = Field(
        8,
        description="Number of top genes to highlight in the volcano plot. Default is 8."
    )
    direction: Literal["regulated", "up", "down", "markers"] = Field(
        "regulated",
        description="Specifies gene regulation direction. Default is 'regulated'."
    )

class DotPlotConfig(BaseModel):
    """
    Configuration for generating dot plots using LLM-based function calls.
    """

    plot_type: Literal["dotplot", "all"] = Field(
        ...,
        description="'dotplot' generates a plot for a single cell type; 'all' generates for all cell types."
    )
    adata_file: str = Field(
        ...,
        description="Path to an AnnData .h5ad file to be loaded."
    )
    cell_type: Optional[str] = Field(
        None,
        description="The specific cell type to visualize in the single-cell-type version."
    )
    gene_symbols: List[Union[str, Dict]] = Field(
        default_factory=list,
        description="List of genes (as strings) or DataFrame-like dictionaries with a 'Gene' key."
    )
    direction: Literal["regulated", "up", "down", "markers"] = Field(
        "regulated",
        description="Specifies gene regulation direction. Default is 'regulated'."
    )
    covariate_index: str = Field(
        ...,
        description="The obs field representing disease/condition."
    )
    cell_type_index: str = Field(
        ...,
        description="Column name in adata.obs representing cell type."
    )
    covariates: List[str] = Field(
        ...,
        description="List of conditions (e.g., diseases) to filter by. This list must be non-empty. If not specified, choose a non-disease condition."
    )

class ViolinPlotConfig(BaseModel):
    """
    Configuration for generating violin plots.
    
    Attributes:
      Required:
        - adata_file: Path to the AnnData file containing the dataset.
        - plot_type: Must be "violin" or "all".
        - gene: Single gene symbol to plot (must exist in adata.var_names).
        - cell_type: The specific cell type to focus on for the plot.
        - covariates: List of conditions (e.g., diseases) to filter by.
        - covariate_index: The primary grouping index in adata.obs (e.g., 'disease').
      
      Optional:
        - cell_type_index: Column name in adata.obs containing cell type information. Defaults to 'cell type' if not provided.
        - display_variables: List of variables to display in the plot. Defaults to [covariate_index] if not provided. The second variable (if present) is used as alt_covariate_index for additional grouping or coloring.
    """
    plot_type: Literal["violin", "all"] = Field(
        ...,
        description="Type of plot to generate. Must be 'violin' or 'all'."
    )
    adata_file: str = Field(
        ...,
        description="Path to the .h5ad AnnData file to be analyzed."
    )   
    gene: str = Field(
        ...,
        description="Single gene symbol to plot. Must exist in adata.var_names."
    )
    cell_type: str = Field(
        ...,
        description="Specific cell type to focus on for the plot."
    )
    covariates: List[str] = Field(
        ...,
        description="List of conditions (e.g., diseases) to filter by. This list must be non-empty."
    )
    covariate_index: str = Field(
        ...,
        description="The primary grouping index in adata.obs, such as 'disease'."
    )
    cell_type_index: str = Field(
        ...,
        description="Column name in adata.obs representing cell type."
    )
    display_variables: List[str] = Field(
        ...,
        description="Mandatory list of all the grouping fields for aggregation/visualization derived from the 'Display Index' field of the dataset metadata."
    )

class VennPlotConfig(BasePlotConfig):
    """
    'venn' or 'all':
    - Venn diagram for 2 or 3 sets of marker genes from different cell types.
    - Must have 'cell_types_to_compare' with length 2 or 3.
    """
    plot_type: Literal["venn", "all"]
    
    cell_types_to_compare: List[str] = Field(
        ...,
        description="List of 2 or 3 cell types to compare in a Venn diagram. Do not select the 'Unknown' cell type."
    )

class UpSetGenesPlotConfig(BasePlotConfig):
    """
    'upset_genes' or 'all':
    Similar to Venn but handles more than 3 sets or more complex overlaps.
    """
    plot_type: Literal["upset_genes", "all"]
    
    cell_types_to_compare: List[str] = Field(
        ...,
        description="List of cell types for intersection in an UpSet plot."
    )

class UmapPlotConfig(BaseModel):

    # Required by figure_generation.py
    plot_type: Literal["umap", "all"] = Field(
        ...,
        description="Plot type: 'umap' or 'all'."
    )
    adata_file: str = Field(
        ...,
        description="Path to an AnnData .h5ad file to be loaded."
    )
    # Maps to 'subset_col' in main_visualizations
    covariate_index: Optional[str] = Field(
        None,
        description="Column in adata.obs to subset data, e.g. 'condition'."
    )
    # Maps to 'subset_values' in main_visualizations
    covariates: List[str] = Field(
        default_factory=list,
        description="Specific values within covariate_index to subset on."
    )

    # Maps to 'color_by' in main_visualizations
    color_by: Optional[str] = Field(
        None,
        description="Column or metadata field to color UMAP (e.g. 'celltype')."
    )
    
    # Maps to 'cluster_by' in main_visualizations
    cell_type_index: Optional[str] = Field(
        None,
        description="Column in adata.obs used for clustering (e.g., 'celltype'). NOTE: Default is similar to field chosen for for 'color_by' if not provided."
    )

    # Maps to 'gene' in main_visualizations
    gene: Optional[str] = Field(
        None,
        description="Gene name to color the UMAP by expression (overrides color_by if present)."
    )

class NetworkPlotConfig(BasePlotConfig):
    """
    Constructs a gene interaction network from 'gene_symbols', coloring nodes by Log Fold Change.

    figure_generation.py references:
      visualize_gene_network_igraph(...) or visualize_gene_networkX(...)
      - gene_symbols must contain 'Gene' and 'Log Fold Change'.
      - cell_type is required.
      - network_technology => 'igraph' or 'networkx'.
      - n_genes default=1000 if not provided.
      - edge_types for interaction coloring; exclude_biogrid flag for filtering BioGRID interactions.
    """
    plot_type: Literal["network", "all"]
    
    gene_symbols: List[dict] = Field(
        default_factory=list,
        description="Must include 'Gene' and 'Log Fold Change' for node color inference."
    )
    cell_type: str = Field(
        ...,
        description="Specific cell type is required for filtering or labeling the network."
    )
    covariate_index: str = Field(
        ...,
        description="The primary grouping index in adata.obs, such as 'disease'."
    )
    covariates: List[str] = Field(
        ...,
        description="covariates: List of conditions (e.g., diseases) to filter by. This list must be non-empty. If not specified, choose a non-disease condition."
    )
    n_genes: Optional[int] = Field(
        1000,
        description="Optional. Number of top genes to include if gene_symbols is empty. Defaults to 1000."
    )
    direction: Literal["regulated", "up", "down", "markers"] = Field(
        "regulated",
        description="Specifies gene regulation direction. Default is 'regulated'."
    )
    network_technology: Optional[Literal["igraph", "networkx"]] = Field(
        "igraph",
        description="Specifies the library to use for network construction. Defaults to 'igraph'."
    )

###############################################################################
# 3. Dictionary: PLOT_GUIDES
###############################################################################

PLOT_GUIDES = {
    "stats": (
        "Stats or 'all' plot:\n"
        "Generates a TSV file of differentially expressed genes (DEGs) or marker genes, depending on the 'direction'.\n"
        "Required fields:\n"
        "  - plot_type: 'stats' or 'all'.\n"
        "  - direction: a variable describing whether to look at differentially expressed genes in disease vs control, either 'regulated', 'up', 'down' or to look at cell type specific 'markers' stored in the H5AD.\n"
        "  - cell_type: Specify the target cell type for analysis.\n"
        "  - disease: Specify the condition or disease to filter by.\n"
        "Optional:\n"
        "  - n_genes: Number of top genes to export.\n"
        "  - restrict_variable1...restrict_variable4 and variable1_index...variable4_index: Optional fields to restrict the dataset based on metadata columns before plotting. These parameters allow finer control over which subset of the data is used by specifying metadata values and their corresponding column names.\n"
    ),
    "heatmap": (
        "Heatmap or 'all' plot:\n"
        "Visualizes gene expression across cells or aggregated groups in a clustered or un-clustered heatmap.\n"
        "\nRequired fields:\n"
        "  - gene_symbols: A list of gene IDs to display in the heatmap. Can be an empty list if not applicable.\n"
        "  - cell_type: A specific cell type to subset the data (must align with 'cell_type_index').\n"
        "  - cell_type_index: Column name representing cell types in adata.obs. Default is 'ann_finest_level'.\n"
        "  - covariate_index: Column name representing the main covariate (e.g., disease) in adata.obs. Default is 'disease'.\n"
        " - direction: a variable describing whether to look at differentially expressed genes in disease vs control, either 'regulated', 'up', 'down' or to look at cell type specific 'markers' stored in the H5AD.\n"
        "  - covariates: A list of covariate values to filter cells. By default, this list should always start with \"normal\" "
        "followed by the specified disease(s). For example, if the user specifies \"pulmonary fibrosis\", "
        "the covariates should be [\"normal\", \"pulmonary fibrosis\"].\n"
        "  - show_individual_cells: Whether to show individual cells (True) or aggregate data by median (False). Default is True.\n"
        "  - cluster_rows / cluster_columns: Enable hierarchical clustering on rows (genes) or columns (groups). Defaults are True.\n"
        "  - median_scale_expression: Normalize expression data by subtracting the median across genes (row-wise). Default is False.\n"
        "  - heatmap_technology: Rendering technology ('seaborn' or 'imshow'). Default is 'seaborn'.\n"
        "  - samples_to_visualize: Specify 'cell-type' to subset by a single cell type or 'all' for all cells. Default is 'cell-type'.\n"
        "  - display_variables: A mandatory list of grouping fields for aggregation or visualization. This is dynamically derived\n"
        "                       from the dataset's metadata and must be explicitly provided.\n"
        "\nOptional:\n"
        "  - covariate: A single covariate value for filtering (deprecated; use 'covariates' instead).\n"
        "\nNotes:\n"
        "  - Ensure 'covariates' align with values in 'covariate_index'.\n"
        "  - Verify 'cell_type' matches entries in 'cell_type_index'.\n"
        "  - Prefer 'covariates' for filtering, even if specifying a single value.\n"
        "  - Default indices ('cell_type_index' and 'covariate_index') can be overridden if necessary.\n"
        "  - If 'covariates' results in an empty subset after filtering, this may cause errors. Validate input parameters accordingly.\n"
        "  - Always provide 'display_variables', as it is essential for defining grouping fields.\n"
        "  - restrict_variable1...restrict_variable4 and variable1_index...variable4_index: Optional fields to restrict the dataset based on metadata columns before plotting. These parameters allow finer control over which subset of the data is used by specifying metadata values and their corresponding column names.\n"
    ),
    "radar": (
        "Radar or 'all' plot:\n"
        "Shows average cell-type frequencies across conditions in a radial plot.\n"
        "Required fields:\n"
        "  - covariate_index: Field in adata.obs representing the grouping variable.\n"
        "  - donor_index: Field in adata.obs representing the donor identifier.\n"
        "  - cell_type_index: Field in adata.obs representing the cell type.\n"
        "Optional fields:\n"
        "  - disease: Specific disease condition used for filtering data.\n"
        "  - covariates: List of covariate values to filter the dataset.\n"
        "  - restrict_variable1...restrict_variable4 and variable1_index...variable4_index: Optional fields to restrict the dataset based on metadata columns before plotting. These parameters allow finer control over which subset of the data is used by specifying metadata values and their corresponding column names.\n"
    ),
    "cell_frequency": (
        "Cell Frequency or 'all' plot:\n"
        "Required fields:\n"
        "  - adata_file => Path to the .h5ad AnnData file to be analyzed.\n"
        "  - covariates => [control, disease2, ...],\n"
        "  - cell_type_index => Field in `obs` representing cell types.\n"
        "  - covariate_index => Field in `obs` representing diseases/conditions.\n"
        "  - donor_index => Field in `obs` representing donor identifiers.\n"
        "  - sex_index => Field in `obs` representing sex or gender annotations.\n"
        "Note:\n"
        "  - Ensure `adata_file` contains the required `obs` fields specified above.\n"
        "Optional fields:\n"
        "  - restrict_variable1...restrict_variable4 and variable1_index...variable4_index: Optional fields to restrict the dataset based on metadata columns before plotting. These parameters allow finer control over which subset of the data is used by specifying metadata values and their corresponding column names.\n"
    ),
    "volcano": (
        "Volcano or 'all' plot:\n"
        "Displays Log2(Fold Change) vs. -log10(FDR) for DE genes.\n"
        "Required fields:\n"
        "  - gene_symbols: A list of gene IDs to display in the heatmap. Can be an empty list if not applicable.\n"
        "  - cell_type: A specific cell type to subset the data (must align with 'cell_type_index').\n"
        "  - direction: A variable describing whether to look at differentially expressed genes in disease vs control, either 'regulated', 'up', 'down' or to look at cell type specific 'markers' stored in the H5AD.\n"
        "  - disease: Specifies the condition or disease used for differential expression analysis. \n"
        "    This should correspond to a grouping variable present in the dataset, which distinguishes control vs. disease samples.\n"
        "Optional fields: \n" 
        "  - top_n=8 to highlight top genes.\n"
        "  - restrict_variable1...restrict_variable4 and variable1_index...variable4_index: Optional fields to restrict the dataset based on metadata columns before plotting. These parameters allow finer control over which subset of the data is used by specifying metadata values and their corresponding column names.\n"
    ),
    "dotplot": (
        "Dotplot or 'all' plot:\n"
        "Plots gene expression (dot size ~ fraction, dot color ~ expression) across groups.\n"
        "Required fields:\n"
        "  - plot_type: Must be 'dotplot' or 'all' to specify the type of plot.\n"
        "  - adata_file => Path to the .h5ad AnnData file to be analyzed.\n"
        "  - gene_symbols: A list of genes to visualize, or an empty list if inferred from metadata.\n"
        "  - cell_type: Specifies the specific cell type to visualize in the single-cell-type version (optional for multi-cell-type plots).\n"
        "  - covariates: List of conditions (e.g., diseases) to filter by. This list must be non-empty. If not specified, choose a non-disease condition.\n"
        "  - direction: A variable describing whether to look at differentially expressed genes in disease vs control, either 'regulated', 'up', 'down' or to look at cell type specific 'markers' stored in the H5AD.\n"
        "  - covariate_index: Column in adata.obs representing covariates or experimental conditions for grouping (optional).\n"
        "  - cell_type_index: Column in adata.obs representing cell types (required for filtering and subsetting).\n"
        "Optional fields:\n"
        "  - restrict_variable1...restrict_variable4 and variable1_index...variable4_index: Optional fields to restrict the dataset based on metadata columns before plotting. These parameters allow finer control over which subset of the data is used by specifying metadata values and their corresponding column names.\n"
    ),
    "violin": (
        "Violin or 'all' plot:\n"
        "Displays the distribution of expression for a single gene across specified covariates.\n"
        "Required fields:\n"
        "  - adata_file (path to the dataset file in AnnData format)\n"
        "  - gene (the single gene symbol to plot; must exist in adata.var_names)\n"
        "  - cell_type (specific cell type to focus on)\n"
        "  - covariates (list of conditions to filter by; must be non-empty)\n"
        "  - covariate_index (primary grouping index in adata.obs, e.g., 'disease')\n"
        "  - cell_type_index: Column name representing cell types in adata.obs. Default is 'ann_finest_level'.\n"
        "  - display_variables: A mandatory list of all the grouping fields for aggregation or visualization. This is dynamically derived\n"
        "                       from the dataset's metadata and must be explicitly provided.\n"
        "Optional fields:\n"
        "  - The first variable in the list **must** be covariate_index. The second variable, if provided, is used for alternative grouping or coloring.)\n"
        "  - restrict_variable1...restrict_variable4 and variable1_index...variable4_index: Optional fields to restrict the dataset based on metadata columns before plotting. These parameters allow finer control over which subset of the data is used by specifying metadata values and their corresponding column names.\n"
    ),
    "venn": (
        "Venn or 'all' plot:\n"
        "Shows overlap of marker genes for 2 or 3 cell types.\n"
        "Required field:\n"
        "  - cell_types_to_compare with length 2 or 3. Do not select the 'Unknown' cell type\n"
        "Optional fields:\n"
        "  - restrict_variable1...restrict_variable4 and variable1_index...variable4_index: Optional fields to restrict the dataset based on metadata columns before plotting. These parameters allow finer control over which subset of the data is used by specifying metadata values and their corresponding column names.\n"
    ),
    "upset_genes": (
        "UpSet_Genes or 'all' plot:\n"
        "Handles complex overlaps of marker genes among multiple sets of cell types.\n"
        "Required field:\n"
        "  - cell_types_to_compare: list of cell types.\n"
        "Optional fields:\n"
        "  - restrict_variable1...restrict_variable4 and variable1_index...variable4_index: Optional fields to restrict the dataset based on metadata columns before plotting. These parameters allow finer control over which subset of the data is used by specifying metadata values and their corresponding column names.\n"
    ),
    "umap": (
        "UMAP or 'all' plot:\n"
        "Required fields:\n"
        "  - adata_file: Path to the AnnData .h5ad file.\n"
        "  - plot_type: Must be 'umap'.\n"
        "Optional fields:\n"
        "  - covariate_index: Column in adata.obs to subset data (e.g., 'condition').\n"
        "  - covariates: List of values within covariate_index to subset (e.g., ['BPD+PH', 'Term infant']).\n"
        "  - color_by: Metadata column or gene to color the UMAP by (e.g., 'celltype' or 'sex').\n"
        "  - cell_type_index: Column in adata.obs for clustering. NOTE: Default is similar to field chosen for for 'color_by' if not provided.\n"
        "  - gene: Gene name to visualize expression (overrides color_by if present).\n"
        "  - restrict_variable1...restrict_variable4 and variable1_index...variable4_index: Optional fields to restrict the dataset based on metadata columns before plotting. These parameters allow finer control over which subset of the data is used by specifying metadata values and their corresponding column names.\n"
    ),
    "network": (
        "Network or 'all' plot:\n"
        "Constructs a gene interaction network from 'gene_symbols' or top genes by Log Fold Change.\n"
        "Required:\n"
        "  - gene_symbols with 'Gene' and 'Log Fold Change'.\n"
        "  - direction: A variable describing whether to look at differentially expressed genes in disease vs control, either 'regulated', 'up', 'down' or to look at cell type specific 'markers' stored in the H5AD.\n"
        "  - cell_type: A specific cell type to subset the data\n"
        "  - covariate_index: Key in adata.obs for filtering covariates.\n"
        "  - covariates: List of specific values for the covariate index to filter data.\n"
        "Optional:\n"
        "  - n_genes: Number of top genes to use if gene_symbols is not provided (default=1000).\n"
        "  - network_technology: 'igraph' or 'networkx' for graphing library selection (default 'igraph').\n"
        "  - restrict_variable1...restrict_variable4 and variable1_index...variable4_index: Optional fields to restrict the dataset based on metadata columns before plotting. These parameters allow finer control over which subset of the data is used by specifying metadata values and their corresponding column names.\n"
    )

}
