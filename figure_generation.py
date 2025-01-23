from preload_datasets import PRELOADED_DATA
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import pandas as pd
import time,sys
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import os

import matplotlib_venn as venn
import upsetplot as upset
import networkx as nx
import circos
from matplotlib_venn import venn2, venn3
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import argparse
import json
import zipfile

import matplotlib
import logging
from matplotlib import rcParams

# Suppress font-related messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Set a default font family to avoid Arial warnings
rcParams['font.family'] = 'DejaVu Sans'


########################################################################################################
# --- Dynamic UMAP Functions ---
def assign_numeric_labels(adata, category_col):
    unique_categories = sorted(adata.obs[category_col].unique(), key=lambda x: x.lower())
    category_mapping = {cat: str(i + 1) for i, cat in enumerate(unique_categories)}
    adata.obs[f'{category_col}_label'] = adata.obs[category_col].map(category_mapping)
    return category_mapping

def generate_color_map(labels):
    color_map = plt.get_cmap('tab20')
    return {label: color_map(i / len(labels)) for i, label in enumerate(labels)}

def subset_data(adata, covariate_col, values):
    if not values:
        return {"All": adata}  # No subset: Return full data
    return {val: adata[adata.obs[covariate_col] == val].copy() for val in values}

def create_legend_table(mapping, dynamic_column_name, dynamic_label_column):
    return pd.DataFrame({
        dynamic_column_name: list(mapping.keys()),
        dynamic_label_column: list(mapping.values())
    })

def main_visualizations(
    adata,
    subset_col=None,
    subset_values=None,
    cluster_by=None,
    color_by=None,
    gene=None,
    save_prefix="UMAP"
):
    """
    Generate UMAP plots with improved legend handling for both gene expression and categorical coloring:
    - Supports a list of genes for side-by-side gene expression UMAPs.
    - Maintains original logic for non-gene plots and single-gene cases.
    """
    generated_plots = []
    global root

    # Determine gene mode and prepare gene list if necessary
    gene_flag = False
    gene_list = []
    if gene:
        if isinstance(gene, list):
            gene_flag = True
            gene_list = gene
        else:
            gene_flag = True
            gene_list = [gene]

    # Create cluster mapping and colors if cluster_by is provided
    cluster_colors = {}
    mapping = None
    if cluster_by:
        unique_clusters = sorted(adata.obs[cluster_by].unique())
        if gene_flag:  # Only create numeric mapping for gene expression plots
            mapping = {cluster: str(i + 1) for i, cluster in enumerate(unique_clusters)}
            adata.obs[f'{cluster_by}_label'] = adata.obs[cluster_by].map(mapping)
        cluster_colors = {cluster: plt.cm.tab20(i/len(unique_clusters)) 
                         for i, cluster in enumerate(unique_clusters)}

    # Adjust color_column assignment to avoid list types during gene mode
    if gene_flag:
        color_column = None
    else:
        # Use color_by if it's not a list; otherwise fall back to cluster_by
        color_column = color_by if color_by and not isinstance(color_by, list) else (cluster_by)

    # Generate color map for categorical plotting only if appropriate
    if color_column and color_column in adata.obs and adata.obs[color_column].dtype.name == 'category':
        global_palette = generate_color_map(adata.obs[color_column].unique())
    else:
        global_palette = None

    # Subset the data
    subsets = subset_data(adata, subset_col, subset_values) if subset_col else {"All": adata}
    subset_names = list(subsets.keys())
    n_subsets = len(subset_names)

    # Setup figure and axes depending on gene_flag and number of genes/subsets
    if gene_flag and gene_list:
        n_genes = len(gene_list)
        # Create grid with rows = number of genes, columns = number of subsets
        fig = plt.figure(figsize=(8 * n_subsets * n_genes, 10 * n_genes))
        gs = fig.add_gridspec(n_genes, n_subsets, wspace=0.4, hspace=0.4)
        axes = [[fig.add_subplot(gs[row, col]) for col in range(n_subsets)] for row in range(n_genes)]
        # Allocate separate legend axis if needed
        legend_ax = None
        if n_subsets == 1:
            legend_ax = fig.add_subplot(gs[:, -1])
            legend_ax.axis('off')
    else:
        if n_subsets == 1:
            if gene_flag:  # Single gene expression plot
                fig = plt.figure(figsize=(12, 10))
                gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])
            else:  # Regular categorical plot
                fig = plt.figure(figsize=(10, 8))
                gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0])
            if gene_flag:
                legend_ax = fig.add_subplot(gs[1])
                legend_ax.axis('off')
            axes = [ax]
        else:
            fig = plt.figure(figsize=(8 * n_subsets, 10))
            gs = fig.add_gridspec(2, n_subsets, height_ratios=[4, 1])
            axes = [fig.add_subplot(gs[0, i]) for i in range(n_subsets)]
            legend_ax = fig.add_subplot(gs[1, :])
            legend_ax.axis('off')

    output_suffix = "side_by_side_umap" if n_subsets > 1 else "umap"
    legend_elements = []

    # Plotting logic
    if gene_flag and gene_list:
        # Loop over each gene (row) and each subset (column)
        for row, gene_val in enumerate(gene_list):
            for col, (name, sub_adata) in enumerate(subsets.items()):
                ax = axes[row][col]
                title_str = f"UMAP: {gene_val} - {name}" if name != "All" else f"UMAP: {gene_val}"
                sc.pl.umap(
                    sub_adata,
                    color=gene_val,
                    ax=ax,
                    show=False,
                    title=title_str,
                    legend_loc=None,  # No legend for gene expression
                    cmap='viridis'
                )

                # Add cluster numbers if cluster_by is provided
                if cluster_by and mapping:
                    for cluster in sub_adata.obs[cluster_by].unique():
                        mask = sub_adata.obs[cluster_by] == cluster
                        cluster_cells = sub_adata[mask]
                        if len(cluster_cells) > 0:
                            center = np.median(cluster_cells.obsm['X_umap'], axis=0)
                            ax.text(center[0], center[1], mapping[cluster],
                                    fontweight='bold', ha='center', va='center',
                                    bbox=dict(facecolor='white', alpha=0.7, 
                                              edgecolor='none', pad=1))
                # Create legend table once in the first row, first column
                if row == 0 and col == 0 and cluster_by and mapping:
                    legend_table_df = create_legend_table(
                        mapping,
                        dynamic_column_name=cluster_by.capitalize(),
                        dynamic_label_column="Cluster"
                    )
                    if legend_ax is None:
                        legend_ax = fig.add_subplot(gs[row, -1])
                        legend_ax.axis('off')
                    table = legend_ax.table(
                        cellText=legend_table_df.values,
                        colLabels=legend_table_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1]
                    )
                    for key, cell in table.get_celld().items():
                        if key[0] == 0:
                            cell.set_text_props(weight='bold', fontsize=12)
                        else:
                            cell.set_text_props(fontsize=10)
                    table.scale(1.5, 1.5)

    else:
        for i, (name, sub_adata) in enumerate(subsets.items()):
            title_str = f"UMAP: {color_column} - {name}" if name != "All" else f"UMAP: {color_column}"
            if gene_flag:
                sc.pl.umap(
                    sub_adata,
                    color=color_column,
                    ax=axes[i],
                    show=False,
                    title=title_str,
                    legend_loc=None,
                    cmap='viridis'
                )
                if cluster_by and mapping:
                    for cluster in sub_adata.obs[cluster_by].unique():
                        mask = sub_adata.obs[cluster_by] == cluster
                        cluster_cells = sub_adata[mask]
                        if len(cluster_cells) > 0:
                            center = np.median(cluster_cells.obsm['X_umap'], axis=0)
                            axes[i].text(center[0], center[1], mapping[cluster],
                                         fontweight='bold', ha='center', va='center',
                                         bbox=dict(facecolor='white', alpha=0.7, 
                                                   edgecolor='none', pad=1))
                    if i == 0:
                        legend_table_df = create_legend_table(
                            mapping,
                            dynamic_column_name=cluster_by.capitalize(),
                            dynamic_label_column="Cluster"
                        )
                        table = legend_ax.table(
                            cellText=legend_table_df.values,
                            colLabels=legend_table_df.columns,
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1]
                        )
                        for key, cell in table.get_celld().items():
                            if key[0] == 0:
                                cell.set_text_props(weight='bold', fontsize=12)
                            else:
                                cell.set_text_props(fontsize=10)
                        table.scale(1.5, 1.5)
            else:
                sc.pl.umap(
                    sub_adata,
                    color=color_column,
                    palette=global_palette,
                    ax=axes[i],
                    legend_loc='right margin' if n_subsets == 1 else None,
                    title=title_str,
                    show=False
                )
                if n_subsets > 1 and i == 0 and color_column in sub_adata.obs:
                    categories = sub_adata.obs[color_column].unique()
                    colors = [global_palette[cat] if global_palette else plt.cm.tab20(j) 
                              for j, cat in enumerate(categories)]
                    legend_elements.extend(zip(categories, colors))

    if legend_elements and not gene_flag:
        patches = [mpatches.Patch(color=color, label=label) 
                   for label, color in legend_elements]
        legend_ax.legend(handles=patches, title=color_column,
                        loc='center', bbox_to_anchor=(0.5, 0.5),
                        ncol=min(5, len(patches)))

    subset_values_str = "_".join(
        sub.replace(" ", "_").replace("+", "plus") for sub in subset_values
    ) if subset_values else "All"
    gene_suffix = f"_{'_'.join(gene_list)}" if gene_flag else ""
    output_path_png = f"{root}/{save_prefix}{gene_suffix}_{subset_values_str}_{output_suffix}.png"
    output_path_pdf = f"{root}/{save_prefix}{gene_suffix}_{subset_values_str}_{output_suffix}.pdf"

    plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
    plt.savefig(output_path_pdf, bbox_inches='tight', dpi=300)
    plt.close()

    description = f"UMAP visualization colored by {color_column}"
    if subset_values:
        description += f" for {', '.join(subset_values)}"

    generated_plots.append([output_path_pdf, description])
    generated_plots.append([output_path_png, description])

    return generated_plots
#########################################################################################################

plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.facecolor'] = 'white'

# Main function to parse JSON and execute tasks
def main(json_input, output_dir="results"):
    """
    Main function to parse JSON input and execute tasks.

    Parameters:
    - json_input: Path to the JSON file containing query parameters.
    - output_dir: Directory to save the results.
    """
    global adata, cell_type_index, covariate_index, donor_index, sex_index, covariates, root
    global variable1_index, variable2_index, variable3_index, variable4_index, gene_symbol
    global restrict_variable1, restrict_variable2, restrict_variable3, restrict_variable4
    global cell_types_to_compare

    root = output_dir
    # Parse the JSON file
    with open(json_input, "r") as f:
        config = json.load(f)

    # Extract input parameters
    adata_file = config["adata_file"]
    plot_type = config["plot_type"]
    cell_type_index = config.get("cell_type_index", "cell type")
    covariate_index = config.get("covariate_index", None)
    donor_index = config.get("donor_index", "donor")
    sex_index = config.get("sex_index", "sex")
    variable1_index = config.get("variable1_index", None)
    variable2_index = config.get("variable2_index", None)
    variable3_index = config.get("variable3_index", None)
    variable4_index = config.get("variable4_index", None)
    restrict_variable1 = config.get("restrict_variable1", None)
    restrict_variable2 = config.get("restrict_variable2", None)
    restrict_variable3 = config.get("restrict_variable3", None)
    restrict_variable4 = config.get("restrict_variable4", None)
    covariates = config.get("covariates", [])
    cell_type = config.get("cell_type", None)
    gene_symbol = config.get("gene", None)
    disease = config.get("disease", None)
    display_variables = config.get("display_variables", [covariate_index])
    gene_symbols = config.get("gene_symbols") or []
    cell_types_to_compare = config.get("cell_types_to_compare", [])
    n_genes = config.get("n_genes", 100)
    if "network" in plot_type:
        n_genes = 1000
    direction = config.get("direction", None)
    heatmap_technology = config.get("heatmap_technology", "seaborn")
    network_technology = config.get("network_technology", "igraph")
    cell_types_to_compare = config.get("cell_types_to_compare", None)
    show_individual_cells = config.get("show_individual_cells", True)
    plots = []

    # Load AnnData file    
    ##print(f"Looking for preloaded dataset: {adata_file}")
    start_time = time.time()

    # Use the preloaded AnnData object instead of reading from disk
    if adata_file in PRELOADED_DATA:
        adata = PRELOADED_DATA[adata_file]
        ##print(f"Using preloaded dataset for {adata_file}")
    else:
        # Fallback or error if the dataset isn't found
        raise ValueError(f"{adata_file} not found in PRELOADED_DATA. Ensure it was preloaded.")

    # Apply variable restrictions
    if restrict_variable1 and variable1_index:
        ##print(f"Filtering {variable1_index} by {restrict_variable1}")
        adata = adata[adata.obs[variable1_index].isin(restrict_variable1)].copy()

    if restrict_variable2 and variable2_index:
        #print(f"Filtering {variable2_index} by {restrict_variable2}")
        adata = adata[adata.obs[variable2_index].isin(restrict_variable2)].copy()

    if restrict_variable3 and variable3_index:
        #print(f"Filtering {variable3_index} by {restrict_variable3}")
        adata = adata[adata.obs[variable3_index].isin(restrict_variable3)].copy()

    if restrict_variable4 and variable4_index:
        #print(f"Filtering {variable4_index} by {restrict_variable4}")
        adata = adata[adata.obs[variable4_index].isin(restrict_variable4)].copy()

    # Filter by covariates
    if covariate_index and covariates:
        #print(f"Filtering by covariates: {covariates}")
        adata = adata[adata.obs[covariate_index].isin(covariates)].copy()

    # Import gene signatures from the h5ad

    if plot_type != 'venn' and plot_type != 'upset_genes':
        gene_symbols, filtered_df = h5ad_gene_signatures(gene_symbols,direction,cell_type,disease,n_genes)


    # Ensure only the specified `covariates` are included
    if covariate_index in adata.obs:
        #print(f"Filtering for sample groups: {covariates}")
        adata = adata[adata.obs[covariate_index].isin(covariates)].copy()
    else:
        #print(f"Disease index '{covariate_index}' not found in adata.obs")
        covariate_index = variable1_index

    if plot_type == 'stats' or plot_type == 'all':
        output_file = f"{root}/Genes-{cell_type}_{disease}_{direction}.tsv"
        filtered_df.to_csv(output_file, sep="\t", index=False)
        plots.append([output_file,f"Differentially expressed {direction} genes in {cell_type} by {disease}"])

    if plot_type == 'heatmap' or plot_type == 'all':
        if cell_type != None and len(gene_symbols)>0:
            if len(gene_symbols)<250 and heatmap_technology == 'seaborn':
                    plots = plot_heatmap(gene_symbols,cell_type,group_by=display_variables,cluster_rows=True,
                            cluster_columns=True,show_individual_cells=show_individual_cells,median_scale_expression=False,
                            samples_to_visualize="cell-type",covariate=disease,plots=plots)
            else:
                plots = plot_heatmap_with_imshow(gene_symbols,cell_type,group_by=display_variables[0],cluster_rows=True,
                        cluster_columns=True,show_individual_cells=show_individual_cells,median_scale_expression=False,
                        samples_to_visualize="cell-type",covariate=disease,plots=plots)

    if plot_type == 'heatmap' or plot_type == 'all':
        for covariate in covariates:
            if len(gene_symbols)<250 and heatmap_technology == 'seaborn':
                plots = plot_heatmap(gene_symbols,cell_type,group_by=[cell_type_index],cluster_rows=True,
                        cluster_columns=False,show_individual_cells=False,median_scale_expression=False,
                        samples_to_visualize="all",covariate=covariate,plots=plots)
            else:
                plots = plot_heatmap_with_imshow(gene_symbols,cell_type,group_by=cell_type_index,cluster_rows=True,
                        cluster_columns=False,show_individual_cells=False,median_scale_expression=False,
                        samples_to_visualize="all",covariate=covariate,plots=plots)

    if plot_type == 'radar' or plot_type == 'all':
        output_pdf=f"{root}/{disease}-radar_cell_frequency.pdf"
        plots = plot_aggregated_cell_frequencies_radar(covariate_index,donor_index,cell_type_index,output_pdf=output_pdf,plots=plots)

    if plot_type == 'cell_frequency' or plot_type == 'all':
        plots = plot_cell_type_frequency_per_donor(control_group=covariates[0],plots=plots)

    if plot_type == 'volcano' or plot_type == 'all':
        plots = plot_volcano(cell_type,disease,gene_symbols,plots=plots)

    if plot_type == 'dotplot' or plot_type == 'all':
        # Capture the return value here
        plots = plot_dot_plot_celltype(
            gene_symbols, cell_type, group_by=covariate_index, plots=plots
        )
        if covariates is None:
            # Also capture the return value here
            plots = plot_dot_plot_all_celltypes(
                gene_symbols, color_map='Reds', plots=plots
            )
        else:
            for covariate in covariates:
                plots = plot_dot_plot_all_celltypes(
                    gene_symbols, covariate=covariate, color_map='Reds', plots=plots
                )

    if plot_type == 'violin' or plot_type == 'all':
        if len(display_variables) > 1:
            dis = display_variables[1]
        elif len(display_variables) > 0:
            dis = display_variables[0]
        else:
            dis = None
        plots = plot_gene_violin(gene_symbol, cell_type, covariates, covariate_index, alt_covariate_index=dis, plots=plots)

    if plot_type == 'venn' or plot_type == 'all':
        if len(cell_types_to_compare)<4:
            plots = compare_marker_genes_and_plot_venn(cell_types_to_compare,plots=plots)
        else:
            plots = compare_marker_genes_and_plot_upset(cell_types_to_compare,plots=plots)

    if plot_type == 'upset_genes' or plot_type == 'all':
        plots = compare_marker_genes_and_plot_upset(cell_types_to_compare,plots=plots)

    if plot_type in ['umap', 'all']:
        # Extract relevant variables for UMAP from JSON configuration
        cell_type_index = config.get("cell_type_index")
        covariate_index = config.get("covariate_index")
        covariates = config.get("covariates", None)  # Subset values for conditions
        display_variables = config.get("display_variables", [])  # Metadata fields for display
        gene_val = config.get("gene", None)  # Gene for expression coloring, optional
        color_by_val = config.get("color_by")  # Retrieve value for color_by
        save_prefix = "UMAP"  # Default save prefix

        # If cell_type_index is not provided, use the value provided to color_by
        if cell_type_index is None:
            cell_type_index = color_by_val

        # Call the dynamic UMAP function
        umap_plots = main_visualizations(
            adata=adata,
            subset_col=covariate_index,  # Use covariate_index to subset data
            subset_values=covariates,    # Subset to these conditions
            cluster_by=cell_type_index,  # Cluster by cell type (or color_by value if cell_type_index was missing)
            color_by=gene_val or cell_type_index,  # Color by gene or cell type
            gene=gene_val,               # Gene name for expression
            save_prefix=save_prefix,
        )
        plots.extend(umap_plots)

    if plot_type == 'network' or plot_type == 'all':
        output_file = f"{root}/{cell_type}_{disease}_{n_genes}_{direction}-network.pdf"
        if network_technology == "igraph":
            plots = visualize_gene_network_igraph(gene_symbols,output_file=output_file,plots=plots)
        else:
            plots = visualize_gene_networkX(gene_symbols,output_file=output_file,plots=plots)

    end_time = time.time()  # End the timer
    #print(f"Execution time: {end_time - start_time:.2f} seconds")
    #print ("plots:",plots)
    return plots

def h5ad_gene_signatures(gene_symbols, direction, cell_type, disease, n_genes):
    """
    Retrieve gene signatures based on markers or differentially expressed genes.

    Parameters:
    - gene_symbols: List of preselected gene symbols (can be empty).
    - direction: 'markers', 'up', or 'down' to determine the type of analysis.
    - cell_type: Cell type to filter by.
    - disease: Condition to filter by.
    - n_genes: Number of top genes to select.

    Returns:
    - List of selected gene symbols (unique, sorted by p-value).
    """
    ##print([gene_symbols, direction, cell_type, disease])
    
    if len(gene_symbols) == 0 and direction is not None:
        if direction == 'markers':
            if "marker_stats" not in adata.uns:
                raise ValueError("The AnnData object does not contain 'marker_stats'. Ensure marker analysis was computed.")
            
            marker_stats_df = adata.uns["marker_stats"]
            
            # Check if "Cell Population" column exists
            if "Cell Population" not in marker_stats_df.columns:
                raise ValueError("The 'Cell Population' column is missing in marker_stats.")
            
            # Filter by cell type
            filtered_df = marker_stats_df[marker_stats_df["Cell Population"] == cell_type]
            
            if filtered_df.empty:
                raise ValueError(f"No marker genes found for cell type '{cell_type}'.")
            
            top_genes = filtered_df
        
        else:  # Differential expression analysis
            if "disease_stats" not in adata.uns:
                raise ValueError("The AnnData object does not contain 'disease_stats'. Ensure differential expression was computed.")
            
            disease_stats_df = adata.uns["disease_stats"]
            
            # Check for required columns
            required_columns = ["Cell Type", "Condition", "Gene", "P-Value", "Log Fold Change"]
            for col in required_columns:
                if col not in disease_stats_df.columns:
                    raise ValueError(f"The column '{col}' is missing in disease_stats.")
            
            # Check if the condition exists in the dataset
            condition_exists = disease in disease_stats_df["Condition"].unique()
            
            if not condition_exists:
                raise ValueError(
                    f"The condition '{disease}' does not exist in the dataset."
                )
            
            # Check if any cell types have DEGs for the condition
            condition_cell_types = disease_stats_df[disease_stats_df["Condition"] == disease]["Cell Type"].unique()
            
            if cell_type not in condition_cell_types:
                # List cell types with DEGs for the condition
                if len(condition_cell_types) > 0:
                    raise ValueError(
                        f"The cell type '{cell_type}' has no DEGs reported for the condition '{disease}'. "
                        f"However, the following cell types have DEGs for '{disease}': {', '.join(condition_cell_types)}."
                    )
                else:
                    raise ValueError(
                        f"No DEGs are reported for any cell type for the condition '{disease}'."
                    )
            
            # Filter by cell type and condition
            filtered_df = disease_stats_df[
                (disease_stats_df["Cell Type"] == cell_type) & 
                (disease_stats_df["Condition"] == disease)]
            
            if filtered_df.empty:
                raise ValueError(f"No DEGs found for cell type '{cell_type}' and condition '{disease}'.")
            
            # Filter by direction
            if direction == "up":
                filtered_df = filtered_df[filtered_df["Log Fold Change"] > 0]
            elif direction == "down":
                filtered_df = filtered_df[filtered_df["Log Fold Change"] < 0]

            if filtered_df.empty:
                raise ValueError(f"No DEGs found for direction '{direction}' in the specified filters.")
        
        # Deduplicate by keeping the row with the smallest p-value for each gene
        filtered_df = filtered_df.sort_values("P-Value").drop_duplicates(subset="Gene", keep="first")

        # Sort by p-value and limit to top n_genes
        top_genes = filtered_df.nsmallest(n_genes, "P-Value")
        
        return top_genes, filtered_df
    else:
        # Return preselected gene symbols
        return gene_symbols, gene_symbols

# Grahing functions
def plot_cell_type_frequency_per_donor(control_group='control',plots=[]):
    """ 
    Produce a box and whisker plot displaying cell frequency by donor (not cells or metacells)
    for disease annotated groups, technology, ethnicity or other variables and assess statisitcal
    differences within cell types. Denote male and female differences
    """

    description = 'Cell frequency barchart by donor with mannwhitneyu differences to control'

    # Group by cell type, disease, and donor
    cell_type_counts_per_donor = (
        adata.obs.groupby([cell_type_index, covariate_index, donor_index])
        .size()
        .reset_index(name='count')
    )
    total_cells_per_donor_condition = (
        adata.obs.groupby([covariate_index, donor_index])
        .size()
        .reset_index(name='total')
    )

    # Remap sex_index to avoid namespace conflicts
    local_sex_index = sex_index if sex_index is not None else 'sex'

    # Merge and map sex information
    if sex_index is None:
        # Set sex to 'unknown' for all donors
        adata.obs[local_sex_index] = 'unknown'
    else:
        # Use the provided sex_index
        adata.obs[local_sex_index] = adata.obs[sex_index]

    # Create donor_sex_map
    donor_sex_map = (
        adata.obs[[donor_index, local_sex_index]]
        .drop_duplicates(subset=donor_index)  # Ensure donor_id is unique
        .set_index(donor_index)[local_sex_index]
    )

    # Replace 'unknown' with NaN to exclude these from analysis or plot them distinctly
    adata.obs[local_sex_index] = adata.obs[local_sex_index].replace('unknown', np.nan)
    merged_df = pd.merge(
        cell_type_counts_per_donor, total_cells_per_donor_condition, on=[covariate_index, donor_index]
    )
    merged_df[local_sex_index] = merged_df[donor_index].map(donor_sex_map)

    # Debugging: #print unique values in 'sex'
    #print(f"Unique values in adata.obs['{local_sex_index}']: {adata.obs[local_sex_index].unique()}")
    #print(f"Unique values in merged_df['{local_sex_index}'] before filtering: {merged_df[local_sex_index].unique()}")

    # Exclude rows with NaN in 'sex'
    merged_df = merged_df.dropna(subset=[local_sex_index])

    # Debugging: #print shape and head of merged DataFrame
    #print(f"Final merged DataFrame (shape): {merged_df.shape}")
    #print(f"Final merged DataFrame (head):\n{merged_df.head()}")

    # Calculate frequency percentages
    merged_df['Frequency (%)'] = (merged_df['count'] / merged_df['total']) * 100

    # Define a custom color palette
    # Dynamically generate a palette matching the number of covariates
    num_categories = len(covariates)
    custom_palette = sns.color_palette("Paired", n_colors=num_categories)

    # Ensure the control group is the first category
    ordered_covariates = [control_group] + [group for group in covariates if group != control_group]
    #print(f"Ordered sample groups: {ordered_covariates}")

    # Plot boxplots with adjusted outlier size
    flierprops = {'marker': 'o', 'markersize': 4, 'color': 'black'}
    plt.figure(figsize=(16, 10))
    ax = sns.boxplot(
        x=cell_type_index,
        y='Frequency (%)',
        hue=covariate_index,
        data=merged_df,
        whis=1.5,
        palette=custom_palette,
        linewidth=0.5,
        showfliers=False,
        hue_order=ordered_covariates,  # Set the control group as the first category
        flierprops=flierprops,
    )

    # Plot male, female, and unknown samples with different markers and adjusted size
    markers = {'male': 'v', 'female': 'o', 'unknown': 's'}
    for sex, marker in markers.items():
        sns.stripplot(
            x=cell_type_index,
            y='Frequency (%)',
            hue=covariate_index,
            data=merged_df[merged_df[local_sex_index] == sex],
            dodge=True,
            jitter=True,
            marker=marker,
            alpha=0.7,
            size=1,
            linewidth=0.5,
            palette=custom_palette,
            ax=ax,
        )

    # Statistical testing: Compare each disease group to the control group
    for i, cell_type in enumerate(merged_df[cell_type_index].unique()):
        control_data = merged_df[
            (merged_df[cell_type_index] == cell_type) & (merged_df[covariate_index] == control_group)
        ]['Frequency (%)'].dropna()

        for disease in covariates:
            if disease != control_group:
                disease_data = merged_df[
                    (merged_df[cell_type_index] == cell_type) & (merged_df[covariate_index] == disease)
                ]['Frequency (%)'].dropna()
                # Ensure both groups have non-empty data
                if len(control_data) > 0 and len(disease_data) > 0:
                    stat, p_value = mannwhitneyu(control_data, disease_data)
                    if p_value < 0.01:
                        max_value = merged_df[
                            merged_df[cell_type_index] == cell_type
                        ]['Frequency (%)'].max()
                        y, h = max_value + 2, 1.5
                        x1, x2 = i - 0.2, i + 0.2
                        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color='black')
                        ax.text(i, y + h + 0.5, '*', ha='center', va='bottom', color='black', fontsize=15)

    # Prepare concise data for export
    export_data = []
    for cell_type in merged_df[cell_type_index].unique():
        control_data = merged_df[
            (merged_df[cell_type_index] == cell_type) & (merged_df[covariate_index] == control_group)
        ]['Frequency (%)'].dropna()
        control_mean = control_data.mean() if len(control_data) > 0 else None
        for disease in covariates:
            if disease != control_group:
                disease_data = merged_df[
                    (merged_df[cell_type_index] == cell_type) & (merged_df[covariate_index] == disease)
                ]['Frequency (%)'].dropna()
                disease_mean = disease_data.mean() if len(disease_data) > 0 else None
                if control_mean is not None and disease_mean is not None:
                    stat, p_value = mannwhitneyu(control_data, disease_data)
                    export_data.append({
                        'Cell Type': cell_type,
                        'Control Group (Healthy) Frequency (%)': control_mean,
                        'Disease Group': disease,
                        'Disease Group Frequency (%)': disease_mean,
                        'P-Value': p_value
                    })

    # Convert export data to DataFrame and save as tab-delimited text file
    concise_export_df = pd.DataFrame(export_data)
    concise_export_filename = f"{root}/cell_frequency_comparisons_summary.tsv"
    concise_export_df.to_csv(concise_export_filename, sep='\t', index=False)

    # Add a combined legend for disease groups and sex
    handles, labels = ax.get_legend_handles_labels()
    unique_disease_handles = handles[:len(covariates)]
    unique_disease_labels = labels[:len(covariates)]
    sex_handles = [
        plt.Line2D([0], [0], marker='v', color='black', linestyle='', label='Male'),
        plt.Line2D([0], [0], marker='o', color='black', linestyle='', label='Female'),
        plt.Line2D([0], [0], marker='s', color='black', linestyle='', label='Unknown'),
    ]
    ax.legend(
        unique_disease_handles + sex_handles,
        unique_disease_labels + ['Male', 'Female', 'Unknown'],
        title='Disease Condition and Sex',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
    )

    output_pdf = f"{root}/cell_type_frequencies_per_donor_boxplot_sig.pdf"
    plt.title('Frequency of Cell Types Across Donors by Disease Condition')
    plt.xlabel('Cell Type')
    plt.ylabel('Frequency (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.savefig(output_pdf.replace("pdf","png"), dpi=250)
    plt.close()

    plots.append([output_pdf, description])
    plots.append([output_pdf.replace("pdf","png"), description])
    plots.append([concise_export_filename, description])
    return plots

def plot_gene_violin(gene_symbol, cell_type, covariates, covariate_index, alt_covariate_index=None,plots=[]):
    """
    Plot a violin plot of gene expression for specified covariates, with an optional
    alternative covariate index for coloring (e.g., assays).

    Parameters:
    - gene_symbol: The gene symbol to plot.
    - cell_type: The cell type to subset.
    - covariates: The groups in the covariate index to filter by (e.g., diseases).
    - covariate_index: The primary index (e.g., disease conditions).
    - alt_covariate_index: The optional alternative index for coloring (e.g., assay type).
    """
    
    description = "Violin plot for a single gene in a single-cell type across conditions with possible confounding variables (optional)"
    covars = [cov for cov in covariates if cov is not None]  # Remove None values

    # Subset the data based on cell type and covariates
    adata_subset = adata[adata.obs[cell_type_index].isin([cell_type]) & adata.obs[covariate_index].isin(covars)].copy()

    if adata_subset.shape[0] == 0:
        #print(f"No data found for cell type '{cell_type}' and specified covariates.")
        return

    # Ensure proper ordering for the main covariate
    adata_subset.obs[covariate_index] = pd.Categorical(adata_subset.obs[covariate_index], categories=covars, ordered=True)

    # Extract expression data for the gene
    if gene_symbol not in adata_subset.var_names:
        #print(f"Gene {gene_symbol} not found in adata.var_names. Available examples: {adata_subset.var_names[:5]}")
        return

    gene_expression = adata_subset[:, gene_symbol].X.toarray().flatten()

    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x=adata_subset.obs[covariate_index],
        y=gene_expression,
        scale="width",
        inner=None,
        palette="muted",
    )

    # Overlay scatter plot for individual data points, colored by alt_covariate_index if provided
    if alt_covariate_index:
        if alt_covariate_index not in adata_subset.obs:
            #print(f"Alternative covariate index '{alt_covariate_index}' not found in adata.obs.")
            return

        # Map alternative covariates to unique colors
        unique_alt_covariates = adata_subset.obs[alt_covariate_index].astype(str).unique()
        color_palette = sns.color_palette("tab10", len(unique_alt_covariates))
        alt_colors = {alt_cov: color for alt_cov, color in zip(unique_alt_covariates, color_palette)}

        # Add scatter plot with colors and jitter
        for i, group in enumerate(covars):
            group_mask = adata_subset.obs[covariate_index] == group
            group_expression = gene_expression[group_mask]
            group_alt_covariates = adata_subset.obs.loc[group_mask, alt_covariate_index].astype(str)

            plt.scatter(
                x=np.random.normal(i, 0.1, size=len(group_expression)),  # Add jitter
                y=group_expression,
                c=[alt_colors[alt_cov] for alt_cov in group_alt_covariates],
                alpha=0.6,
                s=10,  # Decrease dot size
                linewidths=0,  # Remove edge color
            )

        # Add legend for alternative covariate colors
        legend_handles = [
            plt.Line2D(
                [0], [0], marker="o", color=color, label=alt_cov, linestyle="None"
            )
            for alt_cov, color in alt_colors.items()
        ]
        plt.legend(handles=legend_handles, title=alt_covariate_index, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Customize plot
    plt.title(f"{gene_symbol} Expression in {cell_type}")
    plt.xlabel(covariate_index)
    plt.ylabel("Expression")
    plt.xticks(rotation=45)

    # Save the plot
    pdf_filename = f"{root}/{gene_symbol}_{cell_type}_violin_plot.pdf"
    plt.tight_layout()
    plt.savefig(pdf_filename, bbox_inches="tight", dpi=250)
    plt.savefig(pdf_filename.replace("pdf","png"), bbox_inches="tight", dpi=250)
    plt.close()
    plots.append([pdf_filename, description])
    plots.append([pdf_filename.replace("pdf","png"), description])
    return plots


def plot_heatmap(
    gene_symbols, 
    cell_type, 
    group_by=["disease"], 
    figsize=(12, 8), 
    scaling_factor=0.3, 
    cluster_rows=False, 
    cluster_columns=False, 
    show_individual_cells=False,
    median_scale_expression=False,
    samples_to_visualize="cell-type",
    covariate=None,
    plots=[]
):
    description = f"lustered cell-type ({samples_to_visualize}) seaborn heatmap by {group_by[0]} with show_individual_cells {show_individual_cells} median scaling {median_scale_expression} for covariate {covariate}"
    if cluster_columns:
        description = "C" + description
    else:
        description = "Un-c" + description

    from matplotlib.colors import LinearSegmentedColormap
    from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # -- Minimal sanitizing function
    def sanitize(value):
        if value is None:
            return "None"
        return str(value).replace("'", "").replace('"', "").replace(" ", "_")

    # Sanitize filename components
    cell_type_str = sanitize(cell_type)
    group0_str = sanitize(group_by[0]) if isinstance(group_by, list) else sanitize(group_by)
    covariate_str = sanitize(covariate)

    # Subset logic
    if samples_to_visualize == "cell-type":
        subset_adata = adata[adata.obs[cell_type_index].isin([cell_type])]
    else:
        subset_adata = adata[adata.obs[covariate_index].isin([covariate])]

    if subset_adata.n_obs > 10000:
        downsample_indices = np.random.choice(subset_adata.obs_names, size=10000, replace=False)
        subset_adata = subset_adata[downsample_indices]

    try: 
        expression_data = subset_adata[:, gene_symbols].X.toarray()
    except Exception as e:
        if isinstance(gene_symbols, list):
            raise e
        else:
            gene_symbols = gene_symbols["Gene"].tolist()
            expression_data = subset_adata[:, gene_symbols].X.toarray()

    # Build data_to_plot
    if show_individual_cells:
        data_to_plot = expression_data.T  # Genes on rows
        x_labels = subset_adata.obs.index.to_numpy()
    else:
        grouped_obs = subset_adata.obs[group_by[0]].values
        unique_groups = np.unique(grouped_obs)
        median_expression = [
            np.median(expression_data[grouped_obs == group], axis=0)
            for group in unique_groups
        ]
        data_to_plot = np.array(median_expression).T
        x_labels = unique_groups

    if len(x_labels) < 100:
        x_labels_verbose = x_labels
    else:
        x_labels_verbose = False

    if median_scale_expression:
        data_scaled = data_to_plot - np.median(data_to_plot, axis=1, keepdims=True)
    else:
        data_scaled = data_to_plot

    abs_max = np.max(np.abs(data_scaled)) * scaling_factor

    # Clustering rows
    if cluster_rows:
        row_linkage = linkage(data_scaled, method="average")
        row_order = leaves_list(row_linkage)
        data_scaled = data_scaled[row_order, :]
        gene_symbols = [gene_symbols[i] for i in row_order]
        row_colors = dendrogram(row_linkage, no_plot=True)["leaves_color_list"]
    else:
        row_colors = None

    # Generate column colors
    unique_groups1 = subset_adata.obs[group_by[0]].unique()
    covariate1_palette = sns.color_palette("tab10", len(unique_groups1))
    covariate1_color_map = {group: covariate1_palette[i] for i, group in enumerate(unique_groups1)}

    if show_individual_cells:
        col_colors_group1 = np.array([covariate1_color_map.get(value, "gray") for value in subset_adata.obs[group_by[0]]])
    else:
        col_colors_group1 = np.array([covariate1_color_map.get(group, "gray") for group in unique_groups1])

    col_dendro_colors = None
    if cluster_columns:
        col_linkage = linkage(data_scaled.T, method="average")
        col_order = leaves_list(col_linkage)
        data_scaled = data_scaled[:, col_order]
        x_labels = [x_labels[i] for i in col_order]
        col_dendrogram = dendrogram(col_linkage, no_plot=True)
        col_dendro_colors = np.array(col_dendrogram["leaves_color_list"])

        if show_individual_cells:
            col_colors_group1 = np.array([
                covariate1_color_map.get(value, "gray")
                for value in subset_adata.obs[group_by[0]].iloc[col_order]
            ])
        else:
            grouped_obs = subset_adata.obs.groupby(group_by[0]).groups
            col_colors_group1 = np.array([
                covariate1_color_map.get(g, "gray")
                for g in np.array(list(grouped_obs.keys()))[col_order]
            ])

    color_bar2_label = "Clusters"
    if col_colors_group1 is not None and col_dendro_colors is not None:
        col_colors_combined = [col_colors_group1, col_dendro_colors]
    else:
        col_colors_combined = [col_colors_group1]

    # Create custom colormap
    yellow_black_blue = LinearSegmentedColormap.from_list(
        "yellow_black_blue", ["deepskyblue", "black", "yellow"]
    )

    # Create the clustermap
    g = sns.clustermap(
        data_scaled,
        cmap=yellow_black_blue,
        xticklabels=x_labels_verbose,
        yticklabels=gene_symbols,
        col_colors=col_colors_combined,
        row_colors=row_colors,
        figsize=figsize,
        vmin=-abs_max,
        vmax=abs_max,
        row_cluster=cluster_rows,
        col_cluster=cluster_columns,
        colors_ratio=0.015,
    )

    # Draw a thin white line between color bars
    if cluster_columns and col_colors_group1 is not None and col_dendro_colors is not None:
        col_colors_ax = g.ax_col_colors
        line_width = 2
        col_colors_ax.hlines(
            y=0.5,
            xmin=0, xmax=1,
            colors="white",
            linewidth=line_width,
            transform=col_colors_ax.transAxes,
        )

    # Adjust gene label size
    gene_label_size = max(4.5, 100 // len(gene_symbols))
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        fontsize=gene_label_size,
        rotation=0
    )

    # Covariate legend
    legend_handles_covariates1 = [
        plt.Line2D([0], [0], marker="s", color=covariate1_color_map[group], markersize=8, linestyle="", label=group)
        for group in covariate1_color_map
    ]
    legend1 = g.ax_col_dendrogram.legend(
        handles=legend_handles_covariates1,
        title=group_by[0].capitalize(),
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        bbox_transform=plt.gcf().transFigure,
    )

    # If group_by has >1 elements, we do the second legend (unchanged logic)...

    # Adjust the color bar position
    cbar_position = g.cax.get_position()
    new_cbar_position = [
        cbar_position.x0 + 0.01,
        cbar_position.y0,
        cbar_position.width * 0.5,
        cbar_position.height
    ]
    g.cax.set_position(new_cbar_position)

    # Label color bar
    if median_scale_expression:
        g.cax.set_ylabel("Median Norm Log Exp)", fontsize=9)
    else:
        g.cax.set_ylabel("Expression Values (log))", fontsize=9)
    g.cax.yaxis.set_label_position("left")

    # Covariate bar label
    g.ax_col_colors.annotate(
        f"{group_by[0].capitalize()}",
        xy=(1.01, 1),
        xycoords='axes fraction',
        rotation=0,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=7,
        color="black",
    )
    if cluster_columns and col_colors_group1 is not None and col_dendro_colors is not None:
        g.ax_col_colors.annotate(
            color_bar2_label.capitalize(),
            xy=(1.01, 0.4),
            xycoords='axes fraction',
            rotation=0,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=7,
            color="black",
        )

    # Save
    output_pdf = f"{root}/heatmap_{cell_type_str}_{group0_str}_{covariate_str}_{'cells' if show_individual_cells else 'groups'}{'_clustered' if cluster_rows or cluster_columns else ''}.pdf"
    output_png = output_pdf.replace(".pdf", ".png")
    g.savefig(output_pdf, bbox_inches="tight", dpi=150)
    g.savefig(output_png, bbox_inches="tight", dpi=150)
    plt.close()

    # Append both to plots
    plots.append([output_pdf, description])
    plots.append([output_png, description])

    return plots

def plot_heatmap_with_imshow(
    gene_symbols,
    cell_type,
    group_by="disease",
    figsize=(20, 16),
    scaling_factor=0.8,
    cluster_rows=True,
    cluster_columns=True,
    show_individual_cells=False,
    median_scale_expression=False,
    samples_to_visualize="cell-type",
    covariate=None,
    plots=[]
):
    """
    Generates a heatmap using matplotlib without seaborn to reduce file size (rasterized heatmap image).
    """

    description = f"lustered cell-type ({samples_to_visualize}) seaborn heatmap by {group_by[0]} with show_individual_cells {show_individual_cells} median scaling {median_scale_expression} for covariate {covariate}"
    if cluster_columns:
        description = "C" + description
    else:
        description = "Un-c" + description

    from matplotlib.colors import LinearSegmentedColormap
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    import seaborn as sns
    from scipy.cluster.hierarchy import fcluster

    # -- Minimal sanitizing function
    def sanitize(value):
        if value is None:
            return "None"
        return str(value).replace("'", "").replace('"', "").replace(" ", "_")

    # Sanitize filename components
    cell_type_str = sanitize(cell_type)
    group0_str = sanitize(group_by[0]) if isinstance(group_by, list) else sanitize(group_by)
    covariate_str = sanitize(covariate)

    # Subset logic
    if samples_to_visualize == "cell-type":
        subset_adata = adata[adata.obs[cell_type_index].isin([cell_type])]
    else:
        subset_adata = adata[adata.obs[covariate_index].isin([covariate])]

    if subset_adata.n_obs > 10000:
        downsample_indices = np.random.choice(subset_adata.obs_names, size=10000, replace=False)
        subset_adata = subset_adata[downsample_indices]

    try: 
        expression_data = subset_adata[:, gene_symbols].X.toarray()
    except Exception as e:
        if isinstance(gene_symbols, list):
            raise e
        else:
            gene_symbols = gene_symbols["Gene"].tolist()
            expression_data = subset_adata[:, gene_symbols].X.toarray()

    # Build data_to_plot
    if show_individual_cells:
        data_to_plot = expression_data.T
        x_labels = subset_adata.obs.index.to_numpy()
        grouped_obs = subset_adata.obs[group0_str].values if group0_str in subset_adata.obs else subset_adata.obs[group_by[0]].values
    else:
        grouped_obs = subset_adata.obs[group0_str].values if group0_str in subset_adata.obs else subset_adata.obs[group_by[0]].values
        unique_groups = np.unique(grouped_obs)
        median_expression = [
            np.median(expression_data[grouped_obs == group], axis=0)
            for group in unique_groups
        ]
        data_to_plot = np.array(median_expression).T
        x_labels = unique_groups

    x_labels_verbose = []
    if len(x_labels) <= 100:
        x_labels_verbose = x_labels

    if median_scale_expression:
        data_to_plot -= np.median(data_to_plot, axis=1, keepdims=True)

    abs_max = np.max(np.abs(data_to_plot)) * scaling_factor

    # Row clustering
    row_linkage = None
    if cluster_rows:
        row_linkage = linkage(data_to_plot, method="average")
        row_order = leaves_list(row_linkage)
        data_to_plot = data_to_plot[row_order, :]
        gene_symbols = [gene_symbols[i] for i in row_order]

    # Column clustering
    col_linkage = None
    grouped_obs_sorted = grouped_obs
    if cluster_columns:
        col_linkage = linkage(data_to_plot.T, method="average")
        col_order = leaves_list(col_linkage)
        data_to_plot = data_to_plot[:, col_order]
        x_labels = [x_labels[i] for i in col_order]
        grouped_obs_sorted = np.array([grouped_obs[i] for i in col_order])

    # Create custom colormap
    yellow_black_blue = LinearSegmentedColormap.from_list("yellow_black_blue", ["deepskyblue", "black", "yellow"])

    # Figure layout
    fig = plt.figure(figsize=figsize)
    dendro_width = 0.1
    heatmap_width = 0.625
    color_bar_width = 0.015
    heatmap_height = 0.57
    heatmap_left = 0.17

    # Row dendrogram
    if row_linkage is not None:
        ax_row_dendro = fig.add_axes([0.05, 0.25, dendro_width, heatmap_height])
        dendrogram(row_linkage, orientation="left", ax=ax_row_dendro, no_labels=True, link_color_func=lambda k: "black")
        ax_row_dendro.invert_yaxis()
        ax_row_dendro.axis("off")

    def build_color_map(values, coordinates, mode="column"):
        unique_values = np.unique(values)
        color_palette = sns.color_palette("tab10", len(unique_values))
        color_dict = {val: color_palette[i] for i, val in enumerate(unique_values)}
        color_array = [color_dict[val] for val in values]
        ax_colors = fig.add_axes(coordinates)
        if mode == "row":
            for i, color in enumerate(color_array):
                ax_colors.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
            ax_colors.set_xlim(0, 1)
            ax_colors.set_ylim(0, len(color_array))
        elif mode == "column":
            for i, color in enumerate(color_array):
                ax_colors.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
            ax_colors.set_xlim(0, len(color_array))
            ax_colors.set_ylim(0, 1)
        ax_colors.axis("off")
        return unique_values, color_dict

    # Column dendrogram color map
    if cluster_columns and col_linkage is not None:
        col_dendro = dendrogram(col_linkage, no_plot=True)
        col_dendro_colors = col_dendro["leaves_color_list"]
        build_color_map(col_dendro_colors, [heatmap_left, 0.823, heatmap_width, color_bar_width], mode="column")

    # Row clustering color bar
    if cluster_rows and row_linkage is not None:
        row_dendro = dendrogram(row_linkage, no_plot=True)
        row_dendro_colors = row_dendro["leaves_color_list"][::-1]
        build_color_map(row_dendro_colors, [0.153, 0.25, color_bar_width, heatmap_height], mode="row")

    # Column dendrogram
    if col_linkage is not None:
        ax_col_dendro = fig.add_axes([heatmap_left, 0.858, heatmap_width, dendro_width])
        dendrogram(col_linkage, orientation="top", ax=ax_col_dendro, no_labels=True, link_color_func=lambda k: "black")
        ax_col_dendro.axis("off")

    # Dataset grouping color bar
    unique_groups, group_color_map = build_color_map(grouped_obs_sorted, [heatmap_left, 0.84, heatmap_width, color_bar_width], mode="column")

    # Main heatmap
    ax_heatmap = fig.add_axes([heatmap_left, 0.25, heatmap_width, heatmap_height])
    cax = ax_heatmap.imshow(
        data_to_plot,
        cmap=yellow_black_blue,
        vmin=-abs_max,
        vmax=abs_max,
        interpolation="nearest",
        aspect="auto",
    )

    # Heatmap legend
    cbar_ax = fig.add_axes([0.1, 0.85, color_bar_width, 0.1])
    cbar = fig.colorbar(cax, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Expression (log scale)", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    # X-axis labels
    if len(x_labels_verbose) > 0:
        ax_heatmap.set_xticks(range(len(x_labels)))
        ax_heatmap.set_xticklabels(x_labels_verbose, rotation=90, fontsize=6)
    else:
        ax_heatmap.set_xticks([])
        ax_heatmap.set_xticklabels([])

    # Y-axis labels
    font_size = 9
    if len(gene_symbols) <= 20:
        font_size = 12
    elif len(gene_symbols) >= 50:
        font_size = 4
    ax_heatmap.yaxis.tick_right()
    ax_heatmap.set_yticks(range(len(gene_symbols)))
    ax_heatmap.set_yticklabels(gene_symbols, fontsize=font_size)

    # Add legend
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color=group_color_map[group], markersize=10, linestyle="", label=group)
        for group in unique_groups
    ]
    fig.legend(
        handles=legend_handles,
        title="Dataset Groups",
        loc="upper right",
        bbox_to_anchor=(1, 1),
        fontsize=10
    )

    # Save
    output_pdf = f"{root}/heatmap_{cell_type_str}_{group0_str}_{covariate_str}_{'cells' if show_individual_cells else 'groups'}{'_clustered' if cluster_rows or cluster_columns else ''}.pdf"
    output_png = output_pdf.replace(".pdf", ".png")
    plt.savefig(output_pdf, dpi=250, bbox_inches="tight")
    plt.savefig(output_png, dpi=250, bbox_inches="tight")
    plt.close()

    # Append both to plots
    plots.append([output_pdf, description])
    plots.append([output_png, description])

    return plots




def plot_dot_plot_celltype(gene_symbols, cell_type, group_by="disease", figsize=(12, 8), plots=[]):
    """
    Create a dot plot for specified genes and a specific cell type.

    Parameters:
    - gene_symbols: List of gene symbols to include in the dot plot.
    - cell_type: The specific cell type to subset for the plot.
    - group_by: Column in the AnnData object to group data by.
    - figsize: Tuple defining the size of the plot.
    - plots: List to store generated plot file paths and descriptions.
    """
    sc.settings.figdir = root

    description = f"Gene expression dot plot of user-supplied or differentially expressed genes for {cell_type} cells by {group_by}"
    output_file = f"expression_{cell_type}_{group_by}.pdf"  # Base file name

    # Subset the data for the specified cell type
    subset_adata = adata[adata.obs[cell_type_index].isin([cell_type])]
    
    # If gene_symbols is not a list, convert it
    if not isinstance(gene_symbols, list):
        gene_symbols = gene_symbols["Gene"].tolist()

    # Limit the number of genes to avoid cluttered plots
    if len(gene_symbols) > 40:
        gene_symbols = gene_symbols[:40]

    # Generate the dot plot (PDF)
    sc.pl.dotplot(
        subset_adata,
        var_names=gene_symbols,
        groupby=group_by,
        figsize=figsize,
        show=False,  # Do not display interactively
        save=output_file,  # Let Scanpy save the plot
    )

    # Generate the dot plot (PNG)
    sc.pl.dotplot(
        subset_adata,
        var_names=gene_symbols,
        groupby=group_by,
        figsize=figsize,
        show=False,  # Do not display interactively
        save=output_file.replace("pdf", "png"),  # Let Scanpy save the plot
    )

    # Append updated file paths to the plots list with 'dotplot_' prefix
    actual_pdf = f"{root}/dotplot_{output_file}"  # Expect Scanpy's prefix
    actual_png = f"{root}/dotplot_{output_file.replace('pdf','png')}"
    plots.append([actual_pdf, description])
    plots.append([actual_png, description])

    return plots

def plot_dot_plot_all_celltypes(gene_symbols, covariate="control", figsize=(12, 8), color_map="Blues", plots=[]):
    """
    Create a dot plot for specified genes across all cell types.

    Parameters:
    - gene_symbols: List of gene symbols to include in the dot plot.
    - covariate: The specific covariate to subset for the plot.
    - figsize: Tuple defining the size of the plot.
    - color_map: The color map to use for the plot.
    - plots: List to store generated plot file paths and descriptions.
    """
    sc.settings.figdir = root

    description = f"Gene expression dot plot of user-supplied or differentially expressed genes for {covariate}"
    output_file = f"expression_all-cell_types_{covariate}.pdf"  # Base file name

    # Subset the data for the specified covariate
    subset_adata = adata[adata.obs[covariate_index].isin([covariate])]
    
    # If gene_symbols is not a list, convert it
    if not isinstance(gene_symbols, list):
        gene_symbols = gene_symbols["Gene"].tolist()

    # Limit the number of genes to avoid cluttered plots
    if len(gene_symbols) > 40:
        gene_symbols = gene_symbols[:40]

    # Generate the dot plot (PDF)
    sc.pl.dotplot(
        subset_adata,
        var_names=gene_symbols,
        groupby=cell_type_index,
        figsize=figsize,
        color_map=color_map,
        show=False,  # Do not display interactively
        save=output_file,  # Let Scanpy save the plot
    )

    # Generate the dot plot (PNG)
    sc.pl.dotplot(
        subset_adata,
        var_names=gene_symbols,
        groupby=cell_type_index,
        figsize=figsize,
        color_map=color_map,
        show=False,  # Do not display interactively
        save=output_file.replace("pdf", "png"),  # Let Scanpy save the plot
    )

    # Append updated file paths to the plots list with 'dotplot_' prefix
    actual_pdf = f"{root}/dotplot_{output_file}"  # Expect Scanpy's prefix
    actual_png = f"{root}/dotplot_{output_file.replace('pdf','png')}"
    plots.append([actual_pdf, description])
    plots.append([actual_png, description])

    return plots

 
def compare_marker_genes_and_plot_venn(cell_types,plots=[]):
    from matplotlib_venn import venn2, venn3
    """
    Retrieve marker genes for specified cell types and plot a weighted Venn diagram
    showing overlap of marker genes. Saves the Venn diagram to a PDF.

    Parameters:
    - cell_types: List of 2 or 3 cell type names to compare.
    - output_pdf: Path to save the Venn diagram.

    Returns:
    - None
    """
    
    description = "Weighted Venn diagram of top cell-type specific marker genes for selected cell types."
    output_file = f"{root}/venn_ovelapping_markers.pdf"

    if "marker_stats" not in adata.uns:
        raise ValueError("Marker statistics are not available in the AnnData object. Run marker gene identification first.")

    # Retrieve marker statistics DataFrame
    marker_stats_df = adata.uns["marker_stats"]

    # Ensure valid cell types are provided
    available_cell_types = marker_stats_df["Cell Population"].unique()
    for cell_type in cell_types:
        if cell_type not in available_cell_types:
            raise ValueError(f"Cell type '{cell_type}' not found in marker statistics. Available types: {available_cell_types}")

    # Retrieve marker genes for each cell type
    marker_genes = {}
    for cell_type in cell_types:
        genes = marker_stats_df[marker_stats_df["Cell Population"] == cell_type]["Gene"].tolist()
        marker_genes[cell_type] = set(genes)

    # Create a Venn diagram based on the number of cell types
    plt.figure(figsize=(8, 6))
    if len(cell_types) == 2:
        venn = venn2([marker_genes[cell_types[0]], marker_genes[cell_types[1]]], set_labels=cell_types)
    elif len(cell_types) == 3:
        venn = venn3([marker_genes[cell_types[0]], marker_genes[cell_types[1]], marker_genes[cell_types[2]]], set_labels=cell_types)
    else:
        raise ValueError("Only 2 or 3 cell types can be compared for a Venn diagram.")

    # Export intersecting and unique genes
    export_data = {}

    if len(cell_types_to_compare) == 2:
        intersect = marker_genes[cell_types_to_compare[0]] & marker_genes[cell_types_to_compare[1]]
        unique_1 = marker_genes[cell_types_to_compare[0]] - marker_genes[cell_types_to_compare[1]]
        unique_2 = marker_genes[cell_types_to_compare[1]] - marker_genes[cell_types_to_compare[0]]

        export_data["Intersect"] = list(intersect)
        export_data[f"Unique to {cell_types_to_compare[0]}"] = list(unique_1)
        export_data[f"Unique to {cell_types_to_compare[1]}"] = list(unique_2)

    elif len(cell_types_to_compare) == 3:
        intersect_all = marker_genes[cell_types_to_compare[0]] & marker_genes[cell_types_to_compare[1]] & marker_genes[cell_types_to_compare[2]]
        unique_1 = marker_genes[cell_types_to_compare[0]] - marker_genes[cell_types_to_compare[1]] - marker_genes[cell_types_to_compare[2]]
        unique_2 = marker_genes[cell_types_to_compare[1]] - marker_genes[cell_types_to_compare[0]] - marker_genes[cell_types_to_compare[2]]
        unique_3 = marker_genes[cell_types_to_compare[2]] - marker_genes[cell_types_to_compare[0]] - marker_genes[cell_types_to_compare[1]]

        export_data["Intersect All"] = list(intersect_all)
        export_data[f"Unique to {cell_types_to_compare[0]}"] = list(unique_1)
        export_data[f"Unique to {cell_types_to_compare[1]}"] = list(unique_2)
        export_data[f"Unique to {cell_types_to_compare[2]}"] = list(unique_3)

    # Convert export_data to a DataFrame and save to a tab-delimited text file
    output_tsv = output_file.replace("pdf","tsv")
    export_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in export_data.items()]))
    export_df.to_csv(output_tsv, sep="\t", index=False)
    #print(f"Intersecting and unique genes exported to {output_tsv}")
    plots.append([output_tsv, description])

    # Save the Venn diagram to a PDF
    plt.title("Weighted Venn Diagram of Marker Genes")
    plt.savefig(output_file, bbox_inches="tight")
    plt.savefig(output_file.replace("pdf","png"), bbox_inches="tight")
    plt.close()
    plots.append([output_file, description])
    plots.append([output_file.replace("pdf","png"), description])
    return plots

def compare_marker_genes_and_plot_upset(cell_types,plots=[]):
    """
    Retrieve marker genes for specified cell types and plot an UpSet diagram
    showing overlap of marker genes. Saves the UpSet plot to a PDF and exports
    overlapping marker gene data to a text file.

    Parameters:
    - cell_types: List of cell type names to compare.
    - output_pdf: Path to save the UpSet diagram (default: "upset_cell-type_markers.pdf").
    - output_txt: Path to save the overlapping marker gene data (default: "upset_cell-type_markers.txt").

    Returns:
    - None
    """
    output_pdf=f"{root}/upset_cell-type_markers.pdf"
    output_tsv=f"{root}/upset_cell-type_markers.tsv"
    description = "Upset plot of marker genes from different cell types."

    from upsetplot import UpSet, from_contents
    
    if "marker_stats" not in adata.uns:
        raise ValueError("Marker statistics are not available in the AnnData object. Run marker gene identification first.")

    # Retrieve marker statistics DataFrame
    marker_stats_df = adata.uns["marker_stats"]

    # Ensure valid cell types are provided
    available_cell_types = marker_stats_df["Cell Population"].unique()
    for cell_type in cell_types:
        if cell_type not in available_cell_types:
            raise ValueError(f"Cell type '{cell_type}' not found in marker statistics. Available types: {available_cell_types}")

    # Retrieve marker genes for each cell type
    marker_genes = {}
    for cell_type in cell_types:
        genes = marker_stats_df[marker_stats_df["Cell Population"] == cell_type]["Gene"].tolist()
        marker_genes[cell_type] = set(genes)

    # Create UpSet plot data from marker genes
    upset_data = from_contents(marker_genes)

    # Plot the UpSet diagram
    plt.figure(figsize=(10, 8))
    upset_plot = UpSet(upset_data, show_counts=True, sort_categories_by=None)
    upset_plot.plot()
    plt.title("UpSet Diagram of Marker Genes")
    plt.savefig(output_pdf, bbox_inches="tight", dpi=150)
    plt.savefig(output_pdf.replace("pdf","png"), bbox_inches="tight", dpi=150)
    plt.close()
    #print(f"UpSet plot saved to {output_pdf}")

    # Export intersecting and unique genes
    export_data = {}
    for cell_type, genes in marker_genes.items():
        export_data[f"Unique to {cell_type}"] = list(genes)
    
    intersect_all = set.intersection(*marker_genes.values())
    export_data["Intersect All"] = list(intersect_all)

    # Convert export_data to a DataFrame and save to a tab-delimited text file
    export_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in export_data.items()]))
    export_df.to_csv(output_tsv, sep="\t", index=False)
    #print(f"Intersecting and unique genes exported to {output_tsv}")
    plots.append([output_tsv, description])
    plots.append([output_pdf,description])
    plots.append([output_pdf.replace("pdf","png"), description])
    return plots

def plot_aggregated_cell_frequencies_radar(
    covariate_index,
    donor_index,
    cell_type_index,
    output_pdf="radar_cell_frequency.pdf",
    plots=[]
):
    """
    Calculate average cell frequencies per donor and covariate, then visualize using a radar plot.
    Parameters:
    - covariate_index: Column name in `adata.obs` representing the grouping variable (e.g., disease).
    - donor_index: Column name in `adata.obs` representing the donor identifier.
    - cell_type_index: Column name in `adata.obs` representing the cell type.
    - output_pdf: Path to save the radar plot (default: "radar_cell_frequency.pdf").
    Returns:
    - plots: List containing the output file paths and descriptions.
    """
    
    description = 'Radar plot of cell frequencies for different user-supplied conditions'
    
    # Data preparation code remains the same
    cell_type_counts_per_donor = (
        adata.obs.groupby([cell_type_index, covariate_index, donor_index])
        .size()
        .reset_index(name="count")
    )
    
    total_cells_per_donor_condition = (
        adata.obs.groupby([covariate_index, donor_index])
        .size()
        .reset_index(name="total")
    )
    
    merged_df = pd.merge(
        cell_type_counts_per_donor,
        total_cells_per_donor_condition,
        on=[covariate_index, donor_index],
    )
    
    merged_df["Frequency (%)"] = (merged_df["count"] / merged_df["total"]) * 100
    
    aggregated_df = (
        merged_df.groupby([cell_type_index, covariate_index])
        ["Frequency (%)"]
        .mean()
        .reset_index()
    )
    
    radar_data = aggregated_df.pivot(
        index=covariate_index, columns=cell_type_index, values="Frequency (%)"
    ).fillna(0)
    
    categories = radar_data.columns.tolist()
    covariates = radar_data.index.tolist()
    radar_data_matrix = radar_data.values
    
    # Create figure with adjusted size
    plt.figure(figsize=(20, 16))
    
    # Create subplot with extra spacing for labels
    ax = plt.subplot(111, polar=True)
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # Plot data
    for idx, covariate in enumerate(covariates):
        values = radar_data_matrix[idx].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=covariate)
        ax.fill(angles, values, alpha=0.15)
    
    # Set up the axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # Remove default labels
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels outside the plot
    for idx, (label, angle) in enumerate(zip(categories, angles[:-1])):
        angle_rad = angle
        
        # Adjust text alignment based on angle
        if angle_rad > np.pi/2 and angle_rad <= 3*np.pi/2:
            ha = 'right'
            angle_deg = angle_rad*180/np.pi + 180
        else:
            ha = 'left'
            angle_deg = angle_rad*180/np.pi
        
        # Position labels further out from the plot
        label_position = 15
        
        ax.text(angle_rad, label_position, label,
                ha=ha,
                va='center',
                rotation=angle_deg,
                rotation_mode='anchor',
                fontsize=10)
    
    # Create legend with larger font size and box
    legend = ax.legend(loc='center left',
                       bbox_to_anchor=(1.35, 0.5),
                       fontsize=12,
                       frameon=True,
                       framealpha=1,
                       edgecolor='black')
    
    # Make the legend box larger
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_facecolor('white')
    
    # Add title above the legend
    plt.text(1.6, 1.1, "Average Cell Frequencies per Covariate",
             transform=ax.transAxes,
             fontsize=16,
             horizontalalignment='center',
             verticalalignment='center')
    
    # Add radial labels
    ax.set_rlabel_position(0)
    radial_labels = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    ax.set_rticks(radial_labels)
    ax.set_yticklabels([str(x) for x in radial_labels], fontsize=10)
    
    # Adjust plot limits to accommodate labels
    ax.set_rlim(0, 45)  # Increase the outer limit to make room for labels
    
    # Save plots with improved resolution and padding
    plt.savefig(output_pdf, bbox_inches="tight", dpi=300, pad_inches=0.5)
    plt.savefig(output_pdf.replace("pdf", "png"), bbox_inches="tight", dpi=300, pad_inches=0.5)
    plt.close()
    plots.append([output_pdf, description])
    plots.append([output_pdf.replace("pdf","png"), description])
    return plots


def visualize_gene_networkX(
    gene_symbols,
    edge_types={"transcriptional_target": "red", "Arrow": "grey", "Tbar": "blue"},
    exclude_biogrid=True,
    figsize=(12, 12),  # Reduced figure size
    output_file="networkX_interactions.pdf",
    plots=[]
):
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches  # For legend patches

    description = "networkX of user-indicated genes"

    # Check for interaction data
    if "interaction_data" not in adata.uns or adata.uns["interaction_data"].empty:
        raise ValueError("No interaction data found in adata.uns['interaction_data'].")

    interaction_data = adata.uns["interaction_data"]

    # Process gene_symbols input
    if isinstance(gene_symbols, list):
        if gene_symbols and isinstance(gene_symbols[0], dict) and "Gene" in gene_symbols[0] and "Log Fold Change" in gene_symbols[0]:
            gene_symbols = pd.DataFrame(gene_symbols)
        else:
            gene_symbols = pd.DataFrame({"Gene": gene_symbols, "Log Fold Change": [0] * len(gene_symbols)})

    gene_symbols["Log Fold Change"] = gene_symbols["Log Fold Change"].fillna(0)
    gene_names_debug = gene_symbols["Gene"].tolist()
    fold_change_map = gene_symbols.set_index("Gene")["Log Fold Change"].to_dict()

    filtered_interactions = interaction_data[
        (interaction_data["Symbol1"].isin(gene_names_debug)) &
        (interaction_data["Symbol2"].isin(gene_names_debug))
    ]

    filtered_interactions_no_loops = filtered_interactions[
        filtered_interactions["Symbol1"] != filtered_interactions["Symbol2"]
    ]
    if not filtered_interactions_no_loops.empty:
        filtered_interactions = filtered_interactions_no_loops

    if exclude_biogrid:
        filtered_interactions = filtered_interactions[
            ~filtered_interactions["Source"].str.contains("BioGRID", na=False)
        ]

    if filtered_interactions.empty:
        raise ValueError("No interactions found for the provided gene symbols.")

    G = nx.DiGraph()
    for gene in gene_names_debug:
        fold_change = fold_change_map.get(gene, 0)
        color = (
            "lightcoral" if fold_change > 0 else
            "deepskyblue" if fold_change < 0 else
            "gray"
        )
        G.add_node(gene, color=color)

    for _, row in filtered_interactions.iterrows():
        source = row["Symbol1"]
        target = row["Symbol2"]
        edge_type = row["InteractionType"]
        edge_color = edge_types.get(edge_type, "gray")
        G.add_edge(source, target, color=edge_color, arrowstyle="-|>", type=edge_type)

    node_colors = [G.nodes[node].get("color", "gray") for node in G.nodes]
    pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(
        G, pos,
        edge_color=[G.edges[edge]["color"] for edge in G.edges],
        arrows=True,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

    # Create legend for node colors
    red_patch = mpatches.Patch(color="lightcoral", label="Up-regulated")
    blue_patch = mpatches.Patch(color="deepskyblue", label="Down-regulated")
    gray_patch = mpatches.Patch(color="gray", label="Neutral")
    plt.legend(handles=[red_patch, blue_patch, gray_patch], loc='upper left', bbox_to_anchor=(1, 1))

    plt.title("Gene Interaction Network")
    plt.tight_layout()  # Adjust layout

    # Save PDF
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    # Save PNG
    png_output = output_file.replace("pdf", "png")
    plt.savefig(png_output, format="png", bbox_inches="tight", dpi=150)
    plt.close()

    plots.append([output_file, description])
    plots.append([png_output, description])
    return plots


def visualize_gene_network_igraph(
    gene_symbols,
    edge_types={"transcriptional_target": "red", "Arrow": "grey", "Tbar": "blue"},
    exclude_biogrid=True,
    output_file="iGraph_gene_network.pdf",
    plots=[]
):
    from igraph import Graph, plot
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    description = "iGraph network of user-indicated genes"

    # Check for interaction data
    if "interaction_data" not in adata.uns or adata.uns["interaction_data"].empty:
        raise ValueError("No interaction data found in adata.uns['interaction_data'].")

    interaction_data = adata.uns["interaction_data"]

    # Process gene_symbols input
    if isinstance(gene_symbols, list):
        if gene_symbols and isinstance(gene_symbols[0], dict) and "Gene" in gene_symbols[0] and "Log Fold Change" in gene_symbols[0]:
            gene_symbols = pd.DataFrame(gene_symbols)
        else:
            gene_symbols = pd.DataFrame({"Gene": gene_symbols, "Log Fold Change": [0] * len(gene_symbols)})

    gene_symbols["Log Fold Change"] = gene_symbols["Log Fold Change"].fillna(0)

    # Filter interactions for the given gene symbols
    filtered_interactions = interaction_data[
        (interaction_data["Symbol1"].isin(gene_symbols["Gene"])) &
        (interaction_data["Symbol2"].isin(gene_symbols["Gene"]))
    ]

    filtered_interactions_no_loops = filtered_interactions[
        filtered_interactions["Symbol1"] != filtered_interactions["Symbol2"]
    ]
    if not filtered_interactions_no_loops.empty:
        filtered_interactions = filtered_interactions_no_loops

    if exclude_biogrid:
        filtered_interactions = filtered_interactions[
            ~filtered_interactions["Source"].str.contains("BioGRID", na=False)
        ]

    if filtered_interactions.empty:
        raise ValueError("No interactions found for the provided gene symbols.")

    # Extract nodes and edges
    nodes = list(set(filtered_interactions["Symbol1"]).union(set(filtered_interactions["Symbol2"])))
    edges = [
        (row["Symbol1"], row["Symbol2"], edge_types.get(row["InteractionType"], "gray"))
        for _, row in filtered_interactions.iterrows()
    ]

    # Create the iGraph object
    g = Graph(directed=True)
    g.add_vertices(nodes)
    g.vs["name"] = nodes

    g.add_edges([(edge[0], edge[1]) for edge in edges])
    g.es["color"] = [edge[2] for edge in edges]

    # Map colors to nodes based on fold change
    fold_change_map = gene_symbols.set_index("Gene")["Log Fold Change"].to_dict()
    g.vs["color"] = [
        "lightcoral" if fold_change_map.get(node, 0) > 0 else
        "deepskyblue" if fold_change_map.get(node, 0) < 0 else
        "gray"
        for node in g.vs["name"]
    ]

    # Ensure vertex labels are shown
    g.vs["label"] = g.vs["name"]

    # Define layout for the graph
    layout = g.layout("fr")  # Fruchterman-Reingold layout

    # Generate PDF output
    plot(
        g,
        target=output_file,
        layout=layout,
        vertex_size=20,
        margin=50,
        vertex_label_size=11,
        edge_arrow_size=0.5,
    )

    # Generate PNG output
    png_file = output_file.replace("pdf", "png")
    plot(
        g,
        target=png_file,
        layout=layout,
        vertex_size=20,
        margin=50,
        vertex_label_size=11,
        edge_arrow_size=0.5,
    )

    # Overlay legend on PNG using matplotlib
    img = plt.imread(png_file)
    plt.figure(figsize=(12, 12))  # Adjusted figure size for PNG
    plt.imshow(img)
    plt.axis("off")
    red_patch = mpatches.Patch(color="lightcoral", label="Up-regulated")
    blue_patch = mpatches.Patch(color="deepskyblue", label="Down-regulated")
    gray_patch = mpatches.Patch(color="gray", label="Neutral")
    plt.legend(
        handles=[red_patch, blue_patch, gray_patch],
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )

    # Save the overlaid PNG with adjusted DPI
    plt.savefig(png_file, bbox_inches="tight", dpi=300)  # Adjusted DPI
    plt.close()

    plots.append([output_file, description])
    plots.append([png_file, description])
    return plots

if __name__ == '__main__':

    json_input = 'BPD_infant_Sun_normalized_log_deg.json'
    main(json_input, output_dir="results")
