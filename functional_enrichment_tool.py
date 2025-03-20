#functional_enrichment_tool.py
import os
import requests
import json
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import uuid
from typing import List, Literal, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Base URL for file paths
BASE_URL = "https://devapp.lungmap.net"
PLOT_OUTPUT_DIR = os.getenv('PLOT_OUTPUT_DIR')

# Print warning if PLOT_OUTPUT_DIR is not set
if not PLOT_OUTPUT_DIR:
    print("Warning: PLOT_OUTPUT_DIR environment variable is not set. Using current directory for plots.")

def sanitize_filename(filename, max_length=200):
    """
    Remove invalid characters from the filename, replace spaces with underscores,
    append a short UUID to ensure uniqueness, and truncate the filename if it is too long.
    """
    # Replace problematic characters
    sanitized = filename.replace("'", "").replace(" ", "_").replace(",", "_").replace("+", "_")
    
    # Split the filename into name and extension
    name, ext = os.path.splitext(sanitized)
    
    # Append a short UUID for uniqueness
    unique_id = str(uuid.uuid4())[:5]
    new_name = f"{name}_{unique_id}"
    
    # Truncate if too long
    if len(new_name) + len(ext) > max_length:
        new_name = new_name[:max_length - len(ext) - 1]
    
    return new_name + ext

# -----------------------------
# LLM Workflow for Gene Enrichment Analysis
# -----------------------------

# Categories for functional enrichment
ENRICHMENT_CATEGORIES = [
    "GeneOntologyMolecularFunction", 
    "GeneOntologyBiologicalProcess", 
    "GeneOntologyCellularComponent",
    "HumanPheno", 
    "MousePheno", 
    "Domain", 
    "Pathway", 
    "Pubmed", 
    "Interaction", 
    "Cytoband",
    "TFBS", 
    "GeneFamily", 
    "Coexpression", 
    "CoexpressionAtlas", 
    "ToppCell", 
    "Computational",
    "MicroRNA", 
    "Drug", 
    "Disease",
    "all"
]

class EnrichmentAnalysisModel(BaseModel):
    """
    Pydantic model for parsing LLM output for gene enrichment analysis.
    """
    gene_symbols: List[str]
    categories: List[Literal[
        "GeneOntologyMolecularFunction", 
        "GeneOntologyBiologicalProcess", 
        "GeneOntologyCellularComponent",
        "HumanPheno", 
        "MousePheno", 
        "Domain", 
        "Pathway", 
        "Pubmed", 
        "Interaction", 
        "Cytoband",
        "TFBS", 
        "GeneFamily", 
        "Coexpression", 
        "CoexpressionAtlas", 
        "ToppCell", 
        "Computational",
        "MicroRNA", 
        "Drug", 
        "Disease",
        "all"
    ]]

# Initialize the parser
enrichment_parser = PydanticOutputParser(pydantic_object=EnrichmentAnalysisModel)

# Define the prompt template
ENRICHMENT_PROMPT_TEMPLATE = """\
You are an assistant specialized in analyzing gene lists for functional enrichment analysis.

Based on the user's query, extract the following:

1. **Gene Symbols**: 
   - Extract a list of gene symbols mentioned in the query.
   - Only include valid gene symbols (e.g., BRCA1, TP53, etc.).
   - Do not include other terms that might be confused with gene symbols.

2. **Categories for Functional Enrichment**:
   - Determine which categories the user is interested in for the enrichment analysis.
   - If no specific categories are mentioned, use ["all"] as the default.
   - Available categories are:
     - GeneOntologyMolecularFunction: Molecular functions of genes (binding activities, enzyme activities, etc.)
     - GeneOntologyBiologicalProcess: Biological processes involving genes (signaling pathways, metabolic processes, etc.)
     - GeneOntologyCellularComponent: Cellular locations of gene products (nucleus, membrane, organelles, etc.)
     - HumanPheno: Human phenotypes and disease traits associated with genes from HPO database
     - MousePheno: Mouse phenotypes and traits from MGI database associated with mouse homologs
     - Domain: Protein domains and structural elements from InterPro, PFAM, and other databases
     - Pathway: Biochemical and signaling pathways from KEGG, BioCarta, Reactome, and other pathway databases
     - Pubmed: Scientific literature associations and co-occurrence in publications
     - Interaction: Protein-protein interactions and protein interaction partners from BioGRID, STRING, etc.
     - Cytoband: Chromosomal locations and cytogenetic bands of genes
     - TFBS: Transcription factor binding sites and regulatory elements that control gene expression
     - GeneFamily: Gene families and homology-based groupings of related genes
     - Coexpression: Gene coexpression patterns showing which genes typically express together
     - CoexpressionAtlas: Tissue-specific coexpression data showing context-dependent correlations
     - ToppCell: Cell-type markers extracted from diverse sources, useful for finding cell type associations
     - Computational: Computational predictions and in silico analyses of gene function
     - MicroRNA: MicroRNA associations and regulatory relationships with target genes
     - Drug: Drug interactions, drug targets, and pharmacogenomic associations
     - Disease: Disease associations from OMIM, DisGeNET, and other disease databases
     - all: All available categories

User Query:
{user_query}

Your output should be a single JSON object adhering to this schema:
{format_instructions}
"""

# Create the prompt
enrichment_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ENRICHMENT_PROMPT_TEMPLATE),
        ("human", "{user_query}")
    ]
).partial(format_instructions=enrichment_parser.get_format_instructions())

# Initialize the LLM
enrichment_llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

async def run_enrichment_workflow(user_query: str) -> EnrichmentAnalysisModel:
    """
    Run the gene enrichment workflow to extract gene symbols and categories from user query.
    
    Parameters:
    - user_query: The user's query string
    
    Returns:
    - EnrichmentAnalysisModel containing gene_symbols and categories
    """
    chain = enrichment_prompt | enrichment_llm | enrichment_parser
    
    # Invoke the chain asynchronously
    result = await chain.ainvoke({"user_query": user_query})
    
    return result

class FunctionalEnricher:
    """
    A tool for performing gene functional enrichment analysis.
    """
    
    def __init__(self):
        pass
        
    def _perform_enrichment_analysis(self, gene_symbols, selected_categories):
        """
        Performs Gene Functional Enrichment Analysis (GFEA) using the ToppGene API.
        Internal implementation method.
        """
        # --- 1) Convert Gene Symbols to Entrez IDs using ToppGene Lookup API ---
        def get_entrez_ids_toppgene(gene_symbols):
            url = "https://toppgene.cchmc.org/API/lookup"
            payload = {"Symbols": gene_symbols}
            try:
                response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
                if response.status_code == 200:
                    result = response.json()
                    return {gene["Submitted"]: {"Entrez": gene["Entrez"], "OfficialSymbol": gene["OfficialSymbol"]}
                            for gene in result["Genes"]}
                return {}
            except requests.exceptions.RequestException:
                return {}

        # --- 2) Run GFEA Based on Selected Categories ---
        def gene_enrichment(entrez_ids, selected_categories):
            url = "https://toppgene.cchmc.org/API/enrich"
            all_categories = [
                "GeneOntologyMolecularFunction", "GeneOntologyBiologicalProcess", "GeneOntologyCellularComponent",
                "HumanPheno", "MousePheno", "Domain", "Pathway", "Pubmed", "Interaction", "Cytoband",
                "TFBS", "GeneFamily", "Coexpression", "CoexpressionAtlas", "ToppCell", "Computational",
                "MicroRNA", "Drug", "Disease"
            ]
            categories_to_use = all_categories if selected_categories == ["all"] else selected_categories
            payload = {
                "Genes": [data["Entrez"] for data in entrez_ids.values()],
                "Categories": [{"Type": category} for category in categories_to_use]
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json().get("Annotations", [])
            else:
                return []

        # --- 3) Save Raw Results to TSV (No Filtering) ---
        def save_raw_results_tsv(enrichment_results, filename):
            if not enrichment_results:
                return
            keys = [
                "Category", "ID", "Name", "PValue", "QValueFDRBH", "QValueFDRBY", "QValueBonferroni", 
                "TotalGenes", "GenesInTerm", "GenesInQuery", "GenesInTermInQuery", "Source", "URL", "Genes"
            ]
            with open(filename, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=keys, delimiter="\t")
                writer.writeheader()
                for res in enrichment_results:
                    res["Genes"] = ", ".join([gene["Symbol"] for gene in res.get("Genes", [])])
                    writer.writerow(res)

        # --- 4) Filter by QValueFDRBH < 0.05, Organize by Category ---
        def filter_by_category(enrichment_results):
            categories = {}
            for res in enrichment_results:
                if res.get("QValueFDRBH", 1) < 0.05:
                    cat = res["Category"]
                    categories.setdefault(cat, []).append(res)
            return categories

        # --- 4a) Save Filtered Results to TSV (Only Top Results per Category) ---
        def save_filtered_results_tsv(categories, filename, top_count):
            filtered_data = []
            for cat, cat_results in categories.items():
                sorted_results = sorted(cat_results, key=lambda x: x["PValue"])[:top_count]
                filtered_data.extend(sorted_results)
            if not filtered_data:
                return
            keys = [
                "Category", "ID", "Name", "PValue", "QValueFDRBH", "QValueFDRBY", "QValueBonferroni",
                "TotalGenes", "GenesInTerm", "GenesInQuery", "GenesInTermInQuery", "Source", "URL", "Genes"
            ]
            with open(filename, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=keys, delimiter="\t")
                writer.writeheader()
                for res in filtered_data:
                    genes = res.get("Genes", [])
                    if isinstance(genes, list):
                        if genes and isinstance(genes[0], dict):
                            genes_str = ", ".join([gene.get("Symbol", "") for gene in genes])
                        else:
                            genes_str = ", ".join(genes) if all(isinstance(g, str) for g in genes) else str(genes)
                    else:
                        genes_str = str(genes)
                    res["Genes"] = genes_str
                    writer.writerow(res)

        # --- 5) Plot & Save Filtered Results as PNG & PDF with Clean Formatting ---
        def plot_and_save_results(categories, top_count, filename_base):
            """
            Plots the filtered enrichment results (up to top_count per category) as horizontal bar charts,
            placing exactly ONE category per row (i.e., 1 column), with bold text and high-resolution output.
            """

            if not categories:
                return "", ""

            num_categories = len(categories)
            cols = 1
            rows = num_categories

            all_pvals = [r["PValue"] for cat_results in categories.values() for r in cat_results]
            global_min_p = min(all_pvals) * 0.1 if all_pvals else 1e-20
            global_max_p = max(all_pvals) * 10

            fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 4))
            axes = np.atleast_1d(axes)

            pdf_filename = f"{filename_base}.pdf"
            png_filename = f"{filename_base}.png"

            # Ensure parent directories exist
            os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)
            
            with PdfPages(pdf_filename) as pdf:
                for i, (cat, results) in enumerate(categories.items()):
                    ax = axes[i]
                    sorted_results = sorted(results, key=lambda x: x["PValue"])[:top_count]
                    terms = [r["Name"] for r in sorted_results]
                    pvals = [r["PValue"] for r in sorted_results]

                    ax.barh(terms[::-1], pvals[::-1], color='skyblue', edgecolor='black')
                    ax.set_title(cat, fontsize=12, fontweight='bold')
                    ax.set_xlabel("P-Value (log scale)", fontsize=10, fontweight='bold')
                    ax.set_xscale("log")
                    ax.invert_yaxis()

                    # Make term names (y-axis labels) bold
                    for label in ax.get_yticklabels():
                        label.set_fontweight("bold")

                    ax.tick_params(axis="x", labelsize=9)
                    ax.tick_params(axis="y", labelsize=9)

                    ax.set_xlim(left=global_min_p, right=global_max_p)

                for j in range(num_categories, len(axes)):
                    fig.delaxes(axes[j])

                plt.subplots_adjust(left=0.0, right=0.95, top=0.95, bottom=0.05, wspace=0.0, hspace=0.5)

                pdf.savefig(fig, dpi=300, bbox_inches="tight")
                plt.savefig(png_filename, dpi=300, bbox_inches="tight")
                plt.close(fig)

            return os.path.abspath(png_filename), os.path.abspath(pdf_filename)

        # --- Main Workflow ---
        output = {}
        entrez_ids = get_entrez_ids_toppgene(gene_symbols)
        if not entrez_ids:
            return {"png_path": "", "pdf_path": "", "raw_tsv": "", "filtered_tsv": "",
                    "filter_by": {"selected_categories": selected_categories, "applied_QValueFDRBH_threshold": 0.05, "top_results_count": None}}
        enrichment_results = gene_enrichment(entrez_ids, selected_categories)

        # Generate output paths in PLOT_OUTPUT_DIR
        output_dir = PLOT_OUTPUT_DIR if PLOT_OUTPUT_DIR else os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        # Base filename using sanitize_filename function
        genes_str = '_'.join(gene_symbols)
        categories_str = '_'.join(selected_categories)
        filename_base = sanitize_filename(f"gfea_{genes_str}_{categories_str}")
        raw_tsv = os.path.join(output_dir, f"{filename_base}_raw.tsv")

        # Determine top results count: 10 if only one category is selected (and not "all"), otherwise 5.
        top_results_count = 10 if (len(selected_categories) == 1 and selected_categories[0] != "all") else 5

        # Filtered filenames include the top_results_count
        filtered_filename_base = sanitize_filename(f"{filename_base}_top{top_results_count}")
        filtered_filename_base_path = os.path.join(output_dir, filtered_filename_base)
        filtered_tsv = os.path.join(output_dir, f"{filtered_filename_base}_filtered.tsv")

        save_raw_results_tsv(enrichment_results, raw_tsv)
        categories_dict = filter_by_category(enrichment_results)
        save_filtered_results_tsv(categories_dict, filtered_tsv, top_results_count)
        png_path, pdf_path = plot_and_save_results(categories_dict, top_results_count, filtered_filename_base_path)

        # For filter_by info, if selected_categories == ["all"], list all possible categories; else list the selected ones.
        all_possible = [
            "GeneOntologyMolecularFunction", "GeneOntologyBiologicalProcess", "GeneOntologyCellularComponent",
            "HumanPheno", "MousePheno", "Domain", "Pathway", "Pubmed", "Interaction", "Cytoband",
            "TFBS", "GeneFamily", "Coexpression", "CoexpressionAtlas", "ToppCell", "Computational",
            "MicroRNA", "Drug", "Disease"
        ]
        selected_filter = all_possible if selected_categories == ["all"] else selected_categories

        # Add BASE_URL to paths for web access
        if PLOT_OUTPUT_DIR:
            # Ensure proper path joining by handling slashes correctly
            # Remove trailing slash from BASE_URL if exists
            base_url = BASE_URL.rstrip('/')
            # Ensure PLOT_OUTPUT_DIR starts with a slash
            plot_dir = PLOT_OUTPUT_DIR if PLOT_OUTPUT_DIR.startswith('/') else f"/{PLOT_OUTPUT_DIR}"
            
            # Construct clean web URLs that include the full PLOT_OUTPUT_DIR path
            output["png_path"] = f"{base_url}{plot_dir}/{os.path.basename(png_path)}" if png_path else ""
            output["pdf_path"] = f"{base_url}{plot_dir}/{os.path.basename(pdf_path)}" if pdf_path else ""
            output["raw_tsv"] = f"{base_url}{plot_dir}/{os.path.basename(raw_tsv)}" if raw_tsv else ""
            output["filtered_tsv"] = f"{base_url}{plot_dir}/{os.path.basename(filtered_tsv)}" if filtered_tsv else ""
        else:
            # If PLOT_OUTPUT_DIR is not set, fall back to the previous behavior
            output["png_path"] = f"{BASE_URL}/{os.path.basename(png_path)}" if png_path else ""
            output["pdf_path"] = f"{BASE_URL}/{os.path.basename(pdf_path)}" if pdf_path else ""
            output["raw_tsv"] = f"{BASE_URL}/{os.path.basename(raw_tsv)}" if raw_tsv else ""
            output["filtered_tsv"] = f"{BASE_URL}/{os.path.basename(filtered_tsv)}" if filtered_tsv else ""
        
        # Add filter_by information
        output["filter_by"] = {
            "selected_categories": selected_filter,
            "applied_QValueFDRBH_threshold": 0.05,
            "top_results_count": top_results_count
        }
        
        # Additionally include gene symbols and categories for reference
        output["gene_symbols"] = gene_symbols
        output["categories"] = selected_categories
        
        return output
                
    async def functional_enrichment(self, user_query: str) -> dict:
        """
        Process user query to extract gene symbols and categories, then perform
        gene functional enrichment analysis using ToppGene.
        
        Parameters:
        - user_query: The user's query string
        
        Returns:
        - Dictionary with enrichment results (paths to files and analysis details)
        """
        # Process the query with LLM to extract genes and categories
        try:
            analysis_request = await run_enrichment_workflow(user_query)
            
            # If no gene symbols were found, return an error
            if not analysis_request.gene_symbols:
                return {
                    "error": "No valid gene symbols were identified in your query. Please provide a list of gene symbols.",
                    "gene_symbols": [],
                    "categories": []
                }
            
            # Perform the functional enrichment analysis
            result = self._perform_enrichment_analysis(
                gene_symbols=analysis_request.gene_symbols,
                selected_categories=analysis_request.categories
            )
            
            return result
        
        except Exception as e:
            return {
                "error": f"An error occurred during gene functional enrichment analysis: {str(e)}",
                "gene_symbols": [],
                "categories": []
            }

# --- Define a proper async function for the tool ---
async def Functional_Enricher(user_query: str) -> dict:
    """
    Performs gene functional enrichment analysis (GFEA) based on user queries.
    Extracts gene symbols and enrichment categories from natural language queries.
    
    Parameters:
    - user_query: The user's natural language query containing gene names and optional category specifications
    
    Returns:
    - Dictionary with enrichment results including visualization paths and analysis details
    """
    # Create a new instance of the class each time the function is called
    # Note the different names to avoid recursion
    enricher_instance = FunctionalEnricher()
    result = await enricher_instance.functional_enrichment(user_query)
    return result
