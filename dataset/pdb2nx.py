"""
Functions for working with Protein Structure Graphs.
"""
import logging
from typing import Callable, List, Literal, Optional

import networkx as nx
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from biopandas.mmcif import PandasMmcif
from rich.progress import Progress

from graphein.protein.config import GetContactsConfig, ProteinGraphConfig
from graphein.protein.edges.distance import add_distance_to_edges, compute_distmat
from graphein.protein.utils import (
    compute_rgroup_dataframe,
    filter_dataframe,
    get_protein_name_from_filename,
    three_to_one_with_mods,
)
from graphein.rna.constants import RNA_ATOMS
from graphein.utils.utils import annotate_edge_metadata, annotate_graph_metadata, annotate_node_metadata

from .utils_pdb2nx import biopandas_mmcif2pdb

log = logging.getLogger(__name__)


def subset_structure_to_rna(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a subset of atomic dataframe that contains only certain atom names relevant for RNA structures.

    :param df: Protein Structure dataframe to subset
    :type df: pd.DataFrame
    :returns: Subset protein structure dataframe
    :rtype: pd.DataFrame
    """
    return filter_dataframe(
        df, by_column="atom_name", list_of_values=RNA_ATOMS, boolean=True
    )


def read_pdb_to_dataframe(
    pdb_path: Optional[str] = None,
    pdb_code: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    model_index: int = 1,
) -> pd.DataFrame:
    """
    Reads PDB file to ``PandasPDB`` object.

    Returns ``atomic_df``, which is a dataframe enumerating all atoms and their cartesian coordinates in 3D space. Also
    contains associated metadata from the PDB file.

    :param pdb_path: path to PDB file. Defaults to ``None``.
    :type pdb_path: str, optional
    :param pdb_code: 4-character PDB accession. Defaults to ``None``.
    :type pdb_code: str, optional
    :param uniprot_id: UniProt ID to build graph from AlphaFoldDB. Defaults to ``None``.
    :type uniprot_id: str, optional
    :param model_index: Index of model to read. Only relevant for structures containing ensembles. Defaults to ``1``.
    :type model_index: int, optional
    :returns: ``pd.DataFrame`` containing protein structure
    :rtype: pd.DataFrame
    """
    if pdb_code is None and pdb_path is None and uniprot_id is None:
        raise NameError(
            "One of pdb_code, pdb_path or uniprot_id must be specified!"
        )

    if pdb_path is not None:
        if pdb_path.endswith('cif'):
            atomic_df = PandasMmcif().read_mmcif(pdb_path)
            atomic_df = biopandas_mmcif2pdb(atomic_df, model_index)
        else:
            atomic_df = PandasPdb().read_pdb(pdb_path)
    else:
        if uniprot_id is not None:
            atomic_df = PandasPdb().fetch_pdb(
                uniprot_id=uniprot_id, source="alphafold2-v2"
            )
        else:
            atomic_df = PandasPdb().fetch_pdb(pdb_code)

        atomic_df = atomic_df.get_model(model_index)
        if len(atomic_df.df["ATOM"]) == 0:
            raise ValueError(f"No model found for index: {model_index}")

    return pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]])


def label_node_id(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    df["node_id"] = (
        df["chain_id"].apply(str)
        + ":"
        + df["residue_name"]
        + ":"
        + df["residue_number"].apply(str)
    )
    df["residue_id"] = df["node_id"]
    if granularity == "atom":
        df["node_id"] = df["node_id"] + ":" + df["atom_name"]
    elif granularity in {"rna_atom", "rna_centroid"}:
        df["node_id"] = (
            df["node_id"]
            + ":"
            + df["atom_number"].apply(str)
            + ":"
            + df["atom_name"]
        )
    return df


def deprotonate_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Remove protons from PDB dataframe.

    :param df: Atomic dataframe.
    :type df: pd.DataFrame
    :returns: Atomic dataframe with all ``atom_name == "H"`` removed.
    :rtype: pd.DataFrame
    """
    log.debug(
        "Deprotonating protein. This removes H atoms from the pdb_df dataframe"
    )
    return filter_dataframe(
        df, by_column="element_symbol", list_of_values=["H"], boolean=False
    )


def convert_structure_to_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite existing ``(x, y, z)`` coordinates with centroids of the amino acids.

    :param df: Pandas Dataframe protein structure to convert into a dataframe of centroid positions.
    :type df: pd.DataFrame
    :return: pd.DataFrame with atoms/residues positions converted into centroid positions.
    :rtype: pd.DataFrame
    """
    log.debug(
        "Converting dataframe to centroids. This averages XYZ coords of the atoms in a residue"
    )

    centroids = calculate_centroid_positions(df)
    df = df.loc[df["atom_name"] == "CA"].reset_index(drop=True)
    df["x_coord"] = centroids["x_coord"]
    df["y_coord"] = centroids["y_coord"]
    df["z_coord"] = centroids["z_coord"]

    return df


def subset_structure_to_atom_type(
    df: pd.DataFrame, granularity: str
) -> pd.DataFrame:
    """
    Return a subset of atomic dataframe that contains only certain atom names.

    :param df: Protein Structure dataframe to subset.
    :type df: pd.DataFrame
    :returns: Subsetted protein structure dataframe.
    :rtype: pd.DataFrame
    """
    return filter_dataframe(
        df, by_column="atom_name", list_of_values=[granularity], boolean=True
    )


def remove_insertions(df: pd.DataFrame, keep: Literal["first", "last", False] = "first") -> pd.DataFrame:
    """
    This function removes insertions from PDB dataframes.

    :param df: Protein Structure dataframe to remove insertions from.
    :type df: pd.DataFrame
    :param keep: Specifies which insertion to keep. Options are ``"first"`` or ``"last"``.
        Default is ``"first"``
    :type keep: str
    :return: Protein structure dataframe with insertions removed
    :rtype: pd.DataFrame
    """
    # Catches unnamed insertions
    duplicates = df.duplicated(
        subset=["chain_id", "residue_number", "atom_name"], keep=keep
    )
    df = df[~duplicates]

    # Catches explicit insertions
    df = filter_dataframe(
        df, by_column="insertion", list_of_values=[""], boolean=True
    )

    # Remove alt_locs
    df = filter_dataframe(
        df, by_column="alt_loc", list_of_values=["", "A"], boolean=True
    )

    return df


def filter_hetatms(
    df: pd.DataFrame, keep_hets: List[str]
) -> List[pd.DataFrame]:
    """Return hetatms of interest.

    :param df: Protein Structure dataframe to filter hetatoms from.
    :type df: pd.DataFrame
    :param keep_hets: List of hetero atom names to keep.
    :returns: Protein structure dataframe with heteroatoms removed
    :rtype: pd.DataFrame
    """
    return [df.loc[df["residue_name"] == hetatm] for hetatm in keep_hets]


def process_dataframe(
    protein_df: pd.DataFrame,
    atom_df_processing_funcs: Optional[List[Callable]] = None,
    hetatom_df_processing_funcs: Optional[List[Callable]] = None,
    granularity: str = "centroids",
    chain_selection: str = "all",
    insertions: bool = False,
    deprotonate: bool = True,
    keep_hets: List[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Process ATOM and HETATM dataframes to produce singular dataframe used for graph construction.

    :param protein_df: Dataframe to process.
        Should be the object returned from :func:`~graphein.protein.graphs.read_pdb_to_dataframe`.
    :type protein_df: pd.DataFrame
    :param atom_df_processing_funcs: List of functions to process dataframe. These must take in a dataframe and return a
        dataframe. Defaults to None.
    :type atom_df_processing_funcs: List[Callable], optional
    :param hetatom_df_processing_funcs: List of functions to process the hetatom dataframe. These must take in a
        dataframe and return a dataframe
    :type hetatom_df_processing_funcs: List[Callable], optional
    :param granularity: The level of granularity for the graph. This determines the node definition.
        Acceptable values include: ``"centroids"``, ``"atoms"``,
        any of the atom_names in the PDB file (e.g. ``"CA"``, ``"CB"``, ``"OG"``, etc.).
        See: :const:`~graphein.protein.config.GRAPH_ATOMS` and :const:`~graphein.protein.config.GRANULARITY_OPTS`.
    :type granularity: str
    :param insertions: Whether to keep insertions.
    :param insertions: bool
    :param deprotonate: Whether to remove hydrogen atoms (i.e. deprotonation).
    :type deprotonate: bool
    :param keep_hets: Hetatoms to keep. Defaults to an empty list.
        To keep a hetatom, pass it inside a list of hetatom names to keep.
    :type keep_hets: List[str]
    :param verbose: Verbosity level.
    :type verbose: bool
    :param chain_selection: Which protein chain to select. Defaults to ``"all"``. Eg can use ``"ACF"``
        to select 3 chains (``A``, ``C`` & ``F``)
    :type chain_selection: str
    :return: A protein dataframe that can be consumed by
        other graph construction functions.
    :rtype: pd.DataFrame
    """
    protein_df = label_node_id(protein_df, granularity=granularity)
    atoms = filter_dataframe(
        protein_df,
        by_column="record_name",
        list_of_values=["ATOM"],
        boolean=True,
    )
    hetatms = filter_dataframe(
        protein_df,
        by_column="record_name",
        list_of_values=["HETATM"],
        boolean=True,
    )

    # This block enables processing via a list of supplied functions operating on the atom and hetatom dataframes
    # If these are provided, the dataframe returned will be computed only from these and the default workflow
    # below this block will not execute.
    if atom_df_processing_funcs is not None:
        for func in atom_df_processing_funcs:
            atoms = func(atoms)
        if hetatom_df_processing_funcs is None:
            return atoms

    if hetatom_df_processing_funcs is not None:
        for func in hetatom_df_processing_funcs:
            hetatms = func(hetatms)
        return pd.concat([atoms, hetatms])

    if keep_hets:
        hetatms_to_keep = filter_hetatms(hetatms, keep_hets)
        atoms = pd.concat([atoms] + hetatms_to_keep)

    # Deprotonate structure by removing H atoms
    if deprotonate:
        atoms = deprotonate_structure(atoms)

    # Restrict DF to desired granularity
    if granularity == "atom":
        pass
    elif granularity in {"centroids", "rna_centroid"}:
        atoms = convert_structure_to_centroids(atoms)
    elif granularity == "rna_atom":
        atoms = subset_structure_to_rna(atoms)
    else:
        atoms = subset_structure_to_atom_type(atoms, granularity)

    protein_df = atoms

    # Remove alt_loc residues
    if not insertions:
        protein_df = remove_insertions(protein_df)

    # perform chain selection
    protein_df = select_chains(
        protein_df, chain_selection=chain_selection, verbose=verbose
    )

    log.debug(f"Detected {len(protein_df)} total nodes")

    # Sort dataframe to place HETATMs
    protein_df = sort_dataframe(protein_df)

    return protein_df


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts a protein dataframe by chain->residue number->atom number

    This is useful for distributing hetatms/modified residues through the DF.

    :param df: Protein dataframe to sort.
    :type df: pd.DataFrame
    :return: Sorted protein dataframe.
    :rtype: pd.DataFrame
    """
    return df.sort_values(by=["chain_id", "residue_number", "atom_number"])


def select_chains(
    protein_df: pd.DataFrame, chain_selection: str, verbose: bool = False
) -> pd.DataFrame:
    """
    Extracts relevant chains from ``protein_df``.

    :param protein_df: pandas dataframe of PDB subsetted to relevant atoms
        (``CA``, ``CB``).
    :type protein_df: pd.DataFrame
    :param chain_selection: Specifies chains that should be extracted from
        the larger complexed structure.
    :type chain_selection: str
    :param verbose: Print dataframe?
    :type verbose: bool
    :return: Protein structure dataframe containing only entries in the
        chain selection.
    :rtype: pd.DataFrame
    """
    if chain_selection != "all":
        protein_df = filter_dataframe(
            protein_df,
            by_column="chain_id",
            list_of_values=list(chain_selection),
            boolean=True,
        )

    return protein_df


def initialise_graph_with_metadata(
    protein_df: pd.DataFrame,
    raw_pdb_df: pd.DataFrame,
    granularity: str,
    name: Optional[str] = None,
    pdb_code: Optional[str] = None,
    pdb_path: Optional[str] = None,
) -> nx.Graph:
    """
    Initializes the nx Graph object with initial metadata.

    :param protein_df: Processed Dataframe of protein structure.
    :type protein_df: pd.DataFrame
    :param raw_pdb_df: Unprocessed dataframe of protein structure for comparison and traceability downstream.
    :type raw_pdb_df: pd.DataFrame
    :param granularity: Granularity of the graph (eg ``"atom"``, ``"CA"``, ``"CB"`` etc or ``"centroid"``).
        See: :const:`~graphein.protein.config.GRAPH_ATOMS` and :const:`~graphein.protein.config.GRANULARITY_OPTS`.
    :type granularity: str
    :param name: specified given name for the graph. If None, the PDB code or the file name will be used to name the graph.
    :type name: Optional[str], defaults to ``None``
    :param pdb_code: PDB ID / Accession code, if the PDB is available on the PDB database.
    :type pdb_code: Optional[str], defaults to ``None``
    :param pdb_path: path to local PDB file, if constructing a graph from a local file.
    :type pdb_path: Optional[str], defaults to ``None``
    :return: Returns initial protein structure graph with metadata.
    :rtype: nx.Graph
    """

    # Get name for graph if no name was provided
    if name is None:
        if pdb_path is not None:
            name = get_protein_name_from_filename(pdb_path)
        else:
            name = pdb_code

    G = nx.Graph(
        name=name,
        pdb_code=pdb_code,
        pdb_path=pdb_path,
        chain_ids=list(protein_df["chain_id"].unique()),
        pdb_df=protein_df,
        raw_pdb_df=raw_pdb_df,
        rgroup_df=compute_rgroup_dataframe(remove_insertions(raw_pdb_df)),
        coords=np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]]),
    )

    # Create graph and assign intrinsic graph-level metadata
    G.graph["node_type"] = granularity

    # Add Sequences to graph metadata
    for c in G.graph["chain_ids"]:
        if granularity == "rna_atom":
            sequence = protein_df.loc[protein_df["chain_id"] == c][
                "residue_name"
            ].str.cat()
        else:
            sequence = (
                protein_df.loc[protein_df["chain_id"] == c]["residue_name"]
                .apply(three_to_one_with_mods)
                .str.cat()
            )
        G.graph[f"sequence_{c}"] = sequence
    return G


def add_nodes_to_graph(
    G: nx.Graph,
    protein_df: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> nx.Graph:
    """Add nodes into protein graph.

    :param G: ``nx.Graph`` with metadata to populate with nodes.
    :type G: nx.Graph
    :protein_df: DataFrame of protein structure containing nodes & initial node metadata to add to the graph.
    :type protein_df: pd.DataFrame, optional
    :param verbose: Controls verbosity of this step.
    :type verbose: bool
    :returns: nx.Graph with nodes added.
    :rtype: nx.Graph
    """

    # If no protein dataframe is supplied, use the one stored in the Graph object
    if protein_df is None:
        protein_df = G.graph["pdb_df"]
    # Assign intrinsic node attributes
    chain_id = protein_df["chain_id"].apply(str)
    residue_name = protein_df["residue_name"]
    residue_number = protein_df["residue_number"]  # .apply(str)
    coords = np.asarray(protein_df[["x_coord", "y_coord", "z_coord"]])
    b_factor = protein_df["b_factor"]
    atom_type = protein_df["atom_name"]
    nodes = protein_df["node_id"]
    element_symbol = protein_df["element_symbol"]
    G.add_nodes_from(nodes)

    # Set intrinsic node attributes
    nx.set_node_attributes(G, dict(zip(nodes, chain_id)), "chain_id")
    nx.set_node_attributes(G, dict(zip(nodes, residue_name)), "residue_name")
    nx.set_node_attributes(
        G, dict(zip(nodes, residue_number)), "residue_number"
    )
    nx.set_node_attributes(G, dict(zip(nodes, atom_type)), "atom_type")
    nx.set_node_attributes(
        G, dict(zip(nodes, element_symbol)), "element_symbol"
    )
    nx.set_node_attributes(G, dict(zip(nodes, coords)), "coords")
    nx.set_node_attributes(G, dict(zip(nodes, b_factor)), "b_factor")

    return G


def calculate_centroid_positions(
    atoms: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """
    Calculates position of sidechain centroids.

    :param atoms: ATOM df of protein structure.
    :type atoms: pd.DataFrame
    :param verbose: bool controlling verbosity.
    :type verbose: bool
    :return: centroids (df).
    :rtype: pd.DataFrame
    """
    centroids = (
        atoms.groupby("residue_number")
        .mean()[["x_coord", "y_coord", "z_coord"]]
        .reset_index()
    )
    if verbose:
        print(f"Calculated {len(centroids)} centroid nodes")
    log.debug(f"Calculated {len(centroids)} centroid nodes")
    return centroids


def compute_edges(
    G: nx.Graph,
    funcs: List[Callable],
    get_contacts_config: Optional[GetContactsConfig] = None,
) -> nx.Graph:
    """
    Computes edges for the protein structure graph. Will compute a pairwise
    distance matrix between nodes which is
    added to the graph metadata to facilitate some edge computations.

    :param G: nx.Graph with nodes to add edges to.
    :type G: nx.Graph
    :param funcs: List of edge construction functions.
    :type funcs: List[Callable]
    :param get_contacts_config: Config object for ``GetContacts`` if
        intramolecular edges are being used.
    :type get_contacts_config: graphein.protein.config.GetContactsConfig
    :return: Graph with added edges.
    :rtype: nx.Graph
    """
    # This control flow prevents unnecessary computation of the distance matrices
    if "config" in G.graph:
        if G.graph["config"].granularity == "atom":
            G.graph["atomic_dist_mat"] = compute_distmat(G.graph["pdb_df"])
        else:
            G.graph["dist_mat"] = compute_distmat(G.graph["pdb_df"])

    for func in funcs:
        func(G)

    return add_distance_to_edges(G)


def construct_nx_graph(
    config: Optional[ProteinGraphConfig] = None,
    name: Optional[str] = None,
    pdb_path: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    pdb_code: Optional[str] = None,
    chain_selection: str = "all",
    model_index: int = 1,
    df_processing_funcs: Optional[List[Callable]] = None,
    edge_construction_funcs: Optional[List[Callable]] = None,
    edge_annotation_funcs: Optional[List[Callable]] = None,
    node_annotation_funcs: Optional[List[Callable]] = None,
    graph_annotation_funcs: Optional[List[Callable]] = None,
) -> nx.Graph:
    """
    Constructs protein structure graph from a ``pdb_code`` or ``pdb_path``.

    Users can provide a :class:`~graphein.protein.config.ProteinGraphConfig`
    object to specify construction parameters.

    However, config parameters can be overridden by passing arguments directly to the function.

    :param config: :class:`~graphein.protein.config.ProteinGraphConfig` object. If None, defaults to config in ``graphein.protein.config``.
    :type config: graphein.protein.config.ProteinGraphConfig, optional
    :param name: an optional given name for the graph. the PDB ID or PDB file name will be used if not specified.
    :type name: str, optional
    :param pdb_path: Path to ``pdb_file`` when constructing a graph from a local pdb file. Default is ``None``.
    :type pdb_path: Optional[str], defaults to ``None``
    :param pdb_code: A 4-character PDB ID / accession to be used to construct the graph, if available. Default is ``None``.
    :type pdb_code: Optional[str], defaults to ``None``
    :param uniprot_id: UniProt accession ID to build graph from AlphaFold2DB. Default is ``None``.
    :type uniprot_id: str, optional
    :param chain_selection: String of polypeptide chains to include in graph. E.g ``"ABDF"`` or ``"all"``. Default is ``"all"``.
    :type chain_selection: str
    :param model_index: Index of model to use in the case of structural ensembles. Default is ``1``.
    :type model_index: int
    :param df_processing_funcs: List of dataframe processing functions. Default is ``None``.
    :type df_processing_funcs: List[Callable], optional
    :param edge_construction_funcs: List of edge construction functions. Default is ``None``.
    :type edge_construction_funcs: List[Callable], optional
    :param edge_annotation_funcs: List of edge annotation functions. Default is ``None``.
    :type edge_annotation_funcs: List[Callable], optional
    :param node_annotation_funcs: List of node annotation functions. Default is ``None``.
    :type node_annotation_funcs: List[Callable], optional
    :param graph_annotation_funcs: List of graph annotation function. Default is ``None``.
    :type graph_annotation_funcs: List[Callable]
    :return: Protein Structure Graph
    :rtype: nx.Graph
    """

    if pdb_code is None and pdb_path is None and uniprot_id is None:
        raise ValueError(
            "Either a PDB ID, UniProt ID or a path to a local PDB file"
            " must be specified to construct a graph"
        )

    # If no config is provided, use default
    if config is None:
        config = ProteinGraphConfig()
    with Progress(transient=True) as progress:
        task1 = progress.add_task("Reading PDB file...", total=1)
        # Get name from pdb_file is no pdb_code is provided
        # if pdb_path and (pdb_code is None and uniprot_id is None):
        #    pdb_code = get_protein_name_from_filename(pdb_path)
        #    pdb_code = pdb_code if len(pdb_code) == 4 else None
        progress.advance(task1)

        # If config params are provided, overwrite them
        config.protein_df_processing_functions = (
            df_processing_funcs
            if config.protein_df_processing_functions is None
            else config.protein_df_processing_functions
        )
        config.edge_construction_functions = (
            edge_construction_funcs
            if config.edge_construction_functions is None
            else config.edge_construction_functions
        )
        config.node_metadata_functions = (
            node_annotation_funcs
            if config.node_metadata_functions is None
            else config.node_metadata_functions
        )
        config.graph_metadata_functions = (
            graph_annotation_funcs
            if config.graph_metadata_functions is None
            else config.graph_metadata_functions
        )
        config.edge_metadata_functions = (
            edge_annotation_funcs
            if config.edge_metadata_functions is None
            else config.edge_metadata_functions
        )

        raw_df = read_pdb_to_dataframe(
            pdb_path,
            pdb_code,
            uniprot_id,
            model_index=model_index,
        )


        task2 = progress.add_task("Processing PDB dataframe...", total=1)
        # raw_df = label_node_id(raw_df, granularity=config.granularity)
        # raw_df.df["ATOM"] = label_node_id(
        #    raw_df.df["ATOM"], granularity=config.granularity
        # )
        # raw_df.df["HETATM"] = label_node_id(
        #    raw_df.df["HETATM"], granularity=config.granularity
        # )
        raw_df = sort_dataframe(raw_df)
        protein_df = process_dataframe(
            raw_df,
            chain_selection=chain_selection,
            granularity=config.granularity,
            insertions=config.insertions,
            keep_hets=config.keep_hets,
        )
        progress.advance(task2)

        task3 = progress.add_task("Initializing graph...", total=1)
        # Initialise graph with metadata
        g = initialise_graph_with_metadata(
            protein_df=protein_df,
            raw_pdb_df=raw_df,
            name=name,
            pdb_code=pdb_code,
            pdb_path=pdb_path,
            granularity=config.granularity,
        )
        # Add nodes to graph
        g = add_nodes_to_graph(g)
        # Add config to graph
        g.graph["config"] = config
        g.graph["path"] = g.graph["pdb_path"]

        # Annotate additional node metadata
        if config.node_metadata_functions is not None:
            g = annotate_node_metadata(g, config.node_metadata_functions)
        progress.advance(task3)
        task4 = progress.add_task("Constructing edges...", total=1)
        # Compute graph edges
        g = compute_edges(
            g,
            funcs=config.edge_construction_functions,
            get_contacts_config=None,
        )
        progress.advance(task4)

    # Annotate additional graph metadata
    # print(g.graph['dssp_df'])
    if config.graph_metadata_functions is not None:
        g = annotate_graph_metadata(g, config.graph_metadata_functions)

    # Annotate additional edge metadata
    if config.edge_metadata_functions is not None:
        g = annotate_edge_metadata(g, config.edge_metadata_functions)

    return g
