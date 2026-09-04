# run this first because it takes a couple of minutes
import gzip


import navis
from navis import models
print('imported navis version:', navis.__version__)

from fafbseg import flywire
import h5py
from matplotlib import patches, pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import plotly.io as pio
from plotly import graph_objects as go
# import pylustrator
import scipy
from scipy.interpolate import griddata
import seaborn as sbn
import shutil
import svgutils.transform as sg
import time
from svgutils.compose import Unit
import tempfile


sbn.set_style('whitegrid')

# let's build a class for handling connectome data, including a pathway analysis procedure
class Connectome():
    def __init__(self, database_fn='connections_no_threshold.csv', load_fn=None):
        """Initialize the Connectome object.

        Parameters
        ----------
        database_fn : path to .csv file
            The filename of the database to use. If None, then the online database will be used.
        """
        self.database_fn = database_fn
        # load the database
        if '.csv' in database_fn:
            self.database = pd.read_csv(database_fn)
            self.database.columns = ['pre', 'post', 'neuropil', 'weight', 'transmitter']
        else:
            self.database = None
        # if a load filename is provided, then load the data to avoid re-running the analysis
        self.load_fn = load_fn
        if load_fn is not None:
            self.load()

    def set_materialization(self, num):
        self.materialization = 783

    def lookup_root_ids(self, cell_type):
        """Look up the root IDs of a cell type.

        Parameters
        ----------
        cell_type : str or flywire.NeuronCriteria
            The cell type to look up.

        Returns
        -------
        list
            A list of root IDs corresponding to the cell type.
        """
        if isinstance(cell_type, str):
            cell_type = flywire.NeuronCriteria(type=cell_type)
        # loop and retry 5 times, pausing for 5 seconds if the connection fails
        for _ in range(5):
            try:
                if 'materialization' in dir(self):
                    return flywire.search_annotations(cell_type, materialization=self.materialization).root_id.values.tolist()
                else:
                    return flywire.search_annotations(cell_type).root_id.values.tolist()
            except:
                time.sleep(5)
        return flywire.search_annotations(cell_type, materialization=783).root_id.values.tolist()

    def get_paths(self, source, max_hops=5, direction='downstream', skip_recurrents=False, rerun=False):
        """Find all pathways downstream of root_ids up to max_hops.

        Option to search for upstream or downstream connections.

        Parameters
        ----------
        source : list or array-like
            The list of cells from which to start the simulation. This can be a cell type, 
            NeuronCriteria object, or a list of root IDs.
        max_hops : int, default=5
            The maximum number of hops to search downstream. Default is 5.
        direction : str, default='downstream'
            Whether to search for downstream or upstream.
        skip_recurrents : bool, default=False
            Whether to avoid hopping towards cells that have already been included.

        Returns
        -------
        paths : Path
            A Path object with all the specified pathways.
        """
        # allow for cells to be a cell type string, NeuronCriteria object, or a list of root IDs
        if isinstance(source, list):
            root_ids = source
        else:
            root_ids = self.lookup_root_ids(source)
        # todo: allow both upstream and downstream options, so let's use a direction variable instead and allow both
        downstream = direction == 'downstream'
        upstream = direction == 'upstream'
        # if going downstream, we need to start from pre-synaptic and go to post-synaptic cells
        vars = set(['pre', 'post'])        
        if downstream:
            start_var = 'pre'
        # if going upstream, we need to start from post-synaptic and go to pre-synaptic cells
        if upstream:
            start_var = 'post'
        # store the downstream or upstream pathways into a dictionary
        pathways = {}
        stop_var = (vars - set([start_var])).pop()
        # get the edgelist by starting from the root_ids and travelling up- and downstream
        ids = root_ids
        past_ids = np.unique(ids)
        # choose the appropriate list
        edges = []
        # iterate through each hop by replacing _ids with the next list of ids
        for hop in range(max_hops):
            connections = self.database[self.database[start_var].isin(ids)]
            # optional: skip recurrent connections
            if skip_recurrents:
                include = ~connections[stop_var].isin(past_ids)
                connections = connections[include]
            # store
            edges += [connections]
            ids = connections[stop_var].unique().tolist()
            past_ids = np.unique(np.append(past_ids, ids))
            print(f'Completed {hop+1} hops involving {len(ids)} cells and {connections.weight.sum()} edges.')
        # go through all the pathway sets and add the appropriate hop value for each edge 
        # and level for each node (0 is the root IDs, >0 are downstream and <0 are upstream)
        for hop_num, connections in enumerate(edges): 
            connections.loc[:, 'hop'] = hop_num +1
            # self.database.loc[connections.index, 'hop'] = hop_num + 1
        edges = pd.concat(edges)
        # if going upstream, then the hops are negative
        if direction == 'upstream':
            edges.hop *= -1
        # store now that edges were converted to a dataframe
        pathways[direction] = edges
        # combine the pathways
        vals = [val for val in pathways.values()]
        pathways = pd.concat(vals)
        # make into a Paths object
        paths = Paths(pathways, direction=direction)
        return paths

    def get_downstream_convergence(self, group_a, group_b, simulation_use_inits=True, sim_reps=1e5, **path_kwargs):
        """Find all paths downstream of group_a and group_b and compare.

        Parameters
        ----------
        group_a : str
            The name of the first group.
        group_b : str
            The name of the second group.
        simulation_use_inits : bool, default=True
            Whether to use the initial intersecting nodes for the simulation.
        sim_reps : int, default=1e5
            The number of repetitions for the monte carlo simulation.
        max_hops : int, default=5
            The maximum number of hops to consider when finding paths.
        """
        # get the root IDs for each group
        group_a_ids, group_b_ids = [], []
        for info, storage in zip([group_a, group_b], [group_a_ids, group_b_ids]):
            if isinstance(info, int):
                info = [info]
            elif isinstance(info, (str, flywire.NeuronCriteria)):
                info = self.lookup_root_ids(info)
            storage += info
        # for each group, get the downstream pathways up to max_hops
        paths = {}
        for lbl, root_ids in zip([group_a, group_b], [group_a_ids, group_b_ids]):
            local_path_kwargs = dict(path_kwargs)
            if 'direction' in local_path_kwargs:
                local_path_kwargs.pop('downstream', None)
                local_path_kwargs.pop('upstream', None)
            else:
                if local_path_kwargs.pop('upstream', False):
                    local_path_kwargs['direction'] = 'upstream'
                elif 'downstream' in local_path_kwargs:
                    is_downstream = local_path_kwargs.pop('downstream')
                    local_path_kwargs['direction'] = 'downstream' if is_downstream else 'upstream'
                else:
                    local_path_kwargs['direction'] = 'downstream'
            paths[lbl] = self.get_paths(root_ids, **local_path_kwargs)
            print(f"Found {len(paths[lbl].node_ids)} nodes and {len(paths[lbl].edges_df)} edges for {lbl}.")
        # get the intersection of root IDs at each combination of stages
        max_level = int(max([path.node_info.level.max() for path in paths.values()]))
        # keep track of the number of cells and the degree-weighted number of
        # cells in total for each group and their intersection
        overlap = np.zeros((max_level+1, max_level+1), dtype=float)
        total_a = np.zeros(max_level+1, dtype=int)
        total_b = np.zeros(max_level+1, dtype=int)
        overlap_weighted, total_a_weighted, total_b_weighted = np.copy(overlap), np.copy(total_a), np.copy(total_b)
        # note: the degree-weighted overlap should be calculated by forming a new graph 
        labels = paths.keys()
        path_a, path_b = [paths[lbl] for lbl in labels]
        intersection_paths = nx.intersection(path_a.graph, path_b.graph)
        # make a path object from this, using a df of the edges
        # make an edges_df
        path_a_pairs = [(row['pre'], row['post']) for (num, row) in path_a.edges_df.iterrows()]
        path_a.edges_df['pair'] = path_a_pairs
        intersection_edges = path_a.edges_df[path_a.edges_df.pair.isin(intersection_paths.edges)]
        intersection_paths = Paths(intersection_edges)
        # find the cells in the intersection that are the first along their path, so the lowest level along that path
        # todo: we can do this by starting at the end of the intersection graph and hopping upstream, keeping note
        # of which pre-synaptic cells are not present in the next hop's post-synaptic cells
        # aha! this is actually super useful. only the starting points will be present as pre-synaptic but not post-
        # synaptic cells!!! this is much faster to figure out
        pre_edges = set(intersection_edges.pre.values)
        post_edges = set(intersection_edges.post.values)
        initial_nodes = np.array(list(pre_edges - post_edges))
        # todo: this ^^^ doesn't work if we include recursion. how can we find the initial nodes in that case?
        # I'm already getting the initial nodes in the monte carlo simulation, so I can just use that
        intersection_paths.node_info['initial'] = intersection_paths.node_info.root_id.isin(initial_nodes)
        # (so from post to pre-synaptic connections) 
        # for each node, add its level from path_b
        inds = np.searchsorted(path_b.node_info.root_id.values, intersection_paths.node_info.root_id.values)
        intersection_paths.node_info['level_b'] = path_b.node_info.level.values[inds]
        # get the totals for each level
        _, total_a[:] = np.unique(path_a.node_info.level.values, return_counts=True)
        _, total_b[:] = np.unique(path_b.node_info.level.values, return_counts=True)
        # get the totals weighted by node degree
        total_a_weighted[:] = path_a.node_info.groupby('level')['degree'].sum().values
        total_b_weighted[:] = path_b.node_info.groupby('level')['degree'].sum().values
        include = intersection_paths.node_info['initial'].values
        bins = np.arange(max_level+2) - .5
        self.path_a, self.path_b = path_a, path_b
        self.path_intersection = intersection_paths
        # for both paths, do the monte carlo simulation and get the most prominent pathways, 
        # storing the results in the corresponding node info dataframe
        for group_lbl, path, other_path in zip([group_a, group_b], [self.path_a, self.path_b], [self.path_b, self.path_a]):
            root_ids, simulated_paths = path.monte_carlo(reps=sim_reps)
            # replace the indices in simulated_paths with the corresponding root_ids
            simulated_ids = root_ids[simulated_paths]
            # and now replace the IDs with their corresponding cell types / hemibrain types
            simulated_types = path.node_info
            simulated_ids[simulated_paths == -1] = -1
            # and now replace the IDs with their corresponding cell types / hemibrain types
            # simulated_intersections = np.isin(simulated_ids, intersection_paths.node_info.root_id.values)
            # todo: why is the above getting cells that are not in the actual intersection? It feels like I'm finding the union
            # cells in the intersection, based on below, should be DN
            # 1. check that the simulated IDs are actually in the intersection
            simulated_intersections = np.isin(simulated_ids, other_path.node_info.root_id.values)
            simulated_intersections[:, 0] = False
            if simulation_use_inits:
                # now, find the first True along the second axis
                first_included = np.argmax(simulated_intersections, axis=1)
                # now, convert the first_included into a boolean array with the same shape as simulated_ids
                first_included_arr = np.zeros_like(simulated_ids, dtype=bool)
                for row_num, col_num in enumerate(first_included): first_included_arr[row_num, col_num, np.arange(sim_reps).astype(int)] = True
                simulated_intersections[:] = first_included_arr
            # now, reset the first level to True
            simulated_intersections[:, 0] = True
            # now get the counts for each root_id and level
            for lvl in range(max_level+1):
                # get the corresponding subset of the first_included_arr
                include = simulated_intersections[:, lvl]
                # get the simulated IDs for this level
                simulated_ids_lvl = simulated_ids[:, lvl][include]
                # get counts
                included_ids, included_counts =  np.unique(simulated_ids_lvl, return_counts=True)
                # now add these to the corresponding rows of path_a.node_info
                lbl = f'simulated_weight_{lvl}'
                path.node_info[lbl] = 0
                included_rows = path.node_info.root_id.isin(included_ids)
                path.node_info.loc[included_rows, lbl] = included_counts
            # plot a histogram for the probability of each cell type being at the start of their intersection
            # initial_ids = simulated_ids[first_included_arr]
            initial_ids = simulated_ids
            id_vals, id_counts = np.unique(initial_ids, return_counts=True)
            # get the cell types for the initial nodes
            initial_types = path.node_info[path.node_info.root_id.isin(id_vals)]
            # get the counts per cell type
            inds = np.searchsorted(path.node_info.root_id.values, initial_ids)
            initial_types = path.node_info.cell_type.values[inds]
            non_nans = ~pd.isna(initial_types)
            initial_types = initial_types[non_nans]
            initial_types = pd.Series(initial_types)
            # plot the histogram of all types of initial nodes sorted in descending order
            plt.figure()
            plt.suptitle(f"Intersections per Type for {group_lbl}")
            sbn.countplot(y=initial_types, order=initial_types.value_counts().index, stat='probability')
            plt.ylim(10.5, -.5)
            plt.tight_layout()
        # todo: fix below: use the simulated weight at each level to get the product for each 
        # pair of nodes in the intersection and get the totals by summing over each simulated_weight_{lvl} column
        # add the simulated weights to the intersection node info
        # get the ids from the intersection node info
        intersection_ids = intersection_paths.node_info.root_id.values
        # get the simulated weights for each node in the intersection
        # make a column for each level and path
        new_cols = [f'simulated_weight_{path_lbl}_{lvl}' for path_lbl in ['a', 'b'] for lvl in range(max_level+1)]
        old_cols = [f'simulated_weight_{lvl}' for lvl in range(max_level+1)]
        intersection_paths.node_info[new_cols] = 0
        inds = np.searchsorted(path_a.node_info.root_id.values, intersection_ids)
        intersection_paths.node_info[new_cols[:max_level + 1]] = path_a.node_info[old_cols[:max_level + 1]].values[inds]
        inds = np.searchsorted(path_b.node_info.root_id.values, intersection_ids)
        intersection_paths.node_info[new_cols[max_level + 1:]] = path_b.node_info[old_cols[:max_level + 1]].values[inds]
        # now we can get all of the products of the simulated weights for each pair of nodes
        intersection_info = intersection_paths.node_info
        # sum over the levels
        # get the simulated weight totals
        # total_a_sim = path_a.node_info.groupby('level')['simulated_weight'].sum().values
        # total_b_sim = path_b.node_info.groupby('level')['simulated_weight'].sum().values
        total_a_sim = intersection_info[new_cols[:max_level + 1]].sum().values.astype(float)
        total_b_sim = intersection_info[new_cols[max_level + 1:]].sum().values.astype(float)
        # convert simulated totals into probabilities
        total_a_sim /= total_a_sim.max()
        total_b_sim /= total_b_sim.max()
        # get the indices of just the initial nodes
        include = intersection_info['initial'].values
        for df_lbl, df in zip(
            ['All', 'Initial'], 
            [intersection_info, intersection_info.iloc[include]]):
            if len(df) > 0:
                overlap, xvals, yvals  = np.histogram2d(df.level.values, df.level_b.values, bins=(bins, bins))
                overlap_weighted, xvals, yvals  = np.histogram2d(
                    df.level.values, 
                    df.level_b.values,
                    weights=df.degree.values,
                    bins=(bins, bins))
                overlap_sim_weighted = (df[new_cols[:max_level + 1]].values[:, None] * df[new_cols[max_level + 1:]].values[:, :, None]).sum(0).astype(float)
                # convert 2d hist to probabilities
                overlap_sim_weighted /= overlap_sim_weighted.sum()
                # make overlap and overlap_weighted into pandas dataframes
                for lbl, hist_2d, hist_a, hist_b in zip(
                    [f"{df_lbl} Convergence", f"{df_lbl} Weighted Convergence", f"{df_lbl} Simulated Convergence"], 
                    [overlap, overlap_weighted, overlap_sim_weighted], 
                    [total_a, total_a_weighted, total_a_sim], 
                    [total_b, total_b_weighted, total_b_sim]):
                    hist2d = Hist2D()
                    hist2d.add_data(hist_2d, hist_b, hist_a)
                    # add y- and x-labels
                    hist2d.hist_2d.set_xlabel(f"hops from {group_a}")
                    hist2d.hist_2d.set_ylabel(f"hops from {group_b}")
                    hist2d.hist_top.set_ylabel("count")
                    hist2d.hist_right.set_xlabel("count")
                    hist2d.fig.suptitle(lbl)
        # TODO: consider plotting a point for each starting root_id, allowing for 95% confidence intervals
        # test: I am doubtful about the convergence to (1, 1) of downstream pathways from Dm1 and EPG
        # check that the only cells in common between the simulated
        # plot the top 5 intersecting pathways
        # make a simulated weight product matrix
        simulated_weight_prod = intersection_info[new_cols[:max_level + 1]].values[:, None] * intersection_info[new_cols[max_level + 1:]].values[:, :, None]
        intersection_info['weighted_prod'] = simulated_weight_prod.sum((1, 2))
        # # we're going to select specific layer pairs and plot their constituents
        # levels = np.arange(max_level+1)
        # # make a figure with a grid of subplots
        # include = intersection_paths.node_info.initial.values
        # intersection_sub = intersection_paths.node_info[include]
        # # combine the cell_type and hemibrain type columns, using the hemibrain one when both are available
        # isnan = pd.isna(intersection_sub.cell_type)
        # intersection_sub.cell_type[isnan] = intersection_sub.hemibrain_type[isnan]
        # # plot the degree or simulated weighted of each cell type for each combination of path a and b levels
        # for var in ['degree', 'weighted_prod']:
        #     fig, axes = plt.subplots(ncols=max_level+1, nrows=max_level+1, constrained_layout=True)
        #     title = f"Convergence of {group_a} and {group_b} by {var}"
        #     fig.suptitle(title)
        #     for lvla in levels:
        #         for lvlb in levels:
        #             ax = axes[-(lvlb + 1), lvla]
        #             # subset the intersection node info for this specific
        #             # combination of levels
        #             levels_a, levels_b = intersection_sub[['level', 'level_b']].values.T
        #             include = (levels_a == lvla) * (levels_b == lvlb)
        #             if np.any(include):
        #                 # now get the cell type information
        #                 subset = intersection_sub[include]
        #                 # now plot the degree for each cell type involved
        #                 # summary = subset.groupby('cell_type').degree.sum().sort_values()
        #                 summary = subset.groupby('cell_type')[var].sum().sort_values()
        #                 # scatter plot
        #                 plt.sca(ax)
        #                 ax.scatter(summary.index[::-1], summary.values[::-1], color='k')
        #                 ax.set_xlim(-.5, 10)
        #                 ax.set_ylim(0)
        #                 sbn.despine(ax=ax, trim=True)
        #                 # rotate the xlabels by 45 degrees
        #                 plt.xticks(rotation=45)
        #             else:
        #                 ax.axis('off')
        #     # fig.savefig(title.replace(" ", "_"))
        # plt.show()
        return intersection_paths

    def get_connectivity(self, group_a, group_b, sort_by='sum', transmitters=False, data_dir="./", 
                         pixel_size=0.025, neuroglancer_plots=True, matplotlib_plots=True):
        """Get the connectivity between two groups of cells.

        Separate between upstream and downstream connections as well as neurotransmitter type.
        2 columns and k rows, where k = number of neurotransmitter types + 1 (for all types).
        Also, make two such plots, one that is binary (white = 0, black = 1) with the number 
        of cells and the other weighted by the number of synapses.

        For each connectivity histogram, also tabulate the row and column totals.

        Parameters
        ----------
        group_a : str or NeuronCriteria
            The cell type str or NeuronCriteria for the first group.
        group_b : str or NeuronCriteria
            The cell type str or NeuronCriteria for the second group.
        sort_by : str, default='nblast'
            The method to use for sorting the cells. Options are 'nblast', 'sum', or 'hungarian'
        transmitters : bool, default=False
            Whether to plot the the connectivity by neurotransmitter type.
        data_dir : str, default="./"
            The directory to save the .svg files.
        pixel_size : float, default=0.025
            The pixel size for the matplotlib plots.
        neuroglancer_plots : bool, default=False
            Whether to make neuroglancer plots of the connectivity.
        matplotlib_plots : bool, default=True
            Whether to make matplotlib plots (mostly 2D histograms) of the connectivity.
        """
        # make folder for storing the .svg files
        # only use the following when strings are provided for the groups
        if isinstance(group_a, str) and isinstance(group_b, str):
            dirname = os.path.join(data_dir, f"{group_a}_onto_{group_b}")
            if not os.path.isdir(dirname):
                try:
                    os.mkdir(dirname)
                except:
                    breakpoint()
        else:
            # otherwise, make a directory using the current timestamp
            timestamp = time.time()
            dirname = os.path.join(data_dir, f"{timestamp}")
        svgs = []
        hists = []
        # along
        # consider that sometimes we're looking at the connectivity within the same cell type
        # in that case, we only need to look up one set of root ids and paths
        groups = [group_a, group_b]
        same_group = False
        if group_a == group_b:
            groups = [groups[0]]
            same_group = True
        # get the paths for each group
        paths = {}
        # add a dictionary for storing past paths to avoid redundant calculations
        if 'past_paths' not in dir(self):
            self.past_paths = {}
        for group in groups:
            if group in self.past_paths.keys():
                paths[group] = self.past_paths[group]
            else:
                root_ids = self.lookup_root_ids(group)
                path = self.get_paths(root_ids, direction='downstream', max_hops=1, skip_recurrents=False)
                path.root_ids = root_ids
                print(f"Found {len(path.node_ids)} nodes and {len(path.edges_df)} edges for {group}.")
                self.past_paths[group] = path
                if sort_by == 'nblast':
                    # get NBLAST similarity order for root_ids
                    # 1. get skeletons and dotprops for each root_id    
                    skeletons = flywire.get_skeletons(root_ids)
                    # flywire.get_synapses(skeletons, attach=True)
                    # convert to microns
                    skeletons /= (1000./8.)
                    dotprops = navis.make_dotprops(skeletons)
                    # 2. get the NBLAST similarity matrix for each pair of skeletons
                    nblast = navis.nblast_allbyall(dotprops, dotprops)
                    # todo: let's try using the synBLAST instead
                    # nblast = navis.synblast(skeletons, skeletons)
                    # get the mean between up and downstream connections
                    nblast_scores = nblast.values
                    nblast_scores = (nblast_scores + nblast_scores.T) / 2
                    # set diagonals to 0
                    np.fill_diagonal(nblast_scores, 0)
                    # convert to squareform
                    nblast_scores = scipy.spatial.distance.squareform(nblast_scores)
                    # 3. calculate linkage and dendrogram
                    linkage = scipy.cluster.hierarchy.linkage(1 - nblast_scores, method='ward')
                    dendrogram = scipy.cluster.hierarchy.dendrogram(linkage, no_plot=True)
                    order = dendrogram['leaves']
                    path.dendrogram = dendrogram
                    path.nblast_order = order
                # store path object
                paths[group] = path
        main_path = paths[group_a]
        if same_group:
           other_path = main_path
        else:
            other_path = paths[group_b]
        # for upstream and downstream connections:
        direction, hop, start_col, stop_col = 'downstream', 1, 'pre', 'post'
        # for direction, hop, start_col, stop_col in zip(['upstream', 'downstream'], [-1, 1], ['post', 'pre'], ['pre', 'post']):
        include = main_path.edges_df.hop == hop
        include *= (main_path.edges_df[stop_col].isin(other_path.root_ids))
        edges = main_path.edges_df[include]
        # plot the connectivity including all neurotransmitter types
        # get the pivot table of synapse sums with pre-synaptic cells as rows and post-synaptic cells as columns
        pivot = edges.groupby([start_col, stop_col]).agg({'weight': 'sum'}).reset_index()
        # add the missing pre and post-synaptic cells
        missing_pre = np.setdiff1d(main_path.root_ids, pivot[start_col].unique())
        missing_post = np.setdiff1d(other_path.root_ids, pivot[stop_col].unique())
        # add the missing cells
        for missing_id in missing_pre:
            pivot = pd.concat([pivot, pd.DataFrame([{start_col: missing_id, stop_col: -1, 'weight': 0}])], ignore_index=True)
        for missing_id in missing_post:
            pivot = pd.concat([pivot, pd.DataFrame([{start_col: -1, stop_col: missing_id, 'weight': 0}])], ignore_index=True)
        edges_pivot = pivot.pivot(index=start_col, columns=stop_col, values='weight')
        # replace NaNs with 0s
        edges_pivot.fillna(0, inplace=True)
        if sort_by == 'nblast':
            # make a heat map with the rows and columns sorted by the NBLAST similarity order
            main_order, other_order = main_path.nblast_order, other_path.nblast_order
        elif sort_by == 'sum':
            # sort the rows and columns by the sum of synapses
            main_order = edges_pivot.sum(1).sort_values(ascending=False).index
            other_order = edges_pivot.sum(0).sort_values(ascending=False).index
        elif sort_by == 'hungarian':
            # use the Hungarian algorithm to sort the rows and columns  
            # get the cost matrix
            cost_matrix = edges_pivot.values
            # get the row and column indices
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            main_order, other_order = row_ind, col_ind
        # store the transmitter dataframes
        transmitter_dfs = {}
        # sbn.heatmap(edges_pivot.iloc[main_order, other_order].values, ax=ax, cmap='Greys', cbar_kws={'label': 'synapse count'})
        # sort the pivot table by the main and other order
        if sort_by in ['nblast', 'hungarian']:
            pivot_sorted = edges_pivot.iloc[main_order, other_order]
        else:
            pivot_sorted = edges_pivot.loc[main_order, other_order]
        # add the all transmitters to the list of transmitters
        transmitter_dfs['all'] = pivot_sorted
        if matplotlib_plots:
            hist_synapses = Hist2D()
            hist = hist_synapses
            hist.add_data(pivot_sorted.values, pivot_sorted.sum(0).values, pivot_sorted.sum(1).values, margin_type='scatter', log=False,
                        label='synapses', summary_label='synapses', pixel_size=pixel_size)
            # remove the x and y ticks
            hist.hist_2d.set_xticks([])
            hist.hist_2d.set_yticks([])
            # despine the bottom and left axes
            sbn.despine(ax=hist.hist_2d, bottom=True, left=True)
            # save as .svg
            if isinstance(group_a, str) and isinstance(group_b, str):
                fn = os.path.join(dirname, f"{group_a}_onto_{group_b}_synapses.svg")
            else:
                fn = os.path.join(dirname, f"a_onto_b_synapses_{timestamp}.svg")
            try:
                hist.fig.savefig(fn)
            except:
                hist.fig.savefig(fn, dpi=100)
            # todo: figure out why the figsize is wrong for non-square plots
            svgs.append(fn)
            hists += [hist]
            # and now the same plot but for cell counts
            hist_cells = Hist2D()
            hist = hist_cells
            cell_counts = pivot_sorted > 0
            hist.add_data(cell_counts.values, cell_counts.sum(0).values, cell_counts.sum(1).values, margin_type='scatter', log=False,
                        label='cells', summary_label='cells', pixel_size=pixel_size)
            # remove the x and y ticks
            hist.hist_2d.set_xticks([])
            hist.hist_2d.set_yticks([])
            # despine the bottom and left axes
            sbn.despine(ax=hist.hist_2d, bottom=True, left=True)
            # save as .svg
            if isinstance(group_a, str) and isinstance(group_b, str):
                fn = os.path.join(dirname, f"{group_a}_onto_{group_b}_cells.svg")
            else:
                fn = os.path.join(dirname, f"a_onto_b_cells_{timestamp}.svg")
            hist.fig.savefig(fn)
            svgs.append(fn)
            hists += [hist]
        # first, let's plot a simple histogram of the number of synapses for each neurotransmitter type,
        # sorted in descending order
        transmitter_types = self.database.transmitter.unique()
        transmitter_counts = main_path.edges_df[include].groupby('transmitter').weight.sum().sort_values(ascending=False)
        # add any missing transmitters
        missing_transmitters = np.setdiff1d(transmitter_types, transmitter_counts.index)
        for transmitter in missing_transmitters:
            transmitter_counts[transmitter] = 0
        # sort in alphabetical order
        transmitter_counts = transmitter_counts.sort_index()
        transmitter_types.sort()
        # make a hue-based colormap assigning a different color to each neurotransmitter type
        # for each neurotransmitter type:
        # use an hsv colormap to get the main color for each of these transmitters. I want the saturation
        # and values to stay constant so that the base colors vary only by hue
        # let's make the hsv array and then convert it to rgb
        hues = np.linspace(0, .8, len(transmitter_types))
        sats = 1 * np.ones_like(hues)
        vals = .7 * np.ones_like(hues)
        rgb = mpl.colors.hsv_to_rgb(np.stack([hues, sats, vals], axis=1))
        transmitter_counts['color'] = rgb
        if matplotlib_plots:
            # add colors to the transmitter counts
            # make a bar plot of the transmitter counts, coloring each bar by the corresponding color
            fig = plt.figure()
            sbn.barplot(y=transmitter_counts.index, x=transmitter_counts.values, color='k')
            sbn.despine()
            fn = os.path.join(data_dir, f"{group_a}_onto_{group_b}_transmitters.svg")
            fig.savefig(fn)
        # now, for each neurotransmitter type, plot the connectivity
        if transmitters:
            for transmitter, color in zip(transmitter_types, rgb):
                # get the subset of the edges dataframe for this transmitter
                sub_include = main_path.edges_df.transmitter.values == transmitter
                sub_include *= include
                edges = main_path.edges_df[sub_include]
                # like before, generate a pivot table of synapse sums with pre-synaptic cells as rows and post-synaptic cells as columns
                pivot = edges.groupby([start_col, stop_col]).agg({'weight': 'sum'}).reset_index()
                # add the missing pre and post-synaptic cells
                missing_pre = np.setdiff1d(main_path.root_ids, pivot[start_col].unique())
                missing_post = np.setdiff1d(other_path.root_ids, pivot[stop_col].unique())
                # add the missing cells
                for missing_id in missing_pre: 
                    # pivot = pivot.append({start_col: missing_id, stop_col: -1, 'weight': 0}, ignore_index=True)
                    pivot = pd.concat([pivot, pd.DataFrame([{start_col: missing_id, stop_col: -1, 'weight': 0}])], ignore_index=True)
                for missing_id in missing_post: 
                    # pivot = pivot.append({start_col: -1, stop_col: missing_id, 'weight': 0}, ignore_index=True)
                    pivot = pd.concat([pivot, pd.DataFrame([{start_col: -1, stop_col: missing_id, 'weight': 0}])], ignore_index=True)
                edges_pivot = pivot.pivot(index=start_col, columns=stop_col, values='weight')
                # replace NaNs with 0s
                edges_pivot.fillna(0, inplace=True)
                if sort_by in ['nblast', 'hungarian']:
                    pivot_sorted = edges_pivot.iloc[main_order, other_order]
                else:
                    pivot_sorted = edges_pivot.loc[main_order, other_order]
                transmitter_dfs[transmitter] = pivot_sorted
                if matplotlib_plots:
                    # make a heat map with the rows and columns sorted by the corresponding order
                    synapse_hist = Hist2D()
                    # make a linear colormap from white to the color for the 2d histogram
                    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', ['white', color], N=256)
                    synapse_hist.add_data(pivot_sorted.values, pivot_sorted.sum(0).values, pivot_sorted.sum(1).values,
                                        margin_type='scatter', log=False, label='synapses', summary_label='synapses', 
                                        cmap=cmap, color=color, pixel_size=pixel_size)
                    # remove the x and y ticks
                    synapse_hist.hist_2d.set_xticks([])
                    synapse_hist.hist_2d.set_yticks([])
                    # despine the bottom and left axes
                    sbn.despine(ax=synapse_hist.hist_2d, bottom=True, left=True)
                    # and now the same plot but for cell counts
                    cell_hist = Hist2D()
                    cell_counts = pivot_sorted > 0
                    cell_hist.add_data(cell_counts.values, cell_counts.sum(0).values, cell_counts.sum(1).values, 
                                    margin_type='scatter', log=False, label='cells', summary_label='cells', 
                                    cmap=cmap, color=color, pixel_size=pixel_size)
                    # remove the x and y ticks
                    cell_hist.hist_2d.set_xticks([])
                    cell_hist.hist_2d.set_yticks([])
                    # add the transmitter name to the suptitle
                    # synapse_hist.fig.suptitle(f"{transmitter} synapses")
                    # cell_hist.fig.suptitle(f"{transmitter} cells")
                    # despine the bottom and left axes
                    sbn.despine(ax=cell_hist.hist_2d, bottom=True, left=True)
                    # save both as .svg
                    for hist, name in zip([synapse_hist, cell_hist], ['synapses', 'cells']):
                        fn = os.path.join(dirname, f"{group_a}_onto_{group_b}_{transmitter}_{name}.svg")
                        hist.fig.savefig(fn)
                        svgs.append(fn)
                        hists += [hist]
        # Use the combine_svgs function
        if transmitters:
            row_labels = ['All'] + list(transmitter_types)
        else:
            row_labels = ['All']
        col_labels = ['Synapses', 'Cells']
        if transmitters:
            row_labels = ['All'] + list(transmitter_types)
            num_rows = len(transmitter_types) + 1
        else:
            row_labels = ['All']
            num_rows = 1
        col_labels = ['Synapses', 'Cells']
        num_cols = 2
        if len(svgs) > 0:
            # Ensure svgs has the correct shape
            svgs = np.array(svgs).reshape(num_rows, num_cols)
            combine_svgs(
                svgs, 
                os.path.join(data_dir, f"{group_a}_onto_{group_b}_connectivity.svg"),
                row_labels, col_labels)
        # # recursively delete the .svg files
        # fns = os.listdir(dirname)
        # for fn in fns:
        #     new_fn = os.path.join(dirname, fn)
        #     os.remove(new_fn)
        # # and remove the directory
        # os.rmdir(dirname)
        # # and delete all of the figures
        # plt.close('all')
        if transmitters:
            # now make a single plot with simple histograms using the pivot tables
            num_rows = len(list(transmitter_dfs.keys()))
            num_cols = 4
            if matplotlib_plots:
                # we need twice as many columns to plot the counts per source and target cell
                fig, axes = plt.subplots(ncols=num_cols, nrows=num_rows, constrained_layout=True, figsize=(2*num_cols, 2*num_rows))
                # add black the front of rgb list
                rgb = np.insert(rgb, 0, [0, 0, 0], axis=0)
                for row_num, ((transmitter, pivot), row, color) in enumerate(zip(transmitter_dfs.items(), axes, rgb)):
                    pivot_cells = pivot > 0
                    # Use a for loop to create histograms for synapses and cells per target and source cell
                    for ax, data, label in zip(row, 
                                            [pivot.sum(1), pivot.sum(0), pivot_cells.sum(1), pivot_cells.sum(0)], 
                                            ['Synapses per target cell', 'Synapses per source cell', 'Cells per target cell', 'Cells per source cell']):

                        sbn.histplot(data, ax=ax, kde=False, color=color)
                        ax.set_xlabel(label)
                        # get the bootstrapped 99% C.I. of the mean for the data like before and add it to the plot horizontally
                        # randomly sample the data with replacement 100000 times
                        bootstraps = np.random.choice(data, size=(100000, len(data)), replace=True)
                        # get the mean of each bootstrap sample
                        means = bootstraps.mean(1)
                        # get the 99% C.I. of the mean and the mean
                        low_x, high_x = np.percentile(means, [0.5, 99.5])
                        mean = data.mean()
                        # offset the bottom axis and plot the mean and C.I. below the histogram
                        sbn.despine(ax=ax, bottom=False, offset={'left': 0, 'bottom': 10})
                        # add the C.I. to the plot
                        ax.plot([low_x, high_x], [-5, -5], color='w', lw=2, zorder=1)
                        ax.plot([low_x, high_x], [-5, -5], color=color, lw=2, zorder=2)
                        # plot the mean as a colored circle with a white border
                        ax.plot(mean, -5, 'o', color='w', zorder=3)
                        ax.plot(mean, -5, 'o', color='k', zorder=4)
                    # set the titles
                    if row_num == 0:
                        row[0].set_title(f"Synapses per {group_a}")
                        row[1].set_title(f"Synapses per {group_b}")
                        row[2].set_title(f"Cells per {group_a}")
                        row[3].set_title(f"Cells per {group_b}")
                    else:
                        for ax in row: ax.set_title('')
                    # set the y-labels
                    row[0].set_ylabel(transmitter.upper())
                    # for all other columns, remove the y-label
                    for ax in row[1:]: ax.set_ylabel('')
                    # for all axes, set the xmin to 0
                    for ax in row: ax.set_xlim(0)
                # save the figure
                if isinstance(group_a, str) and isinstance(group_b, str):
                    fn = os.path.join(data_dir, f"{group_a}_onto_{group_b}_simple.svg")
                else:
                    fn = os.path.join(data_dir, f"a_onto_b_simple_{timestamp}.svg")
                fig.savefig(fn, dpi=600)
        # store the results in a dictionary
        result = {}
        result['transmitter_dfs'] = transmitter_dfs
        result['svgs'] = svgs
        result['hists'] = hists
        result['paths'] = paths
        if neuroglancer_plots:
            # plot the results in a systematic way
            # make the following groups and colors to add to a neuroglancer viewer:
            # `get_pathway_information` was moved to class-level method to avoid nesting here.
            pass
        root_ids = self.node_ids
        # use only the root_ids that are also in cell_stats
        ids_included = np.isin(root_ids, cell_stats.index)
        lengths, areas, sizes = cell_stats.loc[root_ids[ids_included], ['length_nm', 'area_nm', 'size_nm']].values.T
        self.node_info.loc[root_ids[ids_included], 'length_nm'] = lengths
        self.node_info.loc[root_ids[ids_included], 'area_nm'] = areas
        self.node_info.loc[root_ids[ids_included], 'size_nm'] = sizes
        # let's also manually add histamine as the major neurotransmitter for R1-6, R7 and R8 cells
        histamine_cells = self.node_info.cell_type.isin(['R1-R6', 'R7', 'R8'])
        self.node_info.loc[histamine_cells, 'top_nt'] = 'histamine'
        # systematic differences in source->target/target connectivity: all of the target cells, colored by the number of synapses
        pivot = transmitter_dfs['all']
        # remove -1 from the pre and posts columns
        pivot = pivot.loc[pivot.index != -1, pivot.columns != -1]
        # use a color map to color the target cells by the number of synapses onto the target cell
        target_ids = pivot.sum(0).index
        vals = pivot.sum(0).values
        minval, maxval = vals.min(), vals.max()
        vals = (vals - minval) / (maxval - minval)
        colors = plt.cm.viridis(vals)
        colors = plt.cm.viridis(pivot.sum(0).values / pivot.sum(0).max())
        # print info about the next URL
        url = flywire.encode_url(segments=target_ids, seg_colors=colors)
        print(f"Go here to see the target cells colored by the number of inputs: {url}")
        result['input_coded_url'] = url
        # systematic differences in source->target/source connectivity: all of the source cells, colored by the number of synapses
        source_ids = pivot.sum(1).index
        vals = pivot.sum(1).values
        minval, maxval = vals.min(), vals.max()
        vals = (vals - minval) / (maxval - minval)
        colors = plt.cm.viridis(vals)
        url = flywire.encode_url(segments=source_ids, seg_colors=colors)
        print(f"Go here to see the source cells colored by the number of ouputs: {url}")
        result['output_coded_url'] = url
        if transmitters:
            # Define a function to get the colors, groups, and weights for a given set of root_ids and pivot table
            def get_colors_groups_weights(root_ids, pivot, transmitter_dfs, transmitter_types, rgb, all_df, max_synapses, target=True):
                colors, groups, weights = {}, {}, {}
                key_axis = 0
                if target:
                    key_axis = 1
                for transmitter, transmitter_color in zip(transmitter_types, rgb[1:]):
                    df = transmitter_dfs[transmitter]
                    if target:
                        new_weights = df.loc[:, root_ids]
                        new_weights = new_weights.loc[new_weights.max(1) > 0]
                        inds = np.argmax(new_weights.values, axis=1)
                        max_target_ids = new_weights.columns[inds]
                    else:
                        new_weights = df.loc[root_ids]
                        new_weights = new_weights.loc[:, new_weights.max(0) > 0]
                        inds = np.argmax(new_weights.values, axis=0)
                        max_target_ids = new_weights.index[inds]
                    for (source_id, new_weight), max_target in zip(new_weights.max(key_axis).items(), max_target_ids):
                        if new_weight > weights.get(source_id, 0):
                            weights[source_id] = new_weight
                            total_synapses = all_df.sum(1-key_axis)[max_target]
                            groups[source_id] = f"{max_target} group ({total_synapses} synapses)"
                            color = transmitter_color.tolist() + [new_weight / max_synapses]
                            colors[source_id] = color
                for root_id in root_ids:
                    colors[root_id] = [1.0, 1.0, 1.0, 1.0]
                    total_synapses = all_df.sum(1-key_axis)[root_id]
                    groups[root_id] = f"{root_id} group ({total_synapses} synapses)"
                    weights[root_id] = total_synapses
                return colors, groups, weights
            # Select 5 target cells evenly spaced along the connectivity gradient and plot the source cells that connect to them
            target_inds = np.round(np.percentile(np.arange(len(target_ids)), [0, 25, 50, 75, 100])).astype(int)
            targets = target_ids[np.argsort(pivot.sum(0).values)][target_inds]
            all_df = transmitter_dfs['all']
            all_df = all_df.loc[:, targets]
            max_synapses = all_df.max().max()
            colors, groups, weights = get_colors_groups_weights(targets, pivot, transmitter_dfs, transmitter_types, rgb, all_df, max_synapses, target=True)
            root_ids = list(colors.keys())
            colors = list(colors.values())
            groups = list(groups.values())
            url = flywire.encode_url(segments=root_ids, seg_colors=colors, seg_groups=groups)
            print(f"Go here to see example targets colored by transmitter inputs: {url}")
            result['input_transmitters_url'] = url
            # Select 5 source cells evenly spaced along the connectivity gradient and plot the target cells that they connect to
            source_ids = pivot.sum(1).index
            source_inds = np.round(np.percentile(np.arange(len(source_ids)), [0, 25, 50, 75, 100])).astype(int)
            sources = source_ids[np.argsort(pivot.sum(1).values)][source_inds]
            all_df = transmitter_dfs['all']
            all_df = all_df.loc[sources, :]
            max_synapses = all_df.max().max()
            colors, groups, weights = get_colors_groups_weights(sources, pivot, transmitter_dfs, transmitter_types, rgb, all_df, max_synapses, target=False)
            root_ids = list(colors.keys())
            colors = list(colors.values())
            groups = list(groups.values())
            url = flywire.encode_url(segments=root_ids, seg_colors=colors, seg_groups=groups)
            print(f"Go here to see example sources colored by transmitter outputs: {url}")
            result['output_transmitters_url'] = url
        return result

    def connectivity_panel(self, source_groups, target_groups, **kwargs):
        """Make a panel of connectivity plots for each combination of source and target groups.
        Parameters
        ----------
        source_groups : list
            A list of source groups.
        target_groups : list
            A list of target groups.
        kwargs : dict
            Keyword arguments to pass to the get_connectivity method.
        """
        data_dir = kwargs['data_dir']
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        results = {}
        synapse_svgs = []
        cell_svgs = []
        fns_to_remove = []
        dirnames_to_remove = []
        for source_group in source_groups:
            for target_group in target_groups:
                print(source_group, target_group)
                res = self.get_connectivity(source_group, target_group, sort_by='sum', **kwargs)
                results[(source_group, target_group)] = res
                # recursively delete the .svg files
                dirname = os.path.join(data_dir, f"{source_group}_onto_{target_group}")
                # add the synapse and cell svgs to the corresponding lists
                synapse_fn = os.path.join(dirname, f"{source_group}_onto_{target_group}_synapses.svg")
                synapse_svgs += [synapse_fn]
                cell_fn = os.path.join(dirname, f"{source_group}_onto_{target_group}_cells.svg")
                cell_svgs += [cell_fn]
                dirnames_to_remove += [dirname]
                fns = os.listdir(dirname)
                for fn in fns:
                    new_fn = os.path.join(dirname, fn)
                    if new_fn in [synapse_fn, cell_fn]:
                        fns_to_remove += [new_fn]
                    else:
                        os.remove(new_fn)
                # and delete all of the figures
                plt.close('all')
        # make two combined images using the cell and synapse svgs
        synapse_svgs = np.array(synapse_svgs).reshape(len(source_groups), len(target_groups))
        cell_svgs = np.array(cell_svgs).reshape(len(source_groups), len(target_groups))
        for svgs, lbl in zip([synapse_svgs, cell_svgs], ['synapses', 'cells']):
            fn = os.path.join(data_dir, f"{','.join(source_groups)}_onto_{','.join(target_groups)}_{lbl}_panel.svg")
            try:
                combine_svgs(svgs, fn, source_groups, target_groups, row_title='source', col_title='target')
            except:
                breakpoint()
        # # remove the .svg files
        # for fn in fns_to_remove:
        #     os.remove(fn)        
        # remove the directories
        for dirname in dirnames_to_remove:
            shutil.rmtree(dirname)

    def save(self, fn):
        """Save the connectivity data to a file.

        Parameters
        ----------
        fn : str
            The filename to save the data to.
        """
        with open(fn, 'wb') as f:
            pickle.dump(self, f)

    def load(self):
        """Load the connectivity data from a file."""
        with open(self.load_fn, 'rb') as f:
            data = pickle.load(f)
            for key, value in data.__dict__.items():
                setattr(self, key, value)


class Paths():
    _h5file = None
    _temp_h5_path = None
    _saved = False

    def __init__(self, edges_df, direction='downstream', graph=None):
        """Handle a collection of edges with some useful operations.

        Parameters
        ----------
        edges_df : pd.DataFrame or hdf.DataFrame
            The edges dataframe containing the 'pre', 'post', 'weight', and 'hop' columns.
        direction : str, default='downstream'
            The direction of the flow. Can be 'upstream' or 'downstream'.
        graph : networkx.DiGraph, optional
            An existing graph to use instead of building one from edges_df.
        Raises
        ------
        AssertionError
            If any of the required columns ('pre', 'post', 'weight', 'hop') are missing in the edges dataframe.
        """
        # store the df
        self.edges_df = edges_df
        # store direction
        self.direction = direction
        # make a graph using networkx
        print("Building graph...")
        if graph is not None:
            self.graph = graph
        else:
            cols_needed = ['pre', 'post', 'weight', 'neuropil', 'transmitter', 'hop']
            if isinstance(edges_df, pd.DataFrame):
                # edges dataframe must have 'pre', 'post', 'weight', and 'hop' columns
                assert all([col in edges_df.columns for col in cols_needed]), f"edges_df needs to include {cols_needed}"
                self.graph = nx.from_pandas_edgelist(edges_df, source='pre', target='post', edge_attr=['weight', 'neuropil', 'transmitter', 'hop'], create_using=nx.DiGraph)
            elif isinstance(edges_df, h5py.Group):
                # build the graph from the h5py dataset in chunks
                # edges group must have 'pre', 'post', 'weight', and 'hop' keys
                assert all([key in edges_df.keys() for key in cols_needed]), f"edges_df needs to include {cols_needed}"
                self.graph = nx.DiGraph()
                # go through the 4 datasets in chunks
                chunk_size = edges_df['pre'].chunks[0]
                num_chunks = len(edges_df['pre']) / chunk_size
                # and store into a pandas dataframe for later
                pandas_df = {}
                # for num, (pre, post, weight, neuropil, transmitter, hop) in enumerate(zip(edges_df['pre'].iter_chunks(), edges_df['post'].iter_chunks(), edges_df['weight'].iter_chunks(), edges_df['neuropil'].iter_chunks(), edges_df['transmitter'].iter_chunks(), edges_df['hop'].iter_chunks())):
                for chunk_num, inds in enumerate(edges_df['pre'].iter_chunks()):
                    pre, post, weight, neuropil, transmitter, hop = edges_df['pre'][inds], edges_df['post'][inds], edges_df['weight'][inds], edges_df['neuropil'][inds], edges_df['transmitter'][inds], edges_df['hop'][inds]
                    # add the chunk to the graph
                    edge_tuples = [(pre_id, post_id, {'weight': w, 'neuropil': n, 'transmitter': t, 'hop': h}) for pre_id, post_id, w, n, t, h in zip(pre, post, weight, neuropil, transmitter, hop)]
                    self.graph.add_edges_from(edge_tuples)
                    # add chunk to the pandas dataframe
                    for col in cols_needed:
                        if col not in pandas_df:
                            pandas_df[col] = []
                        pandas_df[col].extend(eval(col))
                    print_progress(chunk_num + 1, num_chunks, prefix='Building graph:', suffix='Complete', bar_length=40)
                # and now, let's build the graph from the pandas dataframe
                self.edges_df = pd.DataFrame(pandas_df)
                # self.graph = nx.from_pandas_edgelist(edges_df, source='pre', target='post', edge_attr=['weight', 'neuropil', 'transmitter', 'hop'], create_using=nx.DiGraph)
        print("Graph built.")
        # get cell info for each node and calculate the level of each node 
        # (0 being the starting set, which is the pre column of hop=1 or post column of hop=-1)
        self.node_ids = np.asarray(self.graph.nodes)
        self.node_ids.sort()
        retries = 5
        for attempt in range(retries):
            try:
                self.node_info = flywire.search_annotations(self.node_ids, materialization=783)
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    breakpoint()
        self.node_info
        # sometimes flywire doesn't return all of the nodes. Let's add empty entries for those
        missing_ids = np.setdiff1d(self.node_ids, self.node_info.root_id.values)
        self.node_info = self.node_info._append(pd.DataFrame({'root_id': missing_ids}), ignore_index=True)
        # sort by root_id
        self.node_info.sort_values('root_id', inplace=True)
        # get the initial nodes based on the edges_df and direction 
        hops = self.edges_df['hop']
        initial_level = 0
        initial_var = 'pre'
        if direction == 'upstream':
            initial_level = -1
            initial_var = 'post'
        if isinstance(self.edges_df, pd.DataFrame):
            # those from the lowest hop number
            self.initial_nodes = self.edges_df[initial_var][hops == initial_level].unique()
        elif isinstance(self.edges_df, h5py.Group):
            self.initial_nodes = set()
            # can we figure out the max or min without iterating through?
            for chunk_num, inds in enumerate(self.edges_df['hop'].iter_chunks()): 
                hops, nodes = self.edges_df['hop'][inds], self.edges_df[initial_var][inds]
                self.initial_nodes.update(nodes[hops == initial_level])
            # now combine into an array
            self.initial_nodes = np.fromiter(self.initial_nodes, int, len(self.initial_nodes))
        # else:
        #     initial_level = 0
        #     initial_var = 'post'
        #     if isinstance(self.edges_df, pd.DataFrame):
        #         # those from the highest hop number
        #         self.initial_nodes = self.edges_df.post[hops == hops.max()].unique()
        #     elif isinstance(self.edges_df, h5py.Group):
        #         # can we figure out the max or min without iterating through?
        #         # go through the hops and post variables, keeping the post
        #         breakpoint()
        # get the hop number from the graph. if absent, then this must be level 0
        levels = []
        degrees = []
        # if edges_df is a pandas dataframe:
        if isinstance(self.edges_df, pd.DataFrame):
            # use the dataframe to get hop counts instead
            # speed up by grouping the dataframe by pre 
            edges_df_grouped_pre = self.edges_df.groupby('pre').first().sort_values('pre')
            edges_df_grouped_pre.reset_index(level=0, inplace=True)
            edges_df_grouped_post = self.edges_df.groupby('post').first().sort_values('post')
            edges_df_grouped_post.reset_index(level=0, inplace=True)
            for node_id in self.node_ids: 
                # if it's a presynaptic cell:
                if node_id in edges_df_grouped_pre.pre.values:
                    hop = edges_df_grouped_pre.hop.values[edges_df_grouped_pre.pre.values == node_id][0]
                    # and the hop is positive:
                    if hop > 0:
                        # then the level is the hop - 1
                        level = hop - 1
                    # but if it's negative:
                    else:
                        # then the level just the hop
                        level = hop
                # otherwise, if it's a post-synaptic cell:
                elif node_id in edges_df_grouped_post.post.values:
                    hop = edges_df_grouped_post.hop.values[edges_df_grouped_post.post.values == node_id][0]
                    # and the hop is positive:
                    if hop > 0:
                        # then the level is just the hop
                        level = hop
                    # but if it's negative:
                    else:
                        # then the level is the hop + 1, such that post-synaptic partners at hop=-1 are level=0
                        level = hop + 1
                levels += [level]
                degrees += [self.graph.degree[node_id]]
        elif isinstance(self.edges_df, h5py.Group):
            # decide on max or min depending on the direction of flow
            compare_func = max if direction == 'upstream' else min
            # let's compare 2 methods: 1) go through the edges in chunks, storing the maximum or minimum hop number
            # or 2) use the graph to find all children or parents of each node
            for node_id in self.node_ids:
                # and now check if the node has successors
                if node_id in self.graph.succ.keys():
                    # get the subgraph
                    subgraph = self.graph[node_id]
                    # get the hop value closest to 0
                    hops = [data['hop'] for _, data in subgraph.items()]
                    level = compare_func(hops)
                    # if the direction is downstream:
                    if direction == 'downstream':
                        # then the level is the minimum hop - 1
                        level = compare_func(hops) - 1
                # otherwise, check if it has predecessors
                elif node_id in self.graph.pred.keys():
                    # get the subgraph
                    subgraph = self.graph.pred[node_id]
                    # get the hop value closest to 0
                    hops = [data['hop'] for _, data in subgraph.items()]
                    level = compare_func(hops)
                    # if the direction is upstream:
                    if direction == 'upstream':
                        # then the level is the maximum hop + 1
                        level = compare_func(hops) + 1
                # now store
                levels += [level]
                degrees += [self.graph.degree[node_id]]
        # store
        try:
            self.node_info['level'] = levels
        except:
            # todo: for some reason, levels isn't matching the side of node_info
            breakpoint()
        # fix the initial node levels to be 0
        self.node_info.loc[self.node_info.root_id.isin(self.initial_nodes), 'level'] = 0
        self.node_info['degree'] = degrees
        self.node_info.set_index('root_id', inplace=True, drop=False)

        # todo: make sure the cell types are aligning properly! For some reason, this change messed up a lot of my LC results
        # check if any of the cells are in the visual cell type dataset
        visual_cell_info = pd.read_csv('visual_neuron_types.csv', index_col='root_id')
        # find the rows in node_info that have IDs in visual_cell_info
        visual_cells = self.node_info.root_id.isin(visual_cell_info.index)
        # add the visual cell types to the node_info dataframe
        root_ids = self.node_info.root_id[visual_cells].values
        old_types = self.node_info.loc[root_ids, 'cell_type']
        new_types = visual_cell_info.loc[root_ids, 'type']
        updates = new_types[new_types != old_types]
        replacements = old_types[new_types != old_types]
        # make this into a dataframe for comparison
        updates = pd.DataFrame({'id': self.node_info.loc[root_ids].index[new_types != old_types],'old': replacements, 'new': updates})
        # what is the new name for the Pm1 cells?
        self.node_info.loc[root_ids, 'cell_type'] = new_types
        # check if any of the NaN cell types can be replaced with a better alternative
        nan_types = pd.isna(self.node_info.cell_type).values
        cell_types = self.node_info.loc[nan_types, 'hemibrain_type'].values
        self.node_info.loc[nan_types, 'cell_type'] = cell_types
        # add the morphology data: surface area, length, and size
        cell_stats = pd.read_csv("cell_stats.csv")
        # re-index using the root_id column
        cell_stats.set_index('root_id', inplace=True, drop=False)        
        root_ids = self.node_ids
        # use only the root_ids that are also in cell_stats
        ids_included = np.isin(root_ids, cell_stats.index)
        lengths, areas, sizes = cell_stats.loc[root_ids[ids_included], ['length_nm', 'area_nm', 'size_nm']].values.T
        self.node_info.loc[root_ids[ids_included], 'length_nm'] = lengths
        self.node_info.loc[root_ids[ids_included], 'area_nm'] = areas
        self.node_info.loc[root_ids[ids_included], 'size_nm'] = sizes
        # let's also manually add histamine as the major neurotransmitter for R1-6, R7 and R8 cells
        histamine_cells = self.node_info.cell_type.isin(['R1-R6', 'R7', 'R8'])
        self.node_info.loc[histamine_cells, 'top_nt'] = 'histamine'

    @property
    def monte_carlo_paths(self):
        # Return the 'paths' dataset from the currently-open HDF5 file, if present.
        # Prefer the grouped layout (/monte_carlo/paths) but fall back to top-level for
        # backward compatibility.
        if self._h5file is None:
            return None
        if 'monte_carlo' in self._h5file:
            mc = self._h5file['monte_carlo']
            if 'paths' in mc:
                return mc['paths']
        if 'paths' in self._h5file:
            return self._h5file['paths']
        return None

    @monte_carlo_paths.setter
    def monte_carlo_paths(self, value):
        import h5py, tempfile, numpy as np
        # If user provided an h5py.Dataset, attach its file and return
        if isinstance(value, h5py.Dataset):
            self._h5file = value.file
            return
        # Otherwise assume array-like and write into (temp) HDF5
        if self._temp_h5_path is None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
            self._temp_h5_path = tmp.name
            tmp.close()
        # open file and write dataset
        self._h5file = h5py.File(self._temp_h5_path, 'a')
        # ensure monte_carlo group exists and write inside it
        if 'monte_carlo' in self._h5file:
            mc_grp = self._h5file['monte_carlo']
        else:
            mc_grp = self._h5file.create_group('monte_carlo')
        if 'paths' in mc_grp:
            del mc_grp['paths']
        mc_grp.create_dataset('paths', data=np.asarray(value), chunks=True, compression='gzip')

    @property
    def monte_carlo_node_ids(self):
        # Prefer grouped location first (/monte_carlo/node_ids) then fall back to top-level
        if self._h5file is None:
            return None
        if 'monte_carlo' in self._h5file:
            mc = self._h5file['monte_carlo']
            if 'node_ids' in mc:
                return mc['node_ids']
        if 'node_ids' in self._h5file:
            return self._h5file['node_ids']
        return None

    @monte_carlo_node_ids.setter
    def monte_carlo_node_ids(self, value):
        import h5py, numpy as np
        # If provided a dataset, attach its file
        if isinstance(value, h5py.Dataset):
            self._h5file = value.file
            return
        if self._temp_h5_path is None:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
            self._temp_h5_path = tmp.name
            tmp.close()
        if self._h5file is None:
            self._h5file = h5py.File(self._temp_h5_path, 'a')
        # ensure monte_carlo group exists and write node_ids inside it
        if 'monte_carlo' in self._h5file:
            mc_grp = self._h5file['monte_carlo']
        else:
            mc_grp = self._h5file.create_group('monte_carlo')
        if 'node_ids' in mc_grp:
            del mc_grp['node_ids']
        mc_grp.create_dataset('node_ids', data=np.asarray(value), chunks=True, compression='gzip')

    def get_monte_carlo_chunk(self, start=0, stop=None):
        mc = self.monte_carlo_paths
        if mc is None:
            return None
        if stop is None:
            stop = mc.shape[0]
        return mc[start:stop]

    def close(self):
        import os
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None
        # Delete temp file if not saved
        if self._temp_h5_path is not None and not self._saved:
            try:
                os.remove(self._temp_h5_path)
            except Exception:
                pass
            self._temp_h5_path = None

    # def save(self, fn):
    #     import shutil, os, h5py, numpy as np, pandas as pd
    #     # Move temp file if needed
    #     if self._temp_h5_path is not None and self._temp_h5_path != fn:
    #         shutil.move(self._temp_h5_path, fn)
    #         self._saved = True
    #         self._temp_h5_path = fn
    #     # Reopen file for writing
    #     if self._h5file is not None:
    #         self._h5file.close()
    #     self._h5file = h5py.File(fn, 'a')
    #     # Save edges dataframe
    #     if hasattr(self, 'edges_df') and isinstance(self.edges_df, pd.DataFrame):
    #         edges_grp = self._h5file.require_group('edges')
    #         # Save columns as datasets and preserve order
    #         edges_grp.attrs['columns'] = np.array(self.edges_df.columns.astype(str).tolist(), dtype='S')
    #         # Save index
    #         try:
    #             index_vals = self.edges_df.index.to_numpy()
    #         except Exception:
    #             index_vals = np.arange(len(self.edges_df))
    #         edges_grp.create_dataset('_index', data=np.array(index_vals.astype(str), dtype='S'))
    #         for col in self.edges_df.columns:
    #             data = self.edges_df[col].to_numpy()
    #             if data.dtype == object or pd.api.types.is_string_dtype(data) or pd.api.types.is_categorical_dtype(data):
    #                 dt = h5py.string_dtype(encoding='utf-8')
    #                 data_to_store = np.array([None if pd.isna(x) else str(x) for x in data], dtype=object)
    #                 edges_grp.create_dataset(col, data=data_to_store.astype('S'), dtype=dt)
    #             else:
    #                 try:
    #                     edges_grp.create_dataset(col, data=data)
    #                 except Exception:
    #                     dt = h5py.string_dtype(encoding='utf-8')
    #                     edges_grp.create_dataset(col, data=np.array([str(x) for x in data], dtype='S'), dtype=dt)
    #     # Save direction as attribute
    #     if hasattr(self, 'direction'):
    #         self._h5file.attrs['direction'] = self.direction
    #     # Save monte carlo results
    #     if self.monte_carlo_paths is not None:
    #         mc_grp = self._h5file.require_group('monte_carlo')
    #         if 'paths' in mc_grp:
    #             del mc_grp['paths']
    #         mc_grp.create_dataset('paths', data=self.monte_carlo_paths)
    #         if self.monte_carlo_node_ids is not None:
    #             if 'node_ids' in mc_grp:
    #                 del mc_grp['node_ids']
    #             mc_grp.create_dataset('node_ids', data=self.monte_carlo_node_ids)


    def __del__(self):
        self.close()

    def monte_carlo(self, reps=1e5, direction='downstream', chunk_size=1000, save_fn=None):
        """Run a monte carlo simulation to find primary paths.
        
        This will start from the lowest level cells.

        Parameters
        ----------
        reps : int, default=1e6
            The number of repetitions per starting cell.
        direction : str, default='downstream'
            The direction of the walk. Can be 'upstream' or 'downstream'.
        chunk_size : int, default=1000
            The chunk size for writing to HDF5.
        save_fn : str, optional
            If provided, the filename to save the monte carlo paths to. Otherwise, it will be
            stored in a temporary HDF5 file.

        Returns
        -------
        all_nodes : array
            The root IDs of all nodes used to convert the numbers in paths to node IDs.
        paths : array
            The resultant paths for each starting cell. Has the shape (num_starting_cells, num_levels, reps).
        """
        # get all of the nodes, sorted for quick lookup
        all_nodes = self.node_info.root_id.values
        initial_nodes = self.node_info[self.node_info.level == 0]
        max_level = self.node_info.level.max()
        min_level = self.node_info.level.min()
        if direction == 'downstream':
            levels = np.arange(max_level + 1)
        elif direction == 'upstream':
            levels = np.arange(min_level, 1)
            levels = levels[::-1]
        else:
            raise ValueError("Invalid direction. Must be 'upstream' or 'downstream'.")
        shape = (len(initial_nodes), len(levels), int(reps))
        # Create HDF5 dataset for paths
        print("Setting up HDF5 storage for simulation...")
        # File selection rules:
        # 1) If save_fn provided, use it as both temp path and active HDF5 file (persistent)
        # 2) Else, if an HDF5 has already been created/loaded, reuse it and set temp path to its filename
        # 3) Else, create a new temporary file for this simulation
        if save_fn is not None:
            # Use provided filename and mark as persistent (do not auto-delete on close)
            if self._h5file is not None and getattr(self._h5file, 'filename', None) != save_fn:
                try:
                    self._h5file.close()
                except Exception:
                    pass
                self._h5file = None
            self._temp_h5_path = save_fn
            # Open in append mode to create the file if it doesn't exist yet
            self._h5file = h5py.File(self._temp_h5_path, 'a')
            # Treat a user-provided path as persistent storage
            self._saved = True
        else:
            if self._h5file is not None:
                # Reuse existing file handle; ensure temp path is aligned
                try:
                    self._temp_h5_path = self._h5file.filename
                except Exception:
                    # Fall back to making a temp file
                    self._h5file = None
            if self._h5file is None:
                if self._temp_h5_path is None:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
                    self._temp_h5_path = tmp.name
                    tmp.close()
                # Open in append mode to create if missing
                self._h5file = h5py.File(self._temp_h5_path, 'a')
        # ensure monte_carlo group exists and create datasets inside it
        mc_grp = self._h5file.require_group('monte_carlo')
        print("Creating paths dataset...")
        chunk_size = (1, shape[1], shape[2])
        paths_ds = mc_grp.require_dataset(
            'paths', shape=shape, dtype='int64', chunks=chunk_size, compression=None,
            fillvalue=-1, exact=True)
        current_nodes = initial_nodes.root_id.values
        # paths_ds[:, 0] = np.searchsorted(all_nodes, current_nodes[:, None])
        initial_inds = np.searchsorted(all_nodes, current_nodes[:, None])
        # Use h5py's iter_chunks for efficient chunked writing
        num_chunks = len(initial_inds)
        print("Starting monte carlo simulation...")
        for node_num, (idx, initial_ind) in enumerate(zip(paths_ds.iter_chunks(), initial_inds)):
            chunk = paths_ds[idx]
            chunk[0, 0] = initial_ind
            for level_ind, level in enumerate(levels[1:]):
                # pre_nodes = chunk[0, level-1]
                pre_nodes = chunk[0, level_ind]
                node_set, indices, counts = np.unique(pre_nodes, return_index=True, return_counts=True)
                node_vals = all_nodes[node_set]
                for node_val, node_ind, count in zip(node_vals, node_set, counts):
                    if node_val in self.graph:
                        if direction == 'downstream':
                            post_nodes = self.graph[node_val]
                        else:
                            post_nodes = self.graph.pred[node_val]
                        if len(post_nodes) > 0:
                            weights = np.array([node['weight'] for node in post_nodes.values()], dtype=float)
                            weights /= weights.sum()
                            sample = np.random.choice(list(post_nodes), size=int(count), p=weights)
                            inds = pre_nodes == node_ind
                            chunk[0, level_ind + 1, inds] = sorted(np.searchsorted(all_nodes, sample))
            paths_ds[idx] = chunk
            # print progress
            print_progress(node_num + 1, num_chunks, prefix='Simulation progress:', suffix='Complete', bar_length=40)
        # invert the paths array so that the initial nodes are last for downstream simulation
        # paths_ds[...] = paths_ds[:, ::-1]
        all_nodes = np.append(all_nodes, np.nan)
        mc_grp.require_dataset('node_ids', data=all_nodes, shape=all_nodes.shape, dtype='int', exact=True)
        # store: let the properties attach to the existing h5 datasets (dataset objects are passed)
        self.monte_carlo_paths = paths_ds
        self.monte_carlo_node_ids = mc_grp['node_ids']
        return all_nodes, paths_ds

    def monte_carlo_matrix(self, reps=1e5, direction='downstream', save_fn=None, seed=None):
        """Matrix-based Monte Carlo — faster drop-in replacement for monte_carlo().

        Produces the same ``paths_ds`` output as ``monte_carlo()`` (HDF5 dataset
        of shape ``(S, H, R)``) and is compatible with all downstream methods
        (``get_simulation_info``, ``get_pathways``, ``get_pathway_information``,
        ``get_synaptic_fields``).

        Speedup over ``monte_carlo()``:

        1. The CSR transition matrix is built once; cumulative row probabilities
           are pre-computed and reused every hop rather than looked up per node
           from the NetworkX dict.
        2. All ``S * R`` walkers at each unique node are sampled in a single
           ``np.searchsorted`` call (vectorized inverse-CDF), replacing the
           Python-level ``np.random.choice`` loop in the NetworkX backend.

        Parameters
        ----------
        reps : int, default=1e5
            Number of random-walk replicates per starting cell.
        direction : str, default='downstream'
            ``'downstream'`` (pre→post) or ``'upstream'`` (post→pre).
        save_fn : str, optional
            HDF5 output path.  If omitted a temporary file is used (same as
            ``monte_carlo()``).
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        all_nodes : ndarray
            Sorted node IDs (matrix index → root_id mapping).
        paths_ds : h5py.Dataset
            Shape ``(S, H, R)``, integer node indices; identical format to the
            dataset produced by ``monte_carlo()``.
        """
        rng = np.random.default_rng(seed)

        # --- setup mirrors monte_carlo() ---
        all_nodes = self.node_info.root_id.values
        initial_nodes = self.node_info[self.node_info.level == 0]
        max_level = self.node_info.level.max()
        min_level = self.node_info.level.min()
        if direction == 'downstream':
            levels = np.arange(max_level + 1)
        elif direction == 'upstream':
            levels = np.arange(min_level, 1)[::-1]
        else:
            raise ValueError("direction must be 'downstream' or 'upstream'")

        reps = int(reps)
        num_starting = len(initial_nodes)
        shape = (num_starting, len(levels), reps)

        # --- HDF5 storage (mirrors monte_carlo()) ---
        print("Setting up HDF5 storage for matrix simulation...")
        if save_fn is not None:
            if self._h5file is not None and getattr(self._h5file, 'filename', None) != save_fn:
                try:
                    self._h5file.close()
                except Exception:
                    pass
                self._h5file = None
            self._temp_h5_path = save_fn
            self._h5file = h5py.File(self._temp_h5_path, 'a')
            self._saved = True
        else:
            if self._h5file is not None:
                try:
                    self._temp_h5_path = self._h5file.filename
                except Exception:
                    self._h5file = None
            if self._h5file is None:
                if self._temp_h5_path is None:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
                    self._temp_h5_path = tmp.name
                    tmp.close()
                self._h5file = h5py.File(self._temp_h5_path, 'a')

        mc_grp = self._h5file.require_group('monte_carlo')
        print("Creating paths dataset...")
        paths_ds = mc_grp.require_dataset(
            'paths', shape=shape, dtype='int64',
            chunks=(1, shape[1], shape[2]), compression=None, fillvalue=-1, exact=True)

        # --- build transition matrix once, extract raw CSR arrays ---
        import scipy.sparse as sp
        P = self._build_transition_matrix()   # row-stochastic CSR (N, N)
        indptr  = P.indptr
        nb_idx  = P.indices
        nb_data = P.data
        rowsum  = np.asarray(P.sum(axis=1)).ravel()

        init_idx = np.searchsorted(all_nodes, initial_nodes.root_id.values)  # (S,)

        print("Starting matrix Monte Carlo simulation...")
        for s, (start_idx, chunk_idx) in enumerate(
                zip(init_idx, paths_ds.iter_chunks())):
            current = np.full(reps, start_idx, dtype=np.int64)
            chunk = np.full((1, len(levels), reps), -1, dtype=np.int64)
            chunk[0, 0, :] = current

            for level_ind in range(1, len(levels)):
                nxt, surv = Paths._csr_sample_step(
                    indptr, nb_idx, nb_data, rowsum, current, rng)
                # dead / leaked walkers get fill value -1, matching NetworkX backend
                current = np.where(surv, nxt, -1)
                chunk[0, level_ind, :] = current

            paths_ds[chunk_idx] = chunk
            print_progress(s + 1, num_starting,
                           prefix='Matrix MC:', suffix='Complete', bar_length=40)

        all_nodes = np.append(all_nodes, np.nan)
        mc_grp.require_dataset('node_ids', data=all_nodes,
                               shape=all_nodes.shape, dtype='int', exact=True)
        self.monte_carlo_paths    = paths_ds
        self.monte_carlo_node_ids = mc_grp['node_ids']
        return all_nodes, paths_ds

    def get_simulation_info(self, chunk_size=1000):
        """Get supplementary info about the monte carlo simulation to help with pathway analysis, using chunked access."""
        import numpy as np
        mc = self.monte_carlo_paths
        if mc is None:
            print("You must run monte_carlo() before get_simulation_info()")
            return None
        types = self.node_info.cell_type.values.astype(str)
        transmitters = self.node_info.top_nt.values.astype(str)
        transmitter_conf = self.node_info.top_nt_conf.values
        lengths, areas, sizes = self.node_info[['length_nm', 'area_nm', 'size_nm']].values.T
        # If dataset is large, process in chunks
        shape = mc.shape
        chunk_size = mc.chunks
        # Create HDF5 datasets for simulation info if not already created
        if not hasattr(self, '_simulation_info_h5') or self._simulation_info_h5 is None:
            if self._temp_h5_path is None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
                self._temp_h5_path = tmp.name
                tmp.close()
            if self._h5file is not None:
                if self._h5file.mode != 'r+':
                    self._h5file.close()
                    self._h5file = None
            if self._h5file is None:
                self._h5file = h5py.File(self._temp_h5_path, 'r+')
            self._h5file.require_group('simulation_info')
            for name, dtype in zip(['cell_type', 'transmitter', 'transmitter_conf', 'length', 'area', 'size'],
                                   [h5py.string_dtype(), h5py.string_dtype(), float, float, float, float]):
                self._h5file['simulation_info'].require_dataset(name, shape=shape, dtype=dtype, chunks=chunk_size, exact=True)
        info = self._h5file['simulation_info']
        # Use h5py's iter_chunks for efficient chunked access
        for idx in mc.iter_chunks():
            mc_chunk = mc[idx]
            for name, data in zip(['cell_type', 'transmitter', 'transmitter_conf', 'length', 'area', 'size'],
                                  [types, transmitters, transmitter_conf, lengths, areas, sizes]):
                info[name][idx] = data[mc_chunk]
            missing = mc_chunk == -1
            if np.any(missing):
                for key in info.keys():
                    chunk_data = info[key][idx]
                    if info[key].dtype != h5py.string_dtype():
                        chunk_data[missing] = np.nan
                    else:
                        chunk_data[missing] = ''
                    info[key][idx] = chunk_data
        return info

    def get_pathways(self, subset=None, stopping_points=None, chunk_size=1000):
        """Generate pathway strings from the monte-carlo simulation and store them on the instance.

        This method extracts a human-readable pathway string for each monte-carlo sample
        # Pre-allocate the pathways array with the correct shape and dtype
        num_rows = self.monte_carlo_paths.shape[0]
        # Estimate max string length (adjust as needed)
        max_str_len = 256
        pathways = np.empty(num_rows, dtype=f"S{max_str_len}")
        current_idx = 0
        chunked access to the simulation to avoid loading everything into memory.

        Parameters
        ----------
        subset : list of str or None, default=None
            A list of cell types to mark as "subset" for downstream conditional
            computations. This method stores the provided `subset` on the instance so
            that `get_pathway_information(..., conditional=True)` can compute conditional
            probabilities relative to this subset.
        stopping_points : list of str or None, default=None
            A list of cell types at which a pathway should be truncated (the stopping
            point is included in the pathway). If None or empty, pathways are truncated
            at the first `nan` entry.
        chunk_size : int, default=1000
            Chunk size to use when iterating over the simulation.

        Returns
        -------
        pathways : ndarray
            Array of pathway strings in the same order as the monte-carlo rows.
        """
        import numpy as np

        if subset is None:
            subset = []
        if stopping_points is None:
            stopping_points = []

        assert self.monte_carlo_paths is not None, "You must run monte_carlo() before get_pathways()"

        # store subset and stopping points so later analysis can use them
        self.pathways_subset = list(subset)
        self.pathways_stopping_points = list(stopping_points)

        # build an MCSimulation object for efficient queries
        self.mcsimulation = MCSimulation(self.monte_carlo_paths, self.node_ids, self.node_info)

        sep = " < "
        if self.direction == 'downstream':
            sep = " > "

        # pathways = []
        # Pre-allocate the pathways array with the correct shape and dtype
        num_rows, num_hops, num_reps = self.monte_carlo_paths.shape
        max_str_len = 100  # Adjust as needed
        pathways = np.empty((num_rows, num_reps), dtype=f"S{max_str_len}")
        current_idx = 0
        print("Processing pathways...")
        for chunk_num, (chunk) in enumerate(self.mcsimulation.query_chunks('cell_type')):
            rows = chunk[0].T.astype(str)
            if rows.size == 0:
                print_progress(chunk_num + 1, len(self.mcsimulation.paths), prefix='Extracting Unique Pathways:', suffix='Complete', bar_length=40)
                continue
            n_rows, n_cols = rows.shape

            # Determine cut indices from stopping_points (include the stopping point)
            if len(stopping_points) > 0:
                stop_mask = np.zeros_like(rows, dtype=bool)
                for sp in stopping_points:
                    stop_mask |= (np.char.find(rows, sp) != -1)
                stop_any = stop_mask.any(axis=1)
                first_stop_idx = np.argmax(stop_mask, axis=1)
                stop_cut = np.where(stop_any, first_stop_idx + 1, n_cols)
            else:
                stop_cut = np.full(n_rows, n_cols, dtype=int)

            # Determine first nan index per row (truncate before 'nan')
            nan_mask = (rows == 'nan')
            nan_any = nan_mask.any(axis=1)
            first_nan_idx = np.where(nan_any, np.argmax(nan_mask, axis=1), n_cols)

            # final truncation index per row: min of stop_cut and first_nan_idx
            final_cut = np.minimum(stop_cut, first_nan_idx).astype(int)

            # chunk_pathways = []
            for i in range(n_rows):
                r = rows[i, :final_cut[i]]
                if r.size == 0:
                    pathways[chunk_num, i] = ''
                    # chunk_pathways.append('')
                else:
                    nonempty = r[r != '']
                    pathways[chunk_num, i] = sep.join(nonempty)
                    # chunk_pathways.append(sep.join(nonempty))
            # pathways += chunk_pathways
            # print_progress(chunk_num + 1, len(self.mcsimulation.paths), prefix='Extracting Unique Pathways:', suffix='Complete', bar_length=40)
            # pathways[chunk_num] = np.array(chunk_pathways, dtype=pathways.dtype)
            # current_idx += n_rows
            print_progress(chunk_num + 1, len(self.mcsimulation.paths), prefix='Extracting Unique Pathways:', suffix='Complete', bar_length=40)
        # pathways = np.array(pathways, dtype=str)
        self.pathways = pathways.astype(str, copy=False)
        return pathways

    def get_pathway_information(self, top_pathways=10, regex_mode=False, subset=[], stopping_points=[], conditional=False, plot=False, **plot_kwargs):
        """Compute statistics for pathway strings previously generated by `get_pathways`.

        Parameters
        ----------
        top_pathways : int or list of str, default=10
            If int, return the top-N most frequent pathways. If list of str, treat each
            element as either 1) an exact pathway string or 2) a regex pattern; any pathway
            matching any pattern will be returned (order preserved by frequency).

            Example regex patterns for pathway searching:
            - To match any pathway that includes 'PN > KC': `r'.*PN\\s>\\sKC.*'`
            - To match any pathway that starts with 'PN': `r'^PN\\s>\\s.*'`
            - To match any pathway that ends with 'KC': `r'.*>\\sKC$'`
        regex_mode : bool, default=False
            If True, treat `top_pathways` as a list of regex patterns to match against
            all unique pathways. Otherwise, treat `top_pathways` as either an int or a
            list of cell types, producing information on any pathway that includes 
            each of those cell types.
        stopping_points : list of str, default=[]
            A list of cell types at which a pathway should be truncated (the stopping
        subset : list of str or None, default=None
            A list of cell types to mark as "subset" for downstream conditional
        conditional : bool, default=False
            If True, compute probabilities conditional on the subset provided to
            `get_pathways` (that subset is stored on the instance as
            `self.pathways_subset`). If no subset was stored, conditional behavior will
            raise an error.
        plot : bool, default=False
            If True, produce diagnostic plots. Additional plotting options can be passed
            via `plot_kwargs` (same behavior as the previous implementation).

        Returns
        -------
        pathway_set : ndarray
            The selected pathway labels (order determined by `top_pathways`).
        pathways_info : dict
            Dictionary of statistics keyed by metric name (e.g. 'probs', 'lengths', ...)
        """
        import numpy as np
        # add the stopping points to the subset list
        subset = subset + stopping_points

        # assert hasattr(self, 'pathways') and self.pathways is not None, "You must run get_pathways() before get_pathway_information()"

        # ensure mcsimulation exists for queries
        if not hasattr(self, 'mcsimulation') or self.mcsimulation is None:
            self.mcsimulation = MCSimulation(self.monte_carlo_paths, self.node_ids, self.node_info)

        num_starts, num_hops, num_reps = self.monte_carlo_paths.shape
        pathway_shape = (num_starts, num_reps)
        # if a pathway_set has already been stored, use that
        pathway_loaded = False
        if 'pathway_set' in dir(self):
            if 'stopping_points' in dir(self):
                if np.array_equal(self.stopping_points, np.array(stopping_points)):
                    pathway_set, pathway_count = self.pathway_set, self.pathway_count
                    pathway_loaded = True
        if not pathway_loaded:
            pathway_set = self.mcsimulation.query_unique_pathways(stopping_points=stopping_points)
            pathway_set, pathway_count = np.array(list(pathway_set.keys())), np.array(list(pathway_set.values()))
        # store for future use
        self.pathway_set = np.copy(pathway_set)
        self.pathway_count = np.copy(pathway_count)
        self.stopping_points = np.copy(stopping_points)

        sep = ' > '
        if self.direction == 'upstream':
            sep = ' < '

        if subset is not None and len(subset) > 0:
            if not isinstance(subset, (list, np.ndarray)):
                subset = [subset]
            assert isinstance(subset[0], str), "subset must be a cell type (str)"
            # included = [any([t in pathway.split(sep) for t in subset]) for pathway in pathway_set]
            included = [np.any(np.char.find(pathway_set[i], subset) > -1) for i in range(len(pathway_set))]
            included = []
            for pathway in pathway_set:
                types = pathway.split(sep)
                included += [np.isin(types, subset).any()]
            pathway_set = pathway_set[included]
            pathway_count = pathway_count[included]

        # sort by count reversed
        sorted_inds = np.argsort(pathway_count)[::-1]
        pathway_set = pathway_set[sorted_inds]

        # Handle top_pathways as either int or list of strings (with regex support)
        self.mode = 0     # self.regex_mode == 0 => exact matching of pathway strings
        if isinstance(top_pathways, int):
            pathway_set = pathway_set[:top_pathways]
        elif isinstance(top_pathways, (list, np.ndarray)):
            # pathway_set = np.array(top_pathways, dtype=str)
            top_pathways = np.asarray(top_pathways, dtype=str)
            if regex_mode:
                self.mode = 1     # self.mode == 1 => use regex matching
                self.regex_mode = True
            else:
                self.mode = 2     # self.mode == 2 => exact matching of cell types
                # for each cell type in top_pathways, get the pertinent list of node ids
                nodes_included = []
                cell_types = self.node_info['cell_type'].values
                for top_pathway in top_pathways:
                    # subset the cell_types column of self.node_info and grab the indices, which should pertain to the node IDs
                    # get the node id for all matches
                    nodes_included += [np.where(cell_types == top_pathway)[0]]

        pathway_probs = {}
        pathway_lengths = {}
        pathway_areas = {}
        pathway_sizes = {}
        pathway_nt = {}
        pathway_nt_conf = {}

        total_length, total_area, total_size = np.nansum(self.mcsimulation.query(['length_nm', 'area_nm', 'size_nm']), axis=1).transpose((2, 0, 1))
        nt = self.mcsimulation.query('top_nt')
        nt_confidence = self.mcsimulation.query('top_nt_conf')

        # build subset mask for conditional probabilities if requested
        if conditional:
            # TODO: use chunking method instead of loading all at once
            if subset is None or len(subset) == 0:
                raise ValueError('Conditional=True but no subset was provided. Call get_pathways(subset=...) first.')
            subset_mask = np.zeros(pathway_shape, dtype=bool)
            for chunk_num, pathway in enumerate(self.mcsimulation.query_pathway_chunks()):
                pathways_chunk = pathway.T.astype(str)
                for cell_type in subset:
                    type_mask = np.char.find(pathways_chunk, cell_type) != -1
                    # chunk_mask |= type_mask
                    subset_mask[chunk_num] |= type_mask
            self.subset_mask = subset_mask
            # subset_mask = np.zeros(pathways.shape, dtype=bool)
            # for cell_type in subset:
            #     type_mask = np.char.find(pathways, cell_type) != -1
            #     subset_mask |= type_mask
        pathway_vals = pathway_set
        self.regex_mode = regex_mode
        if self.regex_mode:
            # build a vectorized regex checker to apply regex logic to this query
            # first, make a helper function for checking each string
            def pathway_matches(pattern, pathway):
                match = re.search(f"({pathway})", pattern)
                return match is not None            
            # then make the vectorized function
            pathway_matcher = np.vectorize(pathway_matches)
            # go throught the top values instead of the pathway set for this
            pathway_vals = top_pathways
        elif self.mode == 2:
            # go through the top_pathways list instead of the pathway set for this
            pathway_vals = top_pathways        
        # go through each unique pathway
        for path_num, pathway in enumerate(pathway_vals):
            if self.mode == 0:
                # simply check if they match -- the regex approach should converge if no special characters were used
                # pathway_mask = pathways == pathway
                pathway_mask_full = np.zeros(pathway_shape, dtype=bool)
                for chunk_num, pathways_chunk in enumerate(self.mcsimulation.query_pathway_chunks(stopping_points=stopping_points)):
                    chunk_mask = pathways_chunk == pathway
                    pathway_mask_full[chunk_num] |= chunk_mask
                pathway_mask = pathway_mask_full
            elif self.mode == 1:
                # check which pathways match the regex
                pathway_mask = pathway_matcher(pathway_set, pathway)
                # then, include all of those in the pathway mask
                # For regex mode, we need to check all pathways in chunks
                pathway_mask_full = np.zeros(pathway_shape, dtype=bool)
                for chunk_num, pathways_chunk in enumerate(
                    self.mcsimulation.query_pathway_chunks(stopping_points=stopping_points)):
                    chunk_mask = np.isin(pathways_chunk, pathway_set[pathway_mask])
                    pathway_mask_full[chunk_num] |= chunk_mask
                pathway_mask = pathway_mask_full
            elif self.mode == 2:
                nodes = nodes_included[path_num]
                pathway_mask_full = np.zeros(pathway_shape, dtype=bool)
                # go through the simulation in chunks, checking if any of the nodes are present
                for chunk_num, chunk in enumerate(self.monte_carlo_paths.iter_chunks()):
                    paths_chunk = self.monte_carlo_paths[chunk]
                    chunk_mask = np.isin(paths_chunk, nodes).any((0, 1))
                    pathway_mask_full[chunk_num] = chunk_mask
                pathway_mask = pathway_mask_full
            # calculate probabilities
            if conditional:
                # calculate the conditional probability for the subset
                probs = pathway_mask.sum(1) / subset_mask.sum(1)
            else:
                # calculate the absolute probabilitiy
                probs = pathway_mask.mean(1)
            # store the probability
            pathway_probs[pathway] = probs
            # calculate important metrics
            total_lengths, total_areas, total_sizes = [], [], []
            transmitters, transmitter_conf = [], []
            for lengths, areas, sizes, nts, nt_confs, mask in zip(total_length, total_area, total_size, nt, nt_confidence, pathway_mask):
                total_lengths += [np.nanmean(lengths[mask])]
                total_areas += [np.nanmean(areas[mask])]
                total_sizes += [np.nanmean(sizes[mask])]
                transmitters += [nts[:, mask]]
                transmitter_conf += [nt_confs[:, mask]]
            # store
            pathway_lengths[pathway] = np.array(total_lengths)
            pathway_areas[pathway] = np.array(total_areas)
            pathway_sizes[pathway] = np.array(total_sizes)
            pathway_nt[pathway] = transmitters
            pathway_nt_conf[pathway] = transmitter_conf
            # progress update
            print_progress(path_num + 1, len(pathway_vals), prefix='Pathway analysis:', suffix='Complete', bar_length=40)
        # combine into a dictionary
        pathways_info = {
            'probs': pathway_probs,
            'lengths': pathway_lengths,
            'areas': pathway_areas,
            'sizes': pathway_sizes,
            'nt': pathway_nt,
            'nt_conf': pathway_nt_conf
        }
        if plot:
            # plot the results
            pathway_set = pathway_set[::-1]
            sbn.set_style('whitegrid')
            confidence = plot_kwargs.get('confidence', 99)
            results = plot_kwargs.get('results', ['probs', 'lengths', 'transmitters'])
            if results == 'all':
                results = ['probs', 'lengths', 'areas', 'sizes', 'transmitters']
            ncols = 0
            potential_xvals = ['probs', 'lengths', 'areas', 'sizes']
            for xval in potential_xvals:
                if xval in results:
                    ncols += 1
            if 'transmitters' in results:
                ncols += 2

            if 'fig' in plot_kwargs and 'axes' in plot_kwargs:
                fig = plot_kwargs['fig']
                axes = plot_kwargs['axes']
                assert len(axes) >= ncols, "Provided axes do not have enough subplots for the requested results."
            else:
                fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(4*ncols, 6), constrained_layout=True, sharey=True)
                if ncols == 1:
                    axes = [axes]

            transmitter_colors = {
                'acetylcholine': 'red',
                'gaba': 'blue',
                'glutamate': 'green',
                'dopamine': 'yellow',
                'serotonin': 'purple',
                'octopamine': 'orange',
            }
            transmitter_shortnames = {
                'acetylcholine': 'ACh',
                'gaba': 'GABA',
                'glutamate': 'Glu',
                'dopamine': 'DA',
                'serotonin': '5HT',
                'octopamine': 'OA',
            }
            alpha = plot_kwargs.get('alpha', 0.2)
            color = plot_kwargs.get('color', 'k')

            def jitterplot_ci(ax, xs, yval=0, yjitter_std=.1, color=color, alpha=alpha):
                yjitter = np.random.normal(0, yjitter_std, size=xs.shape)
                ax.scatter(xs, yval + yjitter, color=color, marker='.', alpha=alpha)
                reps = plot_kwargs.get('reps', 100000)
                bootstraps = np.random.choice(xs, size=(reps, len(xs)), replace=True)
                means = bootstraps.mean(1)
                low_x, high_x = np.percentile(means, [(100 - confidence) / 2, 100 - (100 - confidence) / 2])
                mean = xs.mean()
                ax.errorbar(mean, yval, xerr=[[mean - low_x], [high_x - mean]], fmt='o', color='k')

            log_xvals = ['lengths', 'areas', 'sizes']

            ax_idx = 0
            for xval in potential_xvals:
                if xval in results:
                    ax = axes[ax_idx]
                    for i, pathway in enumerate(pathway_set):
                        yval = i
                        xs = pathways_info[xval][pathway]
                        jitterplot_ci(ax, xs, yval=yval)
                    ax.set_yticks(range(len(pathway_set)))
                    ax.set_yticklabels(pathway_set, rotation=45)
                    ax.set_xlabel(xval.capitalize())
                    if xval in log_xvals:
                        ax.set_xscale('log')
                    ax_idx += 1

            if 'transmitters' in results:
                ax = axes[ax_idx]
                for i, pathway in enumerate(pathway_set):
                    yval = i
                    try:
                        all_transmitters = np.concatenate(pathway_nt[pathway], axis=-1).astype(str)
                    except:
                        breakpoint()
                    num_bars = all_transmitters.shape[0]
                    tot_height = 0.9
                    bar_height = tot_height / num_bars
                    edges = np.linspace(-tot_height/2., tot_height/2., num_bars+1)
                    y_offsets = (edges[:-1:1] + edges[1::1])/2.
                    y_offsets += yval
                    for hop_level, (transmitters, y_offset) in enumerate(zip(all_transmitters[::-1], y_offsets)):
                        try:
                            lbls, counts = np.unique(transmitters, return_counts=True)
                        except:
                            breakpoint()
                        probs = counts / counts.sum()
                        colors = [transmitter_colors.get(lbl, 'gray') for lbl in lbls]
                        left = 0
                        for prob, color, lbl in zip(probs, colors, lbls):
                            ax.barh(y_offset, prob, left=left, color=color, height=bar_height)
                            if prob > 0.1:
                                lbl_short = transmitter_shortnames.get(lbl, lbl)
                                ax.text(left + prob/2., y_offset, lbl_short, color='white', ha='center', va='center', fontsize=bar_height*25*0.8)
                            left += prob
                ax.set_yticks(range(len(pathway_set)))
                ax.set_yticklabels(pathway_set)
                ax.set_xlabel('Transmitter composition')
                ax_idx += 1

                ax = axes[ax_idx]
                for i, pathway in enumerate(pathway_set):
                    yval = i
                    transmitter_conf = pathway_nt_conf[pathway]
                    transmitter_confs = np.array([conf.mean(1) for conf in transmitter_conf]).T
                    all_transmitters = np.concatenate(pathway_nt[pathway], axis=-1)
                    all_transmitters = all_transmitters.astype(str)
                    num_bars = all_transmitters.shape[0]
                    tot_height = 0.9
                    bar_height = tot_height / num_bars
                    edges = np.linspace(-tot_height/2., tot_height/2., num_bars+1)
                    y_offsets = (edges[:-1:1] + edges[1::1])/2.
                    y_offsets += yval
                    for hop_level, (confs, transmitters, y_offset) in enumerate(zip(transmitter_confs[::-1], all_transmitters[::-1], y_offsets)):
                        try:
                            nt, counts = np.unique(transmitters, return_counts=True)
                        except:
                            breakpoint()
                        top_nt = nt[np.argmax(counts)]
                        color = transmitter_colors.get(top_nt, 'gray')
                        jitterplot_ci(ax, confs, yval=y_offset, yjitter_std=bar_height/6., color=color)
                ax.set_yticks(range(len(pathway_set)))
                ax.set_yticklabels(pathway_set)
                ax.set_xlabel('Transmitter confidence')
                ax_idx += 1
        self.pathway_set, self.pathway_count = pathway_set, pathway_count
        self.pathways_info = pathways_info
        return pathway_set, pathways_info
    
    def get_synaptic_fields(self, stopping_points=['L1', 'L2', 'L3', 'L4', 'L5', 'R7', 'R8'], side='right', conditional=False, method='simulation', num_hops=None, column_file='column_assignment.csv', normalize=True, by_hop=False):
        """Get all receptive fields for the target.

        Note: if you want to subset the pathways or apply specific stopping points,
        use get_pathways() first to get the desired pathways, the run this function.

        Parameters
        ----------
        stopping_points : list of str, default=['L1', 'L2', 'L3', 'L4', 'L5', 'R7', 'R8']
            A list of cell types to consider as stopping points for the pathways.
        side : str, default='right'
            The eye side to consider for the receptive fields. Can be 'right' or 'left'.
        conditional : bool, default=False
            If True, compute probabilities conditional on the subset provided to get_pathways()
            or the top_pathways.
        method : str, default='simulation'
            The backend used to compute the fields. 'simulation' uses the Monte Carlo
            results (requires get_pathways()/get_pathway_information() to have been run).
            'analytic' computes the same marginal first-passage probabilities exactly by
            propagating the transition matrix, without needing a Monte Carlo run.
            'matrix_mc' estimates the same quantities by sampling walkers through the
            transition matrix (vectorized inverse-CDF); pass ``reps`` to control accuracy.
        num_hops : int, optional
            Only used when method='analytic'. Number of propagation hops. Defaults to the
            maximum absolute level in node_info (i.e. the graph depth).
        normalize : bool, default=True
            If True (default), the transition matrix is row-stochastic and output values
            are probabilities (each node distributes exactly 1 unit of mass). If False,
            raw synapse weights are used without row-normalization, so output values
            represent total synapse-weighted flow — nodes with many outgoing synapses
            contribute proportionally more, revealing absolute connectivity strength.
            For method='matrix_mc', False keeps raw walker-hit counts instead of
            dividing by reps.

        Returns
        -------
        fields : dict of arrays
            A dictionary of receptive fields for each pathway. The value for each pathway
            is an array of shape (num_targets, pval, qval) representing the receptive fields
            along the square lattice. This can be used to project onto retinal or even visual
            coordinates. 

        Notes
        -----
            Summing and averaging across the first axis will generate an image. If this is done 
            before centering, this will represent the down- or upstream projection onto a cell
            class. If averaging is done after centering for an upstream projection, this will
            represent the average synaptic field for each pathway (akin to a receptive field).
            The same procedure for a downstream projection will represent the amount of downstream
            mixing, characterized best by the 2D impulse response (akin to a point spread function).
        """
        if by_hop and method != 'analytic':
            raise ValueError("by_hop=True is only supported for method='analytic'")
        # dispatch to the analytic backend if requested
        if method == 'analytic':
            return self._get_synaptic_fields_analytic(
                stopping_points=stopping_points, side=side,
                conditional=conditional, num_hops=num_hops, normalize=normalize,
                by_hop=by_hop)
        elif method == 'matrix_mc':
            return self._get_synaptic_fields_matrix_mc(
                stopping_points=stopping_points, side=side,
                conditional=conditional, num_hops=num_hops, normalize=normalize)
        elif method != 'simulation':
            raise ValueError("method must be 'simulation', 'analytic', or 'matrix_mc'")
        # TODO: implement side logic properly -- right now both sides are being combined

        if self.direction == 'upstream':
            sep = ' < '
        # assert self.pathways is not None, "You must run get_pathways() before get_synaptic_fields()"
        info = self.pathways_info
        mcsimulation = self.mcsimulation
        simulation_types_chunked = self.mcsimulation.query_chunks('cell_type')
        trans = [b"gaba", b"acetylcholine", b"glutamate", b"octopamine", b"serotonin", b"dopamine", b"histamine", b"", b"nan"]
        # in the optic lobe:
        effect = [-1, 1, -1, 1, 1, 1, 1, 1, 1]
        effect = dict(zip(trans, effect))
        # apply the dictionary to the array of neurotransmitters
        lookup = np.vectorize(effect.get)
        # to consider all incoming visual streams, using the following cell types as starting points:
        stopping_points = np.array(stopping_points)
        starting_points = stopping_points
        # get the subset of the simulation that passes through the starting points -- useful for getting conditional probabilities
        # input_included = np.zeros_like(pathways, dtype=bool)
        # for cell_type in stopping_points:
        #     input_included |= np.char.find(pathways, cell_type) != -1
        # input_included = self.subset_mask
        # download the visual column assignment file if not already present
        if not os.path.exists(column_file):
            import urllib
            # download and unzip the file from flywire
            url = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz"
            urllib.request.urlretrieve(url, "column_assignment.csv.gz")
            import gzip
            output_file_path = column_file
            with gzip.open("column_assignment.csv.gz", 'rb') as f_in:
                with open(output_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        # load the visual column data
        visual_column_info = pd.read_csv(column_file, index_col=0)
        # grab the retinal coordinates
        all_ps, all_qs = visual_column_info[['p', 'q']].values.T
        # get the coordinate bounds
        pmin, pmax = int(all_ps.min()), int(all_ps.max()) + 1
        qmin, qmax = int(all_qs.min()), int(all_qs.max()) + 1
        width, height = pmax - pmin, qmax - qmin
        # get the monte carlo paths for ease of access
        simulation_res = self.monte_carlo_paths
        # make an empty grid for storing each pathway's field of inputs in retinal coordinates
        # this will allow us to generate synaptic fields
        probs_grid = np.zeros((len(simulation_res), width, height), dtype='float32')
        pvals, qvals = np.meshgrid(np.arange(pmin, pmax), np.arange(qmin, qmax), indexing='ij')
        # get a synaptic field for each pathway
        all_fields = None
        fields = {}
        effects = {}
        # we want to go through each major pathway, generating a N x p x q array of probabilities
        # such that prob[i, p, q] represents the conditional probability of that pathway arriving 
        # at the i-th LC from the retinal coordinate (p, q)
        if self.regex_mode:
            import re
            # make a vectorized regex checker to apply regex logic to this query
            # first, make a helper function for checking each string
            def regex_checker(pattern, string):
                return re.search(f"({pattern})", string) is not None
            pathway_matcher = np.vectorize(regex_checker)
            # pathway_set, pathway_count = np.unique(pathways, return_counts=True)
            # pathway_set = pathway_set[np.argsort(pathway_count)[::-1]]
            pathway_set = self.pathway_set
            sub_pathways = {}
        if self.mode == 2:
            # get the node IDs for each path
            cell_types = self.node_info['cell_type'].values
            top_nodes = []
            for path_lbl in info['probs'].keys():
                # subset the cell_types column of self.node_info and grab the indices, which should pertain to the node IDs
                # get the node id for all matches
                top_nodes += [np.where(cell_types == path_lbl)[0]]
        print(f"Calculating synaptic fields for {len(info['probs'])} pathways...")
        if self.mode == 2:
            # this is the simplest mode:
            starting_nodes, pvals, qvals = [], [], []
            probs_grid = {}
            totals = {}
            # for path_lbl, nodes in zip(info['probs'].keys(), nodes_included):
            for lbl in starting_points:
                nodes = np.where(cell_types == lbl)[0]
                # get the p- and q-values
                root_ids = self.node_info.iloc[nodes].index
                # if a root_id is missing from visual_column_data, omit it
                nodes_included = np.isin(root_ids, visual_column_info.index)
                nodes = nodes[nodes_included]
                # store the starting nodes
                starting_nodes += [nodes]
                # store the p- and q-values
                ps, qs = visual_column_info.loc[root_ids[nodes_included], ['p', 'q']].values.T
                pvals += [ps]
                qvals += [qs]
            # add an empty grid for each pathway
            for path_lbl in info['probs'].keys():
                # make an empty grid for storing each pathway's field of inputs in retinal coordinates
                probs_grid[path_lbl] = np.zeros((width, height), dtype='float32')
                totals[path_lbl] = np.zeros((width, height), dtype=int)
            # now, go through each chunk of the simulation results to build up the probabilities grid
            # we need to reset the simulation_res.iter_chunks() and simulation_types_chunked iterators
            simulation_types_chunked = self.mcsimulation.query_chunks('cell_type')
            for num, (idx, chunk_types) in enumerate(zip(simulation_res.iter_chunks(), simulation_types_chunked)):
                chunk = simulation_res[idx]
                # go throuh all path_lbls
                for path_lbl, nodes in zip(info['probs'].keys(), top_nodes):
                    # we can subset using np.isin the set of included node_ids -- probably faster
                    top_included = np.isin(chunk, nodes)
                    # then we can use those indeces and keep those that are closest to the starting point
                    for starting_lbl, ps, qs, starting_node in zip(starting_points, pvals, qvals, starting_nodes):
                        # only consider those nodes that are in the starting nodes
                        starting_included = np.isin(chunk, starting_node)
                        # only proceed if there are any starting nodes and top nodes in the same pathway
                        pathways_included = starting_included[0].any(0) & top_included[0].any(0)
                        if np.any(pathways_included):
                            chunk_included = chunk[0, :, pathways_included]
                            # find the first occurrence of a starting node in each pathway
                            starting_inds = np.argmax(np.isin(chunk_included, starting_node), axis=1)
                            top_inds = np.argmax(np.isin(chunk_included, nodes), axis=1)
                            # get all of the nodes at those indices
                            included_starting_nodes = chunk_included[0, starting_inds]
                            included_top_nodes = chunk_included[0, top_inds]
                            # get the p and q values for those starting nodes
                            lbls, counts = np.unique(included_starting_nodes, return_counts=True)
                            # go through each label and count and find the appropriate p and q values to increment by count
                            for lbl, count in zip(lbls, counts):
                                # find the appropriate p and q values
                                ind = starting_node == lbl
                                p, q = ps[ind], qs[ind]
                                probs_grid[path_lbl][p - pmin, q - qmin] += count
                                if conditional:
                                    # get the conditional probability
                                    totals[path_lbl][p - pmin, q - qmin] += starting_included.any((0, 1)).sum()
                                else:
                                    # get the total probability
                                    totals[path_lbl][p - pmin, q - qmin] += starting_included.shape[-1]
                # print progress
                print_progress(num + 1, simulation_res.shape[0], prefix="Calculating synaptic fields per chunk:")
            # now, convert counts to probabilities
            fields = {}
            pvals, qvals = np.meshgrid(np.arange(pmin, pmax), np.arange(qmin, qmax), indexing='ij')
            for path_lbl in info['probs'].keys():
                with np.errstate(divide='ignore', invalid='ignore'):
                    field = np.divide(probs_grid[path_lbl], totals[path_lbl], out=np.zeros_like(probs_grid[path_lbl]), where=totals[path_lbl]!=0)
                fields[path_lbl] = SynapticField(field[None], ps=pvals, qs=qvals)
            return SynapticFields(fields)
        elif self.mode in [0, 1]:
            for path_num, path_lbl in enumerate(info['probs'].keys()):
                # allow regex
                if self.mode == 0:
                    # included = pathways == path_lbl        # pathways[included] represents all paths that come from the starting points
                    # what is the starting point of this pathway?
                    # included_points = np.char.find(path_lbl, stopping_points) != -1
                    # results_included = np.any(included_points)
                    results_included = np.isin(path_lbl.split(sep), stopping_points).any()
                elif self.mode == 1:
                    included = pathway_matcher(path_lbl, pathway_set)
                    pathways_included = pathway_set[included]
                    # regex mode is pretty different from normal mode
                    # only include those pathways that have stopping points
                    pathways_included = [p for p in pathways_included if np.any(np.char.find(p, stopping_points)  > -1)]
                    results_included = len(pathways_included) > 0
                # if stopping points are present, use this for calculating the synaptic field
                if results_included:
                    if self.mode == 0:
                        # use just the single pathway label
                        path_lbls = [path_lbl]
                    elif self.mode == 1:
                        # repeat the process for each included pathway
                        path_lbls = pathways_included
                        effects[path_lbl] = 1
                        regex_key = path_lbl
                    # make the N x p x q array of probabilities
                    probs_grid = np.zeros((len(simulation_res), width, height), dtype='float32')
                    # keep track of the starting_points and starting_levels for each pathway
                    starting_coords_arr, starting_levels, sub_types_arr = [], [], []
                    # if using regex, store all of the sub-pathways
                    if self.regex_mode:
                        sub_paths = []
                    for path_num, lbl in enumerate(path_lbls):
                        # what is the starting point of this pathway?
                        # included_points = np.char.find(lbl, stopping_points) != -1
                        included_points = np.isin(stopping_points, lbl.split(sep))
                        # get the relevant information per pathway
                        sub_types = stopping_points[included_points][0]
                        path_types = np.array(lbl.split(sep))
                        if np.any(path_types == sub_types) == False:
                            # skip this pathway -- it doesn't include the stopping point
                            continue
                        starting_level = np.where(path_types == sub_types)[0][0]
                        # let's get the retinal coordinates of all instances of the starting point
                        types = visual_column_info['type']
                        hemisphere = visual_column_info['hemisphere']
                        starting_coords = visual_column_info[(types == sub_types)*(hemisphere == side)][['p', 'q']]
                        starting_coords['node_id'] = np.searchsorted(self.node_ids, starting_coords.index.values)
                        # let's make the dataset searchable by root_id
                        starting_coords['root_id'] = starting_coords.index.values
                        # and sort by node_id to speed up searches
                        starting_coords = starting_coords.set_index('node_id').sort_index()
                        # get the typical neurotransmitter effect for this pathway using the path_types
                        # for each type in path_types, look up it's major transmitter in node_info
                        key_ind = len(lbl.split(sep))
                        if not self.regex_mode:
                            effects[lbl] = np.median(lookup(np.concatenate(info['nt'][lbl], axis=1)[:key_ind - 1].astype(bytes)).prod(0))
                        # else:
                        #     effects[regex_key][lbl] = np.median(lookup(np.concatenate(info['nt'][lbl], axis=1)[:key_ind - 1].astype(bytes)).prod(0))
                        # store
                        starting_coords_arr.append(starting_coords)
                        starting_levels.append(starting_level)
                        sub_types_arr.append(sub_types)
                        if self.regex_mode:
                            sub_paths.append(lbl)
                    # now, go through each chunk of the simulation results to build up the probabilities grid
                    # we need to reset the simulation_res.iter_chunks() and simulation_types_chunked iterators
                    simulation_types_chunked = self.mcsimulation.query_chunks('cell_type')
                    for num, (idx, chunk_types) in enumerate(zip(simulation_res.iter_chunks(), simulation_types_chunked)):
                        chunk = simulation_res[idx]
                        # go through each starting point / pathway combo
                        for path_num, (starting_coords, starting_level, sub_types) in enumerate(zip(starting_coords_arr, starting_levels, sub_types_arr)):
                            # get the chunk paths and types
                            chunk_paths = chunk[0, starting_level]
                            # how many of these involve the starting point?
                            is_starting_point = chunk_types == sub_types
                            # get the unique nodes and their counts
                            nodes, counts = np.unique(chunk_paths, return_counts=True)
                            nodes_included = np.isin(nodes, starting_coords.index.values)
                            # only include those nodes that are in starting_coords -- this mostly skips the ones on the other side of the brain
                            nodes, counts = nodes[nodes_included], counts[nodes_included]
                            ps, qs = starting_coords.loc[nodes][['p', 'q']].values.T
                            # store the probabilities in the probs_grid
                            probs_grid[num, ps - pmin, qs - qmin] += counts / is_starting_point.sum()    # get the starting level
                # generate a SynapticField 
                field = SynapticField(probs_grid, ps=pvals, qs=qvals)
                # fix key for regex mode
                key = path_lbl
                if self.regex_mode:
                    key = regex_key
                fields[key] = field
                # and combine with the overall input field
                # if self.regex_mode:
                #     field = field.sum(0)
                if all_fields is None:
                    all_fields = field
                else:
                    all_fields = all_fields + field
                # in regex mode, store the sub-pathways
                if self.regex_mode:
                    sub_pathways[key] = sub_paths
                # print progress
                print_progress(path_num + 1, len(info['probs']), prefix="Calculating synaptic fields per pathway:")

        fields['all'] = all_fields
        effects['all'] = np.median(list(effects.values()))
        self.synaptic_fields = SynapticFields(fields)
        self.synaptic_effects = effects
        if self.regex_mode:
            self.synaptic_sub_pathways = sub_pathways
            return self.synaptic_fields, self.synaptic_sub_pathways
        return self.synaptic_fields, self.synaptic_effects

    def _build_transition_matrix(self, normalize=True):
        """Build a sparse transition matrix matching the walk direction.

        Parameters
        ----------
        normalize : bool, default=True
            If True, each row is divided by its sum so that the matrix is row-stochastic
            (probability interpretation). If False, raw synapse weights are preserved.

        Returns
        -------
        P : scipy.sparse.csr_matrix, shape (N, N)
            P[i, j] is the (optionally normalized) edge weight from node index i to j,
            where node indices are positions in the sorted ``self.node_ids``. Weights
            are taken from the graph edge 'weight' attribute. When normalize=True each
            row sums to ≤ 1 (rows for dead-end nodes sum to 0). When normalize=False
            each row sums to the total outgoing synapse count for that node.
            Downstream walks move pre -> post; upstream walks move post -> pre.
        """
        import scipy.sparse as sp
        node_ids = self.node_ids
        N = len(node_ids)
        # gather weighted edges from the graph (u -> v with weight w)
        us, vs, ws = [], [], []
        for u, v, w in self.graph.edges(data='weight'):
            us.append(u)
            vs.append(v)
            ws.append(1.0 if w is None else float(w))
        if len(us) == 0:
            return sp.csr_matrix((N, N), dtype=float)
        us = np.asarray(us)
        vs = np.asarray(vs)
        ws = np.asarray(ws, dtype=float)
        u_idx = np.searchsorted(node_ids, us)
        v_idx = np.searchsorted(node_ids, vs)
        # a walker moves along the direction of the walk (downstream: pre->post, upstream: post->pre)
        if self.direction == 'downstream':
            rows, cols = u_idx, v_idx
        else:
            rows, cols = v_idx, u_idx
        # duplicate (row, col) entries are summed by csr_matrix, matching aggregated weights
        P = sp.csr_matrix((ws, (rows, cols)), shape=(N, N))
        if normalize:
            # row-normalize so each row is a probability distribution
            row_sums = np.asarray(P.sum(axis=1)).ravel()
            inv = np.zeros_like(row_sums)
            nonzero = row_sums > 0
            inv[nonzero] = 1.0 / row_sums[nonzero]
            P = sp.diags(inv) @ P
        return P.tocsr()

    @staticmethod
    def _propagate_first_passage(P, absorbing_idx, target_idx, num_hops,
                                 by_hop=False, progress_prefix=None):
        """First-passage mass propagation through the transition matrix.

        Starts one unit of mass at each target row and spreads it with ``P`` for
        ``num_hops`` hops, with the outgoing edges of absorbing nodes zeroed so mass stops on
        first arrival. Returns ``absorbed`` (num_targets, N): the total mass absorbed at each
        node.

        When ``by_hop`` is True, also returns ``hop_absorbed`` -- a list of
        (num_targets, len(absorbing_idx)) arrays, one per hop (0..num_hops), giving the mass
        that *first* reaches each absorbing node at that hop. Only the absorbing columns are
        stored per hop (not the full N), so the per-hop record stays small; by construction
        the entries sum to ``absorbed[:, absorbing_idx]``.
        """
        import scipy.sparse as sp
        N = P.shape[0]
        absorbing_idx = np.asarray(absorbing_idx)
        target_idx = np.asarray(target_idx)
        is_absorbing = np.zeros(N, dtype=bool)
        is_absorbing[absorbing_idx] = True
        # zero the outgoing rows of absorbing nodes so mass stops on first arrival
        P_eff = (sp.diags((~is_absorbing).astype(float)) @ P).tocsr()
        num_targets = len(target_idx)
        dist = sp.csr_matrix((np.ones(num_targets), (np.arange(num_targets), target_idx)),
                             shape=(num_targets, N))
        absorbed = np.zeros((num_targets, N), dtype=np.float64)
        hop_absorbed = [] if by_hop else None
        for hop in range(num_hops):
            cur = dist[:, absorbing_idx].toarray()
            absorbed[:, absorbing_idx] += cur
            if by_hop:
                hop_absorbed.append(cur)
            dist = dist @ P_eff
            if progress_prefix is not None:
                print_progress(hop + 1, num_hops, prefix=progress_prefix, suffix="Complete")
        # capture mass that arrives on the final hop
        final = dist[:, absorbing_idx].toarray()
        absorbed[:, absorbing_idx] += final
        if by_hop:
            hop_absorbed.append(final)
        return (absorbed, hop_absorbed) if by_hop else absorbed

    def _get_synaptic_fields_analytic(self, stopping_points=['L1', 'L2', 'L3', 'L4', 'L5', 'R7', 'R8'],
                                      side='right', conditional=False, num_hops=None, normalize=True,
                                      by_hop=False):
        """Analytic synaptic fields via first-passage propagation of the transition matrix.

        Computes the same marginal quantity the simulation estimates by counting -- the
        probability that a walk from each target (level-0) cell first reaches an input cell
        of a given type at retinal coordinate (p, q) -- but exactly and without a Monte
        Carlo run. Does not require get_pathways()/monte_carlo() to have been called.

        Parameters
        ----------
        stopping_points : list of str
            Cell types treated as absorbing input layers (a walk stops at first arrival).
        side : str, default='right'
            The eye side ('left' or 'right') whose columns provide retinal coordinates.
        conditional : bool, default=False
            If True, normalize each target's field by its total mass reaching any stopping
            point (distribution over origins given the input layer was reached). If False,
            fields are absolute first-passage probabilities (or raw synapse-weighted totals
            when normalize=False).
        num_hops : int, optional
            Number of propagation hops. Defaults to the maximum absolute level in node_info
            (the graph depth).
        normalize : bool, default=True
            If True, the transition matrix is row-stochastic and output values are
            probabilities. If False, raw synapse weights are used without row-normalization;
            output values represent total synapse-weighted flow to each stopping-point cell,
            preserving differences in absolute connectivity strength across ommatidia.
        by_hop : bool, default=False
            If True, additionally build one SynapticFields per hop (the first-passage mass
            arriving at each hop) and store the list on ``self.synaptic_fields_by_hop``; the
            per-hop fields sum to the returned total field. With ``normalize=True`` these are
            per-hop probabilities, with ``normalize=False`` per-hop synapse-weighted path
            counts -- the quantities needed to image how the path counts / contributions
            build up hop by hop. Only the absorbing-node columns are kept during propagation,
            but the assembled per-hop fields still cost ~(num_hops + 1)x the base field
            storage, so it is opt-in.

        Returns
        -------
        synaptic_fields : SynapticFields
            One SynapticField (shape (num_targets, width, height)) per stopping-point type,
            plus an 'all' entry summing across types.
        effects : dict
            Placeholder neurotransmitter effect per field (1.0); analytic mode does not
            compute the trajectory-dependent transmitter signs that the simulation does.
        """
        import scipy.sparse as sp
        # ensure the visual column assignment file is available
        if not os.path.exists("column_assignment.csv"):
            import urllib.request
            url = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz"
            urllib.request.urlretrieve(url, "column_assignment.csv.gz")
            with gzip.open("column_assignment.csv.gz", 'rb') as f_in:
                with open('column_assignment.csv', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        visual_column_info = pd.read_csv("column_assignment.csv", index_col=0)
        # retinal-coordinate grid bounds (shared with the simulation backend)
        all_ps, all_qs = visual_column_info[['p', 'q']].values.T
        pmin, pmax = int(all_ps.min()), int(all_ps.max()) + 1
        qmin, qmax = int(all_qs.min()), int(all_qs.max()) + 1
        width, height = pmax - pmin, qmax - qmin
        pvals, qvals = np.meshgrid(np.arange(pmin, pmax), np.arange(qmin, qmax), indexing='ij')
        # sorted node ids and the row-stochastic transition matrix
        node_ids = self.node_ids
        N = len(node_ids)
        P = self._build_transition_matrix(normalize=normalize)
        # target (level-0) cells form the first axis of each field
        initial = self.node_info[self.node_info.level == 0]
        target_ids = initial.root_id.values
        assert len(target_ids) > 0, "No level-0 (target) cells found in node_info."
        target_idx = np.searchsorted(node_ids, target_ids)
        num_targets = len(target_idx)
        # number of hops defaults to the graph depth
        if num_hops is None:
            num_hops = int(np.abs(self.node_info.level.values).max())
        num_hops = max(int(num_hops), 1)
        # collect stopping cells and their retinal coordinates per type
        stopping_points = np.asarray(stopping_points)
        types_col = visual_column_info['type']
        hemi_col = visual_column_info['hemisphere']
        type_cells = {}
        absorbing_list = []
        for cell_type in stopping_points:
            sel = visual_column_info[(types_col == cell_type) & (hemi_col == side)]
            if len(sel) == 0:
                continue
            in_graph = np.isin(sel.index.values, node_ids)
            sel = sel[in_graph]
            if len(sel) == 0:
                continue
            idx = np.searchsorted(node_ids, sel.index.values)
            ps = sel['p'].values.astype(int)
            qs = sel['q'].values.astype(int)
            type_cells[cell_type] = (idx, ps, qs)
            absorbing_list.append(idx)
        assert len(absorbing_list) > 0, (
            f"None of the stopping points {list(stopping_points)} were found on the "
            f"'{side}' side within this Paths graph.")
        absorbing_idx = np.unique(np.concatenate(absorbing_list))
        is_absorbing = np.zeros(N, dtype=bool)
        is_absorbing[absorbing_idx] = True
        # propagate first-passage mass (optionally recording each hop's arrivals, kept only
        # over the absorbing-node columns to bound memory)
        if by_hop:
            absorbed, hop_absorbed = self._propagate_first_passage(
                P, absorbing_idx, target_idx, num_hops, by_hop=True,
                progress_prefix="Analytic synaptic fields:")
        else:
            absorbed = self._propagate_first_passage(
                P, absorbing_idx, target_idx, num_hops,
                progress_prefix="Analytic synaptic fields:")
        # optional conditioning on reaching any stopping point
        if conditional:
            denom = absorbed[:, absorbing_idx].sum(axis=1)
            denom[denom == 0] = 1.0
        else:
            denom = np.ones(num_targets)

        # positions of each type's cells within the (sorted) absorbing-node columns, so the
        # compact per-hop arrays can be scattered with the same code path as the total
        abs_pos = {ct: np.searchsorted(absorbing_idx, idx)
                   for ct, (idx, _ps, _qs) in type_cells.items()}

        def _fields_from_absorbing(abs_cols):
            """Scatter an (num_targets, len(absorbing_idx)) array into per-type fields."""
            ff = {}
            all_grid = np.zeros((num_targets, width, height), dtype=np.float32)
            for cell_type, (idx, ps, qs) in type_cells.items():
                # when normalize=False, denom is ones and abs_cols holds raw synapse-weighted
                # flow, so no further division is applied here
                mass = abs_cols[:, abs_pos[cell_type]] / denom[:, None]
                grid = np.zeros((num_targets, width, height), dtype=np.float32)
                for col in range(len(idx)):
                    grid[:, ps[col] - pmin, qs[col] - qmin] += mass[:, col]
                ff[cell_type] = SynapticField(grid, ps=pvals, qs=qvals)
                all_grid += grid
            ff['all'] = SynapticField(all_grid, ps=pvals, qs=qvals)
            return SynapticFields(ff)

        self.synaptic_fields = _fields_from_absorbing(absorbed[:, absorbing_idx])
        effects = {key: 1.0 for key in self.synaptic_fields.keys()}
        # per-hop fields: first-passage arrivals at each hop (they sum to the total field)
        self.synaptic_fields_by_hop = (
            [_fields_from_absorbing(h) for h in hop_absorbed] if by_hop else None)
        self.synaptic_effects = effects
        return self.synaptic_fields, self.synaptic_effects

    # ------------------------------------------------------------------
    # Vectorized helper: one matrix-MC hop for many walkers
    # ------------------------------------------------------------------
    @staticmethod
    def _csr_sample_step(indptr, indices, data, rowsum, walker_idx, rng):
        """Sample one hop for W walkers from a CSR transition matrix.

        Parameters
        ----------
        indptr, indices, data : CSR arrays of the row-stochastic matrix P (or P_eff).
        rowsum : (N,) array of per-row probability sums (< 1 for leaky rows).
        walker_idx : (W,) int64 array of current node positions (-1 = dead).
        rng : np.random.Generator

        Returns
        -------
        next_idx : (W,) int64 — next node indices; -1 for dead/dead-end walkers.
        survived : (W,) bool — False for walkers that leaked out or were already dead.
        """
        W = len(walker_idx)
        next_idx = np.full(W, -1, dtype=np.int64)
        survived = np.zeros(W, dtype=bool)

        alive = walker_idx >= 0
        if not alive.any():
            return next_idx, survived

        active = np.where(alive)[0]
        cur = walker_idx[active]

        # Leakage: walkers die with probability 1 - rowsum[cur]
        u = rng.random(len(active))
        alive_after_leak = u < rowsum[cur]
        if not alive_after_leak.any():
            return next_idx, survived

        act2 = active[alive_after_leak]
        cur2 = cur[alive_after_leak]
        u2 = u[alive_after_leak]        # reuse the same draw, rescaled into [0, rowsum]

        # Group by current node for batch inverse-CDF sampling
        order = np.argsort(cur2, kind='stable')
        cur2_o = cur2[order]
        u2_o = u2[order]
        uniq, ustarts = np.unique(cur2_o, return_index=True)
        ustarts = np.append(ustarts, cur2_o.size)

        nxt = np.empty(cur2_o.size, dtype=np.int64)
        for ki in range(len(uniq)):
            node = uniq[ki]
            lo, hi = ustarts[ki], ustarts[ki + 1]
            a, b = int(indptr[node]), int(indptr[node + 1])
            if a == b:          # dead-end
                nxt[lo:hi] = -1
                continue
            row_probs = data[a:b]
            cumprobs = np.cumsum(row_probs)
            cumprobs[-1] = rowsum[node]   # guard against float rounding
            jj = np.searchsorted(cumprobs, u2_o[lo:hi], side='right')
            jj = np.clip(jj, 0, b - a - 1)
            nxt[lo:hi] = indices[a + jj]

        inv_order = np.argsort(order, kind='stable')
        nxt = nxt[inv_order]

        valid = nxt >= 0
        next_idx[act2[valid]] = nxt[valid]
        survived[act2[valid]] = True
        return next_idx, survived

    def _get_synaptic_fields_matrix_mc(self,
                                       stopping_points=None,
                                       side='right',
                                       conditional=False,
                                       num_hops=None,
                                       reps=5000,
                                       seed=None,
                                       normalize=True):
        """Matrix-based Monte Carlo synaptic fields.

        Estimates the same first-passage probabilities as
        ``_get_synaptic_fields_analytic`` but by sampling: each target cell
        launches ``reps`` walkers which step according to the row-stochastic
        transition matrix until they are absorbed by a stopping-point cell or
        exhaust ``num_hops`` steps.  Sampling is vectorized across all walkers
        at each hop via inverse-CDF on pre-sorted CSR rows.

        Parameters
        ----------
        stopping_points : list of str, optional
            Cell types treated as absorbing input layers.  Defaults to
            ``['L1', 'L2', 'L3', 'L4', 'L5', 'R7', 'R8']``.
        side : str, default='right'
            Eye hemisphere whose column coordinates are used.
        conditional : bool, default=False
            If True, normalize each target's field by its total absorbed mass.
        num_hops : int, optional
            Propagation depth; defaults to the graph depth in ``node_info``.
        reps : int, default=5000
            Number of Monte Carlo walkers launched per target cell.
        seed : int or None
            Random seed for reproducibility.
        normalize : bool, default=True
            If True, output values are probabilities (counts / reps, or counts /
            total absorbed when conditional=True). If False, output values are raw
            walker-hit counts (integers as float32), preserving absolute differences
            in how many paths reach each ommatidium.

        Returns
        -------
        synaptic_fields : SynapticFields
        effects : dict
        """
        import scipy.sparse as sp

        if stopping_points is None:
            stopping_points = ['L1', 'L2', 'L3', 'L4', 'L5', 'R7', 'R8']

        rng = np.random.default_rng(seed)

        # --- retinal grid setup (shared with the analytic backend) ---
        if not os.path.exists("column_assignment.csv"):
            import urllib.request
            url = ("https://storage.googleapis.com/flywire-data/codex/data/"
                   "fafb/783/column_assignment.csv.gz")
            urllib.request.urlretrieve(url, "column_assignment.csv.gz")
            with gzip.open("column_assignment.csv.gz", 'rb') as f_in:
                with open('column_assignment.csv', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        visual_column_info = pd.read_csv("column_assignment.csv", index_col=0)
        all_ps, all_qs = visual_column_info[['p', 'q']].values.T
        pmin, pmax = int(all_ps.min()), int(all_ps.max()) + 1
        qmin, qmax = int(all_qs.min()), int(all_qs.max()) + 1
        width, height = pmax - pmin, qmax - qmin
        pvals, qvals = np.meshgrid(np.arange(pmin, pmax),
                                   np.arange(qmin, qmax), indexing='ij')

        # --- transition matrix and absorbing nodes ---
        node_ids = self.node_ids
        N = len(node_ids)
        P = self._build_transition_matrix()          # row-stochastic CSR

        stopping_points = np.asarray(stopping_points)
        types_col = visual_column_info['type']
        hemi_col  = visual_column_info['hemisphere']
        type_cells = {}
        absorbing_list = []
        for cell_type in stopping_points:
            sel = visual_column_info[(types_col == cell_type) & (hemi_col == side)]
            if len(sel) == 0:
                continue
            in_graph = np.isin(sel.index.values, node_ids)
            sel = sel[in_graph]
            if len(sel) == 0:
                continue
            idx = np.searchsorted(node_ids, sel.index.values)
            ps  = sel['p'].values.astype(int)
            qs  = sel['q'].values.astype(int)
            type_cells[cell_type] = (idx, ps, qs)
            absorbing_list.append(idx)

        assert len(absorbing_list) > 0, (
            f"None of the stopping points {list(stopping_points)} were found on "
            f"the '{side}' side within this Paths graph.")

        absorbing_idx = np.unique(np.concatenate(absorbing_list))
        is_absorbing  = np.zeros(N, dtype=bool)
        is_absorbing[absorbing_idx] = True

        # P_eff: zero outgoing edges of absorbing nodes so walkers stop on arrival
        P_eff = (sp.diags((~is_absorbing).astype(float)) @ P).tocsr()
        indptr   = P_eff.indptr
        nb_idx   = P_eff.indices
        nb_data  = P_eff.data
        rowsum   = np.asarray(P_eff.sum(axis=1)).ravel()   # 0 at absorbing rows

        # Provide a rowsum of 1 for absorbing rows so the leak check fires
        # immediately (walker is already absorbed; the step function never
        # reaches sampling for those nodes anyway, but this keeps the logic clean).
        rowsum_eff = rowsum.copy()
        rowsum_eff[absorbing_idx] = 1.0   # treated specially below

        # --- target (level-0) cells ---
        initial     = self.node_info[self.node_info.level == 0]
        target_ids  = initial.root_id.values
        assert len(target_ids) > 0, "No level-0 (target) cells found in node_info."
        target_idx  = np.searchsorted(node_ids, target_ids)
        num_targets = len(target_idx)

        if num_hops is None:
            num_hops = int(np.abs(self.node_info.level.values).max())
        num_hops = max(int(num_hops), 1)

        # --- map absorbing node -> type ---
        node_to_type_idx = {}      # node_index -> list of positions in type_cells arrays
        for cell_type, (t_idx, t_ps, t_qs) in type_cells.items():
            for col, ni in enumerate(t_idx):
                node_to_type_idx.setdefault(int(ni), []).append((cell_type, col))

        # --- run simulation ---
        # absorbed_count[cell_type][t, c] = times target t was absorbed at node c of type
        absorbed_count = {ct: np.zeros((num_targets, len(t_idx)), dtype=np.int64)
                          for ct, (t_idx, _, _) in type_cells.items()}

        # current[t, r] = current node index for target t, rep r
        current = np.broadcast_to(target_idx[:, None],
                                   (num_targets, reps)).copy().astype(np.int64)
        alive   = np.ones((num_targets, reps), dtype=bool)

        for hop in range(num_hops + 1):
            # --- absorb walkers that have reached a stopping-point node ---
            for cell_type, (t_idx, _, _) in type_cells.items():
                for col, ni in enumerate(t_idx):
                    at_node = alive & (current == ni)
                    if at_node.any():
                        absorbed_count[cell_type][:, col] += at_node.sum(axis=1)
                        alive[at_node] = False

            if hop == num_hops or not alive.any():
                break

            # --- advance every still-alive walker one hop ---
            # Process each target independently to keep memory bounded
            for t in range(num_targets):
                alive_t = alive[t]                 # (reps,) bool
                if not alive_t.any():
                    continue
                cur_t = current[t]                 # (reps,) int64
                walker_pos = cur_t.copy()
                walker_pos[~alive_t] = -1          # mark dead walkers

                nxt, surv = Paths._csr_sample_step(
                    indptr, nb_idx, nb_data, rowsum, walker_pos, rng)

                alive[t] &= surv
                advanced = alive[t]
                current[t, advanced] = nxt[advanced]

            print_progress(hop + 1, num_hops,
                           prefix="Matrix-MC synaptic fields:", suffix="Complete")

        # --- build spatial grids ---
        reps_f = float(reps)
        fields    = {}
        all_grid  = np.zeros((num_targets, width, height), dtype=np.float32)

        for cell_type, (t_idx, t_ps, t_qs) in type_cells.items():
            counts = absorbed_count[cell_type]           # (num_targets, n_type_cells)
            if not normalize:
                # raw hit counts — skip all normalization
                reach = counts.astype(np.float32)
            elif conditional:
                denom = counts.sum(axis=1, keepdims=True).astype(float)
                denom[denom == 0] = 1.0
                reach = counts / denom
            else:
                reach = counts / reps_f                  # (num_targets, n_type_cells)
            grid   = np.zeros((num_targets, width, height), dtype=np.float32)
            for col in range(len(t_idx)):
                grid[:, t_ps[col] - pmin, t_qs[col] - qmin] += reach[:, col]
            fields[cell_type] = SynapticField(grid, ps=pvals, qs=qvals)
            all_grid += grid

        fields['all'] = SynapticField(all_grid, ps=pvals, qs=qvals)
        effects = {key: 1.0 for key in fields}
        self.synaptic_fields = SynapticFields(fields)
        self.synaptic_effects = effects
        return self.synaptic_fields, self.synaptic_effects

    def save(self, fn):
        """Save this Paths object to an HDF5 file (using h5py).

        The file will contain:
        - /edges/<column> datasets for each column of edges_df
        - /edges/_index dataset for the dataframe index
        - file attribute 'direction'
        - /monte_carlo/paths dataset (if monte_carlo_paths exists)
        - /monte_carlo/node_ids dataset (if monte_carlo_node_ids exists)

        Parameters
        ----------
        fn : str
            Output HDF5 filename.
        """
        def _write_dataframe_to_group(group, df):
            # save columns as datasets and preserve order
            group.attrs['columns'] = np.array(df.columns.astype(str).tolist(), dtype='S')
            # save index
            try:
                index_vals = df.index.to_numpy()
            except Exception:
                index_vals = np.arange(len(df))
            # index may be non-numeric; store as strings
            group.create_dataset('_index', data=index_vals)
            for col in df.columns:
                data = df[col][:1000].to_numpy()
                # detect string/object dtype
                if data.dtype == object or pd.api.types.is_string_dtype(data) or pd.api.types.is_categorical_dtype(data):
                    # store as variable-length utf-8 strings
                    dt = h5py.string_dtype(encoding='utf-8')
                    # chunk_size = 100000  # adjust as needed for memory
                    n = len(df[col])
                    dset = group.create_dataset(col, shape=(n,), dtype=dt, chunks=True)
                    # for start in range(0, n, chunk_size):
                    for idx in dset.iter_chunks():
                        start = idx[0].start
                        stop = idx[0].stop
                        # chunk = data[start:stop]
                        chunk = df[col][start:stop].to_numpy()
                        # chunk_to_store = np.array([None if pd.isna(x) else str(x) for x in chunk], dtype=object)
                        dset[idx] = chunk.astype('S')
                else:
                    # numeric or boolean or datetime (dates will be converted to ISO strings by caller if needed)
                    try:
                        # chunk_size = 100000  # adjust as needed for memory
                        n = len(df[col])
                        group.create_dataset(col, shape=(n,), dtype=data.dtype, chunks=True)
                        for idx in group[col].iter_chunks():
                            start = idx[0].start
                            stop = idx[0].stop
                            chunk = df[col][start:stop].to_numpy()
                            group[col][idx] = chunk
                    except Exception:
                        # fallback to string storage
                        dt = h5py.string_dtype(encoding='utf-8')
                        n = len(data)
                        dset = group.create_dataset(col, shape=(n,), dtype=dt, chunks=True)
                        for idx in group[col].iter_chunks():
                            start = idx[0].start
                            stop = idx[0].stop
                            chunk = data[start:stop]
                            # chunk_to_store = np.array([str(x) for x in chunk], dtype='S')
                            dset[start:stop] = chunk.astype('S')
        # Ensure we have a writable HDF5 handle to the requested filename.
        # If an HDF5 file is already attached in write mode, reuse it; otherwise open (or create) in append mode.
        reopen = True
        if hasattr(self, '_h5file'):
            if self._h5file is not None:
                try:
                    if self._h5file and self._h5file.mode in ['r+', 'a']:
                        reopen = False
                    else:
                        self._h5file.close()
                except Exception:
                    # If the handle is invalid/closed, fall back to reopening
                    pass
        if reopen:
            # Use append so the file is created if it doesn't exist yet
            self._h5file = h5py.File(fn, 'a')
        f = self._h5file
        # edges dataframe
        # Replace any existing 'edges' group to avoid duplicate-name errors on repeated saves
        breakpoint()
        if 'edges' not in f:
            edges_grp = f.create_group('edges')
            _write_dataframe_to_group(edges_grp, self.edges_df)
        if 'direction' not in f.attrs:
            # write direction as attribute
            try:
                f.attrs['direction'] = self.node_info.level.iloc[0] if False else self.__dict__.get('direction', '')
            except Exception:
                # store provided direction if available
                f.attrs['direction'] = self.__dict__.get('direction', '')
            # Prefer storing explicit attribute if available on object
            if hasattr(self, 'direction'):
                f.attrs['direction'] = self.direction
        # Mark as saved so close() won't remove the file on object deletion
        self._saved = True
        # Track the on-disk file path for safety in close()
        self._temp_h5_path = fn
        # Flush to ensure data is written
        try:
            f.flush()
        except Exception:
            pass
        # also, use pickle to store the networkx graph
        import pickle
        
        graph_fn = fn.replace('.h5', '_graph.pkl')
        with open(graph_fn, 'wb') as graph_file:
            pickle.dump(self.graph, graph_file)
        # mark as saved
        self._saved = True

    def get_main_pathways(self, **monte_carlo_kwargs):
        """Partition the paths into different visual columns and types.
        """
        # if a monte carlo simulation hasn't been run yet, run it now
        if not hasattr(self, 'monte_carlo_paths'):
            self.monte_carlo(**monte_carlo_kwargs)

    def __del__(self):
        """Ensure HDF5 file is closed on deletion."""
        self.close()

    # TODO: make a function for generating synaptic fields from a Paths object
    # def get_synaptic_fields(self, center=(0, 0), top_pathways=10):
    # refer to the prob_grid parts of doc_figures.ipynb


class SynapticField(np.ndarray):
    """SynapticField with a 2D synaptic field.

    Parameters
    ----------
    field_array : numpy.ndarray
        A 2D numpy array representing the synaptic field.
    centered : bool, optional
        Whether the field is already centered. Default is False.
    indices : numpy.ndarray, optional
        An array of shape (p, q, 2) representing the p- and q- indices for each position in the field.
    ps : numpy.ndarray, optional
        An array of p-coordinates corresponding to the retinal coordinates of the field.
    qs : numpy.ndarray, optional
        An array of q-coordinates corresponding to the retinal coordinates of the field.

    Methods
    -------
    __new__(cls, input_array, centered=False)
        Create a new SynapticField instance.
    __array_finalize__(self, obj)
        Finalize the array attributes.
    project(method='retina')
        Project the synaptic field using the specified method.
    center

    """
    def __new__(cls, input_array, qs, ps, centered=False, indices=None):
        # input array must have the same shape as ps and qs
        if input_array.shape[-2:] != ps.shape[-2:] or input_array.shape[-2:] != qs.shape[-2:]:
            raise ValueError("input_array, ps, and qs must have the same shape.")
        obj = np.asarray(input_array).view(cls)
        obj.centered = centered
        obj.ps, obj.qs = ps, qs
        # make an array of p- and q- indices
        if indices is None:
            indices = np.array(np.meshgrid(np.arange(input_array.shape[1]), np.arange(input_array.shape[2]), indexing='xy'))
            indices = indices.transpose(2, 1, 0)
        obj.indices = indices
        # load the retinal-to-visual coordinate table if it is available (optional; the
        # attribute is not required to construct or use the field itself)
        if os.path.exists("mi1_visual_data.csv"):
            obj.visual_info = pd.read_csv("mi1_visual_data.csv")
        else:
            obj.visual_info = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.centered = getattr(obj, 'centered', False)
        self.indices = getattr(obj, 'indices', None)
        self.ps = getattr(obj, 'ps', None)
        self.qs = getattr(obj, 'qs', None)
        self.root_ids = getattr(obj, 'root_ids', None)

    def center(self, method='mean', window=(50, 50)):
        """Center the synaptic field based on a specified method.

        Parameters
        ----------
        method : str, list, np.ndarray optional
            The method to use for centering. Default is 'mean'. If a list or array of indices is provided,
            those indices will be used as the centers for each frame.
        window : tuple, optional
            The window size for the centered field. Default is (50, 50).
        """
        if isinstance(method, (list, np.ndarray)):
            centers = np.array(method)
            num_frames = self.shape[0]
            assert centers.shape == (self.shape[0], 2), f"If method is an array of centers, it must have shape ({num_frames}, 2)."
        elif isinstance(method, str):
            if method == 'mean':
                # get the center of mass for each frame using the probabilities 
                # as weights and indices
                centers = (self.indices[None] * self[..., None]).sum((1, 2))/self.sum((1, 2))[:, None]
            if method == 'max':
                # get the indices of the max value for each frame
                max_vals = self.max(axis=(1, 2))
                peaks = self[:] == max_vals[:, None, None]
                # get the indices of the max value for each frame
                inds = np.where(peaks)
                centers = self.indices[inds[1], inds[2]]
        # now, re-center the field based on centers
        # we'll need to start with an empty array with a prescribed window size
        centered_field = np.zeros((self.shape[0], window[0], window[1]), dtype=self.dtype)
        # centered_indices = np.zeros((self.shape[0], window[0], window[1], 2), dtype=self.indices.dtype)
        centered_ps, centered_qs = np.zeros((self.shape[0], window[0], window[1])), np.zeros((self.shape[0], window[0], window[1]))
        half_window = (window[0] // 2, window[1] // 2)
        for i, center in enumerate(centers):
            if not np.any(np.isnan(center)):
                center_p, center_q = center
                start_p = int(center_p) - half_window[0]
                start_q = int(center_q) - half_window[1]
                end_p = start_p + window[0]
                end_q = start_q + window[1]
                # determine the slice of the original array to copy
                src_start_p = max(0, start_p)
                src_start_q = max(0, start_q)
                src_end_p = min(self.shape[1], end_p)
                src_end_q = min(self.shape[2], end_q)
                # determine the slice of the centered array to paste into
                dest_start_p = src_start_p - start_p
                dest_start_q = src_start_q - start_q
                dest_end_p = dest_start_p + (src_end_p - src_start_p)
                dest_end_q = dest_start_q + (src_end_q - src_start_q)
                # copy the data
                centered_field[i, dest_start_p:dest_end_p, dest_start_q:dest_end_q] = self[i, src_start_p:src_end_p, src_start_q:src_end_q]
                # copy the indices
                # centered_indices[i, dest_start_p:dest_end_p, dest_start_q:dest_end_q] = self.indices[src_start_p:src_end_p, src_start_q:src_end_q]
                # copy and center the ps and qs
                centered_ps[i, dest_start_p:dest_end_p, dest_start_q:dest_end_q] = self.ps[src_start_p:src_end_p, src_start_q:src_end_q] - center_p
                centered_qs[i, dest_start_p:dest_end_p, dest_start_q:dest_end_q] = self.qs[src_start_p:src_end_p, src_start_q:src_end_q] - center_q
        self.centers = centers
        # now, return a new SynapticField with the centered data
        return SynapticField(centered_field, centered=True, ps=centered_ps, qs=centered_qs)

    def plot(self, type='hex', summary_func='sum', axes=None, fwhm=False, mask=None, **plot_kwargs):
        """Plot the synaptic field using some preset styles.

        Note: This assumes that the synaptic field has already been reduced to a 2D array.
        If not, this will apply the summary_func to reduce it or, in the case of 'tile', 
        will plot each frame as a separate subplot.

        Parameters
        ----------
        type : str, optional
            The type of plot to create. Options:
                - 'hex': Hexbin plot of the synaptic field.
                - 'contour': Contour plot of the synaptic field.
                - 'scatter': Scatter plot of the synaptic field using circle markers.
                - 'voronoi': Voronoi diagram of the synaptic field using retinal or visual
                coordinates.
        summary_func : str or function, optional
            The summary function to apply to the synaptic field before plotting. If 
            'tile', will plot each frame as a separate subplot. If a function, will
            apply that function to the array to reduce the field to 2D.
        axes : matplotlib.axes.Axes or array-like of Axes, optional
            The axes to plot on. If None, will create new figure and axes. If 
            summary_func is 'tile', should be an array-like of Axes with one Axes per frame.
        fwhm : bool, optional
            Whether to plot the full-width at half-maximum (FWHM) contour of the synaptic field
            along the horizontal and vertical axes.
        mask : numpy.ndarray, optional
            A boolean mask to apply to the synaptic field before plotting. Must be the same shape
        plot_kwargs : dict, optional
            Additional keyword arguments to pass to the plotting function.
        """
        # choose the appropriate plotting function
        if type == 'hex':
            plot_func = plot_hex_field
        elif type == 'contour':
            plot_func = plot_contour_field

        # if axes is just one axis, make it into a list
        if axes is not None and not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        if axes is None:
            axes = [plt.gca()]
        # reduce the field to 2D if necessary
        if self.ndim > 2:
            if summary_func == 'tile':
                # make sure axes is an array-like of Axes with one Axes per frame
                if axes is None:
                    fig, axes = plt.subplots(1, self.shape[0], figsize=(self.shape[0]*3, 3))
                elif len(axes) != self.shape[0]:
                    raise ValueError(f"If summary_func is 'tile', axes must have one Axes per frame. Instead there are {len(axes)} axes for {self.shape[0]} frames.")
                plots = []
                for i in range(self.shape[0]):
                    ax = axes[i]
                    ax.set_title(f"Frame {i}")
                    plots += [plot_func(self[i], self.ps, self.qs, ax=ax, **plot_kwargs)]
                return plots
            else:
                if callable(summary_func):
                    field_2d = summary_func(self, axis=0)
                else:
                    raise ValueError("summary_func must be 'tile' or a callable function.")
        else:
            field_2d = self
        if self.centered:
            ps, qs = self.indices[..., 0], self.indices[..., 1]
            ps -= np.round(ps.mean()).astype(ps.dtype)
            qs -= np.round(qs.mean()).astype(qs.dtype)
        else:
            ps, qs = self.ps, self.qs
        cmap = plot_kwargs.pop('cmap', 'viridis')
        norm = plot_kwargs.pop('norm', None)
        # make a normalized colormap using the provided norm and cmap
        if mask is not None:
            ret = plot_func(field_2d[mask], ps[mask], qs[mask], ax=axes[0] if axes is not None else plt.gca(), cmap=cmap, norm=norm, **plot_kwargs)
        else:
            ret = plot_func(field_2d, ps, qs, ax=axes[0] if axes is not None else plt.gca(), cmap=cmap, norm=norm, **plot_kwargs)

        if fwhm:
            # get the full-width at half maximum along the x- and y-axes
            # convert ps and qs to xs and ys
            xs, ys = lattice_to_retinal(ps, qs)
            # now, let's find the maximum value of self along each
            # bin the x- and y-values
            x_set = np.unique(np.round(xs, 1))
            y_set = np.unique(np.round(ys, 1))
            # now, decide on bin edges for x and y
            x_bins = np.linspace(x_set.min(), x_set.max(), len(x_set) + 1)
            y_bins = np.linspace(y_set.min(), y_set.max(), len(y_set) + 1)
            # use this partition to find the maximum value in all_gradient for each bin
            binned_max = np.zeros((len(y_bins)-1, len(x_bins)-1))
            x_mid, y_mid = (x_bins[:-1] + x_bins[1:])/2, (y_bins[:-1] + y_bins[1:])/2
            # do this for x and y seperately
            for i in range(len(x_bins)-1):
                for j in range(len(y_bins)-1):
                    in_bin = (xs >= x_bins[i]) & (xs < x_bins[i+1]) & (ys >= y_bins[j]) & (ys < y_bins[j+1])
                    if in_bin.any():
                        binned_max[j, i] = np.abs(field_2d[in_bin]).max()
            binned_max /= binned_max.max()
            # find the FWHM for x and y
            max_xs, max_ys = binned_max.max(0), binned_max.max(1)
            half_max = 0.5
            x_indices = np.where(max_xs >= half_max)[0]
            y_indices = np.where(max_ys >= half_max)[0]
            ymin, xmax = y_mid[max_ys > .1].min(), x_mid[max_xs > .1].max()
            # ymin, xmax = np.percentile(y_mid[max_ys > 0], .5), np.percentile(x_mid[max_xs > 0], 99.5)
            # ymin = min()

            left, right = x_mid[min(x_indices)], x_mid[max(x_indices)]
            bottom, top = y_mid[min(y_indices)], y_mid[max(y_indices)]
            right = xmax
            bottom = ymin
            # plot a horizontal and vertical line from the start and stop of
            # the FWHM
            width, height = right - left, top - bottom
            pad = max(abs(width), abs(height)) * 0.1
            ax = axes[0]
            for vals, mids, orientation, (start, end) in zip(
                [y_indices, x_indices],
                [y_mid, x_mid],
                ['vertical', 'horizontal'],
                [(left, right), (bottom, top)]):
                if len(vals) > 0:
                    # start, end = vals.min(), vals.max()
                    # midpoint = (mids[start] + mids[end]) / 2
                    midpoint = (start + end) / 2
                    if orientation == 'vertical':
                        xval = right + pad
                        ax.plot([xval, xval], [start, end], color='k', linestyle='-', linewidth=2)
                        # annotate the width
                        ax.text(xval, midpoint, f'{width:.0f}', rotation=270, va='center', ha='left', color='k')
                    else:
                        yval = bottom - pad
                        ax.plot([start, end], [yval, yval], color='k', linestyle='-', linewidth=2)
                        # annotate the height above the line
                        ax.text(midpoint, yval, f'{height:.0f}', va='bottom', ha='center', color='k')
            ax.set_aspect('equal')
        return ret

    def save(self, filename):
        """Save the relevant resources to load a SynapticField later.

        Parameters
        ----------
        filename : str
            Path to save the .npz file.
        """
        save_dict = {
            'field_array': self,
            'ps': self.ps,
            'qs': self.qs,
            'centered': self.centered,
            'indices': self.indices
        }
        
        # Save centers if they exist
        if hasattr(self, 'centers'):
            save_dict['centers'] = self.centers
        
        np.savez_compressed(filename, **save_dict)


def load_SynapticField(filename):
    """Load a SynapticField from a saved .npz file.
    
    Parameters
    ----------
    filename : str
        Path to the .npz file created by SynapticField.save().
    
    Returns
    -------
    SynapticField
        The reconstructed SynapticField instance with all attributes.
    """
    data = np.load(filename, allow_pickle=True)
    
    # Extract the required parameters
    field_array = data['field_array']
    ps = data['ps']
    qs = data['qs']
    centered = data['centered'].item() if data['centered'].ndim == 0 else data['centered']
    indices = data['indices']
    
    # Create the SynapticField instance
    synaptic_field = SynapticField(field_array, ps=ps, qs=qs, centered=centered, indices=indices)
    
    # Restore centers if they were saved
    if 'centers' in data:
        synaptic_field.centers = data['centers']
    
    return synaptic_field


class SynapticFields(dict):
    """A collection of SynapticField objects organized by pathway names.
    
    This class extends dict to provide convenient batch operations on multiple
    SynapticField objects, typically representing different pathways from a
    connectivity analysis.
    
    Parameters
    ----------
    fields : dict, optional
        A dictionary mapping pathway names (str) to SynapticField objects.
    
    Examples
    --------
    >>> fields = SynapticFields({'pathway1': field1, 'pathway2': field2})
    >>> centered_fields = fields.center(method='mean')
    >>> fields.save('my_fields.npz')
    >>> loaded_fields = load_SynapticFields('my_fields.npz')
    """
    
    def __init__(self, fields=None):
        """Initialize with an optional dictionary of SynapticField objects."""
        super().__init__(fields or {})
    
    def center(self, method='mean', window=(50, 50), **kwargs):
        """Center all synaptic fields.
        
        Parameters
        ----------
        method : str, list, np.ndarray, or dict, optional
            The centering method. If dict, keys should match pathway names.
        window : tuple, optional
            The window size for centered fields.
        **kwargs
            Additional arguments passed to each SynapticField.center()
        
        Returns
        -------
        SynapticFields
            A new SynapticFields instance with centered fields.
        """
        centered = {}
        for name, field in self.items():
            center_method = method[name] if isinstance(method, dict) else method
            centered[name] = field.center(method=center_method, window=window, **kwargs)
        return SynapticFields(centered)
    
    def plot(self, ncols=3, figsize=None, **plot_kwargs):
        """Plot all synaptic fields in a grid.
        
        Parameters
        ----------
        ncols : int, optional
            Number of columns in the subplot grid.
        figsize : tuple, optional
            Figure size. If None, automatically determined.
        **plot_kwargs
            Additional arguments passed to each SynapticField.plot()
        
        Returns
        -------
        fig, axes
            Matplotlib figure and axes objects.
        """
        import matplotlib.pyplot as plt
        
        n_fields = len(self)
        nrows = int(np.ceil(n_fields / ncols))
        
        if figsize is None:
            figsize = (ncols * 4, nrows * 4)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()
        
        for idx, (name, field) in enumerate(self.items()):
            ax = axes[idx]
            field.plot(axes=ax, **plot_kwargs)
            ax.set_title(name)
        
        # Hide unused subplots
        for idx in range(n_fields, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig, axes
    
    def save(self, filename):
        """Save all synaptic fields to a single file.
        
        Parameters
        ----------
        filename : str
            Output filename (.npz format).
        """
        save_dict = {}
        for name, field in self.items():
            # Create a sub-dict for each field with a prefix
            prefix = f"{name}__"
            save_dict[f"{prefix}field_array"] = np.asarray(field)
            save_dict[f"{prefix}ps"] = field.ps
            save_dict[f"{prefix}qs"] = field.qs
            save_dict[f"{prefix}centered"] = field.centered
            save_dict[f"{prefix}indices"] = field.indices
            if hasattr(field, 'centers'):
                save_dict[f"{prefix}centers"] = field.centers
            if getattr(field, 'root_ids', None) is not None:
                save_dict[f"{prefix}root_ids"] = np.asarray(field.root_ids)
        
        # Save pathway names
        save_dict['__pathway_names__'] = np.array(list(self.keys()), dtype=object)
        np.savez_compressed(filename, **save_dict)
    
    def sum(self, axis=0):
        """Sum all fields along a given axis.
        
        Parameters
        ----------
        axis : int or None, optional
            Axis along which to sum. If None, sum all elements.
        
        Returns
        -------
        dict
            Dictionary mapping pathway names to summed arrays.
        """
        return {name: field.sum(axis=axis) for name, field in self.items()}
    
    def mean(self, axis=0):
        """Compute mean of all fields along a given axis.
        
        Parameters
        ----------
        axis : int or None, optional
            Axis along which to compute mean. If None, mean of all elements.
        
        Returns
        -------
        dict
            Dictionary mapping pathway names to mean arrays.
        """
        return {name: field.mean(axis=axis) for name, field in self.items()}
    
    def __repr__(self):
        """String representation showing pathway names and shapes."""
        items = [f"  '{name}': shape {field.shape}" for name, field in self.items()]
        return f"SynapticFields({{\n" + ",\n".join(items) + "\n})"


def load_SynapticFields(filename):
    """Load a SynapticFields collection from a saved .npz file.
    
    Parameters
    ----------
    filename : str
        Path to the .npz file created by SynapticFields.save().
    
    Returns
    -------
    SynapticFields
        The reconstructed SynapticFields instance.
    """
    data = np.load(filename, allow_pickle=True)
    
    # Get pathway names
    pathway_names = data['__pathway_names__']
    
    # Reconstruct each field
    fields = {}
    for name in pathway_names:
        prefix = f"{name}__"
        field_array = data[f"{prefix}field_array"]
        ps = data[f"{prefix}ps"]
        qs = data[f"{prefix}qs"]
        centered = data[f"{prefix}centered"].item() if data[f"{prefix}centered"].ndim == 0 else data[f"{prefix}centered"]
        indices = data[f"{prefix}indices"]
        
        # Create the SynapticField
        field = SynapticField(field_array, ps=ps, qs=qs, centered=centered, indices=indices)
        
        # Restore centers if saved
        if f"{prefix}centers" in data:
            field.centers = data[f"{prefix}centers"]
        # Restore target root_ids if saved (per-target frame identities)
        if f"{prefix}root_ids" in data:
            field.root_ids = data[f"{prefix}root_ids"]

        fields[name] = field
    
    return SynapticFields(fields)


def lattice_to_retinal(ps, qs):
    """Convert hexagonal lattice coordinates to retinal coordinates.
    
    Parameters
    ----------
    ps : numpy.ndarray
        A 2D numpy array representing the p coordinates in the hexagonal lattice.
    qs : numpy.ndarray
        A 2D numpy array representing the q coordinates in the hexagonal lattice.
    
    Returns
    -------
    xs : numpy.ndarray
        A 2D numpy array representing the x coordinates in retinal space.
    ys : numpy.ndarray
        A 2D numpy array representing the y coordinates in retinal space.
    """
    # convert ps and qs, which are in a square grid to x and y in a hex grid
    new_basis = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    xs, ys = np.dot(new_basis, np.array([ps.flatten(), qs.flatten()]))
    # transform from square to hexagonal lattice
    ys = ys * np.sqrt(3) / 2
    xs = xs * 3 / 2
    xs, ys = xs.reshape(ps.shape), ys.reshape(qs.shape)
    return xs, ys

def plot_hex_field(field, ps, qs, ax=None, cmap='viridis', scale=1.055, thresh=None, vmax=None, vmin=0, norm=None, colorbar=True):
    """Plot a hexagonal plot of the synaptic field.
    
    Parameters
    ----------
    field : numpy.ndarray
        A 2D numpy array representing the synaptic field.
    ps : numpy.ndarray
        A 2D numpy array representing the ps coordinates.
    qs : numpy.ndarray
        A 2D numpy array representing the qs coordinates.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, will create new figure and axes.
    cmap : str, optional
        The colormap to use for the plot.
    scale : float, optional
        The scaling factor for the hexagons.
    thresh : float, optional
        The threshold for the field values.
    vmax : float, optional
        The maximum value for the color scale.
    vmin : float, optional
        The minimum value for the color scale.
    norm : matplotlib.colors.Normalize, optional
        The normalization for the color scale.
    colorbar : bool, optional
        Whether to include a colorbar.
    """
    if ax is None:
        ax = plt.gca()
    if vmax is None:
        vmax = field.max()
    if vmin is None:
        vmin = field.min()
    # convert ps and qs, which are in a square grid to x and y in a hex grid
    xs, ys = lattice_to_retinal(ps, qs)
    # make a hexagon for each x and y, centered at (x, y) with color based on field
    # add a scaling factor to shrink or expand the hexagons
    radius = scale * 2./3.
    field_normed = (field - vmin) / (vmax - vmin)
    colors = plt.get_cmap(cmap)(field_normed)
    if norm is not None:
        # make a normalized colormap using the provided norm and cmap
        colors = plt.get_cmap(cmap)(norm(field))
    if thresh is not None:
        included = field > thresh
    else:
        included = np.ones_like(field, dtype=bool)
    # get the included x- and y-ranges
    xmin, xmax = xs[included].min(), xs[included].max()
    ymin, ymax = ys[included].min(), ys[included].max()
    
    # Handle both 1D and 2D cases
    if xs.ndim == 1:
        # 1D case: iterate directly over arrays
        for i in range(len(xs)):
            if included[i]:
                color = colors[i]
                # make a hexagonal patch
                hexagon = patches.RegularPolygon(
                    (xs[i], ys[i]), numVertices=6, radius=radius, orientation=np.pi/6,
                    facecolor=color, edgecolor=color)
                ax.add_patch(hexagon)
    else:
        # 2D case: iterate over both dimensions
        for i in range(len(xs)):
            for j in range(len(xs[0])):
                if included[i, j]:
                    color = colors[i, j]
                    # make a hexagonal patch
                    hexagon = patches.RegularPolygon(
                        (xs[i, j], ys[i, j]), numVertices=6, radius=radius, orientation=np.pi/6,
                        facecolor=color, edgecolor=color)
                    ax.add_patch(hexagon)
    ax.set_aspect('equal')
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(ymin - 1, ymax + 1)
    # if 'colorbar' in plot_kwargs and plot_kwargs['colorbar']:
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm if norm is not None else plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax)
    return ax

def plot_contour_field(field, ps, qs, ax=None, cmap='viridis', levels=7, **plot_kwargs):
    """Plot a contour plot of the synaptic field.
    
    Parameters
    ----------
    field : numpy.ndarray
        A 2D numpy array representing the synaptic field.
    ps : numpy.ndarray
        A 2D numpy array representing the ps coordinates.
    qs : numpy.ndarray
        A 2D numpy array representing the qs coordinates.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, will create new figure and axes.
    cmap : str, optional
        The colormap to use for the plot.
    levels : int, optional
        The number of contour levels to plot.
    """
    if ax is None:
        ax = plt.gca()
    xs, ys = lattice_to_retinal(ps, qs)
    included = field > plot_kwargs.get('thresh', 0)
    contour = ax.contourf(
        xs, ys, field, levels=levels, cmap=cmap, **plot_kwargs)
    plt.colorbar(contour, ax=ax)
    ax.set_aspect('equal')
    return ax

class MCSimulation():
    """Monte Carlo Simulation wrapper for chunked property access."""
    def __init__(self, monte_carlo_paths, monte_carlo_ids, node_info, direction='downstream'):
        self.paths = monte_carlo_paths
        self.ids = monte_carlo_ids
        self.info = node_info.copy()
        self.direction = direction
        # reindex info to have a simple index
        if 'root_id' in self.info.columns:
            del self.info['root_id']
        self.info.reset_index(inplace=True)
        self.chunk_size = getattr(self.paths, 'chunks', 10000)

    def query_chunks(self, key):            
        """Query the Monte Carlo simulation in chunks for a given variable."""
        # get the pertinent values
        values = self.info[key].values
        # let's add an empty value at the end because our simulation uses -1 to indicate Nothing, but this will result in an error
        extra_row = np.repeat(np.nan, 1)
        if values.ndim > 1:
            extra_row = np.repeat(np.nan, values.shape[-1])
            values = np.append(values, extra_row[None], axis=0)
        else:
            values = np.append(values, extra_row, axis=0)
        # run through the simulation in chunks, converting to the pertinent values and storing
        for idx in self.paths.iter_chunks():
            sub_res = self.paths[idx]
            yield values[sub_res]

    def query_pathway_chunks(self, stopping_points=[]):
        """Query the Monte Carlo simulation for pathways, optionally stopping at given points."""
        sep = " < "
        if self.direction == 'upstream':
            sep = " > "
        # get the cell types in chunks
        for arr in self.query_chunks('cell_type'):
            # reshape and convert to strings
            rows = arr[0].T.astype(str)
            if rows.size > 0:
                n_rows, n_cols = rows.shape
                # Determine cut indices from stopping_points (include the stopping point)
                if len(stopping_points) > 0:
                    stop_mask = np.zeros_like(rows, dtype=bool)
                    for sp in stopping_points:
                        stop_mask |= (np.char.find(rows, sp) != -1)
                    stop_any = stop_mask.any(axis=1)
                    first_stop_idx = np.argmax(stop_mask, axis=1)
                    stop_cut = np.where(stop_any, first_stop_idx + 1, n_cols)
                else:
                    stop_cut = np.full(n_rows, n_cols, dtype=int)
                # Determine first nan index per row (truncate before 'nan')
                nan_mask = (rows == 'nan')
                nan_any = nan_mask.any(axis=1)
                first_nan_idx = np.where(nan_any, np.argmax(nan_mask, axis=1), n_cols)
                # final truncation index per row: min of stop_cut and first_nan_idx
                final_cut = np.minimum(stop_cut, first_nan_idx).astype(int)
                # now, combine and output
                res = []
                res = np.zeros(n_rows, dtype=f"S{100}")
                for i in range(n_rows):
                    r = rows[i, :final_cut[i]]
                    if r.size == 0:
                        res[i] = ''
                        # pathways[chunk_num, i] = ''
                        # chunk_pathways.append('')
                    else:
                        nonempty = r[r != '']
                        res[i] = sep.join(nonempty)
                        # pathways[chunk_num, i] = sep.join(nonempty)
                        # chunk_pathways.append(sep.join(nonempty))\
                yield res.astype(str)

    def query_unique_pathways(self, **kwargs):
        """Get all unique pathways in the simulation."""
        # iteratively apply the unique function to each chunk
        unique_pathways = {}
        for arr in self.query_pathway_chunks(**kwargs):
            unique, counts = np.unique(arr, return_counts=True)
            for pathway, count in zip(unique, counts):
                unique_pathways[pathway] = unique_pathways.get(pathway, 0) + count
        return unique_pathways

    def query_pathways(self, max_str_len=100, **kwargs):
        """Get all pathways in the simulation."""
        # get the number of 
        num_rows, num_hops, num_reps = self.paths.shape
        res = np.zeros((num_rows, num_reps), dtype=f"S{max_str_len}")
        for chunk_num, arr in enumerate(self.query_pathway_chunks(**kwargs)):
            res[chunk_num, ...] = arr

    def query(self, key):
        # use query_chunks and collate the results
        res = [arr for arr in self.query_chunks(key)]
        return np.concatenate(res)


# make a class for plotting 2D histogram-like data
class Hist2D():
    def __init__(self):
        """"Plot 2D histogram-like data with marginal histograms."""

    def add_data(self, hist_2d, hist_top, hist_right, cmap='Greys', margin_type='bar', log=True, color='k',
                 label=None, summary_label=None, pixel_size=.025, margin_size=1):
        """Add the histogram data.

        Parameters
        ----------
        hist_2d : array-like
            The 2D histogram data.
        hist_top : array-like
            The histogram data for the top axis.
        hist_right : array-like
            The histogram data for the right axis.
        cmap : str, default='Greys'
            The colormap to use for the 2D histogram.
        margin_type : str, default='bar'
            The type of margin to use for the top and right histograms. Options are 'bar' or 'scatter'.
        log : bool, default=True
            Whether to use a log scale for the 2D histogram.
        color : str, default='k'
            The color to use for the 2D histogram and scatter plot.
        label : str, default=None
            The label to use for the colorbar.
        summary_label : str, default=None
            The label to use for the summary plots.
        pixel_size : float, default=.025
            The size of each pixel in inches.
        margin_size : float, default=1
            The size of the margin in inches.
        """
        # how can I make the subplots such that the top and right histograms are the same size as the 2D histogram?
        # todo: make the width and height ratios and figsize based on the dimensions of the 2D histogram,
        # s.t. the height of the top and width of the right as well as the pixel size in the 2d histograms 
        # are the same regardless of the size of the 2D histogram
        # fig height should be the height of the 2d histogram * pixel_size + margin_size
        # and fig width should be the width of the 2d histogram * pixel_size + margin_size
        # and the width and height ratios should be [img_width, 1] and [1, img_height]
        # calculate the fig
        hist_height, hist_width = hist_2d.shape
        img_height, img_width = pixel_size * np.array(hist_2d.shape)
        pad_scale = .6
        if margin_type == 'scatter':
            # when the plot's a scatter plot, we add some space to the right and top for the jittered points
            hist_width += pad_scale * margin_size
            hist_height += pad_scale * margin_size
        img_height = hist_height * pixel_size
        img_width = hist_width * pixel_size
        fig_width = img_width + margin_size
        fig_height = img_height + margin_size
        summary_pad_x = int(round(pad_scale/2.0 * (margin_size / fig_width) * hist_width))
        summary_pad_y = int(round(pad_scale/2.0 * (margin_size / fig_height) * hist_height))
        self.fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True, dpi=600)
        # self.hist_2d = self.fig.add_gridspec(top=.75, right=.75).subplots()
        top = fig_height / (fig_height + margin_size)
        right = fig_width / (fig_width + margin_size)
        self.hist_2d = self.fig.add_gridspec(top=top, right=right).subplots()
        # self.hist_top, self.ax_empty = self.axes[0]
        # self.hist_2d, self.hist_right = self.axes[1]
        # make the top and right axes using inset_axes instead so that the resizing is better
        # let's define the padding in absolute units instead of relative units
        pad = .1 * margin_size
        # convert to relative units
        h_pad, w_pad = pad / fig_height, pad / fig_width
        self.hist_top = self.hist_2d.inset_axes([0, 1.0 + h_pad, 1, 1 - top], sharex=self.hist_2d)
        self.hist_right = self.hist_2d.inset_axes([1.0 + w_pad, 0, 1 - right, 1], sharey=self.hist_2d)
        # plot the 2D histogram
        height, width = hist_2d.shape
        if log:
            img = self.hist_2d.imshow(hist_2d, cmap=cmap, norm='log', vmax=hist_2d.max())
        else:
            img = self.hist_2d.imshow(hist_2d, cmap=cmap, vmax=hist_2d.max())
        # make a custom axis in the 2d hist for the colorbar
        if np.any(hist_2d > 0) and len(np.unique(hist_2d)) > 2:
            # get the bar width and height as a funciton of margin_size
            bar_height = margin_size
            bar_width = .1 * bar_height
            bar_height /= fig_height
            bar_width /= fig_width
            bar_x = 1 - 3*bar_width
            bar_y = 1 - bar_height
            inset_ax = self.hist_2d.inset_axes([bar_x, bar_y, bar_width, bar_height])
            plt.colorbar(img, cax=inset_ax)
            # if label is not None:
            #     inset_ax.set_ylabel(label)
        # self.hist_2d.colorbar()
        # plot the top histogram
        xvals = np.arange(width+1) - .5
        yvals = np.arange(height+1) - .5
        if margin_type == 'bar':
            self.hist_top.stairs(hist_top, xvals, color='cyan', fill=True, zorder=2)
            self.hist_top.stairs(hist_2d.sum(0), xvals, color='black', fill=True, zorder=3)
            # make a step plot of the sum along the vertical and horizontal axes
            # and the right histogram
            self.hist_right.stairs(hist_right, yvals, color='red', orientation='horizontal', fill=True, zorder=2)
            self.hist_right.stairs(hist_2d.sum(1), yvals, color='black', orientation='horizontal', fill=True, zorder=3)
        elif margin_type == 'scatter':
            # use the halfway point for the scatter xvals
            xs = np.arange(width)
            ys = np.arange(height)
            self.hist_top.scatter(xs, hist_top, color=color, zorder=2, marker='.', edgecolors='none')
            self.hist_right.scatter(hist_right, ys, color=color, zorder=2, marker='.', edgecolors='none')
            # and add summary plots for each marginal plot showing the mean and 95% confidence intervals for the marginal sums
            # get the mean and 95% confidence intervals for the top and right vectors
            # bootstrap 10000 times 
            means, lows, highs = [], [], []
            for vals in [hist_top, hist_right]:
                # get random indices
                inds = np.random.choice(np.arange(len(vals)), size=(10000, len(vals)))
                # use to get the means
                boostrap_distro = vals[inds].mean(1)
                low, high = np.percentile(boostrap_distro, [.5, 99.5])
                lows += [low]
                highs += [high]
                means += [vals.mean()]
            # x_summary = 1.1 * width
            # y_summary = height - 1.1 * height
            # use objective measuremnts instead
            # get padding in pixels
            x_summary = width + summary_pad_x
            y_summary = -summary_pad_y
            # x_summary = width + (.1 * margin_size)
            # y_summary = - .1 * margin_size
            # plot the summary points and lines
            # self.hist_top.scatter(x_summary, means[0], color=color, zorder=5, marker='o', edgecolor='w')
            self.hist_top.plot([x_summary, x_summary], [lows[0], highs[0]], color='k', zorder=4, lw=1, solid_capstyle='butt')
            self.hist_top.plot([x_summary, x_summary], [lows[0], highs[0]], color='w', zorder=3, lw=2, solid_capstyle='projecting')
            # self.hist_right.scatter(means[1], y_summary, color=color, zorder=4, marker='s', edgecolor='w')
            self.hist_right.plot([lows[1], highs[1]], [y_summary, y_summary], color='k', zorder=4, lw=1, solid_capstyle='butt')
            self.hist_right.plot([lows[1], highs[1]], [y_summary, y_summary], color='w', zorder=3, lw=2, solid_capstyle='projecting')
            # plot a line through the two means
            self.hist_top.axhline(means[0], color='k', zorder=1, linestyle='--', lw=.5)
            self.hist_right.axvline(means[1], color='k', zorder=1, linestyle='--', lw=.5)
            # annotate the two means rounded to 2 decimal places
            # self.hist_top.annotate(f"{means[0]:.2f}", (x_summary, means[0]), color=color, zorder=5)
            # ymax = self.hist_right.get_ylim()[1]
            # self.hist_right.annotate(f"{means[1]:.2f}", (means[1], ymax), color=color, zorder=5, clip_on=False, ha='center', va='bottom')
            # for both the top and right histograms, make a jitterplot at 1.05 * width and 1.05 * height
            jitter_width = 10
            jitter_height = 10
            xjitter = np.random.normal(0, jitter_width/6, len(hist_top))
            yjitter = np.random.normal(0, jitter_height/6, len(hist_right))
            self.hist_top.scatter(x_summary + xjitter, hist_top, color=color, zorder=2, alpha=.2, edgecolor='none', marker='.')
            self.hist_right.scatter(hist_right, y_summary + yjitter, color=color, zorder=2, alpha=.2, edgecolor='none', marker='.')
        # format
        # self.hist_top.set_yscale('log')
        # self.hist_right.set_xscale('log')
        self.hist_2d.set_aspect('equal')
        self.hist_2d.set_xticks(np.arange(width))
        self.hist_2d.set_yticks(np.arange(height))
        self.hist_2d.invert_yaxis()
        # specify the y- and x-ranges
        sbn.despine(ax=self.hist_top, trim=False, bottom=True, left=True, right=False)
        self.hist_top.set_xticks([])
        # add the ylabel for the top axis, on the right side of the plot
        # self.hist_top.set_ylabel(summary_label)
        # self.hist_top.yaxis.set_label_position('right')
        # add the xlabel for the right axis
        # self.hist_right.set_xlabel(summary_label)
        if margin_type == 'scatter':
            low_x, high_x = -.5, x_summary + summary_pad_x
            low_y, high_y = y_summary - summary_pad_y, height
            self.hist_2d.set_xlim(low_x, high_x)
            self.hist_2d.set_ylim(low_y, high_y)
            for ax in [self.hist_2d, self.hist_top]: ax.set_xlim(low_x, high_x)
            for ax in [self.hist_2d, self.hist_right]: ax.set_ylim(low_y, high_y)
            # add the means to the corresponding y- and x-ticks
            self.hist_top.set_ylim(-.5)
            yticks = self.hist_top.get_yticks()
            if np.any(yticks < 0):
                yticks = yticks[yticks >= 0]
            yticks = list(yticks)
            if 0 not in yticks: yticks += [0]
            yticks += [means[0]]
            yticks = sorted(yticks)
            ytickvals = [f"{tick:.0f}" for tick in yticks]
            # set the tick nearest to the mean to be empty
            closest_ind = np.argsort(abs(np.array(yticks) - means[0]))[1]
            ytickvals[closest_ind] = ''
            self.hist_top.set_yticks(yticks, ytickvals)
            # do the same but for the right axis
            xticks = list(self.hist_right.get_xticks())
            xticks += [means[1]]
            xticks = sorted(xticks)
            xtickvals = [f"{tick:.0f}" for tick in xticks]
            # set the tick nearest to the mean to be empty
            closest_ind = np.argsort(abs(np.array(xticks) - means[1]))[1]
            xtickvals[closest_ind] = ''
            self.hist_right.set_xticks(xticks, xtickvals)
            self.hist_right.set_xlim(-.5)
        sbn.despine(ax=self.hist_2d, trim=True)
        sbn.despine(ax=self.hist_right, trim=False, left=True, bottom=False)
        self.hist_right.set_yticks([])

def combine_svgs(svgs, output_filename, row_labels, col_labels, padding=0.5, dpi=96, row_title=None, col_title=None, font_size=12):
    """Combine an array of SVG files into a single SVGFigure.

    Parameters
    ----------
    svgs : np.ndarray
        A 2D array of SVG file paths.
    output_filename : str
        The filename to save the combined SVG.
    row_labels : list
        A list of labels for the rows.
    col_labels : list
        A list of labels for the columns.
    padding : float, optional
        Padding between the SVGs in inches, by default 0.5.
    dpi : int, optional
        The DPI of the SVG, by default 96.
    row_title, col_title : str, optional
        A title for the rows and columns, by default None.
    font_size : int, optional
        Font size for the labels, by default 12.
    """
    import svgutils.transform as sg

    # Get the shape of the svgs array
    num_rows, num_cols = svgs.shape

    # Load all SVGs to get their dimensions
    svg_dimensions = []
    for row_svgs in svgs:
        row_dimensions = []
        for svg in row_svgs:
            fig = sg.fromfile(svg)
            width, height = fig.get_size()
            width, height = float(width[:-2]), float(height[:-2])
            row_dimensions.append((width, height))
        svg_dimensions.append(row_dimensions)

    # Calculate the maximum width and height for each column and row
    col_widths = [max(svg_dimensions[row][col][0] for row in range(num_rows)) for col in range(num_cols)]
    row_heights = [max(svg_dimensions[row][col][1] for col in range(num_cols)) for row in range(num_rows)]

    # Calculate the total figure size including padding
    grid_padding = .1 * dpi
    total_width = sum(col_widths) + num_cols*grid_padding + padding * dpi
    total_height = sum(row_heights) + num_rows*grid_padding + padding * dpi

    # Create the composite figure with the right size, including padding
    try:
        fig_composite = sg.SVGFigure(f"{total_width / dpi} in", f"{total_height / dpi} in")
    except:
        breakpoint()

    # Store the plot and text elements to append to the figure
    plots, txts = [], []

    y_offset = padding * dpi
    for row in range(num_rows):
        x_offset = padding * dpi
        for col in range(num_cols):
            fig = sg.fromfile(svgs[row, col])
            plot = fig.getroot()
            plot.moveto(x_offset, y_offset)
            plots.append(plot)

            # Add column labels
            if row == 0:
                col_txt = sg.TextElement(x_offset + col_widths[col] / 2, padding * dpi / 2, col_labels[col], size=font_size, anchor='middle')
                txts.append(col_txt)

            # Add row labels
            if col == 0:
                row_txt = sg.TextElement(padding * dpi / 2, y_offset + row_heights[row] / 2, row_labels[row], size=font_size, anchor='middle')
                row_txt.rotate(-90, padding * dpi / 2, y_offset + row_heights[row] / 2)
                txts.append(row_txt)

            x_offset += col_widths[col] + grid_padding
        y_offset += row_heights[row] + grid_padding

    # Add row and column titles
    if row_title is not None:
        yval = total_height / 2
        txt = sg.TextElement(font_size / 2, yval, row_title, size=font_size, anchor='middle')
        txt.rotate(-90, font_size / 2, yval)
        txts.append(txt)
    if col_title is not None:
        xval = total_width / 2
        txt = sg.TextElement(xval, font_size / 2, col_title, size=font_size, anchor='middle')
        txts.append(txt)

    # Add the plots and text to the figure
    fig_composite.append(plots)
    fig_composite.append(txts)

    # Save the composite figure
    fig_composite.save(output_filename)
    print(f"Saved composite figure to {output_filename}")

# connectome = Connectome(database_fn='connections_no_threshold.csv')
# connectome.save(fn)
# intersection = connectome.get_downstream_convergence('Dm1', 'EPG', max_hops=8, skip_recurrents=True)
# intersection_w_recurrents = connectome.get_downstream_convergence('Dm1', 'EPG', max_hops=4, skip_recurrents=False)
# breakpoint()
# plot some paths of interest based on the above analysis


def load_Paths(fn):
    """Load a Paths object previously saved with `Paths.save`.

    Parameters
    ----------
    fn : str
        Path to the HDF5 file to load.

    Returns
    -------
    Paths
        The reconstructed Paths instance.
    """
    def _read_dataframe_from_group(group):
        cols = [c.decode('utf-8') if isinstance(c, bytes) else c for c in group.attrs.get('columns', [])]
        # read index
        idx_raw = group.get('_index')
        if idx_raw is not None:
            index = [x.decode('utf-8') if isinstance(x, bytes) else x for x in idx_raw[()]]
        else:
            index = None
        data = {}
        for col in cols:
            ds = group.get(col)
            if ds is None:
                data[col] = [None] * (len(index) if index is not None else 0)
                continue
            arr = ds[()]
            # decode bytes to str when necessary
            if arr.dtype.kind in ('S', 'O'):
                arr = np.array([x.decode('utf-8') if isinstance(x, bytes) else x for x in arr])
                # convert 'None' strings back to NaN
                if np.any(pd.isna(arr) | (arr == 'None')):
                    arr = np.where(pd.isna(arr) | (arr == 'None'), 'None', arr)
            data[col] = arr
        df = pd.DataFrame(data)
        if index is not None:
            df.index = index
        return df

    # TODO: why is it taking SOOO long to load a path file?
    f = h5py.File(fn, 'r+')
    direction = f.attrs.get('direction', 'downstream')
    if isinstance(direction, bytes):
        direction = direction.decode('utf-8')

    # check if the graph pickle file exists
    graph = None
    if os.path.exists(fn.replace('.h5', '_graph.pkl')):
        import pickle
        graph_fn = fn.replace('.h5', '_graph.pkl')
        with open(graph_fn, 'rb') as graph_file:
            graph = pickle.load(graph_file)
        # set flag
    # graph = None
    # otherwise, make the graph using the edges dataset
    edges_df = None
    if 'edges' not in f:
        f.close()
        raise ValueError('HDF5 file does not contain "edges" group; not a valid Paths file')
    edges_grp = f['edges']
    edges_df = edges_grp
    # edges_df = _read_dataframe_from_group(edges_grp)
    # construct Paths
    paths_obj = Paths(edges_df, direction=direction, graph=graph)

    # lazy monte carlo access
    if 'monte_carlo' in f:
        mc = f['monte_carlo']
        # attach monte carlo datasets explicitly to make intent clear
        # setters accept h5py.Dataset and will attach the file handle
        if 'paths' in mc:
            paths_obj.monte_carlo_paths = mc['paths']
        if 'node_ids' in mc:
            paths_obj.monte_carlo_node_ids = mc['node_ids']
    # attach file handle for any other lazy access
    paths_obj._h5file = f
    return paths_obj

# make a progress bar function called print_progress
def print_progress(part, whole, prefix='', suffix='', bar_length=40):
    import sys
    if whole != 0:
        prop = float(part) / float(whole)
        block = int(round(bar_length * prop))
        text = f"\r{prefix} [{'#' * block + '-' * (bar_length - block)}] {int(prop * 100)}% {suffix}"
        sys.stdout.write(text)
        sys.stdout.flush()
    else:
        text = f"\r{prefix}: {part}"
        sys.stdout.write(text)
        sys.stdout.flush()


from scipy.spatial import cKDTree
import numpy as np
from scipy.spatial import ConvexHull
import urllib

# class SynapticProjection():
#     """Allows projecting a synaptic field into visual coordinates in real world or spherical coordinates.

#     Parameters
#     ----------

#     Methods
#     -------
#     spherical_projection : Project the synaptic field onto a sphere surrounding the fly, making one mesh per eye.
#     plot_3d : Make a 3D plot of the spherical projection using a 3D plotly mesh.
#     """
#     def __init__(self):
#         # we should project to visual coordinates and make one mesh for each eye
#         # import the visual projection data:
#         if not os.path.exists('visual_column_info.csv'):
#             # otherwise, download it from flywire
#             url = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz"
#             # download the url and read it into a pandas DataFrame
#             test = urllib.request.urlretrieve(url, filename='visual_column_info.csv.gz')
#             # unzip the file
#             gz_file_path = 'visual_column_info.csv.gz'
#             output_file_path = 'visual_column_info.csv'
#             with gzip.open(gz_file_path, 'rb') as f_in:
#                 with open(output_file_path, 'wb') as f_out:
#                     shutil.copyfileobj(f_in, f_out)
#         # now, read the visual column info
#         self.visual_column_info = pd.read_csv('visual_column_info.csv')
#         self.optical_info = pd.read_csv("mi1_visual_data.csv")

#     def spherical_projection(self, radius=1e5, num_points=1e6):
#         """Project the synaptic field onto a sphere surrounding the fly, making one mesh per eye.

#         Parameters
#         ----------
#         radius : float, default=1e5
#             The radius of the sphere onto which to project the synaptic field.
#         num_points : int, default=1e6
#             The number of points to generate on the sphere for projection.
#         """
#         # make one mesh per eye
#         data = self.optical_info
#         pts = data[['position_x', 'position_y', 'position_z']].values
#         dirs = data[['optical_x', 'optical_y', 'optical_z']].values
#         left_eye = data['side'] == 'left'
#         proj_pts = project_coords(pts, dirs, center=np.zeros(3), radius=radius, convex=False)
#         # make the regularly spaced sphere
#         sphere_points = fibonacci_sphere(int(num_points))
#         sphere_points *= radius
#         # label the points on the sphere by finding the nearest projected point, per eye
#         tree_left = cKDTree(proj_pts[left_eye])
#         distances_left, indices_left = tree_left.query(sphere_points, k=1) 
#         tree_right = cKDTree(proj_pts[~left_eye])
#         distances_right, indices_right = tree_right.query(sphere_points, k=1)
#         # for distances > some threshold, set index to -1
#         indices_left[distances_left > 10000] = -1
#         indices_right[distances_right > 10000] = -1
#         # get the retinal coordinates
#         ps, qs = data[['lattice_p_corrected', 'lattice_q_corrected']].values.T.astype(int)
#         self.pvals_left, self.qvals_left = ps[left_eye][indices_left], qs[left_eye][indices_left]
#         self.pvals_right, self.qvals_right = ps[~left_eye][indices_right], qs[~left_eye][indices_right]
#         self.ps, self.qs = ps, qs
#         self.left_eye = left_eye
#         self.indices_left = indices_left
#         self.indices_right = indices_right
#         self.left_points, self.right_points = sphere_points[indices_left > -1], sphere_points[indices_right > -1]
#         # TODO: remember to copy the pixel_values logic from the other file
#         self.convex_hull_left = ConvexHull(sphere_points[indices_left > -1])
#         self.convex_hull_right = ConvexHull(sphere_points[indices_right > -1])

#     def plot_3d(self, colorvals=None, side='left', normalize=False, cmap='magma', fig=None, row=None, col=None):
#         """Make a 3D plot of the spherical projection using a 3D plotly mesh.
        
#         Parameters
#         ----------
#         colorvals : array-like, default=None
#             The values to use for coloring the mesh. If None, uses uniform color.
#         side : str, default='left'
#             The side of the eye to plot ('left' or 'right').
#         normalize : bool, default=False
#             Whether to normalize the colorvals to [0, 1].
#         cmap : str, default='magma'
#             The colormap to use for the mesh.
#         fig : plotly.graph_objects.Figure, default=None
#             An existing figure to add the mesh to. If None, creates a new figure.
#         row : int, default=None
#             The row in the figure to add the mesh to. If None, uses row 1.
#         col : int, default=None
#             The column in the figure to add the mesh to. If None, uses column 1.
#         """
#         # filter the ps and qs based on side
#         include = self.left_eye if side == 'left' else ~self.left_eye
#         indices = self.indices_left if side == 'left' else self.indices_right
#         ps, qs = self.ps[include], self.qs[include]
#         # pvals, qvals = (self.pvals_left, self.qvals_left) if side == 'left' else (self.pvals_right, self.qvals_right)
#         hull = self.convex_hull_left if side == 'left' else self.convex_hull_right
#         if colorvals is None:
#             # make uniform colorvals
#             colorvals = np.ones(include.sum())
#         elif isinstance(colorvals, SynapticField):
#             field = colorvals
#             # this method only works if the field is already reduced to 2D
#             assert field.ndim == 2, "SynapticField must be 2D for this plotting method."
#             sf_ps, sf_qs = field.ps, field.qs
#             inds = np.where((sf_ps[None] == ps[:, None, None]) * (sf_qs[None] == qs[:, None, None]))
#             colorvals = np.zeros(ps.shape[0]) * np.nan
#             colorvals[inds[1]] = field[inds[1], inds[2]]
#         assert len(colorvals) == include.sum(), "colorvals must have the same length as the included points."
#         # normalize the colorvals 
#         if normalize:
#             colorvals = colorvals / np.nanmax(colorvals)
#         # repeat the colorvals to match the appropriate sphere points
#         cvals = colorvals[indices]
#         # get the appropriate sphere points
#         eye_points = self.left_points if side == 'left' else self.right_points
#         # get the convex hull simplices
#         i, j, k = hull.simplices.T
#         # make the 3D plotly mesh using the convex hull
#         from plotly import graph_objects as go
#         if fig is None:
#             fig = go.Figure()
#         if row is None:
#             row = 1
#         if col is None:
#             col = 1
#         fig.add_trace(go.Mesh3d(
#             x=eye_points[:, 0],
#             y=eye_points[:, 1],
#             z=eye_points[:, 2],
#             i=i, j=j, k=k,
#             # facecolor=cvals,
#             intensity=cvals,
#             colorscale=cmap, 
#             opacity=1,
#             flatshading=True,
#             lighting=dict(ambient=0.6, diffuse=0.5, roughness=0.9, specular=0.2),
#             lightposition=dict(x=100, y=200, z=0),
#             name=f"{side} eye"),
#             row=row, col=col)
#         fig.update_layout(scene=dict(
#             xaxis_title='X Axis',
#             yaxis_title='Y Axis',
#             zaxis_title='Z Axis',
#             aspectmode='data'
#         ))
#         return fig


def project_coords(pos_vectors, dir_vectors, center=np.zeros(3), radius=1e5,
                   convex=False):
    """Project 3D vectors onto an encompassing sphere ('world referenced' coordinates).

    
    Parameters
    ----------
    pos_vectors : array-like, shape=(N, 3)
        The coordinates specifying the origin of each vector.
    dir_vectors : array-like, shape=(N, 3)
        The coordinates specifying the directional components of each vector.
    center : array-like, default=(0, 0, 0)
        The coordinate of the center of the sphere. 
    radius : float, default=1e5
        The radius of the sphere.
    convex : bool, default=False
        Whether to assume the vectors are on a concave or convex surface,
        using the projection further from the center instead of the nearer one.
    """    
    proj_coords = []
    # Shift all position vectors so the sphere center is at the origin
    pos_vectors_centered = pos_vectors - center[None, :]
    # For each lens, project its anatomical axis onto the sphere
    for p_vector, d_vector in zip(pos_vectors_centered, dir_vectors):
        # Compute the discriminant for the quadratic equation of intersection
        a = np.dot(d_vector, d_vector)
        b = 2 * np.dot(d_vector, p_vector)
        c = np.dot(p_vector, p_vector) - radius ** 2
        descr = b ** 2 - 4 * a * c
        # descr = (2 * np.dot(p_vector, d_vector)) ** 2 - 4 * np.dot(d_vector, d_vector) * (np.dot(p_vector, p_vector) - radius ** 2)
        # diff = const ** 2 - np.linalg.norm(p_vector) ** 2 + radius ** 2
        proj_pt = np.empty(3)
        if descr >= 0:
            # calculate the distances using the quadratic formula
            dists = np.asarray([
                (-b - np.sqrt(descr))/(2*a),
                (-b + np.sqrt(descr))/(2*a)
            ])
            # Choose the intersection based on convex/concave geometry
            if convex:
                dist = dists[np.argmax(abs(dists))]
            else:
                dist = dists[np.argmin(abs(dists))]
            # Calculate the projected point on the sphere
            proj_pt[:] = p_vector + d_vector * dist
        else:
            # If no intersection, return NaNs for this point
            proj_pt[:] = np.nan
        proj_coords += [proj_pt]
    # Convert list of projected points to a numpy array
    proj_coords = np.asarray(proj_coords)
    return proj_coords

def fibonacci_sphere(num_points, radius=1.0):
    """
    Generates num_points ~evenly distributed on the surface of a sphere.

    Args:
        num_points (int): The number of points to generate.
        radius (float): The radius of the sphere. Defaults to a unit sphere (radius=1.0).

    Returns:
        numpy.ndarray: An array of shape (num_points, 3) containing the (x, y, z) coordinates.
    """
    points = np.zeros((num_points, 3))
    phi = np.pi * (3 - np.sqrt(5))  # Golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        r = np.sqrt(1 - y * y)  # radius at this height
        theta = phi * i  # angle for this point

        x = np.cos(theta) * r
        z = np.sin(theta) * r

        points[i] = [x, y, z]

    return points * radius

from scipy.interpolate import griddata

class SynapticProjection():
    """Allows projecting a synaptic field into visual coordinates in real world or spherical coordinates.

    Parameters
    ----------

    Methods
    -------
    spherical_projection : Project the synaptic field onto a sphere surrounding the fly, making one mesh per eye.
    plot_3d : Make a 3D plot of the spherical projection using a 3D plotly mesh.
    """
    def __init__(self):
        # we should project to visual coordinates and make one mesh for each eye
        # import the visual projection data:
        if not os.path.exists('visual_column_info.csv'):
            # otherwise, download it from flywire
            url = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz"
            # download the url and read it into a pandas DataFrame
            test = urllib.request.urlretrieve(url, filename='visual_column_info.csv.gz')
            # unzip the file
            gz_file_path = 'visual_column_info.csv.gz'
            output_file_path = 'visual_column_info.csv'
            with gzip.open(gz_file_path, 'rb') as f_in:
                with open(output_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        # now, read the visual column info
        self.visual_column_info = pd.read_csv('visual_column_info.csv')
        self.optical_info = pd.read_csv("mi1_visual_data.csv")

    def spherical_projection(self, radius=1e5, num_points=1e5):
        """Project the synaptic field onto a sphere surrounding the fly, making one mesh per eye.

        Parameters
        ----------
        radius : float, default=1e5
            The radius of the sphere onto which to project the synaptic field.
        num_points : int, default=1e6
            The number of points to generate on the sphere for projection.
        """
        # Extract position and direction vectors from optical data
        self.pts = self.optical_info[['position_x', 'position_y', 'position_z']].values
        self.dirs = self.optical_info[['optical_x', 'optical_y', 'optical_z']].values
        self.left_eye = self.optical_info['side'] == 'left'
        self.radius = radius

        # Project optical axes onto sphere
        self.proj_pts = project_coords(self.pts, self.dirs, center=np.zeros(3), radius=radius, convex=False)
        
        # Generate evenly distributed sphere points
        sphere_points = fibonacci_sphere(int(num_points)) * radius
        
        # Map sphere points to nearest ommatidium for each eye
        tree_left = cKDTree(self.proj_pts[self.left_eye])
        distances_left, indices_left = tree_left.query(sphere_points, k=1)
        
        tree_right = cKDTree(self.proj_pts[~self.left_eye])
        distances_right, indices_right = tree_right.query(sphere_points, k=1)
        
        # Exclude points too far from any ommatidium (threshold: 10000)
        indices_left[distances_left > 10000] = -1
        indices_right[distances_right > 10000] = -1
        
        # Extract retinal lattice coordinates
        self.ps, self.qs = self.optical_info[['lattice_p_corrected', 'lattice_q_corrected']].values.T.astype(int)
        
        # Store lattice coordinates for sphere points (per eye)
        self.pvals_left = self.ps[self.left_eye][indices_left]
        self.qvals_left = self.qs[self.left_eye][indices_left]
        self.pvals_right = self.ps[~self.left_eye][indices_right]
        self.qvals_right = self.qs[~self.left_eye][indices_right]
        
        # Filter valid points and store
        valid_left = indices_left > -1
        valid_right = indices_right > -1
        self.points_left = sphere_points[valid_left]
        self.points_right = sphere_points[valid_right]
        self.indices_left = indices_left[valid_left]
        self.indices_right = indices_right[valid_right]
        
        # Create convex hulls for mesh triangulation
        self.convex_hull_left = ConvexHull(self.points_left)
        self.convex_hull_right = ConvexHull(self.points_right)

    def generate_acceptance_maps(self, scale=2.0, save_fn='acceptance_masks.pkl', rerun=False):
        """Estimate acceptance fields for each ommatidium based on the optical measurements in Zhao et al (2025).


        Parameters
        ----------
        scale : float, default=2.0
            The scaling factor to apply to the acceptance angles.
        save_fn : str, default='acceptance_masks.pkl'
            The filename to save/load the acceptance masks.
        rerun : bool, default=False
            If True, regenerate the acceptance maps even if a saved file exists.
        """
        # use pickle to load the acceptance masks if they exist, 
        # otherwise generate and save them for future attempts
        if os.path.exists(save_fn) and not rerun:
            # use pickle to load the acceptance masks
            with open(save_fn, 'rb') as f:
                self.masks_per_side, self.probs_per_side, self.fwhms_per_side = pickle.load(f)
        else:
            # we need to measure the mean IO angle per ommatidium, given that there are as many as 6 neighbors for each ommatidium
            pts, dirs = self.pts, self.dirs
            proj_pts = self.proj_pts
            data = self.optical_info
            io_angles = np.zeros_like(proj_pts[:, 0])
            for eye_side in ['left', 'right']:
                include = data['side'] == eye_side
                # for each point, find its 6 nearest neighbors and compute the mean angle between them, given that we know their projected positions and radius
                tree = cKDTree(proj_pts[include])
                distances, indices = tree.query(proj_pts[include], k=4)  # k=7 to include self
                distances, indices = distances[:, 1:], indices[:, 1:]  # exclude self
                radius = 1e5
                angles = np.arcsin(distances / (2 * radius)) * 2 * (180 / np.pi)  # in degrees
                # mean_io_angles = angles.mean(axis=1)
                mean_io_angles = angles[:, 0]
                # store the mean io angles
                io_angles[include] = mean_io_angles
            # scale the io angles to get acceptance angles
            acceptance_angles = scale * io_angles
            # now, for each ommatidium, generate an acceptance mask on the sphere points
            self.masks_per_side, self.probs_per_side, self.fwhms_per_side = {}, {}, {}
            # self.eye_masks = {}
            for side, eye_points, on_side in zip(['left', 'right'], [self.points_left, self.points_right], [self.left_eye, ~self.left_eye]):
                masks = []
                probs = []
                fwhms = []
                # eye_mask = np.zeros(eye_points.shape[0], dtype=bool)
                # for position and direction vector, measure the angular deviation of each sphere point
                for num, (pt, dir, acceptance_angle) in enumerate(zip(pts[on_side], dirs[on_side], acceptance_angles[on_side])):
                # pt, dir, acceptance_angle = pts[left_eye.astype(bool)][0], dirs[left_eye.astype(bool)][0], acceptance_angles[left_eye.astype(bool)][0]
                    dir /= np.linalg.norm(dir)
                    vecs = eye_points - pt[None, :]
                    vecs /= np.linalg.norm(vecs, axis=1)[:, None]
                    dot_prods = np.dot(vecs, dir)
                    angles = np.arccos(dot_prods) * (180 / np.pi)
                    # the acceptance angle represents the full width at half maximum of a Gaussian distribution
                    # make a similar distribution
                    # how do I convert from the known FWHM to the standard deviation?
                    # FWHM = 2 * sqrt(2 * ln(2)) * sigma => sigma = FWHM / (2 * sqrt(2 * ln(2)))
                    sigma = acceptance_angle / (2 * np.sqrt(2 * np.log(2)))
                    normal = scipy.stats.norm(loc=0, scale=sigma)
                    # measure the probability density at each angle
                    # we're only concerned points within 3*acceptance_angle
                    included = angles < 5 * acceptance_angle
                    prob = normal.pdf(angles[included])
                    prob /= prob.max()
                    # store the mask, even if there are no included points
                    masks += [included]
                    if np.any(included):
                        prob = normal.pdf(angles[included])
                        # find the points within the acceptance angle
                        prob /= prob.max()
                        probs += [prob]
                        # measure the FWHM from the plotted data
                        fwhm = 2 * angles[included][prob >= 0.5].max()
                        fwhms += [fwhm]
                        # and add to the maximum intensity eye mask
                        # eye_mask |= included
                    else:
                        probs += [np.array([])]
                        fwhms += [0]
                self.masks_per_side[side] = masks
                self.probs_per_side[side] = probs
                self.fwhms_per_side[side] = fwhms
                # self.eye_masks[side] = eye_mask 
                # make a new 

    def get_cvals(self, colorvals=None, side='left', normalize=False, apply_acceptance=False, acceptance_method='sum'):
        """Make a 3D plot of the spherical projection using a 3D plotly mesh.
        
        Parameters
        ----------
        colorvals : array-like, default=None
            The values to use for coloring the mesh. If None, uses uniform color.
        side : str, default='left'
            The side of the eye to plot ('left' or 'right').
        normalize : bool, default=False
            Whether to normalize the color values to [0, 1].
        cmap : str, default='magma'
            The colormap to use for coloring the mesh.
        apply_acceptance : bool, default=False
            Whether to apply the acceptance angle mask to the color values.
        acceptance_method : str, default='sum'
            The method to combine probabilities within the acceptance angle ('sum' or 'max').
            Sum: first, scale each acceptance field by the corresponding colorval, then sum across all fields.
            Max: first, scale each acceptance field by the corresponding colorval, then take the maximum across all fields.
        """
        # filter the ps and qs based on side
        include = self.left_eye if side == 'left' else ~self.left_eye
        indices = self.indices_left if side == 'left' else self.indices_right
        ps, qs = self.ps[include], self.qs[include]
        # pvals, qvals = (self.pvals_left, self.qvals_left) if side == 'left' else (self.pvals_right, self.qvals_right)
        if colorvals is None:
            # make uniform colorvals
            colorvals = np.ones(include.sum())
        elif isinstance(colorvals, SynapticField):
            field = colorvals
            # this method only works if the field is already reduced to 2D
            assert field.ndim == 2, "SynapticField must be 2D for this plotting method."
            sf_ps, sf_qs = field.ps, field.qs
            inds = np.where((sf_ps[None] == ps[:, None, None]) * (sf_qs[None] == qs[:, None, None]))
            # colorvals = np.zeros(ps.shape[0]) * np.nan
            colorvals = field[inds[1], inds[2]]
        assert len(colorvals) == include.sum(), "colorvals must have the same length as the included points."
        # normalize the colorvals 
        if normalize:
            colorvals = colorvals / np.nanmax(colorvals)
        if apply_acceptance:
            cvals = np.zeros_like(indices, dtype=float)
            # apply the acceptance masks
            if acceptance_method in ['sum', 'max']:
                acceptance_func = np.sum if acceptance_method == 'sum' else np.max
                for omma_num, (mask, probs, colorval) in enumerate(zip(self.masks_per_side[side], self.probs_per_side[side], colorvals)):
                    if len(probs) == 0:
                        continue
                    # scale the probs by the colorval
                    scaled_probs = probs * colorval
                    # apply to the cvals
                    try:
                        cvals[mask] = acceptance_func(np.array([cvals[mask], scaled_probs]), axis=0)
                    except:
                        print(f"Error applying acceptance for ommatidium {omma_num} on side {side}.")
            elif acceptance_method == 'OR':
                # let's calculate the probability of success in at least one of accptance fields
                # this is 1 - Pr(no response) 
                # Pr(no response) = product(1 - p_i) for each acceptance field i
                prob_no_response = np.ones_like(indices, dtype=float)
                for omma_num, (mask, probs, colorval) in enumerate(zip(self.masks_per_side[side], self.probs_per_side[side], colorvals)):
                    # scale the probs by the colorval
                    scaled_probs = probs * colorval
                    # apply to the prob_no_response
                    prob_no_response[mask] *= (1 - scaled_probs)
                # now get the probability of at least one response
                cvals = 1 - prob_no_response
        else:
            # repeat the colorvals to match the appropriate sphere points
            cvals = colorvals[indices]
        return cvals

    def plot_3d(self, colorvals=None, side='left', normalize=False, apply_acceptance=False, acceptance_method='sum', 
                cmap='magma', projection=None, fig=None, row=None, col=None):
        """Make a 3D plot of the spherical projection using a 3D plotly mesh.

        Parameters
        ----------
        colorvals : array-like, default=None
            The values to use for coloring the mesh. If None, uses uniform color.
        side : str, default='left'
            The side of the eye to plot ('left' or 'right').
        normalize : bool, default=False
            Whether to normalize the colorvals to [0, 1].
        apply_acceptance : bool, default=False
            Whether to apply the acceptance angle mask to the color values.
        acceptance_method : str, default='sum'
            The method to use for applying acceptance ('sum', 'max', or 'OR').
        cmap : str, default='magma'
            The colormap to use for the mesh.
        projection : str, default=None
            The type of 2D projection to use ('spherical', 'mercator', 'molleweide', or None for 3D).   
        fig : plotly.graph_objects.Figure, default=None
            An existing figure to add the mesh to. If None, creates a new figure.
        row, col : int, default=None
            The row and column in the figure to add the mesh to. If None, uses row 1 and col 1.
        """
        hull = self.convex_hull_left if side == 'left' else self.convex_hull_right
        cvals = self.get_cvals(colorvals=colorvals, side=side, normalize=normalize, apply_acceptance=apply_acceptance, acceptance_method=acceptance_method)
        # get the appropriate sphere points
        eye_points = self.points_left if side == 'left' else self.points_right
        # check the pyplot theme. If it's dark, make the axes white, otherwise black
        axis_color = 'white' if pio.templates.default == "plotly_dark" else 'black'
        if fig is None:
            from plotly import graph_objects as go
            fig = go.Figure()
        if projection == None:
            # get the convex hull simplices
            i, j, k = hull.simplices.T
            # reduce the simplices to just those that are on the surface of the eye
            # based on the radial distance of the centroid of each simplex
            simplex_centroids = eye_points[hull.simplices].mean(axis=1)
            radial_distances = np.linalg.norm(simplex_centroids, axis=1)
            # only include those within .05 * radius of the sphere
            surface_simplices = np.abs(radial_distances - self.radius) < (0.05 * self.radius)
            i, j, k = i[surface_simplices], j[surface_simplices], k[surface_simplices]
            # make the 3D plotly mesh using the convex hull
            from plotly import graph_objects as go
            fig.add_trace(go.Mesh3d(
                x=eye_points[:, 0],
                y=eye_points[:, 1],
                z=eye_points[:, 2],
                i=i, j=j, k=k,
                # facecolor=cvals,
                # interpolate the colors across the faces
                intensity=cvals,
                colorscale=cmap, 
                opacity=1,
                # flatshading=True,
                # lighting=dict(ambient=0.6, diffuse=0.5, roughness=0.9, specular=0.2),
                # lightposition=dict(x=100, y=200, z=0),
                name=f"{side} eye",
                colorbar=dict(x=-0.15, len=0.75)  # Move colorbar to the left
            ), row=row, col=col)
            # 
            # fig.update_layout(scene=dict(
            #     xaxis_title='X Axis',
            #     yaxis_title='Y Axis',
            #     zaxis_title='Z Axis',
            #     aspectmode='data'
            # ))
            # let's make some custom spherical axes
            # first, let's make 3 great circles, one for each axis
            axis_length = self.radius * 1.2
            theta = np.linspace(0, 2 * np.pi, 100)
            x_circle = axis_length * np.cos(theta)
            y_circle = axis_length * np.sin(theta)
            z_circle = np.zeros_like(theta)
            # make the 3 great circles

            fig.add_trace(go.Scatter3d(x=x_circle, y=y_circle, z=z_circle, mode='lines', line=dict(color=axis_color, width=2), name='X Axis', showlegend=False), row=row, col=col)
            fig.add_trace(go.Scatter3d(x=x_circle, y=z_circle, z=y_circle, mode='lines', line=dict(color=axis_color, width=2), name='Y Axis', showlegend=False), row=row, col=col)
            fig.add_trace(go.Scatter3d(x=z_circle, y=x_circle, z=y_circle, mode='lines', line=dict(color=axis_color, width=2), name='Z Axis', showlegend=False), row=row, col=col)
            # and add single letter labels for the 6 points of intersection: Dorsal (above), Ventral (below), Anterior (front), Posterior (back), Left and Right
            label_scale = self.radius * 1.35
            # dorsal: (0, 0, label_scale), ventral: (0, 0, -label_scale), anterior: (label_scale, 0, 0), posterior: (-label_scale, 0, 0), left: (0, label_scale, 0), right: (0, -label_scale, 0)
            for x, y, z, lbl in zip(
                [0, 0, label_scale, -label_scale, 0, 0],
                [0, 0, 0, 0, label_scale, -label_scale],
                [label_scale, -label_scale, 0, 0, 0, 0],
                ['D', 'V', 'A', 'P', 'L', 'R']):
                fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='text', text=[lbl], textposition='middle center', name=f'{lbl} Label', showlegend=False), row=row, col=col)
            # and remove the grid and axes that plotly adds by default
            fig.update_layout(scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showgrid=False, zeroline=False, visible=False)
            ))

            return fig
        # otherwise, implement other projections
        elif projection in ['spherical', 'mercator', 'molleweide']:
            # get the 2D projection
            proj_pts = project(eye_points, type=projection)
            # make a 2D mesh interpolation of the new projected points
            xgrid, ygrid = np.mgrid[proj_pts[:, 0].min():proj_pts[:, 0].max():500j, proj_pts[:, 1].min():proj_pts[:, 1].max():500j]
            # use griddata to interpolate the colorvals onto the grid
            grid_z = griddata(proj_pts, cvals, (xgrid, ygrid), method='linear')
            # remove values below thresh
            include = grid_z > 0
            grid_z[~include] = np.nan

            # plot the 2D mesh using plotly
            from plotly import graph_objects as go
            fig.add_trace(go.Heatmap(
                z=grid_z.T,
                x=xgrid[:,0],
                y=ygrid[0,:],
                colorscale=cmap,
                colorbar=dict(title='Intensity', yanchor='top', y=1, x=0),
                zsmooth='best'
            ), row=row, col=col)
            # plot the major circles for reference
            # plot the visual horizon, visual midline, and vertical meridian
            angles = np.linspace(-np.pi, np.pi, 100)
            # map these onto 3D coordinates
            xs_list, ys_list = [], []
            for pos in [
                self.radius * np.vstack([np.cos(angles), np.zeros_like(angles), np.sin(angles)]).T,
                self.radius * np.vstack([np.zeros_like(angles), np.cos(angles), np.sin(angles)]).T
            ]:
                proj_circle = project(pos, type=projection)
                xs, ys = proj_circle[:, 0], proj_circle[:, 1]
                ys = ys[xs >= 0]
                xs = xs[xs >= 0]
                xs = np.concatenate([-xs[::-1], xs])
                ys = np.concatenate([ys[::-1], ys])
                xs_list.append(xs)
                ys_list.append(ys)

            # combine the two curves into a single trace (insert a NaN to break the line)
            combined_x = np.concatenate([xs_list[0], [np.nan], xs_list[1]])
            combined_y = np.concatenate([ys_list[0], [np.nan], ys_list[1]])
            fig.add_trace(go.Scatter(x=combined_x, y=combined_y, mode='lines',
                                     line=dict(color=axis_color, width=1, dash='dash'),
                                     name='Great Circles'), row=row, col=col)
            # label the 4 poles: 
            key_angles = np.array([[-np.pi, 0], [0, 0], [np.pi, 0], [0, -0]])
            # convert to positions on the sphere
            key_positions = np.array([
                [np.cos(az), np.sin(az), np.sin(el)] for az, el in key_angles
            ]) * self.radius
            # project to 2D
            key_proj = project(key_positions, type=projection)
            key_lbls = ["-π", "π", "π", "-π"]
            scale = 1.1
            for (x, y), lbl, text_pos in zip(key_proj, key_lbls, ['middle left', 'top center', 'middle right', 'bottom center']):
                fig.add_trace(
                    go.Scatter(
                        x=[scale * x], y=[scale * y], 
                        mode='text', text=[lbl], 
                        textposition=text_pos, name=f'{lbl} Label',
                        textfont=dict(color=axis_color, size=12)
                        ), row=row, col=col)
            # remove the standard gridlines
            fig.update_layout(
                title=f"{side.capitalize()} Eye - {projection.capitalize()} Projection",
                # xaxis_title='X',
                # yaxis_title='Y',
                yaxis=dict(scaleanchor="x", scaleratio=1)  # keep aspect ratio
            )
            # remove the x and y gridlines
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
            return fig

def project(pts, type='spherical'):
    """Convert from cartesian to spherical coordinates of azimuth and elevation."""
    # 
    if type == 'spherical':
        azimuth, elevation = np.arctan2(pts[:, 1], pts[:, 0]), np.arcsin(pts[:, 2] / np.linalg.norm(pts, axis=1))
        return np.vstack([azimuth, elevation]).T
    elif type == 'mercator':
        azimuth, elevation = np.arctan2(pts[:, 1], pts[:, 0]), np.arcsin(pts[:, 2] / np.linalg.norm(pts, axis=1))
        x = azimuth
        y = np.log(np.tan((np.pi / 4) + (elevation / 2)))
        return np.vstack([x, y]).T
    elif type == 'molleweide':
        azimuth, elevation = np.arctan2(pts[:, 1], pts[:, 0]), np.arcsin(pts[:, 2] / np.linalg.norm(pts, axis=1))
        theta = np.arcsin(elevation / (np.pi / 2))
        x = (2 * np.sqrt(2) / np.pi) * azimuth * np.cos(theta)
        y = np.sqrt(2) * np.sin(theta)
        return np.vstack([x, y]).T
