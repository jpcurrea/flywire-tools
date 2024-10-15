# run this first because it takes a couple of minutes
import navis
from navis import models
print('imported navis version:', navis.__version__)

from fafbseg import flywire
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import scipy
import seaborn as sbn
import shutil
import svgutils.transform as sg
import time
from svgutils.compose import Unit

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
                return flywire.search_annotations(cell_type).root_id.values.tolist()
            except:
                time.sleep(5)
        return flywire.search_annotations(cell_type).root_id.values.tolist()

    def get_paths(self, root_ids, max_hops=5, downstream=True, upstream=False, skip_recurrents=True):
        """Find all pathways downstream of root_ids up to max_hops.

        Option to search for upstream or downstream connections.

        Parameters
        ----------
        root_ids : list or array-like
            The list of root node IDs from which to start the downstream search.
        max_hops : int, default=5
            The maximum number of hops to search downstream. Default is 5.
        downstream, upstream : bool
            Whether to search for downstream or upstream paths.
        skip_recurrents : bool, default=True
            Whether to avoid hopping towards cells that have already been included.

        Returns
        -------
        paths : Path
            A Path object with all the specified pathways.
        """
        # if going downstream, we need to start from pre-synaptic and go to post-synaptic cells
        vars = set(['pre', 'post'])        
        start_vars = []
        if downstream:
            start_vars += ['pre']
        # if going upstream, we need to start from post-synaptic and go to pre-synaptic cells
        if upstream:
            start_vars += ['post']
        # store the downstream or upstream pathways into a dictionary
        pathways = {}
        for start_var in start_vars:
            stop_var = (vars - set([start_var])).pop()
            # get the edgelist by starting from the root_ids and travelling up- and downstream
            ids = root_ids
            past_ids = np.unique(ids)
            # choose the appropriate list
            direction = 'downstream'
            if start_var == 'post':
                direction = 'upstream'
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
                connections['hop'] = hop_num +1
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
        paths = Paths(pathways)
        return paths


    def get_downstream_convergence(self, group_a, group_b, simulation_use_inits, sim_reps=1e5, **path_kwargs):
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
            paths[lbl] = self.get_paths(root_ids, downstream=True, **path_kwargs)
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

    def get_connectivity(self, group_a, group_b, sort_by='nblast', transmitters=False, data_dir="./", pixel_size=0.025, neuroglancer_plots=True):
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
        neuroglancer_plots : bool, default=False
            Whether to make neuroglancer plots of the connectivity.
        """
        # make folder for storing the .svg files
        dirname = os.path.join(data_dir, f"{group_a}_onto_{group_b}")
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
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
            if group in self.past_paths:
                paths[group] = self.past_paths[group]
            else:
                root_ids = self.lookup_root_ids(group)
                path = self.get_paths(root_ids, downstream=True, upstream=False, max_hops=1, skip_recurrents=False)
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
            pivot = pivot.append({start_col: missing_id, stop_col: -1, 'weight': 0}, ignore_index=True)
        for missing_id in missing_post:
            pivot = pivot.append({start_col: -1, stop_col: missing_id, 'weight': 0}, ignore_index=True)
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
        hist_synapses = Hist2D()
        hist = hist_synapses
        if sort_by in ['nblast', 'hungarian']:
            pivot_sorted = edges_pivot.iloc[main_order, other_order]
        else:
            pivot_sorted = edges_pivot.loc[main_order, other_order]
        hist.add_data(pivot_sorted.values, pivot_sorted.sum(0).values, pivot_sorted.sum(1).values, margin_type='scatter', log=False,
                      label='synapses', summary_label='synapses', pixel_size=pixel_size)
        # remove the x and y ticks
        hist.hist_2d.set_xticks([])
        hist.hist_2d.set_yticks([])
        # despine the bottom and left axes
        sbn.despine(ax=hist.hist_2d, bottom=True, left=True)
        # save as .svg
        fn = os.path.join(dirname, f"{group_a}_onto_{group_b}_synapses.svg")
        try:
            hist.fig.savefig(fn)
        except:
            hist.fig.savefig(fn, dpi=100)
        # todo: figure out why the figsize is wrong for non-square plots
        svgs.append(fn)
        hists += [hist]
        # add the all transmitters to the list of transmitters
        transmitter_dfs['all'] = pivot_sorted
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
        fn = os.path.join(dirname, f"{group_a}_onto_{group_b}_cells.svg")
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
        # add colors to the transmitter counts
        # make a bar plot of the transmitter counts, coloring each bar by the corresponding color
        fig = plt.figure()
        sbn.barplot(y=transmitter_counts.index, x=transmitter_counts.values, color='k')
        sbn.despine()
        transmitter_counts['color'] = rgb
        fn = os.path.join(data_dir, f"{group_a}_onto_{group_b}_transmitters.svg")
        fig.savefig(fn)
        # plt.figure()
        # # test: plot these colors to see if they're correct
        # plt.scatter(np.arange(len(transmitters)), np.ones_like(transmitters), color=rgb)
        # plt.show()
        # breakpoint()
        if transmitters:
            for transmitter, color in zip(transmitter_types, rgb):
                # make a linear colormap from white to the color for the 2d histogram
                cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', ['white', color], N=256)
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
                for missing_id in missing_pre: pivot = pivot.append({start_col: missing_id, stop_col: -1, 'weight': 0}, ignore_index=True)
                for missing_id in missing_post: pivot = pivot.append({start_col: -1, stop_col: missing_id, 'weight': 0}, ignore_index=True)
                edges_pivot = pivot.pivot(index=start_col, columns=stop_col, values='weight')
                # replace NaNs with 0s
                edges_pivot.fillna(0, inplace=True)
                if sort_by in ['nblast', 'hungarian']:
                    pivot_sorted = edges_pivot.iloc[main_order, other_order]
                else:
                    pivot_sorted = edges_pivot.loc[main_order, other_order]
                transmitter_dfs[transmitter] = pivot_sorted
                # make a heat map with the rows and columns sorted by the corresponding order
                synapse_hist = Hist2D()
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
            # we need twice as many columns to plot the counts per source and target cell
            fig, axes = plt.subplots(ncols=num_cols, nrows=num_rows, constrained_layout=True, figsize=(2*num_cols, 2*num_rows))
            # add black the front of rgb list
            rgb = np.insert(rgb, 0, [0, 0, 0], axis=0)
            # 
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
            fn = os.path.join(data_dir, f"{group_a}_onto_{group_b}_simple.svg")
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
    def __init__(self, edges_df):
        """Handle a collection of edges with some useful operations.

        Parameters
        ----------
        edges_df : DataFrame
            The edges dataframe containing the 'pre', 'post', 'weight', and 'hop' columns.
        Raises
        ------
        AssertionError
            If any of the required columns ('pre', 'post', 'weight', 'hop') are missing in the edges dataframe.
        """
        # store the edges dataframe. It must have 'pre', 'post', 'weight', and 'hop' columns
        cols_needed = ['pre', 'post', 'weight', 'hop']
        assert all([col in edges_df.columns for col in cols_needed]), f"edges_df needs to include {cols_needed}"
        # store the df
        self.edges_df = edges_df
        # make a graph using networkx
        self.graph = nx.from_pandas_edgelist(edges_df, source='pre', target='post', edge_attr=['weight', 'neuropil', 'transmitter', 'hop'], create_using=nx.DiGraph)
        # get cell info for each node and calculate the level of each node 
        # (0 being the starting set, which is the pre column of hop=1 or post column of hop=-1)
        self.node_ids = np.array(self.graph.nodes)
        self.node_ids.sort()
        retries = 5
        for attempt in range(retries):
            try:
                self.node_info = flywire.search_annotations(self.node_ids)
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    breakpoint()
        self.node_info.sort_values('root_id', inplace=True)
        # get the hop number from the graph. if absent, then this must be level 0
        levels = []
        degrees = []
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
        # store
        self.node_info['level'] = levels
        self.node_info['degree'] = degrees

    def monte_carlo(self, reps=1e5):
        """Run a monte carlo simulation to find primary paths.
        
        This will start from the lowest level cells.

        Parameters
        ----------
        reps : int, default=1e6
            The number of repetitions per starting cell. 

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
        levels = np.arange(max_level + 1)
        # each cell in the following matrix will correspond to the index of the node in the node_info dataframe
        paths = np.zeros((len(initial_nodes), len(levels), int(reps)), dtype='int64')
        paths.fill(-1)
        # run through the nodes in batches
        current_nodes = initial_nodes.root_id.values
        paths[:, 0] = np.searchsorted(all_nodes, current_nodes[:, None])
        # for each id, get the downstream partners and use a weighted distribution for sampling reps times
        for node_num, root_id in enumerate(current_nodes):
            # go through each level above 0
            for level in levels[1:]:
                pre_nodes = paths[node_num, level-1]
                # use the unique nodes to get the next nodes without redundancy
                node_set, indices, counts = np.unique(pre_nodes, return_index=True, return_counts=True)
                # get the next nodes for each unique node
                node_vals = all_nodes[node_set]
                starts = indices
                stops = np.append(indices[1:], [None])
                for node_val, node_ind, count in zip(node_vals, node_set, counts):
                    if node_val in self.graph:
                        # get the next nodes
                        post_nodes = self.graph[node_val]
                        if len(post_nodes) > 0:
                            # get a weighted random sample of the next nodes
                            weights = np.array([node['weight'] for node in post_nodes.values()], dtype=float) 
                            weights /= weights.sum()
                            sample = np.random.choice(post_nodes, size=int(count), p=weights)
                            # get the indices of the sample
                            inds = pre_nodes == node_ind
                            paths[node_num, level, inds] = sorted(np.searchsorted(all_nodes, sample))
        # store the results
        self.monte_carlo_paths = paths
        self.monte_carlo_node_ids = all_nodes
        # return the paths and nodes
        return all_nodes, paths

    # def get_all_unique_paths(self):
    #     """Trace all paths starting from the lowest level.

    #     Note: this will keep a running list of all paths, which can take
    #     a lot of memory.
    #     """

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