"""
Module for creating the SpatialGraph class.

The SpatialGraph class is a child class of NetworkX's Graph class
and inherits all its functionality. It has the additional
attributes dists, locs and part which to store the
pairwise distances between nodes, node spatial locations
and the graph's community structure, if known.


Additionally it has two class methods which allow it to be
instantiated as a random spatial network from a list of parameters.
"""

import networkx as nx
from networkx import Graph, DiGraph, isolates
import matplotlib.pyplot as plt
import networkx.convert_matrix as nxmat
from sklearn.metrics import pairwise
import numpy as np
from numpy import array
from numpy.linalg import norm
import scipy.sparse as sp
import pyperclip


def decay_func(dists, ell=1., method="invpow"):
    """
    Compute the distance decay function according specified method.


    Parameters:
    ----------
    dists : numpy.ndarray
        matrix of distances between nodes
    ell : float or string, optional (default=1.)
        parameter to use in distance decay function. If string must
        be one of ['mean', 'max'] and uses mean or max of dists
    method : string, optional (default='invpow')
        distay decay function to use. If 'invpow' computes
        elementwise dists^{-ell}, if 'invexp' computes
        elementwise exp(-dists * ell)
    """
    if method not in ["invpow", "invexp"]:
        raise ValueError(f"'{method} not a valid distance decay method")
    if type(ell) == str:
        if ell in ["max", "mean"]:
            ell = 1 / getattr(np, ell)(dists[dists != np.inf])
        else:
            raise ValueError(f"invalid entry '{ell}' for ell:"
                             " must be a float or one of ['max', 'mean']")
    # actually calculate distance decay matrix
    if method == "invpow":
        return dists**(-ell), ell
    elif method == "invexp":
        return np.exp(-dists * ell), ell


class SpatialDiGraph(DiGraph):
    def __init__(self):
        super().__init__()
        self.fmat = array([])
        self.dists = array([])
        self.part = {}
        self.locs = array([])

    def __str__(self):
        out_str = "SpatialDiGraph with "\
            f"{self.number_of_nodes()} nodes and "\
            f"{self.number_of_edges()} edges\n"
        return out_str

    def add_distances(self, dists):
        if type(dists) != np.ndarray:
            raise TypeError("dists must be a numpy.ndarray not"
                            f"a {type(dists).__name__}")
        self.dists = dists

    def add_flows(self, fmat):
        if type(fmat) != np.ndarray:
            raise TypeError("flow mat must be a numpy.ndarray not "
                            f"a {type(fmat).__name__}")
        self.fmat = fmat

    def copy(self):
        # __copy__() wasn't being implemented when I tried
        new_graph = super().copy()
        new_graph.dists = self.dists.copy()
        new_graph.fmat = self.fmat.copy()
        return new_graph

    def export_matrices(self, out_fmat=None, out_dmat=None,
                        copy_to_clip=True):
        """
        Produce a sparse flow matrix .npz file and a distance matrix .npy file.

        Parameters:
        -----------
        out_fmat : str, optional (default=None)
            path (without post fixes) to save fmat to. If None saves to the
            current directory with name. 'spatialgraph_fmat'
        out_dmat : str, optional (default=None)
            path (without post fixes) to save fmat to. If None saves to the
            current directory with name. 'spatialgraph_dmat'

        Examples:
        ---------
        >>> g = SpatialGraph.from_gravity_benchmark(0.5, 1)
        >>> g.export()
        >>> fmat = sp.load_npz("spatialgraph_fmat.npz")
        >>> dmat = np.load("spatialgraph_dmat.npy")
        >>> print(fmat)
        >>> print(dmat)
        """
        if out_fmat is None:
            out_fmat = "spatialgraph_fmat"
        if out_dmat is None:
            out_dmat = "spatialgraph_dmat"
        sp.save_npz(out_fmat, sp.csr_matrix(self.fmat))
        np.save(out_dmat, self.dists)
        if copy_to_clip:
            pyperclip.copy(out_fmat)
        print(f"\nsuccessfully exported {type(self).__name__} "
              f"distance and flow files as:\n"
              f"    flow matrix: {out_fmat}.npz\n"
              f"    dist matrix: {out_dmat}.npy\n\n"
              f"    path to flow_mat copied to clipboard\n")

    def export_matrices_new(self, outpath=None,
                            copy_to_clip=True):
        """
        Save flow matrix, distance matrix and partition to .npz file.

        Parameters:
        -----------
        outpath : str, optional (default=None)
            path (without post fixes) to save arrays to. If None saves to the
            current directory with name. 'spatialgraph'
        copy_to_clip : bool, optional (default=True)
            whether to automatically copy the outpath to the clipboard

        Examples:
        ---------
        >>> g = SpatialGraph.from_gravity_benchmark(0.5, 1)
        >>> g.export()
        >>> TODO
        >>> print(fmat)
        >>> print(dmat)
        """
        if outpath is None:
            outpath = "spatialgraph"
        np.savez(outpath, fmat=self.fmat, dmat=self.dists, partition=self.part)
        if copy_to_clip:
            pyperclip.copy(outpath)
        print(f"\nsuccessfully exported {type(self).__name__} "
              f"flow, distance and partition files to:\n"
              f"    {outpath}.npz\n")

    def draw_spatial(self, coords=None, figsize=(20, 10), cols=None,
                     node_mag=2., edge_mag=0.1,
                     edge_col='#2db7d6', ax=None):
        if not cols:
            if self.part:
                cols = ['#2a57eb' if x ==
                        1 else '#2aebc4' for x in self.part.values()]
            else:
                cols = '#2a57eb'
        weights = [0.1 + edge_mag * self[u][v]['weight']
                   for u, v in self.edges()]
        degs = [node_mag *
                x for x in dict(self.degree(weight="weight")).values()]

        if self.locs:
            pos = self.locs
        elif coords:
            pos = coords
        else:
            raise ValueError("Specify coordinates to plot spatially")
        if ax == None:
            fig, ax = plt.subplots(figsize=figsize)
        nx.draw(self, pos, node_size=degs, node_color=cols,
                edge_color=edge_color, width=weights, alpha=0.85, ax=ax)
        limits = plt.axis('on')
        ax.tick_params(left=True, bottom=True,
                       labelleft=True, labelbottom=True)
        plt.show()

    @classmethod
    def from_numpy_array(cls, fmat, cleanup=True, **kwargs):
        """
        Construct SpatialGraph using a numpy ndarray.

        Parameters:
        -----------
        fmat : numpy.ndarray
            flow matrix used to initialise the SpatialGraph
        cleanup : bool, optional (default=True)
            remove any isolated nodes from the graph
        kwargs :
            any additional attributes to assign to the
            SpatialGraph

        Note:
        -----
        From numpy array uses the networkx method
        networkx.convert_matrix.from_numpy_array(), self
        -loops of weight 1 are represented as 1s in the
        matrix but contribute 2 to the node degree.
        """
        if sp.issparse(fmat):
            fmat = fmat.todense()
        if fmat.shape[0] != fmat.shape[1]:
            raise ValueError("flow matrix must be square")

        G = nxmat.from_numpy_array(fmat, create_using=cls)

        G.add_flows(fmat)
        # poss not best way to dot this
        for key, value in kwargs.items():
            setattr(G, key, value)

        return G

        # needs fixin'

    @classmethod
    def from_gravity_benchmark(cls, lamb: float, rho: float, N=20,
                               len=10., ell=1., decay_method="invpow", seed=None):
        """
        Create an undirected gravity benchmark spatial graph.

        Parameters:
        ----------
        lamb : float
            assortativity parameter; lamb < 1 creates a graph with assortative
            community structure while lamb > 1 creates a disassortative
            community structure.
        rho : float
            density of the edges and magnitude of the weights; the synthetic
            graph will have L = rho * N * (N - 1) edges where N is the number
            of nodes in the graph.
        N : int, optional (default=20)
            number of nodes in the graph
        len : float, optional (default=10.)
            length of the l x l square for node assignments
        ell : float, optional (default=1.)
            link probabilities between nodes distance d apart will decrease
            proportional to d^{-gamma}
        seed : int, optional (default=None)
            seed to initialise the random number generator, can be set to
            an integer for reproducible graphs

        Returns:
        -------
        g : SpatialDiGraph
            The synthetic spatial graph

        References:
        -----------
        .. 1. Expert et al., Uncovering space-independent communities in
        spatial networks. Proceedings of the National Academy of Sciences,
        vol. 108, pp. 7663--7668(2011)

        Examples:
        ---------
        >>> from community import SpatialGraph
        >>> g = SpatialGraph.from_gravity_benchmark(0.5, 2)
        """

        L = int(rho * N * (N - 1))
        nodes = [x for x in range(N)]

        # place nodes in space and calculate distances
        rng = np.random.default_rng(seed)
        nlocs = len * rng.random((N, 2))
        dists = np.array([[norm(a - b, 2) for b in nlocs] for a in nlocs])

        # assign random binary communities
        communities = rng.integers(0, 2, N, dtype=int)
        partition = {node: community for node, community in
                     zip(nodes, communities)}

        # begin making matrix of link probabilities
        probs = np.array([[1 if i == j else lamb for j in communities]
                         for i in communities], dtype=float)

        # take entries of upper triangle of probs matrix
        # NOTE: changed this a little must check it works
        upper = np.arange(N)
        mask = upper[:, None] > upper
        probs[mask] += 0.1
        probs[~mask] = np.maximum(probs[~mask]-0.1, 0)

        with np.errstate(divide='ignore'):
            dists_cost, _ = decay_func(dists, ell, decay_method)
            dists_cost[dists_cost == np.infty] = 0.
        probs = probs * dists_cost

        probs = probs.flatten()
        z = probs.sum()
        probs /= z

        fmat = np.zeros([N, N])

        indices = [(i, j) for i in range(N) for j in range(N)]
        selection = rng.choice(indices, size=L, p=probs)
        for ix in selection:
            fmat[ix[0], ix[1]] += 1

        g = cls.from_numpy_array(fmat, dists=dists)
        g.part = partition

        nlocs = {node: np.array(loc) for node, loc in zip(nodes, nlocs)}
        g.locs = nlocs

        return g

    @classmethod
    def from_cp_benchmark(cls, p=0.2, N=20, len=10, rho=1., seed=0):
        """Create a spatial network with core-periphery structure.

        TODO: tidy docs
        Construct a synthetic spatial network that combines the directed core-periphery
        block structure of [1] and the spatial effects of [2].
        Parameters:
        ----------
        p : float
            set probabilites for directed core-periphery block matrix of probabilities.
            p=0 returns an Erdos-Renyi random graph and p=0.5 returns the idealised
            block structure as in [1].


        References:
        ----------
        ..[1] Elliot et al., Core–periphery structure in directed networks
        """

        L = int(rho * N * (N - 1))
        nodes = [x for x in range(N)]

        # assign to cores and peripheries
        n = N // 4
        part = np.zeros(N)
        part[0:n] = 0              # out-periphery
        part[n: 2 * n] = 1         # in-core
        part[2 * n: 3 * n] = 2     # out-core
        part[3 * n:] = 3           # in-periphery

        # idealised block matrix
        M = np.zeros([N, N])  # noqa
        M[:, n: 2 * n] = 1.
        M[2 * n: 3 * n, :] = 1.
        M[:, :n] = 0.
        M[3 * n:, :] = 0.

        # matrix of probabilities based on community
        pmat = (0.5 + p) * M + (0.5 - p) * (1 - M)

        # same as Rodrigo did (a little confused)
        i, j = np.triu_indices_from(pmat, k=1)  # i < j
        r, s = np.tril_indices_from(pmat, k=-1)  # i > j
        i = np.concatenate((i, r))
        j = np.concatenate((j, s))

        probas = pmat[i, j]
        probas /= probas.sum()  # normalizations

        rng = np.random.default_rng(seed)
        draw = rng.multinomial(L, probas)
        (idx,) = draw.nonzero()
        fmat = sp.coo_matrix((draw[idx], (i[idx], j[idx])), shape=(N, N))

        fmat = fmat.toarray()
        partition = {node: int(community) for node, community in
                     zip(nodes, part)}

        # construct the SpatialGraph
        g = cls.from_numpy_array(fmat)
        g.part = partition
        return g

    @classmethod
    def from_cpsp_benchmark(cls, p=0.2, N=20, len=10, rho=1., ell=2., seed=0):
        """Create a spatial network with core-periphery structure.

        TODO: tidy docs
        Construct a synthetic spatial network that combines the directed core-periphery
        block structure of [1] and the spatial effects of [2].
        Parameters:
        ----------
        p : float
            set probabilites for directed core-periphery block matrix of probabilities.
            p=0 returns an Erdos-Renyi random graph and p=0.5 returns the idealised
            block structure as in [1].
        ell : float
            parameter for the distance decay function as in [2]


        References:
        ----------
        ..[1] Elliot et al., Core–periphery structure in directed networks
        ..[2] Expert et al.,
        """

        L = int(rho * N * (N - 1))
        nodes = [x for x in range(N)]

        # place nodes in space and calculate distances
        rng = np.random.default_rng(seed)
        nlocs = len * rng.random((N, 2))
        dmat = np.array([[norm(a - b, 2) for b in nlocs] for a in nlocs])

        # assign to cores and peripheries
        n = N // 4
        part = np.zeros(N)
        part[0:n] = 0              # out-periphery
        part[n: 2 * n] = 1         # in-core
        part[2 * n: 3 * n] = 2     # out-core
        part[3 * n:] = 3           # in-periphery

        # idealised block matrix
        M = np.zeros([N, N])  # noqa
        M[:, n: 2 * n] = 1.
        M[2 * n: 3 * n, :] = 1.
        M[:, :n] = 0.
        M[3 * n:, :] = 0.

        # matrix of probabilities based on community
        pmat = (0.5 + p) * M + (0.5 - p) * (1 - M)
        # add random space same as Expert
        with np.errstate(divide='ignore'):
            pmat /= dmat ** (-ell)
            pmat[pmat == np.inf] = 0.

        # same as Rodrigo did (a little confused)
        i, j = np.triu_indices_from(pmat, k=1)  # i < j
        r, s = np.tril_indices_from(pmat, k=-1)  # i > j
        i = np.concatenate((i, r))
        j = np.concatenate((j, s))

        probas = pmat[i, j]
        probas /= probas.sum()  # normalisations

        draw = rng.multinomial(L, probas)
        (idx,) = draw.nonzero()
        fmat = sp.coo_matrix((draw[idx], (i[idx], j[idx])), shape=(N, N))

        fmat = fmat.toarray()
        nlocs = {node: np.array(loc) for node, loc in zip(nodes, nlocs)}
        partition = {node: int(community) for node, community in
                     zip(nodes, part)}

        # construct the SpatialGraph
        g = cls.from_numpy_array(fmat, dists=dmat)
        g.part = partition
        g.locs = nlocs

        return g


class SpatialGraph(Graph):
    """
    The SpatialGraph class is a child of NetworkX's Graph class.

    The SpatialGraph class is a child class of NetworkX's Graph class
    and inherits all its functionality. It has the additional
    attributes dists, locs and part which to store the
    pairwise distances between nodes, node spatial locations
    and the graph's community structure, if known.

    Additionally it has two class methods which allow it to be
    instantiated as a random spatial network from a list of parameters.

    Parameters
    ----------
    incoming_graph_data : input graph, optional (default=None)
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, NumPy matrix
        or 2d ndarray, SciPy sparse matrix, or PyGraphviz graph.
    attr : keyword arguments, optional (default=no attributes)
        Attributes to add to graph as key=value pairs.

    Examples:
        ---------
    >>> from community import SpatialGraph
    >>> g = SpatialGraph.from_gravity_benchmark(0.5, 2)
    """

    def __init__(self):
        super().__init__()
        self.fmat = array([])
        self.dists = array([])
        self.part = {}
        self.locs = array([])

    def __str__(self):
        out_str = "SpatialGraph with "\
            f"{self.number_of_nodes()} nodes and "\
            f"{self.number_of_edges()} edges\n"
        return out_str

    def add_distances(self, dists):
        if type(dists) != np.ndarray:
            raise TypeError("dists must be a numpy.ndarray not"
                            f"a {type(dists).__name__}")
        self.dists = dists

    def add_flows(self, fmat):
        if type(fmat) != np.ndarray:
            raise TypeError("flow mat must be a numpy.ndarray not "
                            f"a {type(fmat).__name__}")
        self.fmat = fmat

    def copy(self):
        # __copy__() wasn't being implemented when I tried
        new_graph = super().copy()
        new_graph.dists = self.dists.copy()
        new_graph.fmat = self.fmat.copy()
        return new_graph

    def export_matrices(self, out_fmat=None, out_dmat=None,
                        copy_to_clip=True):
        """
        Produce a sparse flow matrix .npz file and a distance matrix .npy file.

        Parameters:
        -----------
        out_fmat : str, optional (default=None)
            path (without post fixes) to save fmat to. If None saves to the
            current directory with name. 'spatialgraph_fmat'
        out_dmat : str, optional (default=None)
            path (without post fixes) to save fmat to. If None saves to the
            current directory with name. 'spatialgraph_dmat'

        Examples:
        ---------
        >>> g = SpatialGraph.from_gravity_benchmark(0.5, 1)
        >>> g.export()
        >>> fmat = sp.load_npz("spatialgraph_fmat.npz")
        >>> dmat = np.load("spatialgraph_dmat.npy")
        >>> print(fmat)
        >>> print(dmat)
        """
        if out_fmat is None:
            out_fmat = "spatialgraph_fmat"
        if out_dmat is None:
            out_dmat = "spatialgraph_dmat"
        sp.save_npz(out_fmat, sp.csr_matrix(self.fmat))
        np.save(out_dmat, self.dists)
        if copy_to_clip:
            pyperclip.copy(out_fmat)
        print(f"\nsuccessfully exported {type(self).__name__} "
              f"distance and flow files as:\n"
              f"    flow matrix: {out_fmat}.npz\n"
              f"    dist matrix: {out_dmat}.npy\n\n"
              f"    path to flow_mat copied to clipboard\n")

    def export_matrices_new(self, outpath=None,
                            copy_to_clip=True):
        """
        Save flow matrix, distance matrix and partition to .npz file.

        Parameters:
        -----------
        outpath : str, optional (default=None)
            path (without post fixes) to save arrays to. If None saves to the
            current directory with name. 'spatialgraph'
        copy_to_clip : bool, optional (default=True)
            whether to automatically copy the outpath to the clipboard

        Examples:
        ---------
        >>> g = SpatialGraph.from_gravity_benchmark(0.5, 1)
        >>> g.export_matrices_new('test')
        >>> test = np.load('test.npz', allow_pickle=True)
        >>> fmat = test['fmat']
        >>> dmat = test['dmat']
        >>> partition = test['partition'][()]
        """
        if outpath is None:
            outpath = "spatialgraph"
        if self.part:
            np.savez(outpath, fmat=self.fmat,
                     dmat=self.dists, partition=self.part)
        else:
            np.savez(outpath, fmat=self.fmat, dmat=self.dists)
        if copy_to_clip:
            pyperclip.copy(f"{outpath}.npz")
        print(f"\nsuccessfully exported {type(self).__name__} "
              f"flow, distance and partition files to:\n"
              f"    {outpath}.npz\n")

    def draw_spring(self, figsize=(20, 10), cols=None, node_mag=2, edge_mag=0.1):
        """Quickly draw graph with spring-block layout."""
        plt.subplots(figsize=(20, 10))
        degs = [node_mag *
                x for x in dict(self.degree(weight="weight")).values()]
        pos = nx.drawing.layout.spring_layout(
            self, weight="weight", iterations=50)
        if not cols:
            if self.part:
                cols = ['#2a57eb' if x ==
                        1 else '#2aebc4' for x in self.part.values()]
            else:
                cols = '#2a57eb'
        weights = [0.1 + edge_mag * self[u][v]['weight']
                   for u, v in self.edges()]
        nx.draw(self, pos, node_size=degs, node_color=cols,
                edge_color='#2db7d6', width=weights, alpha=0.85)

    def draw_spatial(self, coords=None, figsize=(20, 10), cols=None, node_mag=2., node_add=0., edge_mag=0.1, ax=None, edge_col='#2db7d6'):
        if not cols:
            if self.part:
                cols = ['#2a57eb' if x ==
                        1 else '#2aebc4' for x in self.part.values()]
            else:
                cols = '#2a57eb'
        weights = [0.1 + edge_mag * self[u][v]['weight']
                   for u, v in self.edges()]
        degs = [node_add + node_mag *
                x for x in dict(self.degree(weight="weight")).values()]

        if self.locs:
            pos = self.locs
        elif coords:
            pos = coords
        else:
            raise ValueError("Specify coordinates to plot spatially")
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        nx.draw(self, pos, node_size=degs, node_color=cols,
                edge_color=edge_col, width=weights, alpha=0.85, ax=ax)
        limits = plt.axis('on')
        ax.tick_params(left=True, bottom=True,
                       labelleft=True, labelbottom=True)
        plt.show()

    @classmethod
    def from_numpy_array(cls, fmat, cleanup=True, **kwargs):
        """
        Construct SpatialGraph using a numpy ndarray.

        Parameters:
        -----------
        fmat : numpy.ndarray
            flow matrix used to initialise the SpatialGraph
        cleanup : bool, optional (default=True)
            remove any isolated nodes from the graph
        kwargs :
            any additional attributes to assign to the
            SpatialGraph

        Note:
        -----
        From numpy array uses the networkx method
        networkx.convert_matrix.from_numpy_array(), self
        -loops of weight 1 are represented as 1s in the
        matrix but contribute 2 to the node degree.
        """
        if sp.issparse(fmat):
            fmat = fmat.todense()
        if fmat.shape[0] != fmat.shape[1]:
            raise ValueError("flow matrix must be square")

        G = nxmat.from_numpy_array(fmat, create_using=cls)

        G.add_flows(fmat)
        for key, value in kwargs.items():
            setattr(G, key, value)

        return G

    @classmethod
    def from_gravity_benchmark(cls, lamb: float, rho: float, N=20,
                               len=10., ell=1., decay_method="invpow", seed=None):
        """
        Create an undirected gravity benchmark spatial graph.

        Parameters:
        ----------
        lamb : float
            assortativity parameter; lamb < 1 creates a graph with assortative
            community structure while lamb > 1 creates a disassortative
            community structure.
        rho : float
            density of the edges and magnitude of the weights; the synthetic
            graph will have L = rho * N * (N - 1) edges where N is the number
            of nodes in the graph.
        N : int, optional (default=20)
            number of nodes in the graph
        len : float, optional (default=10.)
            length of the l x l square for node assignments
        ell : float, optional (default=1.)
            link probabilities between nodes distance d apart will decrease
            proportional to d^{-gamma}
        seed : int, optional (default=None)
            seed to initialise the random number generator, can be set to
            an integer for reproducible graphs

        Returns:
        -------
        g : SpatialGraph
            The synthetic spatial graph

        References:
        -----------
        .. 1. Expert et al., Uncovering space-independent communities in
        spatial networks. Proceedings of the National Academy of Sciences,
        vol. 108, pp. 7663--7668(2011)

        Examples:
        ---------
        >>> from community import SpatialGraph
        >>> g = SpatialGraph.from_gravity_benchmark(0.5, 2)
        """

        L = int(rho * N * (N - 1) / 2)
        nodes = [x for x in range(N)]

        # place nodes in space and calculate distances
        rng = np.random.default_rng(seed)
        nlocs = len * rng.random((N, 2))
        dists = np.array([[norm(a - b, 2) for b in nlocs] for a in nlocs])

        # assign random binary communities
        communities = rng.integers(0, 2, N, dtype=int)
        partition = {node: community for node, community in
                     zip(nodes, communities)}

        # begin making matrix of link probabilities
        probs = np.array([[1 if i == j else lamb for j in communities]
                         for i in communities], dtype=float)

        with np.errstate(divide='ignore'):
            dists_cost, _ = decay_func(dists, ell, decay_method)
            dists_cost[dists_cost == np.infty] = 0.

        probs = probs * dists_cost
        fmat = np.zeros([N, N])

        # take entries of lower triangle of probs matrix
        upper = np.arange(N)
        mask = upper[:, None] > upper
        probs = probs[mask]
        z = probs.sum()
        probs /= z
        indices = [(i, j) for i in range(N) for j in range(i)]
        selection = rng.choice(indices, size=L, p=probs)
        for ix in selection:
            fmat[ix[0], ix[1]] += 1
        fmat += fmat.T  # symmetrise flow matrix

        g = cls.from_numpy_array(fmat, dists=dists)
        g.part = partition

        nlocs = {node: np.array(loc) for node, loc in zip(nodes, nlocs)}
        g.locs = nlocs

        return g

    @classmethod
    def from_cerina_benchmark(cls, N, rho, ell, beta, epsilon, L=1.0, directed=False, seed=0):
        """Create a benchmark network of the type proposed by Cerina et al.

        Code by Rodrigo Leal Cervantes: Rodrigo.LealCervantes@maths.ox.ac.uk.
        """

        nb_edges = int(N * (N - 1) * rho)
        if not directed:
            nb_edges //= 2

        rng = np.random.RandomState(seed)

        # Coordinates
        ds = rng.exponential(scale=ell, size=N)
        alphas = 2 * np.pi * rng.rand(N)
        shift = L * np.ones(N)
        shift[N // 2:] *= -1  # // makes it an int!

        xs = ds * np.cos(alphas) + shift
        ys = ds * np.sin(alphas)
        coords = np.vstack((xs, ys)).T

        # Attibute assignment
        idx_plane = xs > 0
        # which are correctly attributed
        idx_success = rng.rand(N) < 1 - epsilon

        n = N // 2
        comm_vec = np.zeros(N, dtype=int)
        comm_vec[np.bitwise_and(idx_plane, idx_success)] = 1
        comm_vec[np.bitwise_and(idx_plane, ~idx_success)] = -1
        comm_vec[np.bitwise_and(~idx_plane, idx_success)] = -1
        comm_vec[np.bitwise_and(~idx_plane, ~idx_success)] = 1

        # Edge selection
        smat = comm_vec[:, np.newaxis] * comm_vec  # vec of attributes
        dmat = pairwise.euclidean_distances(coords)
        pmat = np.exp(beta * smat - dmat / ell)

        i, j = np.triu_indices_from(pmat, k=1)  # i < j
        if directed:
            r, s = np.tril_indices_from(smat, k=-1)  # i > j
            i = np.concatenate((i, r))
            j = np.concatenate((j, s))

        probas = pmat[i, j]
        probas /= probas.sum()  # normalization

        draw = rng.multinomial(nb_edges, probas)
        (idx,) = draw.nonzero()
        fmat = sp.coo_matrix((draw[idx], (i[idx], j[idx])), shape=(N, N))
        if not directed:
            fmat = (fmat + fmat.T).tocoo()  # addition changes to csr

        # change to ndarray for us
        fmat = fmat.toarray()

        # more useful values in attribute vector
        comm_vec[comm_vec == -1] = 0
        # put things in nice dictionaries for g
        part = {node: com for node, com in zip(range(N), comm_vec)}
        locs = {node: coord for node, coord in zip(range(N), coords)}

        # construct the SpatialGraph
        g = cls.from_numpy_array(fmat, dists=dmat)
        g.part = part
        g.locs = locs

        return g

    @classmethod
    def from_kojaku_benchmark(cls, lamb=0.1, mu=0.01, rho=1, N=200, length=10, ell=2., seed=0):
        """Generate synthetic network with multiple core-periphery pairs.

        Network formulation inspired by [1]
        Parameters:
        -----------
        lamb : float
            'strength' of CP structure. lamb=0 corresponds to ideal CP
            structure.
        mu : float
            discrepancy between degree distribution in cores vs. in
            peripheries;
        rho : float
            density of the edges and magnitude of the weights; the synthetic
            graph will have L = rho * N * (N - 1) edges where N is the number
            of nodes in the graph.
        N : int, optional (default=200)
            number of nodes in the graph
        length : float, optional (default=10.)
            length of the l x l square for node assignments
        ell : float, optional (default=1.)
            link probabilities between nodes distance d apart will decrease
            proportional to d^{-gamma}
        seed : int, optional (default=None)
            seed to initialise the random number generator, can be set to
            an integer for reproducible graphs


        References:
        ..[1] Kojaku, S., & Masuda, N. (2018). Core-periphery structure requires something else in the network
        """

        num_cores = 4  # int # NOTE: can't change yet
        rng = np.random.default_rng(seed)

        nodes = [x for x in range(N)]
        # assign communities and CPs
        coms = [0 if node < N//2 else 1 for node in nodes]
        partition = [0] * N
        partition = [x for i in range(num_cores) for x in [i] * (N//num_cores)]
        cps = [(x + 1) % 2 for x in partition]
        # [ core, periph, core, periph]

        # generate expected degrees
        k = [200 if cp >= 1 else 50 for cp in cps]
        kmap = {200: 50, 50: 200}
        switch_nodes = rng.choice(nodes, size=int(N * mu), replace=False)
        k = [kmap[k] if node in switch_nodes else k for node,
             k in zip(nodes, k)]

        # calculate total degree per community
        D = {}
        for node in nodes:
            com = coms[node]
            cp = cps[node]
            D[(com, cp)] = D.get((com, cp), 0.) + k[node]

        # calculate planted edge weights (lower triangle)
        m_plant = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if coms[i] != coms[j]:
                    # if different communities
                    m_plant[i, j] = 0.
                else:
                    com = coms[i]
                    if cps[i] != cps[j]:
                        # D_periph
                        m_plant[i, j] = D[(com, 0)]
                    elif cps[i] == 0:
                        m_plant[i, j] = 0.
                    elif cps[i] == 1:
                        m_plant[i, j] = D[(com, 1)] - D[(com, 0)]

        # calculate config model for m_rand
        M = sum(k)  #  technically 2 * M
        m_rand = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                m_rand[i, j] = D[(coms[i], cps[i])] * D[(coms[j], cps[j])] / M

        # combine to give expected edges
        pmat = lamb * m_rand + (1 - lamb) * m_plant

        # add distance component
        nlocs = length * rng.random((N, 2))
        dmat = np.array([[norm(a - b, 2) for b in nlocs] for a in nlocs])
        with np.errstate(divide='ignore'):
            pmat /= dmat ** (-ell)
            pmat[pmat == np.inf] = 0.

        # make network
        i, j = np.triu_indices_from(pmat, k=1)  # i < j
        probas = pmat[i, j]
        probas /= probas.sum()  # normalizations

        L = int(rho * N * (N - 1))
        rng = np.random.default_rng(seed)
        draw = rng.multinomial(L, probas)
        (idx,) = draw.nonzero()
        fmat = sp.coo_matrix((draw[idx], (i[idx], j[idx])), shape=(N, N))
        fmat = fmat.toarray()
        fmat += fmat.T

        nlocs = {node: np.array(loc) for node, loc in zip(nodes, nlocs)}
        # community assignments
        partition = {node: int(community) for node, community in
                     zip(nodes, coms)}

        # boolean for is node is a core node
        cps = {node: int(cp) for node, cp in zip(nodes, cps)}

        # construct the SpatialGraph
        g = cls.from_numpy_array(fmat, dists=dmat)
        g.part = partition
        g.cores = cps
        g.locs = nlocs
        return g
