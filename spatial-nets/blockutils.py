import numpy as np
import scipy.sparse as sp
import graph_tool.all as gt
from shapely import geometry
from tqdm import tqdm

from spatial_nets import utils


def mean_comm_sizes(state: gt.BlockState, dmat: np.ndarray):
    # TODO: pass dmat as property (how about spatialNet type?)
    B = state.get_nonempty_B()

    out = np.zeros(B, dtype=int)

    for c in range(B):
        idx = state.b.a == c
        comm_ds = dmat[idx, :][:, idx]
        i, j = np.triu_indices_from(comm_ds, k=1)
        out[c] = comm_ds[i, j].mean()

    return out


def mean_composition(
    state: gt.BlockState,
    format_vec: np.ndarray,
    norm: bool = True,
):
    # TODO: pass format_vec as vertex properties
    B = state.get_nonempty_B()

    nb_formats = len(np.unique(format_vec))
    out = np.zeros((B, nb_formats), dtype=int)

    for c in range(B):
        idx = state.b.a == c
        u, cs = np.unique(format_vec[idx], return_counts=True)
        out[c, u] = cs

    if norm:
        out = out / out.sum(axis=1)[:, np.newaxis]

    return out


def jaccard_score(state: gt.BlockState, coords: np.ndarray):
    # TODO: pass coords as vertex properties
    B = state.get_nonempty_B()

    mercator_coords = utils.project_mercator
    geoms = []

    for c in range(B):
        idx = state.b.a == c
        geoms.append(geometry.MultiPoint(coords[idx]).convex_hull)

    idx_i = []
    idx_j = []
    data = []

    for i in range(B):
        for j in range(i + 1, B):
            if geoms[i].overlaps(geoms[j]):
                idx_i.append(i)
                idx_j.append(j)

                intersect = (geoms[i] & geoms[j]).area
                union = (geoms[i] | geoms[j]).area
                data.append(intersect / union)

    return sp.csr_matrix((data, (idx_i, idx_j)), shape=(B, B))


def repeat_sbm_fit(G, nb_repeats, **gt_kwargs):
    states = []
    entropies = np.zeros(nb_repeats)
    Bs = np.zeros(nb_repeats, dtype=int)

    for k in tqdm(range(nb_repeats)):
        s = gt.minimize_blockmodel_dl(G, **gt_kwargs)
        states.append(s)
        entropies[k] = s.entropy()
        Bs[k] = s.get_nonempty_B()

    return states, entropies, Bs
