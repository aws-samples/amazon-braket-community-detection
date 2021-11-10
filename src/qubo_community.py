# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import collections
import networkx as nx
import numpy as np
import scipy.sparse as sp

from networkx.algorithms import community


def modularity_mx(nx_G):
    """
    Create a sparse matrix for the modularity of a graph nx_G

    :param nx_G: networkX graph
    :return: scipy coo sparse matrix, the modularity matrix for nx_G with weighted edges
    """
    
    # Generate a sparse adjacency matrix using networkx
    adj = nx.adjacency_matrix(nx_G, nodelist=sorted(list(nx_G)), weight='weight')
    
    # Create a numpy array for node's degree. 
    # Here we assumed an undirected graph, so `degrees` refers to both the in-degree and out-degree
    sorted_degrees = sorted(nx_G.degree(weight='weight'))
    degrees = np.array([degree[1] for degree in sorted_degrees])
    m = sum(degrees) / 2

    # Calculate the expected number of edges between two nodes for a null model.
    # Note we use sparse matrix format here but this degree matrix is a dense one by definition
    degrees = np.expand_dims(degrees.squeeze(), axis=0)
    degree_mx = sp.csr_matrix(degrees.T).multiply(sp.csr_matrix(degrees)) / (2 * m)
    
    # Create a modularity matrix and convert it into coo sparse matrix format (Torch sparse tensor compatible)
    modu_mx_sparse = (sp.csr_matrix(adj) - degree_mx) / (2 * m)
    modu_mx_sparse = sp.coo_matrix(modu_mx_sparse)

    return modu_mx_sparse


def qubo_matrix_community_sparse(nx_G, k, alpha=5):
    """
    Create a sparse matrix as a QUBO matrix for a graph nx_G with k-community detection

    :param nx_G: networkX graph
    :param k: int, the number of communities to detect for the graph nx_G
    :param alpha: float, the penalty coefficient for the constraint term in the QBUO matrix
    :return: scipy coo sparse matrix, the QUBO matrix to minimize for a graph nx_G with k-community detection
    """
    
    # get the number of nodes for a networkx graph
    num_nodes = nx_G.number_of_nodes()

    # create the modularity matrix in coo sparse format
    modu_mx_sparse = modularity_mx(nx_G)

    # define the coefficient value for the QUBO constraint term that a node can only be in one community
    gamma_v = alpha / num_nodes
    
    # create sparse diagonal matrix for the linear constraint term in the QUBO matrix
    gamma_mx = sp.eye(num_nodes) * gamma_v
    
    # create a block diagonal matrix for k-commnuity problem where k > 2
    # this block diagonal matrix is for the linear constraint term in the QUBO matrix 
    gamma_blockmatrix_sparse = sp.block_diag([gamma_mx] * k)

    # create a k x k sparse block matrix with each block being a diagonal matrix
    # this block matrix is for the quadratic constraint term in the QUBO matrix
    constraint_mx = [[gamma_mx] * k] * k
    constraint_blockmatrix_sparse = sp.bmat(constraint_mx)
    
    # create a sparse block diagonal matrix with the diagonal value equal to the modularity matrix
    # this is the modularity matrix for k communities in QUBO format
    modu_mx_sparse_k = [modu_mx_sparse] * k
    modu_block_sparse = sp.block_diag(modu_mx_sparse_k)

    # create a QUBO sparse matrix (for minimization) by combinding the modularity matrix and the constraint
    # term matrix for a k-community problem
    q_blockmatrix_sparse = -1 * modu_block_sparse + constraint_blockmatrix_sparse - 2 * gamma_blockmatrix_sparse
    q_blockmatrix_sparse = sp.coo_matrix(q_blockmatrix_sparse)

    return q_blockmatrix_sparse

def qbsolv_response_to_community(nx_G, response_sample, k):
    """
    Extract communities from QBSolv responses and calculate its modularity

    :param nx_G: networkX graph
    :param response_sample: QBSolv responses
    :param k: int, the number of communities to detect for the graph nx_G
    :return: dict, a dictionary of node sets as community groups and the graph modularity
    """

    num_nodes = nx_G.number_of_nodes()

    # Split result out into binary assignments within each community
    result = response_sample.squeeze()
    result = result.reshape(k, num_nodes)

    # Extract node IDs belonging to each community, based on results
    communities = []
    for i in range(k):
        node_c = np.where(result[i] == 1)[0]
        if len(node_c) > 0:
            communities.append(set(node_c))

    # Check if there is multi-community assignment for a node or a node without community assignment
    for i in range(num_nodes):
        if result[:, i].sum() > 1:
            raise ValueError('Multi-community assignment!')
            break
        if result[:, i].sum() == 0:
            raise ValueError('Node without community assignment!')
            break
    
    # Order communities according to lowest-ID nodes in each set, ascending order
    communities.sort(key=min)

    modu = community.modularity(nx_G, communities)
    k = len(communities)  # in case any communities returned no hits

    return {"modularity": modu, "num_comm": k, "comm": communities}
