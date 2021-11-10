# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os


class CommunityGraphFactory(object):
    """
    Create a graph for community detection from two options: 1) load from a local data file; 2) create a synthetic network via networkx's random partition graph
    """
    
    def __init__(self, seed=1):
        """
        Estbalish the seed for numpy's RNG downstream

        :param seed: int, seed value to use when constructing random graphs
        """
        self.seed = seed

    @staticmethod
    def _sort_nodes(nx_G):
        """
        Relabel node index in order of 0, 1, ...., N-1

        :param nx_G: networkx.classes.graph.Graph, Graph object to be sorted
        """
        nodes = sorted(nx_G.nodes())
        new_nodes = list(range(len(nodes)))
        nodes_mapping = dict(zip(nodes, new_nodes))

        return nx.relabel_nodes(nx_G, nodes_mapping)
    
    def load_graph(self, graph_name, graph_file_dict):
        """
        Load a graph by name from a dict: {graph_name: [data_file_path, delimiter]}

        :param graph_name: str, Identifier (name) of graph to be loaded in
        :param graph_file_dict: dict, Contains location and delimiter specification for known graphs
        """
        
        nx_G = nx.read_weighted_edgelist(
            graph_file_dict[graph_name][0],
            delimiter=graph_file_dict[graph_name][1],
            create_using=nx.Graph(),
            nodetype=int
        )
        print(nx.info(nx_G))
        return self._sort_nodes(nx_G)
    
    def create_random_graph(self, total_nodes, num_comm, prob_in, prob_out, seed=None):
        """
        Create a random partition graph with minimum degree >=1, save the group labels for
        graph nodes into a local file. If seed is provided, overwrite initialized seed value.

        :param total_nodes: int, Desired number of nodes in generated graph
        :param num_comm: int, Desired number of communities in graph
        :param prob_in: float, Probability of an edge between nodes in the same community
        :param prob_out: float, Probability of an edge between nodes in different communities
        :param seed: int, optional, If provided, use as seed value for generating random graphs
        """

        if seed is not None:
            self.seed = seed

        np.random.seed(self.seed)
        community_size = np.random.multinomial(total_nodes, np.ones(num_comm)/num_comm, size=self.seed)[0]
        nx_G = nx.random_partition_graph(community_size, prob_in, prob_out, self.seed)
        
        out_degree = dict(nx_G.degree(weight='weight'))
        seed_ = 1
        while (min(out_degree.values()) == 0) & (seed_ <= 50):
            seed_ += 1
            print(f"iterating graph generation with random seed {seed_}")
            nx_G = nx.random_partition_graph(community_size, prob_in, prob_out, seed=seed_)
            out_degree = dict(nx_G.degree(weight='weight'))
        
        if min(out_degree.values()) == 0:
            raise ValueError("some nodes have ZERO degree! Change random graph input settings and re-generate a graph.")
        
        print(nx.info(nx_G))
        
        # save node labels
        labels = [node_attributes['block'] for _, node_attributes in nx_G.nodes(data=True)]
        node_labels = list(enumerate(labels))
        
        if not os.path.exists('./data/synthetic'):
            os.makedirs('./data/synthetic')
        
        label_outfile = os.path.join(
            "./data/synthetic", f"node_labels_{total_nodes}_{num_comm}_{prob_in}_{prob_out}.node_labels")

        # Store node-label mapping to csv-string file
        with open(label_outfile, 'w') as file:
            file.writelines(','.join(str(j) for j in i) + '\n' for i in node_labels)
        
        return self._sort_nodes(nx_G)
    
    def draw_graph(self, nx_G):
        """
        Simply draw a graph for a graph with less than 200 nodes
        """
        
        if len(nx_G.nodes()) <= 200:
            pos = nx.kamada_kawai_layout(nx_G)
            return nx.draw(nx_G, pos, with_labels=True)
        else:
            print("Too many nodes (>= 200) to display!")


# for labeled networks
def load_node_labels(file_path, delimiter=','):
    """
    Load node label data from a local file

    :param file_path: str, a local file path to node label data for a graph
    :param delimiter: str, delimiter used to separate node records
    :return: label_nodes_dict, label_community, labels_array
    """

    with open(file_path, 'r') as f:
        node_labels = [list(map(int, line.split(delimiter))) for line in f.read().splitlines()]

    # Sort according to node values
    node_labels_sorted = sorted(node_labels, key=lambda x: x[0])
    node_labels = np.array(node_labels_sorted)

    # Center node and label IDs
    node_labels[:, 0] -= node_labels[:, 0].min()
    node_labels[:, 1] -= node_labels[:, 1].min()

    # Create one-hot encoding of labels (node classes)
    labels_array = np.zeros((node_labels.shape[0], len(set(node_labels[:, 1]))))
    labels_array[node_labels[:, 0], node_labels[:, 1]] = 1.0

    # Map {label_id: [nodes, with, that, label]}
    label_nodes_dict = {label: [node for node, label_ in node_labels if label_ == label] \
                        for label in set(node_labels[:, 1])}

    # Prep list of nodes under each community for networkX function
    label_community = list(label_nodes_dict.values())

    return label_nodes_dict, label_community, labels_array


# draw network graph with nodes colored by community groups
def draw_graph_community(nx_G, communities, comm_order=None, color_map='rainbow', color_list=[], seed=42):
    """
    Draw network with nodes colored based on community groups

    :param nx_G: networkX graph
    :param communities: list, a list with nodes grouped by communities, e.g., [{0, 1, 3}, {2, 4, 5}]
    :param comm_order: list, order of communities to map communities to colors
    :param color_map: str, one of the existing color map in matplotlib.pyplot
    :param color_list: list, a list of color names for specifying a color for a community, e.g.,
        color_list = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:cyan']
    :param seed: int, random seed for networkX layout
    :return: a draw of network
    """

    if comm_order is None:
        # re-order communities based on the sum of node index for consistent coloring:
        #  we may still have inconsistent coloring among results from different number of communities
        sum_nodes = [sum(i) for i in communities]
        comm_order = sorted(range(len(sum_nodes)), key=lambda k: sum_nodes[k])
    communities = [communities[i] for i in comm_order]

    class_map = {}
    
    if len(color_list) == 0:
        for cl in range(len(communities)):
            for n in communities[cl]:
                class_map.update({n: cl})

        class_map = dict(sorted(class_map.items()))

        pos = nx.spring_layout(nx_G, seed=seed)
        nx.draw(nx_G, cmap=plt.get_cmap(color_map), pos=pos, node_color=list(class_map.values()), with_labels=True,
                font_color='white', node_size=500, font_size=10)
    else:
        assert len(color_list) >= len(communities), \
            "Number of colors in color_list is less than the number of communities!!"
        for cl in range(len(communities)):
            for n in communities[cl]:
                class_map.update({n: color_list[cl]})

        class_map = dict(sorted(class_map.items()))

        pos = nx.spring_layout(nx_G, seed=seed)
        nx.draw(nx_G, pos=pos, node_color=list(class_map.values()), with_labels=True,
                font_color='white', node_size=500, font_size=10)
