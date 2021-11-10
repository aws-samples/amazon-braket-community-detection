# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import collections
import datetime
import minorminer
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import time
import warnings

from copy import deepcopy
from dwave_qbsolv import QBSolv
from dwave.system.composites import FixedEmbeddingComposite
from braket.ocean_plugin import BraketDWaveSampler
from collections import defaultdict
from networkx.algorithms import community

from qubo_community import qubo_matrix_community_sparse, qbsolv_response_to_community


def create_qubo_dict(nx_G, k, alpha=5):
    """
    Create a QUBO matrix in dict format for community detection

    :param nx_G: networkX graph
    :param k: int, number of communities to detect
    :param alpha: int, constraint coefficient to force a solution with one node for one community assignment only
    :return: QUBO sparse matrix in dict format
    """
    
    qubo_matrix_sparse = qubo_matrix_community_sparse(nx_G, k, alpha=alpha)

    indices = list(zip(qubo_matrix_sparse.row, qubo_matrix_sparse.col))
    values = qubo_matrix_sparse.data

    qubo_sparse = defaultdict(int)
    for idx, val in zip(indices, values):
        qubo_sparse[idx] = val

    print(f"The size of the QUBO matrix in dictionary format for {k}-community is {len(qubo_sparse)}")
    
    return qubo_sparse


class QbsolvCommunity(object):
    """
    Use QBSolv to solve community detection via either the classical solver or the hybrid solver 
    """
    def __init__(self, graph, solver_limit=40, num_repeats=1, num_reads=1000, seed=1, alpha=5):
        
        """
        Set hyperparameter values for QBSolv

        :param graph: a networkX graph
        :param solver_limit: int, the maximum number of variables (n) for sub-QUBOs
        :param num_repeats: int, the maximum iterations to repeat QBSolv solver execution to discover a new best solution
        :param num_reads: int, the number of times the annealing to be performed
        :param seed: int, random seed value
        :param alpha: int, the penalty coefficient to enforce assigning only one community to each node
        """
        
        self.graph = graph
        self.solver_limit = solver_limit
        self.num_repeats = num_repeats
        self.num_reads = num_reads
        self.seed = seed
        self.alpha = alpha
    
    def solve_classical(self, num_comm):
        """
        Call QUBO classical solver for community detection

        :param num_comm: int, number of communities to solve for
        :return: dict, two dictionaries for graph's community results and QBSolv reponse results
        """
        t0 = time.time()

        q_dict = create_qubo_dict(self.graph, num_comm, self.alpha)
    
        # execute optimization task using QBSolv classical (run on your local notebook instance) 
        response_classical = QBSolv().sample_qubo(
            q_dict, num_repeats=self.num_repeats, solver_limit=self.solver_limit, seed=self.seed
        )
        print(f"Mode: Classical, time spent is {round(time.time()-t0, 2)} seconds for {self.num_repeats} repetitions")
        print(response_classical)
        
        # extract the best solution that has the lowest energy
        sample_classical = np.array(list(response_classical.first.sample.values()))
        comm_classical = qbsolv_response_to_community(self.graph, sample_classical, num_comm)
        
        return comm_classical, response_classical
    
    def solve_hybrid(self, num_comm, s3_folder, device_arn):
        """
        Call QUBO hybrid solver for community detection
        
        :param num_comm: int, number of communities to solve for
        :param s3_folder: str, the Amazon Braket S3 path to store solver response files
        :return: dict, two dictionaries for graph's community results and QBSolv reponse results
        :param device_arn: str, D-Wave QPU Device ARN (only needed for QBSolv Hybrid solver)
        """
        q_dict = create_qubo_dict(self.graph, num_comm, self.alpha)
        self.qpu_cost_warning()
        
        execution = input("Continue to execute QBSolv Hybrid job: Y or N?")
        
        if execution.lower() in ["y", "yes"]:
            t0 = time.time()
            system = BraketDWaveSampler(s3_folder, device_arn)

            # find embedding of subproblem-sized complete graph to the QPU
            G_sub = nx.complete_graph(self.solver_limit)
            embedding = minorminer.find_embedding(G_sub.edges, system.edgelist)

            # use the FixedEmbeddingComposite() method with a fixed embedding
            solver = FixedEmbeddingComposite(system, embedding)

            # execute optimization task using QBSolv hybrid on D-Wave QPU 
            response_hybrid = QBSolv().sample_qubo(q_dict, solver=solver, num_repeats=self.num_repeats,
                                                   solver_limit=self.solver_limit, num_reads=self.num_reads,
                                                   seed=self.seed)
            print(f"Mode: Hybrid, time spent is {round(time.time()-t0, 2)} seconds for {self.num_repeats} repeats")
            print(response_hybrid)
            
            # extract the best solution that has the lowest energy
            sample_hybrid = np.array(list(response_hybrid.first.sample.values()))
            comm_hybrid = qbsolv_response_to_community(self.graph, sample_hybrid, num_comm)

            return comm_hybrid, response_hybrid
        else:
            raise ValueError("Hybrid job execution declined by the user!")

    def qpu_cost_warning(self):
        
        def custom_formatwarning(msg, *args, **kwargs):
            return str(msg) + '\n'

        warnings.formatwarning = custom_formatwarning
        stmt = f"\033[91mWARNING:\033[0m Additional cost for using D-Wave QPU. " \
               f"Please evaluate potential cost before executing this QBSolv hybrid job"
        warnings.warn(stmt)
        
        
class CommunityDetectionSolver(object):
    """
    Call QUBO classical/hybrid solver to find community structure of a graph and save results
    """
    def __init__(self, graph_name, graph, num_comm, solver_limit=100, num_repeats=1, num_reads=1000, 
                 seed=1, alpha=10, mode='classical', s3_folder='N/A', device_arn = 'N/A'):
        """
        Input for graph and QBSolv hyperparameter values
        
        :param graph_name: str, the name of a graph for result saving
        :param graph: a networkX graph
        :param num_comm: int, number of communities to solve for
        :param solver_limit: int, the maximum number of variables (n) for sub-QUBOs
        :param num_repeats: int, the maximum iterations to repeat QBSolv solver execution to discover a new best solution
        :param num_reads: int, the number of times the annealing to be performed
        :param seed: int, random seed value
        :param alpha: int, the penalty coefficient to enforce assigning only one community to each node
        :param mode, str, must be either 'classical' or 'hybrid'. Determines whether the classical or hybrid solver is called
        :param s3_folder, str, Amazon Braket S3 folder path (only needed for QBSolv Hybrid solver)
        :param device_arn: str, D-Wave QPU Device ARN (only needed for QBSolv Hybrid solver)
        
        """
        self.graph_name = graph_name
        self.graph = graph
        self.num_comm = num_comm
        self.solver_limit = solver_limit
        self.num_repeats = num_repeats
        self.num_reads = num_reads
        self.seed = seed
        self.alpha = alpha
        self.mode = mode
        self.s3_folder = s3_folder
        self.device_arn = device_arn
        
    def _solve_single_graph(self):
        """
        Call QUBO classical/hybrid solver to process a single graph for community detection
        
        :param graph: a networkX graph
        :param num_comm: int, number of communities to solve for
        """
        
        QbsolvComm = QbsolvCommunity(
            self.graph, self.solver_limit, self.num_repeats, self.num_reads, self.seed, self.alpha)
        
        t0 = time.time()
        if self.mode == 'classical':
            comm_results, response = QbsolvComm.solve_classical(self.num_comm)
        elif self.mode == 'hybrid':
            comm_results, response = QbsolvComm.solve_hybrid(self.num_comm, self.s3_folder, self.device_arn)
        else:
            raise ValueError(f"Invalid qbsolv mode {self.mode}. Mode has to be in ['classical', 'hybrid']!")

        time_spent = round(time.time()-t0, 2)
        
        return comm_results, response, time_spent

    def run_community_detection(self, save=False):
        """
        Call QUBO classical/hybrid solver for community detection and save results
        
        :param save: boolean, True or False to set whether to save results locally or not
        :return: dict, two dictionaries: the first one 'results_graph' contains graph-level results about graph, qbsolv seetings, modularity values, 
        and execution time; the second one 'track' contains node-level results about graph edge connections and community assignment.
        """
        results_graph = collections.defaultdict(list)
        timenow = str(datetime.datetime.now())[:19].replace(' ', '_')
        date_today = str(datetime.date.today())
        track = {'graphs': [], 'responses': [], 'communities': []}

        output_parent = f'./output/{date_today}'
        result_file = f"result_{self.graph_name}_{timenow}_sl{self.solver_limit}_rp{self.num_repeats}_shot{self.num_reads}_seed{self.seed}.csv"

        if save and (not os.path.exists(output_parent)):
            print(f'Creating parent folder(s): {output_parent}')
            print('Will create required sub-directories quietly')
            os.makedirs(output_parent)

        # run QBSolv for community detection
        comm_results, response, time_spent = self._solve_single_graph()
        print(f"Modularity from QBSolv with {comm_results['num_comm']} communities is {round(comm_results['modularity'], 4)}")

        # Save graph information
        results_graph['graph_name'].append(self.graph_name)
        results_graph['total_nodes'].append(self.graph.number_of_nodes())
        results_graph['num_edge'].append(self.graph.number_of_edges())
        results_graph['num_comm'].append(self.num_comm)
            
        # Save results from qbsolv
        results_graph['modu_qbsolv'].append(comm_results['modularity'])
        results_graph['num_comm_qbsolv'].append(comm_results['num_comm'])
        results_graph['wall_time_s'].append(time_spent)
        results_graph['solver_limit'].append(self.solver_limit)
        results_graph['num_repeats'].append(self.num_repeats)
        results_graph['num_reads'].append(self.num_reads)
        results_graph['seed'].append(self.seed)
        results_graph['alpha'].append(self.alpha)
        results_graph['mode'].append(self.mode)
        results_graph['s3_folder'].append(self.s3_folder)

        graph_file = f"graph_{self.graph_name}_QBS_sl{self.solver_limit}_rp{self.num_repeats}_shot{self.num_reads}_seed{self.seed}.gpickle"
        response_file = f"response_{self.graph_name}_QBS_sl{self.solver_limit}_rp{self.num_repeats}_shot{self.num_reads}_seed{self.seed}.p"
        comm_file = f"modu_{self.graph_name}_QBS_sl{self.solver_limit}_rp{self.num_repeats}_shot{self.num_reads}_seed{self.seed}.p"

        track['graphs'].append((graph_file, deepcopy(self.graph)))
        track['responses'].append((response_file, deepcopy(response)))
        track['communities'].append((comm_file, deepcopy(comm_results)))

        if save:
            self._save_results(track, results_graph, output_parent, result_file)

        return results_graph, track
        
    @staticmethod
    def _save_results(track_dict, results_graph, output_parent, result_file):
        
        results_df = pd.DataFrame.from_dict(results_graph)
        for top_key, item_list in track_dict.items():
            output_subdir = f'{output_parent}/{top_key}'
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            for item in item_list:
                out_loc = f'{output_subdir}/{item[0]}'
                print(f'Writing to disk: {out_loc}')
                if top_key == 'graphs':
                    nx.write_gpickle(item[1], out_loc)
                else:
                    pickle.dump(item[1], open(out_loc, 'wb'))

        print(f'Writing to disk: {output_parent}/{result_file}')
        results_df.to_csv(f'{output_parent}/{result_file}', index=False)