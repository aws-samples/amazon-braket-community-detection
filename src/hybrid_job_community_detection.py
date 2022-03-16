# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
from braket.jobs import save_job_result
from braket.jobs.metrics import log_metric
import json
import networkx as nx

# Load community detection specific library
from src.qbsolv_community import QbsolvCommunity

def main():
    # Print statements can be viewed in cloudwatch
    print(os.environ)

    input_dir = os.environ["AMZN_BRAKET_INPUT_DIR"]
    hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
    job_name = os.environ["AMZN_BRAKET_JOB_NAME"]
    s3_bucket = os.environ["AMZN_BRAKET_OUT_S3_BUCKET"]
    device_arn = os.environ["AMZN_BRAKET_DEVICE_ARN"]

    # Read the hyperparameters
    with open(hp_file, "r") as f:
        hyperparams = json.load(f)
    print(hyperparams)

    # Graph related parameters
    input_graph_file = str(hyperparams["input_graph_file"]).strip('"')
    k = int(hyperparams["num_community"])

    # QBSolv related parameters
    solver_mode = str(hyperparams["solver_mode"]).strip('"')
    solver_limit = int(hyperparams["solver_limit"])
    num_repeats = int(hyperparams["num_repeats"])
    num_reads = int(hyperparams["num_reads"])
    seed = int(hyperparams["seed"])
    alpha = int(hyperparams["alpha"])

    print(f"Load graph file from {input_dir}/input-graph/{input_graph_file}")
    nx_G = nx.read_weighted_edgelist(
        f"{input_dir}/input-graph/{input_graph_file}",
        delimiter=None, # check the input graph file and update the delimiter here
        create_using=nx.Graph(),
        nodetype=int)
    print(f"Input graph information: {nx.info(nx_G)}")

    # Initialize QbsolvCommunity class
    qbsolv_comm = QbsolvCommunity(nx_G, solver_limit, num_repeats, num_reads, seed, alpha)

    if solver_mode == "classical":
        print("Executing QBSolv Classical solver for community detection")
        comm_results, qbsolv_output = qbsolv_comm.solve_classical(k)
    elif solver_mode == "hybrid":
        # QBSolv Hybrid solver specific input
        s3_task_prefix = f"jobs/{job_name}/tasks" # the folder name in the S3 braket bucket to save QBSolv task output
        s3_folder = (s3_bucket, s3_task_prefix)

        print("Executing QBSolv Hybrid solver for community detection")
        comm_results, qbsolv_output = qbsolv_comm.solve_hybrid(k, s3_folder, device_arn, ack_QPUcost=True)
    else:
        raise ValueError(f"Invalid qbsolv solver mode {solver_mode}. Solver mode has to be in ['classical', 'hybrid']!")

    log_metric(
        metric_name="Modularity",
        value=comm_results["modularity"],
    )
    
    # We're done with the job, so save the result.
    # This will be returned in job.result()    
    print('Save results')
    save_job_result({"community_results": str(comm_results), "hyperparams": str(hyperparams), "qbsolv_output": str(qbsolv_output)})

if __name__ == "__main__":
    main()