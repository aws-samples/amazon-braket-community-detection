## Community Detection in Complex Networks using Hybrid Quantum Annealing on Amazon Braket

This code package is for solving network community detection problems using hybrid quantum annealing on Amazon Braket. 

### Quick Start

The `Notebook_QBSolv_community_detection` notebook provides a step-by-step guide on how to formulate community detection as a Quadratic Unconstrained Binary Optimization (QUBO) problem, similar to the work done by [Negre et. al](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0227538). We then demonstrate how to use the open source QBSolv library, which provides quantum-classical hybrid solvers for QUBO problems, using a combination of classical compute resources and D-Wave quantum annealers, to solve community detection problems on Amazon Braket.

#### Table of Contents for the Tutorial Notebook
* Modularity-based Community Detection
* Community Detection as a QUBO Problem
    * Two communities (k=2)
    * Multiple communities
        * One-Hot Encoding
        * Generalized QUBO formulation
* Datasets
* List of Key Functions for Community Detection
* Set Up Environment
    * Helper Functions for Community Detection
* Download Graph Data
* Prepare Graph for Community Detection
    * Built-In Graph
    * Graph from a Local Data File
    * Random Partition Graph
* Apply QBSolv for Community Detection
    * Set Community Parameters for a Graph
    * Set Hyperparameters for QBSolv
    * Execute QBSolv Classical Solver
    * Execute QBSolv Hybrid Solver
* Detect Communities for Synthetic and Real-World Graphs
    * Study a Set of Synthetic Graphs
    * Study a Set of Real-World Graphs


### Datasets

The graph datasets used for demonstration here can be downloaded from http://networkrepository.com/ under a Creative Commons Attribution-ShareAlike License. Dataset reference: Rossi, Ryan A.  and Ahmed, Nesreen K. (2015) The Network Data Repository with Interactive Graph Analytics and Visualization. AAAI https://networkrepository.com.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

