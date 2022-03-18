## Community Detection in Complex Networks using Hybrid Quantum Annealing on Amazon Braket

This code package is for solving network community detection problems using hybrid quantum annealing on Amazon Braket. To learn more background information, you can read our AWS quantum blog post series on community detection [Part 1](https://aws.amazon.com/blogs/quantum-computing/community-detection-in-complex-networks-using-hybrid-quantum-annealing-on-amazon-braket-part-i/) and [Part 2](https://aws.amazon.com/blogs/quantum-computing/community-detection-using-hybrid-quantum-annealing-on-amazon-braket-part-2/). 

### Quick Start

The tutorial notebook [`Notebook_QBSolv_community_detection`](Notebook_QBSolv_community_detection.ipynb) provides a step-by-step guide on how to formulate community detection as a Quadratic Unconstrained Binary Optimization (QUBO) problem, similar to the work done by [Negre et. al](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0227538). We then demonstrate how to use the open source QBSolv library, which provides quantum-classical hybrid solvers for QUBO problems, using a combination of classical compute resources and D-Wave quantum annealers, to solve community detection problems on Amazon Braket.

**Table of Contents for the Tutorial Notebook**
* Modularity-based community detection
* Community detection as a QUBO problem
* Datasets
* List of key functions for community detection
* Set up environment
* Download graph data
* Prepare graph data for community detection
* Apply QBSolv for community detection
* Detect communities for synthetic and real-world networks

The Amazon Braket Hybrid Jobs notebook [`Hybrid_jobs_for_community_detection`](Hybrid_jobs_for_community_detection.ipynb) provides a step-by-step guide on how to use Amazon Braket Hybrid Jobs to seamlessly manage and execute quantum annealing-based community detection tasks at scale. 

**Table of Contents for the Amazon Braket Hybrid Jobs Notebook**
* Set up environment
* Prepare input data
* Create an algorithm script
* Specify hyperparameters
* Submit a hybrid job
* View results
* Run hyperparameter tuning


### Datasets

The graph datasets used for demonstration here can be downloaded from http://networkrepository.com/ under a Creative Commons Attribution-ShareAlike License. Dataset reference: Rossi, Ryan A.  and Ahmed, Nesreen K. (2015) The Network Data Repository with Interactive Graph Analytics and Visualization. AAAI https://networkrepository.com.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

