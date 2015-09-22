# Node types
Each variable in the model is described by a node in a directed graph. Code in this folder defines several common types:
- `DirichletNode.py`: Dirichlet distribution.
- `GammaNode.py`: Gamma distribution.
- `GaussianNode.py`: Normal distribution.
- `HMM.py`: Node defining a Hidden Markov or semi-Markov model.
- `NormalGammaNode.py`: Node with mean and precision defined by a Normal-Gamma distribution.
- `helpers.py`: Code involved in input checking and convenience methods for setting up specialized combinations of hierarchical priors used in the model.
- `utility_nodes.py`: Code defining certain additional types of nodes (constant node, product of two variables) useful in defining graphs.