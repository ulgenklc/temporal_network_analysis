# temporal_network_analysis
Temporal network object for multilayer network analysis of the calcium dynamics obtained by Two Photon Microscopy from the epileptic mice. 

Using the helper functions, a time series of calcium dynamics(in general any time series) can be converted into cross-correlation matrices of given window size. Then, the `temporal_network` object converts the list of adjacency matrices into an ordinally and diagonally coupled multilayer network, i.e. interlayer edges are given between a node and it's future self in the adjacent layers only. 

Then, one can perform community detection using Multilayer Modularity Maximization(MMM) (PJ. Mucha et al. 2010) using a Louvain-like greedy algorithm Leiden (VA. Traag et al. 2019). 


More documentation is coming soon.
