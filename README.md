# temporal_network_analysis
Temporal network object for multilayer network analysis of the calcium dynamics obtained by Two Photon Microscopy from the epileptic mice. 

Using the helper functions, a time series of calcium dynamics(in general any time series) can be converted into cross-correlation matrices of given window size. Then, the `temporal_network` object converts the list of adjacency matrices into an ordinally and diagonally coupled multilayer network, i.e. interlayer edges are given between a node and it's future self in the adjacent layers only. 

Then, one can perform several different community detection algorithms such as:

Multilayer Modularity Maximization(MMM) (PJ. Mucha et al. 2010) using a Louvain-like greedy algorithm Leiden (VA. Traag et al. 2019). 

Infomap with Map equation https://www.mapequation.org.

DPPM with https://github.com/nathanntg/dynamic-plex-propagation

DSBM with https://graph-tool.skewed.de

Tensor factorization as explained in 	

Gauvin L, Panisson A, Cattuto C. Detecting the community structure and activity patterns of temporal networks: a non-negative tensor factorization approach. PLoS One. 2014;9(1):e86028. Published 2014 Jan 31. doi:10.1371/journal.pone.0086028


See the[documenation](https://temporal-network-analysis.readthedocs.io/en/latest/index.html) for details.
