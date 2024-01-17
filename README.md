# Temporal Networks
Temporal network is a python package for multilayer network analysis of time series data via dynamic community detection (DCD).

We present our skeleton coupling algorithm within the package accompanied by the paper:
Bengier Ülgen Kilic, Sarah Feldt Muldoon, Skeleton coupling: a novel interlayer mapping of community evolution in temporal networks, 2023,https://arxiv.org/abs/2301.10860

## Available DCD algorithms
Additionally, one can perform several different community detection algorithms such as:

Multilayer Modularity Maximization(MMM) (PJ. Mucha et al. 2010) using a Louvain-like greedy algorithm Leiden (VA. Traag et al. 2019). 

Infomap with Map equation https://www.mapequation.org.

DPPM with https://github.com/nathanntg/dynamic-plex-propagation

DSBM with https://graph-tool.skewed.de

Tensor factorization as explained in 	

Gauvin L, Panisson A, Cattuto C. Detecting the community structure and activity patterns of temporal networks: a non-negative tensor factorization approach. PLoS One. 2014;9(1):e86028. Published 2014 Jan 31. doi:10.1371/journal.pone.0086028

## Documentation
See the [documenation](https://temporal-network-analysis.readthedocs.io/en/latest/index.html) for details.
