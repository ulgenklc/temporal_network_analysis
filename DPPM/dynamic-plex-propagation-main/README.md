# Dynamic Plex Propagation

This repository contains a Matlab implementation of the dynamic plex propagation algorithm. The algorithm was originally developed in (Viles 2013) and provides a mechanism for identifying and tracking communities in functional networks over time under uncertainty. This has specific relevance to the processing of brain connectivity (for example, as measured by ECoG) leading up to a seizure.

Inspired by the clique percolation method in (Palla 2005), the algorithm identifies k-plexes at each time step in a function network (the use of k-plexes makes the algorithm robust to noise) and connects them both within each time step and across time steps. As a result, this allows for the birth and death of functional communities to be tracked over time and under uncertainty.

This Matlab implementation is a basic implementation of the algorithm, as well as a number of related tools for:

* Simulating dynamic networks
* Aggregating statistics
* Visualizing dynamic networks

This is still a work in progress. This document and the repository will be updated with documentation and examples of how the algorithm is used.

## Running the algorithm

In order to run the algorithm, you must have a dynamic adjacency matrix for an undirected binary graph. Given `n` vertices and `t` time steps, the dynamic adjacency matrix `C` should be `n × n × t`. Each value in the matrix represents whether an edge exists between two vertices at a specific time. For example `C(a, b, d) = 1` means that node `a` is connected to node `b` at time `d`. The diagonal of the matrix should always be zero (`C(a, a, :) = 0`).

The algorithm can be run as follows:

```matlab
[vertices, communities] = dpp(C, k, m);
```

**Inputs:** The matrix `C` is the dynamic adjacency matrix. The optional parameter `k` specifies the k-plexes to identify and defaults to 2 (meaning that the algorithm will identify 2-plexes). The optional parameter `m` specifies the minimum size k-plex and defaults to `k + 2`. Omitting both `k` and `m` means the algorithm will look for 2-plexes that include 4 or more vertices.

**Outputs:** Both `vertices` and `communities` are cell arrays of length `t` (corresponding to the number of time steps in the dynamic adjacency matrix). The cell array `communities` contains a vector of the dynamic communities identified at that time step. Each dynamic community receives a unique number, and they are numbered in sequential, ascending order. You can track the life of a dynamic community by looking for all time steps where the community number appears in the `communities` cell array. Note that one community may appear multiple times in the same timestep, if the two communities are connected at either an earlier or later time step.

The `vertices` cell array contains a logical matrix with `n` columns (corresponding to the number of vertices in the dynamic adjacency matrix) and `d` rows (corresponding to the dynamic communities listed in the `communities` vector for the same time step). The logical matrix represents which vertices compose the community at that time step.

## Visualizing the output

To help visualize the dynamic communities, functions are available to create a movie file showing the progression of dynamic communities over time. This code was originally written by Weston Viles (see references) and adapted to this dynamic plex propagation implementation.

A movie can be generated using the following command:

```matlab
movie_dynamic_communities('movie.mp4', C, vertices, communities, xy, taxis, frame_rate);
```

**Inputs:** The first parameter is a file name for the generated movie. Matrix `C` is the dynamic adjacency matrix. Arguments `vertices` and `communities` are the outputs of the dynamic plex propagation algorithm. Matrix `xy` is a `n` by 2 matrix providing coordinates of each vertex in 2D space for plotting purposes. The optional parameter `taxis` provides timestamps for each frame, and the optinal parameter `frame_rate` sets the framerate for the saved video file (which defaults to 5 frames / second).

## More information

For more information, check out the Wiki, which includes details on:

* [Networks and dynamic networks](https://github.com/nathanntg/dynamic-plex-propagation/wiki/Networks-and-dynamic-networks)
* [The algorithm itself](https://github.com/nathanntg/dynamic-plex-propagation/wiki/The-algorithm)

And other helpful resources and documentation.

## References

Palla, Gergely, Imre Derényi, Illés Farkas, and Tamás Vicsek. 2005. “Uncovering the Overlapping Community Structure of Complex Networks in Nature and Society.” Nature 435 (7043): 814–18.

Viles, Weston. 2013. “Network Data Analysis.” PhD diss., Boston University.

## Authors

**L. Nathan Perkins** wrote much of the code in this repository (unless otherwise noted in the header of files).

- <https://github.com/nathanntg>
- <http://www.nathanntg.com>

**Weston Viles** developed the approach and early prototypes of the code, as well as created some of the code used in this repository for visualizing dynamic communities (as noted in the header of the files).

- <http://math.bu.edu/people/wesviles/>

