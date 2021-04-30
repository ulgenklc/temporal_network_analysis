Introduction
===============

Installation/Usage
*********************
As the package has not been published on PyPi yet, it CANNOT be installed using ``pip``.

For now, the suggested method is to put the file `Temporal_Community_Detection.py` in the same directory as your source files and call ``from Temporal_Community_Detection import temporal_network``

Initiate a ``temporal_network`` object
*********************************************
Temporal networks are a subclass of multilayer/multiplex networks encoding the dynamical systems change over time as a set of networks. 
Our main object is a ``temporal_network`` and it accepts multiple type of data. One can provide the temporal connectivity as a list of adjacency matrices, a tensor or a dictionary of edge list. See API for more information.

    
