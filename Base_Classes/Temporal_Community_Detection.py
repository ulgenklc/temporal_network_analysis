#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from math import floor
import random

import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from scipy.spatial.distance import jensenshannon

import leidenalg as la
import igraph as ig

try: 
    from graph_tool.all import *
except: 
    from infomap import Infomap, MultilayerNode

    import tensorly as tl
    from tensorly.decomposition import non_negative_parafac
    warnings.warn(message = 'Graph-tool requires its own environment. Restart the kernel with a gt environment to run DSBM, otherwise you can proceed.')


# In[ ]:


class temporal_network:
    """
    Temporal network object to run dynamic community detection and other multilayer network diagnostics on. 
    Temporal network is a memoryless multiplex network where every node exists in every layer. 
    
    Attributes
    ------------
    temporal_network.size: int
        Number of nodes in any given layer.
    temporal_network.length: int
        Total number of layers.
    temporal_network.nodes: list
        A list of node ids starting from 0 to ``size-1``.
    temporal_network.windowsize: int
        Assuming that temporal network is created from a continous time-series data, windowsize is the size
        of the windows we are splitting the time-series into.
    temporal_network.supra_adjacency: array, ``size*length x size*length``
        The supra adjacency matrix to encode the connectivity information of the multilayer network.
    temporal_network.list_adjacency: list, [array1, array2, ...]
        A list of arrays of length ``length`` where each array is ``size x size`` encoding the connectivity 
        information of each layer.
    temporal_network.edge_list: list, [list1, list2, ...]
        A list of length ``length`` of lists where each element of the sublist is a 4-tuple (i,j,w,t) indicating
        there is an edge from node i to node j of nonzero weight w in the layer t. So, every quadruplet in the
        t'th sublist in ``edge_list`` has 4th entry t.
    
    Parameters
    ------------
    size: int
        Number of nodes in any given layer.
    length: int
        Total number of layers.
    window_size: int
        Size of the windows the time series will be divided into.
    data: str
        ``temporal_network`` accepts three types of connectivity input, ``supra_adjacency``, ``list_adjacency`` 
        and ``edge_list`` (see the attributes). So, we must specify which one of these types we are submitting 
        the connectivity information to the ``temporal_network``. Accordingly, this parameter can be one of the 
        ``supra__adjacency``, ``list__adjacency`` and ``edge__list``, respectively.
        
        Once the data type is understood, object converts the given input into the other two data types so that
        if it needs to use one of the other types(it is easier to work with ``list_adjacency`` for example, but 
        some helper functions from different libraries such as ``igraph``, processes ``edge_list`` better), 
        it can switch back and forth quicker.
    **kwargs:
        supra_adjacency: array, ``size*length x size*length``
            The supra adjacency matrix to encode the connectivity information of the multilayer network. Should 
            be provided if ``data = supra__adjacency``.
    **kwargs:
        list_adjacency: list, [array1, array2, ...]
            A list of arrays of length ``length`` where each array is ``size x size`` encoding the connectivity 
            information of each layer. Should be provided if ``data = list__adjacency``.
    **kwargs:
        edge_list: list, [list1, list2, ...]
            A list of length ``length`` of lists where each element of the sublist is a 4-tuple (i,j,w,t) 
            indicating there is an edge from node i to node j of nonzero weight w in the layer t. So, every 
            quadruplet in the t'th sublist in ``edge_list`` has 4th entry t. Should be provided if 
            ``data = edge__list``.
    **kwargs:
        omega: int
            Interlayer edge coupling strength. Should be provided if data is ``list__adjacency`` or 
            ``edge__list``. For now, we will assume all the coupling is going to be diagonal with a constant 
            strength.
            
            TODO: extend omega to a vector(for differing interlayer diagonal coupling strengths) and to
            a matrix(for non-diagonal coupling).
    **kwargs:
        kind:
            Interlayer coupling type. Can be either ``ordinal`` where only the adjacent layers are coupled or 
            ``cardinal`` where all layers are pairwise coupled with strength ``omega``. Should be provided if 
            data is ``list__adjacency`` or ``edge__list``.
    
    """
    def __init__(self, size, length, window_size, data, **kwargs):
        
        if length < 1: raise ValueError('Object should be a multilayer network with at least 2 layers')
        if size < 3: raise ValueError('Layers must have at least 3 nodes')
        
        self.size = size # number of nodes in every layer
        self.length = length # number of layers
        self.nodes = [i for i in range(self.size)]
        self.windowsize = window_size
                    
        if  data == 'supra__adjacency':
            self.supra_adjacency = kwargs['supra_adjacency']
            list_adjacency = [ [] for i in range(length) ]
            
            for i in range(self.length):
                list_adjacency[i] = self.supra_adjacency[i*self.size:(i+1)*self.size,i*self.size:(i+1)*self.size]
            
            self.list_adjacency = list_adjacency
            
            edge_list = []
            for i in range(self.length):
                A = self.list_adjacency[i]
                firing = np.transpose(np.nonzero(A))
                for j,m in enumerate(firing):
                    quadreplet =(m[0],m[1],A[m[0],m[1]],i)
                    edge_list.append(quadreplet)
            self.edgelist = edge_list
                
        
        elif data == 'edge__list':
            self.edgelist = kwargs['edge_list']
            supra_adjacency = np.zeros((self.size*self.length,self.size*self.length))
            list_adjacency = [ [] for i in range(self.length) ]
            for q in range(self.length):
                list_adjacency[q]=np.zeros((self.size,self.size))
            
            for k,e in enumerate(self.edgelist):
                i,j,w,t = e[0], e[1], e[2],e[3]
                supra_adjacency[self.size*(t)+i][self.size*(t)+j] = w
                list_adjacency[t][i][j] = w

        
            ##filling off-diagonal blocks
            if kwargs['kind'] == 'ordinal':
                for n in range(self.size*(self.length-1)):
                    supra_adjacency[n][n+self.size] = kwargs['omega']
                    supra_adjacency[n+self.size][n] = kwargs['omega']
                
            elif kwargs['kind'] == 'cardinal':
                i = 0
                while self.length-i != 0:
                    i = i+1
                    for n in range(self.size*(self.length-i)):
                        supra_adjacency[n][n+i*self.size] = kwargs['omega']
                        supra_adjacency[n+i*self.size][n] = kwargs['omega']
            
            self.supra_adjacency = supra_adjacency
            self.list_adjacency = list_adjacency
            
        elif data == 'list__adjacency':
            self.list_adjacency = kwargs['list_adjacency']
            supra_adjacency = np.zeros((self.size*self.length,self.size*self.length))
            
            for i in range(self.length):
                supra_adjacency[i*self.size:(i+1)*self.size,i*self.size:(i+1)*self.size] = self.list_adjacency[i]
            
            ##filling off-diagonal blocks
            if kwargs['kind'] == 'ordinal':
                for n in range(self.size*(self.length-1)):
                    supra_adjacency[n][n+self.size] = kwargs['omega']
                    supra_adjacency[n+self.size][n] = kwargs['omega']
                
            elif kwargs['kind'] == 'cardinal':
                i = 0
                while self.length-i != 0:
                    i = i+1
                    for n in range(self.size*(self.length-i)):
                        supra_adjacency[n][n+i*self.size] = kwargs['omega']
                        supra_adjacency[n+i*self.size][n] = kwargs['omega']
            
            self.supra_adjacency = supra_adjacency
            
            edge_list = []
            for i in range(self.length):
                A = self.list_adjacency[i]
                firing = np.transpose(np.nonzero(A))
                for j,m in enumerate(firing):
                    quadreplet =(m[0],m[1],A[m[0],m[1]],i)
                    edge_list.append(quadreplet)
            self.edgelist = edge_list
            
    def aggragate(self, normalized = True):
        """
        Helper function to aggragate layers of the temporal network.
        
        Parameters
        --------------
        normalized: Bool
            divides the total edge weight of each edge by the number of layers(``self.length``).
            
        Returns
        --------------
        ``n x n`` aggragated adjacecy array.
        
        """
        t = self.length
        n = self.size
        aggragated = np.zeros((n,n))
        
        for i,c in enumerate(self.list_adjacency):
            aggragated = aggragated + c
            
        if normalized: return (aggragated/t)
        else: return (aggragated)
    
    def binarize(self, array, thresh = None):
        """
        Helper function to binarize the network edges.
        
        Parameters
        ------------
        array: np.array
            Input array corresponding to one layer of the temporal network.
        thresh: float (Default: None)
            if provided, edges with weight less than ``thresh`` is going to be set to 0 and 1 otherwise. If not provided, thresh = 0.
        
        Returns
        ------------
        binary_spikes: np.array
            ``n x n`` binary adjacency matrix.
        
        """
        n,t = array.shape
        binary_spikes = np.zeros(array.shape)
        for i in range(n):
            for j in range(t):
                if thresh is None:
                    if array[i][j] <= 0: pass
                    else: binary_spikes[i][j] = 1
                else:
                    if array[i][j] < thresh: array[i][j] = 0
                    else: binary_spikes[i][j] = 1
        return(binary_spikes)
    
    def threshold(self, array, thresh):
        """
        Helper function to threshold the network edges.
        
        Parameters
        ----------
        array: np.array
            Input array corresponding to one layer of the temporal network.
        thresh: float
            Threshold to keep the edges stronger than this value where weaker edges are going to be set to 0.
        
        Returns
        ----------
        thresholded_array: np.array
            ``n x n`` thresholded adjacency matrix.
            
        """
        n,t = array.shape
        thresholded_array = np.copy(array)
        for i in range(n):
            for j in range(t):
                if array[i][j] < thresh: thresholded_array[i][j] = 0
                else: pass
        return(thresholded_array)
    
    def bin_time_series(self, array, gaussian = True, **kwargs):
        """
        Helper function for windowing a given time series of spikes into a desired size matrices.
        
        Parameters
        ------------
        array: np.array
            ``n x t`` array where n is the number of neurons and t is the length of the time series.
        gaussain: bool (Default: True)
            If True, every spike in the time series is multiplied by a 1d-gaussian of size ``sigma``.
        **kwargs:
            sigma: size of the gaussian (See gaussian_filter).
        
        Return
        ------------
        A: np.array
            Matrix of size ``l x n x windowsize`` where l is the number of layers ``(= t/self.windowsize)``, n is the number of neurons.
            
        """
        binsize = self.windowsize
        n = array.shape[0] # number of neurons
        totalsize = array.shape[1] # total duration of spikes
        gauss_array = np.zeros((n,totalsize))
        l = int(totalsize/binsize) # number of resulting layers
        
        if gaussian:
            for i in range(n):
                gauss_array[i] = gaussian_filter(array[i],kwargs['sigma'])
        else: gauss_array= array
            
        A = np.zeros((l,n,binsize))
        for i in range(l):
            A[i] = gauss_array[:,i*binsize:(i+1)*binsize]
        return(A)
    
    def edgelist2edges(self):# helper to create igraphs
        """
        Helper function for creating edge lists for iGraph construction.
        
        Returns
        --------
        all_edges: list ([list1,list2,...])
            A list of length `length` lists where each list contains node pairs (i,j) in the corresponding layer.
        all_weights: list ([list1, list2,...])
            A list of length `length` lists where each list contains floats in the corresponding layer indicating the edge weight between the node pair.
        """
        T = self.length
        all_edges = [[] for i in range(T)]
        all_weights = [[] for i in range(T)]
        dtype = [('row',int),('column',int),('weight',float),('layer',int)]
        for k,e in enumerate(np.sort(np.array(self.edgelist, dtype=dtype),order='layer')):
            i,j,w,t = e[0], e[1], e[2],e[3]
            pair = (i,j)
            all_edges[t].append(pair)
            all_weights[t].append(w)
        return (all_edges, all_weights)
    
    def neighbors(self, node_id, layer):
        """
        Helper function for finding the neighbors of a given node.
        
        Parameters
        -----------
        node_id: int
            ID of the node to be found the neighbors of.
        layer: int
            Layer ID of the node that it belongs to.
            
        Return
        -----------
        neighbors: list
            list of node IDs of the neighbors of ``node_id`` in layer ``layer``.
        """
        
        if node_id > self.size: return('Invalid node ID')
        if layer > self.length: return('Invalid layer')
        neighbors = []
        
        for k,e in enumerate(self.edgelist):
            i,j,w,t = e[0],e[1],e[2],e[3]
            if t != layer:pass
            else:
                if i != node_id:pass
                else:neighbors.append(j)
                    
        return(neighbors)
    
    def get_attrs_or_nones(self, seq, attr_name):
        """
        Helper method.
        """
        try:
            return seq[attr_name]
        except KeyError:
            return [None] * len(seq)
    
    def disjoint_union_attrs(self, graphs):
        """
        Helper function to take the disjoint union of igraph objects. See ``slices_to_layers``.
        """
        
        G = ig.Graph.disjoint_union(graphs[0], graphs[1:])
        vertex_attributes = set(sum([H.vertex_attributes() for H in graphs], []))
        edge_attributes = set(sum([H.edge_attributes() for H in graphs], []))

        for attr in vertex_attributes:
            attr_value = sum([self.get_attrs_or_nones(H.vs, attr) for H in graphs], [])
            G.vs[attr] = attr_value
        for attr in edge_attributes:
            attr_value = sum([self.get_attrs_or_nones(H.es, attr) for H in graphs], [])
            G.es[attr] = attr_value
        return G

    def time_slices_to_layers(self, graphs, interlayer_indices, interlayer_weights, update_method,
                              interslice_weight=1,
                              slice_attr='slice',
                              vertex_id_attr='id',
                              edge_type_attr='type',
                              weight_attr='weight'):
        """
        Helper function for implementing non-diagonal coupling with Modularity Maximization. See ``slices_to_layers``.
        
        """
        
    
        G_slices = ig.Graph.Tree(len(graphs), 1, mode=ig.TREE_UNDIRECTED)
        G_slices.es[weight_attr] = interslice_weight
        G_slices.vs[slice_attr] = graphs
    
        return self.slices_to_layers(G_slices, interlayer_indices, interlayer_weights, update_method ,slice_attr,vertex_id_attr,edge_type_attr,weight_attr)

    def slices_to_layers(self, G_coupling, interlayer_indices, interlayer_weights, update_method,
                         slice_attr='slice',
                         vertex_id_attr='id',
                         edge_type_attr='type',
                         weight_attr='weight'):
        """
        Actual function implementing non-diagonal coupling with Modularity Maximization. Leiden algorithm's python package 
        inherently only allows diagonal coupling. So, this function is needed for non-diagonal coupling.
        """
        
        if not slice_attr in G_coupling.vertex_attributes():
            raise ValueError("Could not find the vertex attribute {0} in the coupling graph.".format(slice_attr))

        if not weight_attr in G_coupling.edge_attributes():
            raise ValueError("Could not find the edge attribute {0} in the coupling graph.".format(weight_attr))

        # Create disjoint union of the time graphs
        for v_slice in G_coupling.vs: 
            H = v_slice[slice_attr]
            H.vs[slice_attr] = v_slice.index
            if not vertex_id_attr in H.vertex_attributes():
                raise ValueError("Could not find the vertex attribute {0} to identify nodes in different slices.".format(vertex_id_attr ))
            if not weight_attr in H.edge_attributes():
                H.es[weight_attr] = 1

        G = self.disjoint_union_attrs(G_coupling.vs[slice_attr])
        G.es[edge_type_attr] = 'intraslice'

        for i in range(len(G_coupling.vs[slice_attr])-1):
            v_slice = G_coupling.vs[i]
            nodes_v = sorted([v for v in G.vs if v[slice_attr] == v_slice.index and v[vertex_id_attr] in G.vs.select(lambda v: v[slice_attr]==v_slice.index)[vertex_id_attr]], key=lambda v: v[vertex_id_attr])
            for j,v in enumerate(nodes_v):
                if update_method == 'neighborhood': w, nbr = self.neighborhood_flow(i, j, interlayer_indices, interlayer_weights, thresh = 0.1)
                elif update_method == 'skeleton': w, nbr = interlayer_weights['%d,%d'%(i,j)], interlayer_indices['%d,%d'%(i,j)]
                edges = []
                a = G.vs[int(i*self.size + j)]
                for n in nbr:
                    b = G.vs[int((i+1)*self.size + n)]
                    edges.append((a,b))
                    edges.append((b,a))
                e_start = G.ecount()
                G.add_edges(edges)
                e_end = G.ecount()
                e_idx = range(e_start,e_end)
                G.es[e_idx][weight_attr] = w
                G.es[e_idx][edge_type_attr] = 'interslice'

        # Convert aggregate graph to individual layers for each time slice.
        G_layers = [None]*G_coupling.vcount()
        for v_slice in G_coupling.vs:
            H = G.subgraph_edges(G.es.select(_within=[v.index for v in G.vs if v[slice_attr] == v_slice.index]), delete_vertices=False)
            H.vs['node_size'] = [1 if v[slice_attr] == v_slice.index else 0 for v in H.vs]
            G_layers[v_slice.index] = H

        # Create one graph for the interslice links.
        G_interslice = G.subgraph_edges(G.es.select(type_eq='interslice'), delete_vertices=False)
        G_interslice.vs['node_size'] = 0
    
        return G_layers, G_interslice, G
    
    def create_igraph(self):
        """
        Helper function that creates igraphs for modularity maximization.
        """
        T = self.length
        N = self.size
        G = []
        edges = self.edgelist2edges()[0]
        weights = self.edgelist2edges()[1]
        for i in range(T):
            G.append(ig.Graph())
            G[i].add_vertices(N)
            G[i].add_edges(edges[i])
            G[i].es['weight'] = weights[i]
            G[i].vs['id'] = list(range(N))
            G[i].vs['node_size'] = 0
        return(G)

    
    def leiden(self, G, interslice, resolution):
        """
        Function that runs Multilayer Modularity Maximization using Leiden solver.
        
        Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing well-connected communities. 
        Sci Rep 9, 5233 (2019). https://doi.org/10.1038/s41598-019-41695-z
        
        Parameters
        ------------
        G: list ([g1,g2,...])
            A list of igraph objects corresponding to different layers of the temporal network.
        interslice: float
            Leidenalg package automatically utilizes diagonal coupling of layers. If a float is provided as ``interslice``
            a uniform interlayer coupling weight is going to be applied for all nodes in all layers. If a list of length ``size`` 
            is provided, every node will be coupled with themselves with given weight. If a list of, length ``length -1``, 
            lists is provided, then you can tune individual interlayer weights as well.
        resolution: float
            Resolution parameter.
            
        Returns
        -----------
        partitions: leidenalg object. See https://leidenalg.readthedocs.io/en/stable/
        interslice_partitions: leidenalg object. See https://leidenalg.readthedocs.io/en/stable/
        """
        
        layers, interslice_layer, G_full = la.time_slices_to_layers(G, interslice_weight = interslice)
        
        partitions = [la.RBConfigurationVertexPartition(H, 
                                            weights = 'weight', 
                                            resolution_parameter = resolution) for H in layers]
        
        interslice_partition = la.RBConfigurationVertexPartition(interslice_layer, 
                                                                 weights = 'weight',
                                                                 resolution_parameter = 0)
                                                     
        optimiser = la.Optimiser()
        
        diff = optimiser.optimise_partition_multiplex(partitions + [interslice_partition])

        return(partitions, interslice_partition)
    
    def MMM_static(self):
        """
        Running leiden algorithm on the individual layers of temporal network for skeleton coupling.
        
        Returns
        ==========
        inter_membership: triple list
            List that contain layer, membership and node information respectively.
        """
        G_ind = self.create_igraph()
        inter_membership = []
        for i in range(self.length):
            partitions = la.find_partition(G_ind[i], la.ModularityVertexPartition, weights = 'weight')
            membership = [[] for i in range(max(partitions.membership)+1)]
            for j,e in enumerate(partitions.membership):
                membership[e].append(j)
            inter_membership.append(membership)
        return(inter_membership)
    
    def infomap(self, inter_edge, threshold, update_method = None, **kwargs):
        '''
        Function that runs Infomap algorithm on the temporal network. 
        
        https://www.mapequation.org
        
        Parameters
        ---------------
        inter_edge: float
            Interlayer edge weight.
        threshold: float
            Value for thresholding the network edges. Functional networks obtained by correlation is going to 
            need thresholding with infomap.
        update_method: None, ``local``, ``global``, ``neighborhood`` or ``skeleton``. Default None.
            Updating the interlayer edges according to either of these methods.
        **kwargs:
            spikes: array
                if ``local`` or ``global`` update method is being used, initial ``spikes`` that is used to obtain the correlation
                matrices needs to be provided.
        '''
        im = Infomap("--two-level --directed --silent")
            ######### Make Network
            ## add intra edges
        thresholded_adjacency = []
        for l in range(self.length):
            thresholded_adjacency.append(self.threshold(self.list_adjacency[l], thresh = threshold))
            for n1,e in enumerate(thresholded_adjacency[l]):
                for n2,w in enumerate(e):
                    s = MultilayerNode(layer_id = l, node_id = n1)
                    t = MultilayerNode(layer_id = l, node_id = n2)
                    im.add_multilayer_link(s, t, w)
                    im.add_multilayer_link(t, s, w)
                
        ## add inter edges
        if update_method == 'local' or update_method == 'global': 
        
            updated_interlayer = self.update_interlayer(kwargs['spikes'], 0, inter_edge, 0.1, update_method) 
        
            for l in range(self.length-1):
                for k in range(self.size):
                    s = MultilayerNode(layer_id = l, node_id = k)
                    t = MultilayerNode(layer_id = l+1, node_id = k)
                    im.add_multilayer_link(s, t, updated_interlayer[l][k])
                    im.add_multilayer_link(t, s, updated_interlayer[l][k])
                
        elif update_method == 'neighborhood':
        
            updated_interlayer_indices, updated_interlayer_weights = self.get_normalized_outlinks(thresholded_adjacency, inter_edge)
            for l in range(self.length-1):
                for k in range(self.size):
                    w, nbr = self.neighborhood_flow(l, k, updated_interlayer_indices, updated_interlayer_weights, threshold)
                    for n in nbr:
                        s = MultilayerNode(layer_id = l, node_id = k)
                        t = MultilayerNode(layer_id = l+1, node_id = n)
                        im.add_multilayer_link(s, t, w)
                        im.add_multilayer_link(t, s, w)
        
        elif update_method == 'skeleton':
            membership_static = self.infomap_static(thresholded_adjacency)
            bridge_links = self.find_skeleton(membership_static)
            
            for l in range(self.length-1):
                for k in range(self.size):
                    if bridge_links['%d,%d'%(l,k)]:
                        for i, inter in enumerate(bridge_links['%d,%d'%(l,k)]):
                            s = MultilayerNode(layer_id = l, node_id = k)
                            v = MultilayerNode(layer_id = l+1, node_id = inter)
                            im.add_multilayer_link(s, v, inter_edge)
                            im.add_multilayer_link(v, s, inter_edge)
            
                    
        elif update_method == None:
            for l in range(self.length-1):
                for k in range(self.size):# number of nodes which is 60 in the multilayer network
                    s = MultilayerNode(layer_id = l, node_id = k)
                    t = MultilayerNode(layer_id = l+1, node_id = k)
                    im.add_multilayer_link(s, t, inter_edge)
                    im.add_multilayer_link(t, s, inter_edge)
        
        im.run()
        return(im)
    
    def infomap_static(self, thresholded_adjacency):
        """
        Helper function for running infomap on the individual layers of temporal network.
        
        Parameters
        ============
        thresholded_adjacency : list
            List of adjacency matrices.
            
        Returns
        ===========
        inter_membership: triple list
            List that contain layer, membership and node information respectively.
        """
        inter_membership = []
        clean_memberships = {}
        for l in range(self.length):
            im = Infomap("--two-level --directed --silent")
            for n1,e in enumerate(thresholded_adjacency[l]):
                for n2,w in enumerate(e):
                    im.add_link(n1,n2, w)
            im.run()
            membership = [[] for i in range(im.num_top_modules)]
            for node in im.nodes:
                membership[int(node.module_id-1)].append(node.node_id)
            inter_membership.append(membership)         
        return(inter_membership)
    
    def find_comm_size(self, n, list_of_lists):
        """
        Helper function for finding the comunities in the next layer of a given node.
        
        Parameters
        ============
        n : int
            Node id to be found whose communities of.
        list_of_lists : list
            First dimension of output of ``infomap_static``.
            
        Returns
        ==========
        comm : list
            Community membership of the given node in the next time step.
        len : int
            Size of that community.
        """
        
        for i,comm in enumerate(list_of_lists):
            if n in comm:
                break
        return(comm, len(comm))
    
    def find_skeleton(self, static_memberships):
        """
        Function that finds links of skeleton coupling.
        
        Parameters
        ===========
        static_membership : list
            Output of ``infomap_static``.
        
        Returns
        ==========
        bridge_links : dict
            Dictionary of skeleton links.
        """
        
        bridge_links = {}
        for t in range(self.length-1):    
            for i, comm1 in enumerate(static_memberships[t]):
                for j, node in enumerate(comm1):
                    comm2, comm2_size = self.find_comm_size(node, static_memberships[t+1])
                    if self.find_comm_size(node, static_memberships[t])[1] > 1:
                        if comm2_size > 1:
                            bridge_links['%d,%d'%(t,node)] = comm2
                        else:
                            bridge_links['%d,%d'%(t,node)] = []
                    else:
                        if comm2_size > 1:
                            bridge_links['%d,%d'%(t,node)] = []
                        else:
                            bridge_links['%d,%d'%(t,node)] = [node]
        return(bridge_links)
    
    def membership(self, interslice_partition): 
        """
        Returns the community assignments from the Leiden algorithm as tuple (n,t) where ``n`` is the node id ``t`` is the layer 
        that node belongs to.
        """
        n = self.size
        membership = [[] for i in range(interslice_partition._len)]
        for i,m in enumerate(interslice_partition._membership):
            time = floor(i/n)
            node_id = i%n
            membership[m].append((node_id,time))
        return(membership, len(membership))
    
    def community(self, membership, ax):
        """
        Helper function to visualize the community assignment of the nodes. At every run, a random set of colors are generated
        to indicate community assignment.
        
        Parameters
        ---------------
        membership: list
            A list of length ``number of communities`` where each list contains (node,time) pairs indicating the possesion of that
            ``node`` at the ``time`` to that community.
        ax: matplotlib.axis object
            An axis for plotting of the communites.
            
        Returns
        -----------
        comms: array of shape ``n x t``
            array to be visualized.
        color: list
            list of colors to be plotted for future use.
        
        """
        n = self.size
        t = self.length
        number_of_colors = len(membership)

        comms = np.zeros((n,t))

        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

        for i, l in enumerate(membership):
            for j,k in enumerate(l):
                comms[k[0]][k[1]] = i

        cmap = mpl.colors.ListedColormap(color)

        ax.imshow(comms, interpolation = 'none', cmap = cmap, aspect = 'auto', origin = 'lower', extent = [-0.5,t-0.5,-0.5,n-0.5])
        return(comms, color)
    
    def raster_plot(self, spikes, ax, color = None, **kwargs):
        """
        Plots the raster plot of the spike activity on a given axis. if ``color`` provided, raster includes the community assignments.
        
        Parameters
        ------------
        spikes: array ``n x t``
            Initial spike train array for ``n`` nodes of length ``t``.
        ax: matplotlib.axis object
            axis to be plotted.
        color: list
            Second output of the ``self.community``. List of colors of length ``number of communities``.
        **kwargs:
            comm_assignment: array
                First output of the ``self.community``. If not provided raster is going to be plotted blue.
        """
        binsize = self.windowsize
        binarized_spikes = self.binarize(spikes)
        binned_spikes = self.bin_time_series(binarized_spikes, gaussian = False)
        l,n,t = binned_spikes.shape
                    
        sp = np.nonzero(binned_spikes)
        
        if color is None: 
            col = [0]*l
            clr = [col for i in range(n)]
            color = ['#0000ff']
        else: clr = kwargs['comm_assignment']
        
        cmap = mpl.colors.ListedColormap(color)
        
        for i in range(len(sp[0])):
            ax.scatter(sp[0][i]*binsize+sp[2][i],  sp[1][i], 
                       s = 5, 
                       c = color[int(clr[sp[1][i]][sp[0][i]])], 
                       marker = 'x', 
                       cmap = cmap)
            
        ax.set_title('Raster Plot', fontsize = 20)
        ax.set_xlabel('Time (Frames)', fontsize = 15)
        ax.set_ylabel('Neuron ID', fontsize = 15)
        ax.set_xticks([t*i for i in range(l+1)])
        ax.set_yticks([5*i for i in range(int(n/5)+1)]+[n])
        ax.tick_params(axis = 'x', labelsize = 10)
        ax.tick_params(axis = 'y', labelsize = 13)
    
    def trajectories(self, thresh = 0.9, node_id = None, community = None, edge_color = True, pv = None):
        """
        Function graphing the edge trajcetories of the temporal network.
        
        Parameters
        -------------
        thresh: float
            Threshold for keeping filtering the edge weights.
        node_id: int (Default: None)
            If None, function is going to graph all of the nodes's trajectories.
        community: array (Default: None)
            First output of ``self.community`` indicating the community assignment of the nodes if exists.
        edge_color: bool
            Different colors on each layer if True, black otherwise.
        pv: list
            Pass a list of pv cell indices or None --dashes the pv cells.
        
        """
            
        layers = []

        if edge_color == True: ed_color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(self.length)]
        else: e_color = 'black' #["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(self.length)]

            
        if community is None: node_color = 'r'     
        else:
            colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(int(np.max(community)))]
            comap = mpl.colors.ListedColormap(colors)
            node_color = community
            norm = plt.Normalize(0,int(np.max(community)))

        if node_id == None:
            for k in self.nodes:
                for j in range(1,self.length):
                    for i in self.neighbors(k, j):
                        if self.list_adjacency[j][k][i] > thresh:
                            layers.append((j-1, j))
                            layers.append((k, i))
                            try: layers.append('%s' %ed_color[j])
                            except: layers.append('%s'%e_color)
                            
            fig,ax = plt.subplots(1,1,figsize = (20,10))
            plt.plot(*layers,figure = fig)
            plt.title('Temporal trajectories of all the cells that are stronger than %f'%(thresh), fontsize = 20)
            plt.xlabel('Layers',fontsize = 15)
            plt.ylabel('Nodes',fontsize = 15)


            for i in range(self.size):
                x = np.linspace(0, self.length -1, self.length)
                y = np.linspace(i,i, self.length)
                try:plt.scatter(x, y, s = 15, c = node_color, figure = fig, alpha = 1)
                except: plt.scatter(x, y, s = 15, c = node_color[i], norm = norm, figure = fig, alpha = 1, cmap = comap)


        else:
            for j in range(1,self.length):
                for i in self.neighbors(node_id,j):
                    if self.list_adjacency[j][node_id][i] > thresh:
                        layers.append((j-1, j))
                        layers.append((node_id, i))
                        try: layers.append('%s' %ed_color[j])
                        except: layers.append('%s'%e_color)
                            
            fig,ax = plt.subplots(1, 1, figsize = (20,10))
            plt.plot(*layers, figure = fig)
            plt.title('Temporal trajectories of the cell %d that are stronger than %f'%(node_id,thresh), fontsize = 20)
            plt.xlabel('Layers', fontsize = 15)
            plt.ylabel('Nodes', fontsize = 15)
            
            for i in range(self.size):
                x = np.linspace(0, self.length-1, self.length)
                y = np.linspace(i, i, self.length)
                try:plt.scatter(x, y, s = 15, c = node_color, figure = fig, alpha = 1)
                except: plt.scatter(x, y, s = 15, c = node_color[i], norm = norm, figure = fig, alpha = 1, cmap = comap)
        
        if community is not None:
            cbar = plt.colorbar(cmap = cmap)
        
            cbar.set_ticks([i for i in np.arange(0,int(np.max(community)),3)])
            cbar.set_ticklabels([i for i in np.arange(0,int(np.max(community)),3)])
            cbar.set_label('Colorbar for node communities - total of %d communities'%int(np.max(community)), rotation = 270)
        if pv is not None:
            plt.hlines(pv, 0, self.length-1, color = 'b', alpha = 0.4, linestyle = 'dashed')
            plt.yticks(pv, color = 'b')
        plt.tight_layout()
    
    def get_normalized_outlinks(self, thresholded_adjacency, interlayer): 
        """
        Helper function for neighborhood coupling that finds the interlayer neighbors of a every node in the next and 
        previous layers and normalizes edge weights.
        
        Parameters
        -------------
        thresholded_adjacency: list
            List of adjacency matrices corresponding to every layer of the temporal network.
        interlayer: float
            The node itselves edge weight that is connected to its future(or past) self that is the maximal among other 
            interlayer neighbors.
            
        Returns
        -----------
        interlayer_indices: dict (dict['t,i'])
            Dictionary of interlayer neighbors of a node i in layer t.
        interlayer_weights: dict (dict['t,i'])
            Dictionary of interlayer weights corresponding to indices of node i in layer t.
        """
        #interlayer is the node itselves edge weight that is connected to its future self that is the maximal
        interlayer_indices = {}
        interlayer_weights = {}
        for i in range(self.length):
            layerweights = []
            for j in range(self.size):
                maximal_neighbors = [[int(interlayer),j]]
                for nonzero in np.nonzero(thresholded_adjacency[i][j,:])[0]:
                    maximal_neighbors.append([thresholded_adjacency[i][j,nonzero], nonzero])
                weights = np.array(sorted(maximal_neighbors, reverse = True))[:,0]
                indices = np.array(sorted(maximal_neighbors, reverse = True))[:,1]
                norm_weights = weights/np.sum(weights)
                indices, norm_weights
                interlayer_indices['%d,%d'%(i,j)] = indices
                interlayer_weights['%d,%d'%(i,j)] = norm_weights
        return(interlayer_indices,interlayer_weights)
    
    def neighborhood_flow(self, layer, node, interlayer_indices, interlayer_weights, thresh):
        """
        Helper function to evaluate the weights of the individual non-diagonal interlinks using jensenshannon entropy.
        We also threshold weaker interlinks and keep only the ones that have maximal interlayer edge weight for computational 
        purposes. In this sense, we are coupling a maximal neighborhood around a node with previous and future layer.
        
        Parameters
        --------------
        layer: int
            Layer that node belongs to.
        node: int
            Node ID
        interlayer_indices: dict
            First output of the ``get_normalized_outlinks``.
        interlayer_weights: dict
            Second output of the ``get_normalized_outlinks``.
        thresh: float
            Value for thresholding the weakest ``thresh`` percentage of interlinks that this node has.
            
        Return:
        ---------
        w: float
            Neighborhood coupling weight.
        nbr: dict
            Thresholded list of maximal interlinks
        """
        length = int(min(len(interlayer_weights['%d,%d'%(layer,node)]),len(interlayer_weights['%d,%d'%(layer+1,node)]))*thresh)
        w = 1-jensenshannon(interlayer_weights['%d,%d'%(layer,node)][:length],interlayer_weights['%d,%d'%(layer+1,node)][:length])**2
        nbr = interlayer_indices['%d,%d'%(layer,node)][:length]
        return(w,nbr)
        
        
    def update_interlayer(self, spikes, X, omega_global, percentage, method):
        """
        Function for local and global updates. This function assumes diagonal coupling and evaluates the interlink weights
        according to the ``local`` or ``global`` change in some nodal property, spike rates in our case.
        
        Parameters
        --------------
        spikes: array
            Initial spike train array.
        X: float
            Value for determining if the nodal property between consecutive layers(local), or compared to global average, is 
            less than ``X`` standard deviation.
        omega_global: float
            Initial interlayer value for all diagonal links.
        percentage: float
            If the nodal property is less than ``X`` standard deviation, for a given node, interlayer edge weight is adjusted
            so that new weight is equal to ``omega_global x percentage``.
        method: 'local' or 'global'
            Method for updating the interlayer edges. If local a comparison between consecutive layers is made and if global,
            overall average of the spike rates are hold as a basis.
            
        Returns:
        ------------
        interlayers: list
            A list of length ``(length-1) x size`` indicating interlayer edge weights of every node.
            
        """
        ## all three methods in this function assumes the diagonal coupling
        ## i.e. output is the list(of length layers -1) of lists (each of length number of neuorns)
        ## corresponding to a node's interlayer coupling strength with it's future self.
        binned_spikes = self.bin_time_series(spikes, gaussian = False)
        sp = np.nonzero(binned_spikes)
        
        layers ,num_neurons, t = self.length, self.size, self.windowsize
        
        count_spikes = np.zeros((layers, num_neurons))
    
        if method == 'local':
            for i in range(len(sp[0])):
                l, n, t = sp[0][i], sp[1][i], sp[2][i]
                count_spikes[l][n] = count_spikes[l][n] + 1
            interlayers = []
            for i in range(layers-1):
                zscores = zscore(np.diff(count_spikes, axis = 0)[i])
                layerweights = []
                for j in range(num_neurons):
                    if zscores[j] <= X: layerweights.append(percentage*omega_global)
                    else: layerweights.append(omega_global)
                interlayers.append(layerweights)

        elif method == 'global':
            for i in range(len(sp[0])):
                l, n, t = sp[0][i], sp[1][i], sp[2][i]
                count_spikes[l][n] = count_spikes[l][n] + 1
            interlayers = []
            zscores = zscore(sum(np.diff(count_spikes, axis = 0)))
            for i in range(layers-1):
                layerweights = []
                for j in range(num_neurons):
                    if zscores[j] <= X: layerweights.append(percentage*omega_global)
                    else: layerweights.append(omega_global)
                interlayers.append(layerweights)
        return(interlayers)
    
    def make_tensor(self, rank, threshold, update_method = None, **kwargs):
        """
        Helper function to utilize Tensor Factorization Approach described in:
        
        https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0086028
        
        Parameters
        ---------------
        rank: int
            Input for predetermined number of communites to be found.
        threshold: float
            Edge threshold for adjacency matrices.
        update_method: ``local``, ``global`` or ``neighborhood``(Default: None)
            Updating the edges according to one of these methods although this is not an applied technique in the literature. 
            Go with None unless you know what you are doing.
        **kwargs:
            spikes: array
                Initial spike train matrix of size ``n x t``
        Returns
        -------------
        weights_parafac: array
            See the paper.
        factors_parafac: array
            See the paper.
        """
        #Make tensor according to one of four methods
        if update_method == 'local' or update_method == 'global': 
            tensor = np.zeros((self.size, self.size, int((2*self.length)-1)))
            inters = self.update_interlayer(kwargs['spikes'], 0.5, 1, 0.01, update_method)
            for i in range(int((2*self.length)-1)):
                if i%2 == 0:
                    tensor[:,:,i] = self.threshold(self.list_adjacency[int(i/2)], threshold)
                else:
                    tensor[:,:,i] = np.diag(inters[int((i-1)/2)])
            X = tl.tensor(tensor)
        
        elif update_method == 'neighborhood':
            tensor = np.zeros((self.size, self.size, int((2*self.length)-1)))
            updated_interlayer_indices, updated_interlayer_weights = self.get_normalized_outlinks(self.list_adjacency, 1)
            for i in range(int((2*self.length)-1)):
                if i%2 == 0:
                    tensor[:,:,i] = self.threshold(self.list_adjacency[int(i/2)], threshold)
                else:
                    inter_layer = np.zeros((self.size,self.size))
                    for k in range(self.size):
                        w, nbr = self.neighborhood_flow(int(i/2), k, updated_interlayer_indices, updated_interlayer_weights, threshold)
                        if np.isnan(w):
                            w = 1.0
                        for n in nbr:
                            inter_layer[k,int(n)] = w
                    tensor[:,:,i] = inter_layer
            X = tl.tensor(tensor)
    
        elif update_method == None:
            tensor = np.zeros((self.size, self.size, self.length))
            for i in range(self.length):
                tensor[:,:,i] = self.threshold(self.list_adjacency[i], threshold)
            X = tl.tensor(tensor)
            
        #solve for PARAFAC decomposition
        weights_parafac, factors_parafac = non_negative_parafac(X, rank = rank, n_iter_max = 500, init = 'random')

        return(weights_parafac, factors_parafac)
    
    def process_tensor(self, factors, rank):
        """
        Helper function for converting the output of ``make_tensor`` as in the function ``membership``.
        
        Parameters
        ------------
        factors: array
            First output of ``make_tensor``.
        rank: int
            Number of communities to be found which is an ad-hoc parameter in this algorithm.
            
        Returns
        -----------
        membership: list ([list1,list2,...])
            List of length ``rank`` of lists where each list contains the membership information of the nodes belonging to
            corresponding community.
        comms: list
            List of length ``length x size`` for community assignment.
        """
        comms = []
        membership = [[] for r in range(rank)]
        for i in range(self.length):
            for j in range(self.size):
                comm_id = np.argmax(((factors[0][j]+factors[1][j])/2)*factors[2][i])
                comms.append(comm_id)
                membership[comm_id].append((j,i))
        return(membership, comms)
    
    def process_matrices(self, threshs):
        """
        Helper function preparing adjacency matrices into the pipeline for DSBM converting the matrix into an edge_list.
        
        Parameters
        ------------
        threshs: 1-D array
            Set of threshold values.
        
        Returns:
        -----------
        processed_matrices: dict
            Dictionary of edge list values corresponding to each given threshold value.
        """
        processed_matrices = {}
        for k, f in enumerate(threshs):
            edge_lists = [[] for i in range(self.length)]
            for i in range(self.length):
                A = self.list_adjacency[i]
                firing = np.transpose(np.nonzero(np.triu(A)))
                for j,m in enumerate(firing):
                    if A[m[0],m[1]]<f: pass
                    else: 
                        quadreplet =(m[0], m[1], A[m[0], m[1]], i)
                        edge_lists[i].append(quadreplet)
            processed_matrices['%.2f'%f] = edge_lists
        return(processed_matrices)
    
    def dsbm_via_graphtool(self, edge_list, deg):
        """
        Running DSBM using https://graph-tool.skewed.de
        
        Overlap is True by default according to https://journals.aps.org/pre/abstract/10.1103/PhysRevE.92.042807
        
        Parameters
        --------------
        edge_list: list ([list1,list2,...])
            List of lists of length ``length``where each list contains the edge list of the corresponding layer. Output of 
            ``process_matrices``.
        deg: Bool
            If True degree_corrected model will be used and vice versa.
            
        Returns
        -----------
        membership: list([list1,list2,...])
            List of lists of length ``number_of_communities`` where each list contains the community assignment of the nodes 
            it contains.
        labels: list
            List of length ``length x size`` containing all the community assignments.
        """
        graphs = []

        g = Graph()
        graphs.append(g)
        graphs[0].add_vertex(self.size)
        e_weight = graphs[0].new_ep("double")
        e_layer = graphs[0].new_ep("int")
        n_id = graphs[0].new_vp("int", vals = [i for i in range(self.size)])
        graphs[0].add_edge_list(edge_list[0], eprops=[e_weight, e_layer])
        graphs[0].edge_properties["edge_weight"] = e_weight
        graphs[0].edge_properties["edge_layer"] = e_layer
    
    
        G = graphs[0]

        for l in range(1,self.length):
            g = Graph()
            graphs.append(g)
            graphs[l].add_vertex(self.size)
            e_weight = graphs[l].new_ep("double")
            e_layer = graphs[l].new_ep("int")
            n_id = graphs[l].new_vp("int", vals = [i for i in range(self.size)])
            graphs[l].add_edge_list(edge_list[l], eprops=[e_weight, e_layer])
            graphs[l].edge_properties["edge_weight"] = e_weight
            graphs[l].edge_properties["edge_layer"] = e_layer
        
            G = graph_union(G, graphs[l], include = False, internal_props = True)

        state = LayeredBlockState(G, deg_corr = deg, ec = G.ep.edge_layer,  recs=[G.ep.edge_weight], rec_types=["real-exponential"],  layers = True, overlap = True)
    
        labels = [comm_id for comm_id in state.get_nonoverlap_blocks()]
        
        number_of_colors = len(np.unique(labels))
        membership = [[] for i in range(number_of_colors)]
        for i in range(self.size):#(num_neurons*layers):
            for j in range(self.length):
                node_id = labels[j*self.size+i]
                membership[node_id].append((i,j))
        return(membership, labels)
        
    
    def community_consensus_iterative(self, C):
        """
        Function finding the consensus on the given set of partitions. See the paper:
        
        'Robust detection of dynamic community structure in networks', Danielle S. Bassett, 
        Mason A. Porter, Nicholas F. Wymbs, Scott T. Grafton, Jean M. Carlson et al.
    
        We apply Leiden algorithm to maximize modularity.
        
        Parameters
        ---------------
        C: array
            Matrix of size ``parameter_space x (length x size)`` where each row is the community assignment of the corresponding
            parameters.
            
        Returns
        ------------
        partition: Leidenalg object
            See https://leidenalg.readthedocs.io/en/stable/
            
        """
        
        npart,m  = C.shape 
        C_rand3 = np.zeros((C.shape)) #permuted version of C
        X = np.zeros((m,m)) #Nodal association matrix for C
        X_rand3 = X # Random nodal association matrix for C_rand3

        # randomly permute rows of C
        for i in range(npart):
            C_rand3[i,:] = C[i,np.random.permutation(m)]
            for k in range(m):
                for p in range(m):
                    if int(C[i,k]) == int(C[i,p]): X[p,k] = X[p,k] + 1 #(i,j) is the # of times node i and j are assigned in the same comm
                    if int(C_rand3[i,k]) == int(C_rand3[i,p]): X_rand3[p,k] = X_rand3[p,k] + 1 #(i,j) is the # of times node i and j are expected to be assigned in the same comm by chance
        #thresholding
        #keep only associated assignments that occur more often than expected in the random data

        X_new3 = np.zeros((m,m))
        X_new3[X>(np.max(np.triu(X_rand3,1)))/2] = X[X>(np.max(np.triu(X_rand3,1)))/2]
        
        ##turn thresholded nodal association matrix into igraph
        edge_list = []
        weight_list = []
        for k,e in enumerate(np.transpose(np.nonzero(X_new3))):
            i,j = e[0], e[1]
            pair = (i,j)
            edge_list.append(pair)
            weight_list.append(X_new3[i][j])
        
        G = ig.Graph()
        G.add_vertices(m)
        G.add_edges(edge_list)
        G.es['weight'] = weight_list
        G.vs['id'] = list(range(m))
        
        optimiser = la.Optimiser()
        partition = la.ModularityVertexPartition(G, weights = 'weight')
        diff = optimiser.optimise_partition(partition, n_iterations = -1)
        
        return(partition)
    
    def run_community_detection(self, method, update_method = None, consensus = False, **kwargs):
        '''
        Wrap-up function to run community detection using one of the 4 methods:
        
        1) Multilayer Modularity Maximization (MMM): https://leidenalg.readthedocs.io/en/stable/
        
        P. J. Mucha, T. Richardson, K. Macon, M. A. Porter and J.-P. Onnela, Science 328, 876-878 (2010).
        
        2) Infomap: https://www.mapequation.org
        
        Mapping higher-order network flows in memory and multilayer networks with Infomap, Daniel Edler, Ludvig Bohlin, 
        and Martin Rosvall, arXiv:1706.04792v2.
        
        3) Non-negative tensor factorization using PARAFAC: http://tensorly.org/stable/index.html
        
        Detecting the Community Structure and Activity Patterns of Temporal Networks: A Non-Negative Tensor Factorization 
        Approach, Laetitia Gauvin , André Panisson, Ciro Cattuto. 
        
        4) Dynamical Stochastic Block Model (DSBM): https://graph-tool.skewed.de
        
        Inferring the mesoscale structure of layered, edge-valued, and time-varying networks, Tiago P. Peixoto, Phys. Rev. E, 2015.
        
        Parameters
        ---------------
        method: str
            Either ``MMM``, ``Infomap``, ``PARA_FACT``(Tensor Factorization) or ``DSBM`` indicating the community detection method.
        update_method: str (Default: None)
            Interlayer edges will be processed based on one of the three methods, either 
            'local', 'global', 'neigborhood' and 'skeleton'. Available only for ``MMM`` and ``Infomap``.
        consensus: bool
            Statistically significant partitions will be found from a given set of parameters. See ``community_consensus_iterative``.
        **kwargs:
            interlayers: 1-D array like
                A range of values for setting the interlayer edges of the network. Pass this argument if you are using 
                ``MMM`` or ``Infomap``.
        **kwargs:
            resolutions: 1-D array like
                A range of values for the resolution parameters. Pass this argument if you are using ``MMM``.
        **kwargs:
            thresholds: 1-D array like
                A range of values to threshold the network. Pass this argumment if you are using ``Infomap``, ``PARA_FACT`` 
                or ``DSBM``.
        **kwargs:
            ranks: 1-D array like
                A range of integers for ad-hoc number of communities. Pass this argument if you are using ``PARA_FACT``.
        **kwargs:
            degree_correction: list
                A list of boolean values(either True or False) for degree correction. Pass this argument if you are using ``DSBM``.
        **kwargs:
            spikes: 2-D array
                Initial spike train array containing the spikes of size ``n x t``. Pass this argument if your ``update_method``
                is ``local`` or ``global``.
                
        Returns
        --------------
        membership_partitions: dict
            Dictionary with keys as first set of parameters lists and second set of parameters list indices indicating the 
            community assignment of each node.
        C: array
            Matrix of size ``parameter_space x(length x size)``. This is the input for ``community_consensus_iterative``.
        
        '''
        
        if method == 'MMM':
            grid = len(kwargs['interlayers'])
            membership_partitions = {}
            C = np.zeros((grid*grid, self.size*self.length))
            for i,e in enumerate(kwargs['interlayers']):
                membership_labels = []
                igraphs = self.create_igraph()
    
                ##update interlayer edges
                if update_method == 'local' or update_method == 'global': 
            
                    inter_edge = self.update_interlayer(kwargs['spikes'], X = 0.5, omega_global = e, percentage = 0.01, method = update_method)    
                    for j,f in enumerate(kwargs['resolutions']):
                        parts, inter_parts = self.leiden(igraphs, inter_edge, f)
                    
                        C[i*grid+j,:] = inter_parts.membership
                        comm_labels, comm_size  = self.membership(inter_parts)
                        membership_labels.append(comm_labels)
                        
                elif update_method == 'neighborhood':
                    interlayer_indices, interlayer_weights = self.get_normalized_outlinks(self.list_adjacency, e)
                    for j,f in enumerate(kwargs['resolutions']):
                        layers, interslice_layer, G_full = self.time_slices_to_layers(igraphs, interlayer_indices, interlayer_weights, update_method = update_method)
                        partitions = [la.RBConfigurationVertexPartition(H, weights = 'weight', resolution_parameter = f) for H in layers]
        
                        interslice_partition = la.RBConfigurationVertexPartition(interslice_layer, weights = 'weight', resolution_parameter = 0)
                                                     
                        optimiser = la.Optimiser()
        
                        diff = optimiser.optimise_partition_multiplex(partitions + [interslice_partition])
                        C[i*grid+j,:] = interslice_partition.membership
                        comm_labels, comm_size  = self.membership(interslice_partition)
                        membership_labels.append(comm_labels)
                        
                elif update_method == 'skeleton':
                    membership_static = self.MMM_static()
                    bridge_links = self.find_skeleton(membership_static)
                    bridge_weights = {}
                    for l in range(self.length):
                        for n in range(self.size):
                            bridge_weights['%d,%d'%(l,n)] = e
                    for j,f in enumerate(kwargs['resolutions']):
                        layers, interslice_layer, G_full = self.time_slices_to_layers(igraphs, bridge_links, bridge_weights, update_method = update_method)
                        partitions = [la.RBConfigurationVertexPartition(H, weights = 'weight', resolution_parameter = f) for H in layers]
        
                        interslice_partition = la.RBConfigurationVertexPartition(interslice_layer, weights = 'weight', resolution_parameter = 0)
                                                     
                        optimiser = la.Optimiser()
        
                        diff = optimiser.optimise_partition_multiplex(partitions + [interslice_partition])
                        C[i*grid+j,:] = interslice_partition.membership
                        comm_labels, comm_size  = self.membership(interslice_partition)
                        membership_labels.append(comm_labels)
                        
                elif update_method == None:
                    
                    for j,f in enumerate(kwargs['resolutions']):
                        parts, inter_parts = self.leiden(igraphs, e, f)
                    
                        C[i*grid+j,:] = inter_parts.membership
                        comm_labels, comm_size  = self.membership(inter_parts)
                        membership_labels.append(comm_labels)
                    
                membership_partitions['interlayer=%.3f'%e] = membership_labels
            
        elif method == 'infomap':
            grid1 = len(kwargs['interlayers'])
            grid2 = len(kwargs['thresholds'])
            membership_partitions = {}
            C = np.zeros((grid1*grid2, self.size*self.length))
            dtype = [('layer',int),('nodeid',int),('module', int)]

            for i, interlayer in enumerate(kwargs['interlayers']):
                inter_membership = []
                for j, thresh in enumerate(kwargs['thresholds']):
                    
                    IM = self.infomap(interlayer, thresh, update_method, **kwargs)
                    
                    membership = [[] for i in range(IM.num_top_modules)]
                    for node in IM.nodes:
                        membership[int(node.module_id-1)].append((node.node_id, node.layer_id))
                    inter_membership.append(membership)
        
                    ordered_set = []
                    for node in IM.nodes:
                        ordered_set.append((node.layer_id, node.node_id, node.module_id))
                    ordered_nodes = np.array(ordered_set , dtype = dtype)
                    
                    C[i*grid2+j,:] = [node[2] for node in np.sort(ordered_nodes, order = ['layer', 'nodeid'])]
        
                membership_partitions['interlayer=%.3f'%interlayer] = inter_membership
        
        elif method == 'PARA_FACT':
            grid1 = len(kwargs['ranks'])
            grid2 = len(kwargs['thresholds'])
            membership_partitions = {}
            C = np.zeros((grid1*grid2, self.size*self.length))
            
            for i, r in enumerate(kwargs['ranks']):
                inter_membership = []
                for j, thresh in enumerate(kwargs['thresholds']):
                    weights, factors = self.make_tensor(r, thresh, update_method, **kwargs)
                    membership, comm  = self.process_tensor(factors, r)
                    inter_membership.append(membership)
                    C[i*grid2+j,:] = comm
                membership_partitions['rank=%d'%r] = inter_membership
                
        elif method == 'DSBM':
            
            grid1 = len(kwargs['degree_correction'])
            grid2 = len(kwargs['thresholds'])
            membership_partitions = {}
            C = np.zeros((grid1*grid2, self.size*self.length))
            edge_list = self.process_matrices(kwargs['thresholds'])
            for i, deg in enumerate(kwargs['degree_correction']):
                inter_membership = []
                for j, thresh in enumerate(kwargs['thresholds']):
                    membership, comm  = self.dsbm_via_graphtool(edge_list['%.2f'%thresh], deg)
                    inter_membership.append(membership)
                    C[i*grid2+j,:] = comm
                membership_partitions['degree_correction=%s'%deg] = inter_membership
        
        if consensus: 
            return(self.membership(self.community_consensus_iterative(C))[0], C)
        else: 
            return(membership_partitions, C)

        
def normalized_cross_corr(x,y):
    """
    Function to compute normalized cross-correlation between two vectors.
    
    Parameters
    ------------
    x: 1_D array like
        First vector.
    y: 1_D array like
        Second vector.
    Returns
    ------------
    corr_array: array
        Correlation array between x and y.
    
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    x_cov_std = np.nanmax(np.sqrt(np.correlate(x - x_mean, x - x_mean, 'full')))
    y_cov_std = np.nanmax(np.sqrt(np.correlate(y - y_mean, y - y_mean, 'full')))

    normalization = x_cov_std * y_cov_std
        

    unnormalized_correlation = np.correlate(x - x_mean, y - y_mean, 'full')
    
    corr_array = unnormalized_correlation/normalization

    return(corr_array)

def max_norm_cross_corr(x1, x2):
    """
    Function for computing maximum cross-correlation.
    
    Parameters
    -----------
    x1: 1_D array like
        First vector.
    x2: 1_D array like
        Second vector.
    Returns
    -----------
    max_corr:int
        Maximum cross-correlation between the two input vectors.
    lag: int
        Lag difference where the maximum cross-correlation occurs.
    """
    
    correlation= normalized_cross_corr(x1, x2)
    
    lag = abs(correlation).argmax() - len(x1)+1
    
    max_corr = max(abs(correlation))
    
    return(max_corr, lag)

def cross_correlation_matrix(data):
    """
    Main function to call for computing cross-correlation matrix of a time series.
    
    Parameters
    ------------
    data: array
        ``n x t`` matrix where n is the number of neuorons and t is the length of the time series.
    Returns
    ------------
    X_full: array
        ``n x n`` symmetric cross-correlation matrix.
    X: array
        ``n x n`` upper triangular cross-correlation matrix.
    lag: array
        ``n x n`` lag matrix.
    """
    #input: n x t matrix where n is the number of rois and t is the duration of the time series
    #return: n x n symmetric cross correlation matrix, nxn uppertriangular cross correlation matrix and lag matrix
    n, t = data.shape
    X = np.zeros((n,n))
    lag = np.zeros((n,n))
    
    for i in range(n-1):
        for j in range(i+1,n):
            X[i][j],lag[i][j] = max_norm_cross_corr(data[i,:],data[j,:])
    X[np.isnan(X)] = 0
    lag[np.isnan(lag)] = 0
    
    X_full = X + X.T
    lag = lag + lag.T
    return(X_full, X, lag)


def bin_time_series(array, binsize, gaussian = True, **kwargs):
    """
    Helper function for windowing the time series into smaller chunks.
    
    Parameters
    ------------
    array: 2_D array
        ``n x t`` matrix where n is the number of neuorons and t is the length of the time series.
    binsize: int
        Size of each window. This number needs to be smaller than ``t`` and a positive divider of ``t``.
    gaussian: bool (Default: True)
        If True, each spike is going to be multiplied by a 1-D gaussian of length ``sigma``.
    **kwargs:
        sigma: float
            Size of the gaussian. See ``gaussian_filter``.
    Returns
    ------------
    A: array
        Matrix of size ``l x n x t`` where l is the number of windows(=t/binsize), n is number of neurons and t is the length of the time series.
        
    """
    n = array.shape[0] # number of neurons
    totalsize = array.shape[1] # total duration of spikes
    gauss_array = np.zeros((n,totalsize))
    l = int(totalsize/binsize) # number of resulting layers
        
    if gaussian:
        for i in range(n):
            gauss_array[i] = gaussian_filter(array[i],kwargs['sigma'])
    else: gauss_array = array
            
    A = np.zeros((l,n,binsize))
    for i in range(l):
        A[i] = gauss_array[:,i*binsize:(i+1)*binsize]
    return(A)

def binarize(array, thresh = None):
    """
    Function for binarizing adjacency matrices.
    
    Parameters
    ------------
    array: array like
        Cross-correlation matrix.
    thresh: float (Default: None)
        If None, entries that are non-zero are going to be set to 1. If a value between [0,1] is given, then every entry smaller than ``thresh`` will be set to 0 and 1 otherwise.
        
    Returns
    -----------
    binary_spikes:array like
        Binarized cross-correlation matrix of same size.
        
    """
    n,t = array.shape
    binary_spikes = np.zeros((n,t))
    for i in range(n):
        for j in range(t):
            if thresh is not None:
                if array[i][j] <=thresh: pass
                else: binary_spikes[i][j] = 1
            else:
                if array[i][j] == 0: pass
                else: binary_spikes[i][j] = 1
    return(binary_spikes)

def threshold(array, thresh):
    """
    Function for thresholding the adjacency matrices.
    
    Parameters
    ------------
    array: array like
        Cross-correlation matrix.
    thresh: float 
        Value in which every entry smaller than ``thresh`` will be set to 0 and entries greater than ``thresh`` will stay the same.
        
    Returns
    -----------
    threhsolded_array:array like
        Thresholded cross-correlation matrix of same size.
    """
    n,t = array.shape
    thresholded_array = np.copy(array)
    for i in range(n):
        for j in range(t):
            if array[i][j] < thresh: thresholded_array[i][j] = 0
            else: pass
    return(thresholded_array)

def gaussian_filter(array,sigma):
    """
    Function that multiplies vectors with a gaussian.
    
    Parameters
    --------------
    array: 1_D array like
        Input vector.
    sigma: float
        1 spike turns into 3 non-zero spikes(one at each side of smaller magnitude) with sigma=0.25. 1 spike turns into 5 non-zero spikes(two at each side of smaller magnitude) with sigma=0.50. 1 spike turns into 9 non-zero spikes(four at each side of smaller magnitude) with sigma=1, and so on..
        
    Returns
    ----------
    array: 1_D array like
        Gaussian vector.
    """
    #sigma=0.25==gaussian kernel with length 3
    #sigma=0.5==gaussian kernel with length 5
    #sigma=1==gaussian kernel with length 9
    return(gaussian_filter1d(array,sigma))

def jitter(spike, k):
    """
    Function for randomly jittering spikes when generating communities.
    
    Parameters
    -------------
    spike: array
        Spike train to be jittered.
    k: int
        Number of time frames, to the right or to the left, for a spike to be jittered.
    Returns
    -------------
    jittered: array
        Jittered spike train.
    """
    #jittering the given spike train
    jittered = np.zeros(spike.shape)
    for i in np.nonzero(spike)[1]:
        jitt = random.randint(-k,k)
        try:jittered[0,i+jitt] = 1
        except:jittered[0,i] = 1
    return(jittered)

def spike_count(spikes, ax, num_bins = None, t_min = None, t_max = None):
    """
    Helper function to visualize the distribution of the number of spikes in a given spike train.
    
    Parameters
    ------------
    spikes: array
        Spike train matrix of size ``n x t``.
    ax: matplotlib axis
        Axis for distribution to be plotted.
    num_bins: int (Default: None)
        If None, this will be the difference between maximum and minimum number of spikes in a population.
    t_min: int (Default: None)
        if None, this will be ``0``, otherwise spikes will be counted in the given range.
    t_max: int (Default: None)
        if None, this will be ``t``, otherwise spikes will be counted in the given range.
        
    Returns
    ------------
    n: array
        The values of the histogram bins.
    bins: array
        The edges of the bins. 
    """
    n,t = spikes.shape
    if t_min is None: t_min = 0
    if t_max is None: t_max = t
    if t_max<=t_min: raise ValueError('t_min should be less than t_max')
    spike_count = []
    binary = binarize(spikes)
    for i in range(n):
        spike_count.append(np.sum(binary[i][t_min:t_max]))
    if num_bins is None: num_bins = int(np.max(spike_count) - np.min(spike_count))
    n, bins, patches = ax.hist(spike_count, num_bins, color = 'blue')
    ax.set_title("Spike Rate Distribution")
    ax.set_xlabel("Total Number of Spikes", fontsize = 22)
    ax.set_ylabel("Number of Neurons", fontsize = 22)
    return(n,bins)

def find_repeated(l):
    """
    Helper function for generating transient communities.
    """
    k = []
    repeated = []
    for i in l:
        if i not in k:
            k.append(i)
        else:
            if i not in repeated:
                repeated.append(i)
    return(repeated)

def get_repeated_indices(l):
    """
    Helper function for generating transient communities. 
    """
    repeated = find_repeated(l)
    k = []
    for r in repeated:
        first = l.index(r)
        reversed_l = l[::-1]
        last = len(l) - reversed_l.index(r)
        k.append((first,last))
    return(k)

def getOverlap(a, b):
    """
    Helper function for generating transient communities. Finds repeated indices.
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def generate_ground_truth(comm_sizes, method = 'scattered', pad = False, community_operation = 'grow'):
    """
    Main function that generates ground truth labels for the experiments. Community labels according to two methods one in which the rest of the network except the planted communities are scattered i.e. they all have their own community or they are all in one community, integrated.
    
    Parameters
    ------------
    comm_sizes: list, or list of lists
        If the ``community_operation`` is ``grow`` (or ``contract``), this should be a list indicating the number of neurons joining (or leaving from) the main community. If the ``community_operation`` is ``merge`` or ``transient``, then this should be a list of lists([list1,list2,...]) where each list contains sizes of the communities in that layer. For example, [[6,1,1,1,1],[6,4]] indicates a 6 neuron community in the first layer and additional 4 neurons, that are independently firing, merges into 1 community in the second layer.
    method: ``scattered`` or ``integrated`` (Default: 'scattered')
        If the ``community_operation`` is ``grow`` (or ``contract``), two types of ground truths can be prepared. Integrated is the one where independently firing neurons are grouped together into one single community and scattered is the one with independently firing neurons.
    pad: bool (Default: False)
        If True, the truth will be padded from the beginning and the end by the exact same community membership.
    community_operation: ``grow``, ``contract``, ``merge`` or ``transient`` (Default: 'grow') 
        Type of community events that are available. Community expansion, community contraction, community merge and transient communities.
        
    Returns
    -----------
        truth_labels:list
            List of truth labels, of length ``n*t`` where n is the number of neuorons and t is the number of layers. If pad, length will be ``n*(t+2)``.
    """
    if community_operation == 'grow':
        layers = len(comm_sizes)
        if method == 'scattered':
            truth_labels = [0 for i in range(sum(comm_sizes[:1]))] + list(np.arange(1, sum(comm_sizes[1:])+1))
            
            truth_labels_tip = truth_labels
                     
            for j in range(2,layers):
                truth_labels = truth_labels + [0 for i in range(sum(comm_sizes[:j]))] + truth_labels_tip[sum(comm_sizes[:j]):]
            
            truth_labels = truth_labels + [0 for i in range(sum(comm_sizes[:layers]))]
            
            if pad:
                truth_labels = truth_labels_tip + truth_labels
                truth_labels = truth_labels + [0 for i in range(sum(comm_sizes[:layers]))]  
        
        if method == 'integrated':
        
            truth_labels = [0 for i in range(sum(comm_sizes[:1]))] + [1 for i in range(sum(comm_sizes[1:]))]
            if pad: truth_labels = truth_labels + truth_labels
            for j in range(2,layers):
                truth_labels = truth_labels + [0 for i in range(sum(comm_sizes[:j]))] + [1 for i in range(sum(comm_sizes[j:]))]

            if pad:
                truth_labels = truth_labels + [0 for i in range(sum(comm_sizes[:layers]))]
                truth_labels = truth_labels + [0 for i in range(sum(comm_sizes[:layers]))]
    
    elif community_operation == 'contract':
        layers = len(comm_sizes)
        if method == 'scattered':
            truth_labels_tip = [0 for i in range(sum(comm_sizes))]
            truth_labels_end = [0 for i in range(sum(comm_sizes[:1]))] + list(np.arange(1, sum(comm_sizes[1:])+1))
            truth_labels = truth_labels_tip
            
            for j in range(1,layers):
                truth_labels = truth_labels + [0 for i in range(sum(comm_sizes[:(layers-j)]))] + truth_labels_end[sum(comm_sizes[:(layers-j)]):]
            
            
            if pad:
                truth_labels = truth_labels_tip + truth_labels + truth_labels_end
                
        if method == 'integrated':
            truth_labels = [0 for i in range(sum(comm_sizes))]
            truth_labels_tip = truth_labels
            for j in range(1,layers):
                truth_labels = truth_labels + [0 for i in range(sum(comm_sizes[:(layers-j)]))] + [1 for i in range(sum(comm_sizes[(layers-j):]))]
                
            truth_labels_end = truth_labels[sum(comm_sizes)*(layers-1):]
            if pad:
                truth_labels = truth_labels_tip + truth_labels +truth_labels_end
            
                
    elif community_operation == 'merge': ##only for two layers
        truth_labels = []
        
        for j,f in enumerate(comm_sizes[0]):
            truth_labels = truth_labels + [j for k in range(f)]
    
        for j,f in enumerate([6,3,7]):##communities that are merged in the first layer are assigned one of the labels of 
            #merged communities 
            truth_labels = truth_labels + [f for i in range(comm_sizes[1][j])]
            
        if pad:
            l1 = truth_labels[:sum(comm_sizes[0])]
            l2 = truth_labels[sum(comm_sizes[0]):]
            truth_labels = l1 + truth_labels +l2
     
    elif community_operation == 'transient':
        truth_labels = []
        maks = 0
        layers = len(comm_sizes)
        num_neurons = sum(comm_sizes[0])
        for i,f in enumerate(comm_sizes[0]):
            truth_labels = truth_labels + [i for j in range(f)]
        for k in range(1,layers):
            maks = max(truth_labels)
            for i,f in enumerate(comm_sizes[k]):
                truth_labels = truth_labels + [i + maks+1 for j in range(f)]
        
        for l in range(1,layers):
            current = get_repeated_indices(truth_labels[l*num_neurons:(l+1)*num_neurons])
            prev = get_repeated_indices(truth_labels[(l-1)*num_neurons:(l)*num_neurons])
            for c in current:
                for p in prev:
                    O = getOverlap(p,c)
                    if O/(c[1]-c[0]) >= 1/2:
                        truth_labels[l*num_neurons+c[0]:l*num_neurons+c[1]] = [truth_labels[(l-1)*num_neurons+p[0]]]*(c[1]-c[0])
            
            for i in range(num_neurons):
                try:
                    if truth_labels[l*num_neurons+i] == truth_labels[l*num_neurons+i+1] or truth_labels[l*num_neurons+i] == truth_labels[l*num_neurons+i-1]: pass
                    else:
                        if truth_labels[(l-1)*num_neurons+i] == truth_labels[(l-1)*num_neurons+i+1] or truth_labels[(l-1)*num_neurons+i] == truth_labels[(l-1)*num_neurons+i-1]: pass
                        else:
                            truth_labels[l*num_neurons + i] = truth_labels[(l-1)*num_neurons+ i]
                except:
                    if truth_labels[l*num_neurons+i] == truth_labels[l*num_neurons+i-1]: pass
                    else:
                        if truth_labels[(l-1)*num_neurons+i] == truth_labels[(l-1)*num_neurons+i+1] or truth_labels[(l-1)*num_neurons+i] == truth_labels[(l-1)*num_neurons+i-1]: pass
                        else:
                            truth_labels[l*num_neurons + i] = truth_labels[(l-1)*num_neurons+ i]
                        
        if pad:
            l1 = truth_labels[:sum(comm_sizes[0])]
            l2 = truth_labels[sum(comm_sizes[0])*(layers-1):]
            truth_labels = l1 + truth_labels +l2
            
    return(truth_labels)

def information_recovery(pred_labels, comm_size, truth, interlayers, other_parameter, com_op):
    """
    Function for calculating the quality of the resulting partitions on a parameter plane and visualizes the quality landscape according to NMI, ARI and F1-Score.
    
    Parameters
    -----------
    pred_labels: list
        List of truth labels appended in the order of layers. This should be the same length as the output of ``generate_ground_truth``.
    comm_size: list, or list of lists
        This will be passed to ``generate_ground_truth`` that assumes pad to be True.
    truth: ``integrated`` or ``scattered``
        Same as in ``generate_ground_truth``.
    interlayers: 1_D array like
        To get the landscape information on a plane of parameters, we pass two array like object. This one is the y-axis one on the result.
    other_parameter: 1_D array like
        This is the x-axis array for quality.
    com_op: ``grow``, ``contract``, ``merge`` or ``transient``
        Same as in ``generate_ground_truth``.
        
    Returns
    ------------
    fig: matplotlib object
        Figure object for the plots.
    ax: matplotlib object
        Axis objects for the plots.
    """
    
    NMI1 = np.zeros((len(interlayers), len(other_parameter)))
    ARI1 = np.zeros((len(interlayers), len(other_parameter)))
    F1S1 = np.zeros((len(interlayers), len(other_parameter)))
    
    if truth == 'Scattered': true_labels = generate_ground_truth(comm_size, method = 'scattered', pad = True, community_operation = com_op)
    if truth == 'Integrated': true_labels = generate_ground_truth(comm_size, method = 'integrated', pad = True, community_operation = com_op)

    
    for i in range(len(interlayers)):
        for j in range(len(other_parameter)):
            NMI1[i][j] = normalized_mutual_info_score(true_labels, list(pred_labels[i*len(other_parameter)+j].astype(int)), average_method = 'max')
            ARI1[i][j] = adjusted_rand_score(true_labels, list(pred_labels[i*len(other_parameter)+j].astype(int)))
            F1S1[i][j] = f1_score(true_labels, list(pred_labels[i*len(other_parameter)+j].astype(int)), average = 'weighted')
        
    fig,ax = plt.subplots(1,3, figsize = (85, 50))
    normalize = Normalize(vmin=0, vmax=1)
    c = ax[0].imshow(NMI1, origin = 'lower', 
                     interpolation = 'none', 
                     cmap = 'Reds', aspect = 'auto',
                     norm = normalize,
                     extent = [other_parameter[0]-0.005, other_parameter[-1]+0.005, interlayers[0]-0.005, interlayers[-1]+0.005])

    c = ax[1].imshow(ARI1, origin = 'lower', 
                     interpolation = 'none', 
                     cmap = 'Reds', aspect = 'auto',
                     norm = normalize,
                     extent = [other_parameter[0]-0.005, other_parameter[-1]+0.005, interlayers[0]-0.005, interlayers[-1]+0.005])

    c = ax[2].imshow(F1S1, origin = 'lower',
                     interpolation = 'none', 
                     cmap = 'Reds', aspect = 'auto',
                     norm = normalize,
                     extent = [other_parameter[0]-0.005, other_parameter[-1]+0.005, interlayers[0]-0.005, interlayers[-1]+0.005])

    ax[0].set_title('NMI wrt %s Ground Truth'%truth, fontsize = 60)
    ax[0].set_xlabel('Threshold', fontsize = 50)
    ax[0].set_ylabel('Interlayer Coupling', fontsize = 50)
    ax[0].set_xticks([i*0.1 for i in range(9)])
    ax[0].set_yticks(interlayers)
    ax[0].tick_params(axis = 'both', labelsize = 30)

    ax[1].set_title('ARI wrt %s Ground Truth'%truth, fontsize = 60)
    ax[1].set_xlabel('Threshold', fontsize = 50)
    ax[1].set_ylabel('Interlayer Coupling', fontsize = 50)
    ax[1].set_xticks([i*0.1 for i in range(9)])
    ax[1].set_yticks(interlayers)
    ax[1].tick_params(axis = 'both', labelsize = 30)

    ax[2].set_title('F1-Score wrt %s Ground Truth'%truth, fontsize = 60)
    ax[2].set_xlabel('Threshold', fontsize = 50)
    ax[2].set_ylabel('Interlayer Coupling', fontsize = 50)
    ax[2].set_xticks([i*0.1 for i in range(9)])
    ax[2].set_yticks(interlayers)
    ax[2].tick_params(axis = 'both', labelsize = 30)
    
    cbar = fig.colorbar(c, ax = ax.flat, orientation = 'horizontal')
    cbar.ax.tick_params(labelsize = 40) 
    
    return(fig,ax)
    
def display_truth(comm_sizes, community_operation, ax = None):
    """
    Function for displaying the ground truths.
    
    Parameters
    -------------
    comm_sizes: list, or list of lists
        This will be passed to ``generate_ground_truth`` where pad is True by default.
    community_operation: ``grow``, ``contract``, ``merge`` or ``transient``
        Type of the community event which will also be passed to ``generate_ground_truth``.
    ax: matplotlib object (Default: None)
        If None, a new axis will be created, otherwise the ground truth will be plotted to the provided axis.
    
    """
    if community_operation == 'grow' or community_operation == 'contract':
        n = sum(comm_sizes)
        layers = len(comm_sizes)
        l = layers + 2
    
        scattered_truth = generate_ground_truth(comm_sizes, 
                                                method = 'integrated', 
                                                pad = False, 
                                                community_operation = community_operation)
        
        number_of_colors = max(scattered_truth)+1
    
        membership = [[] for i in range(number_of_colors)]
        for i,m in enumerate(scattered_truth):
            time = floor(i/n)
            node_id = i%n
            membership[m].append((node_id,time))

        if ax is None: 
            fig,ax = plt.subplots(1,2, figsize = (16,8))

        comms = np.zeros((n,layers+2))

        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

        for i, l in enumerate(membership):
            for j,k in enumerate(l):
                comms[k[0]][k[1]] = i

        cmap = mpl.colors.ListedColormap(color)

        ax[0].imshow(comms, interpolation = 'none', cmap = cmap, aspect = 'auto', origin = 'lower', extent = [-0.5,layers+2-0.5,-0.5,n-0.5])
        ax[0].set_xticks([i for i in range(layers+2)])
        ax[0].set_yticks([i*10 for i in range(int(n/10)+1)])
        ax[0].tick_params(axis = 'both', labelsize = 15)
        ax[0].set_xlabel('Layers (Time)', fontsize = 18)
        ax[0].set_ylabel('Neuron ID', fontsize = 18)
        ax[0].set_title('Integrated Ground Truth with %d Communities' %len(color), fontsize = 20)
    
        integrated_truth = generate_ground_truth(comm_sizes, 
                                                 method = 'scattered', 
                                                 pad = False, 
                                                 community_operation = community_operation)
        number_of_colors = max(integrated_truth)+1
        membership = [[] for i in range(number_of_colors)]
        for i,m in enumerate(integrated_truth):
            time = floor(i/n)
            node_id = i%n
            membership[m].append((node_id,time))

        comms = np.zeros((n,layers+2))

        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

        for i, l in enumerate(membership):
            for j,k in enumerate(l):
                comms[k[0]][k[1]] = i

        cmap = mpl.colors.ListedColormap(color)

        ax[1].imshow(comms, interpolation = 'none', cmap = cmap, aspect = 'auto', origin = 'lower', extent = [-0.5,layers+2-0.5,-0.5,n-0.5])
        ax[1].set_xticks([i for i in range(layers+2)])
        ax[1].set_yticks([i*10 for i in range(int(n/10)+1)])
        ax[1].tick_params(axis = 'both', labelsize = 15)
        ax[1].set_xlabel('Layers (Time)', fontsize = 18)
        ax[1].set_ylabel('Neuron ID', fontsize = 18)
        ax[1].set_title('Scattered Ground Truth with %d Communities' %len(color), fontsize = 20)
    
    elif community_operation == 'merge' or community_operation == 'transient':
        n = sum(comm_sizes[0])
        layers = len(comm_sizes)
        l = layers + 2
        
        truth = generate_ground_truth(comm_sizes, pad = True, community_operation = community_operation)
        
        number_of_colors = max(truth)+1
    
        membership = [[] for i in range(number_of_colors)]
        for i,m in enumerate(truth):
            time = floor(i/n)
            node_id = i%n
            membership[m].append((node_id,time))

        fig,ax = plt.subplots(1,1, figsize = (8,8))

        comms = np.zeros((n,layers+2))

        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

        for i, l in enumerate(membership):
            for j,k in enumerate(l):
                comms[k[0]][k[1]] = i

        cmap = mpl.colors.ListedColormap(color)

        ax.imshow(comms, interpolation = 'none', cmap = cmap, aspect = 'auto', origin = 'lower', extent = [-0.5,layers+2-0.5,-0.5,n-0.5])
        ax.set_xticks([i for i in range(layers+2)])
        ax.set_yticks([i*10 for i in range(int(n/10)+1)])
        ax.tick_params(axis = 'both', labelsize = 15)
        ax.set_xlabel('Layers (Time)', fontsize = 18)
        ax.set_ylabel('Neuron ID', fontsize = 18)
        ax.set_title('Ground Truth with %d Communities' %len(np.unique(truth)), fontsize = 20)
        
        
def space_comms(comm_size):
    """
    Helper function for spacing the communities randomly for the transient communities.
    """
    lll = []
    rrr = []
    for c in comm_size:
        for i in range(random.randint(8,2*sum(comm_size))):
            lll.append(1)
            rrr.append(random.randint(10,30))
        lll.append(c)
        rrr.append(random.randint(10,30))
    return(lll,rrr)

def generate_transient(comm_per_layer):
    """
    Helper function for creating the time series for the transient communities.
    
    Parameters
    -------------
    comm_per_layer:list of lists
        List of lists of length ``number of layers`` where each list contains the number of communities at that layer.
    
    Returns
    -----------
    comm_sizes: list of lists
        Sizes of the communities in the corresponding layers which will be passed to ``create_time_series``.
    spike_rate: list of lists
        Randomly selected corresponding spike rates for the Homogeneous Poisson process that generates spike trains.
    num_neurons: int
        Number of neurons.
    """
    layer = len(comm_per_layer)
    comm_size = []
    for i in range(layer):
        comm_size.append([int(np.random.power(3/2)*20)  for j in range(comm_per_layer[i])])
        
    num_neurons = int(max([sum(comm_size[i]) for i in range(len(comm_size))])*(2))
    
    comm_sizes = []
    spike_rate = []
    for c in comm_size:
        flag = True
        while flag:
            temp_size, temp_rate = space_comms(c)
            if sum(temp_size) < num_neurons:
                temp_rate = temp_rate + [random.randint(10,30) for j in range(num_neurons-sum(temp_size))]
                temp_size = temp_size + [1 for i in range(num_neurons-sum(temp_size))]
                comm_sizes.append(temp_size)
                spike_rate.append(temp_rate)
                flag = False
    return(comm_sizes, spike_rate, num_neurons)


def create_time_series(operation, community_sizes, spiking_rates, spy = True, windowsize = 1000, k = 5):
    """
    Main function for creating spike trains using Homogeneous Poisson Process.
    
    Parameters
    --------------
    operation: ``grow``, ``contract``, ``merge`` or ``transient``
        Community operation.
    community_sizes:list or list of lists
        If the ``operation`` is ``grow`` (or ``contract``), this should be a list indicating the number of neurons joining (or leaving from) the main community. If the ``operation`` is ``merge`` or ``transient``, then this should be a list of lists([list1,list2,...]) where each list contains sizes of the communities in that layer.
    spike_rates: list or list of lists
        This should be of same size and shape as ``community_sizes`` indicating the spike rates of the corresponding communities.
    spy: bool (Deafult: True)
        Displays the time series if True.
    windowsize: int (Default: 1000)
        Length of the window size for a new layer of events to be created.
    k: int (Default: 5)
        Constant for jittering the spikes when creating new communities.
    Returns
    -------------
    spikes: array
        Matrix of size ``n x t`` where n is the number of neuorons and t is the length of the time series.
    """
    
    binsize = windowsize
    layers = len(community_sizes)
    total_duration = int(layers*binsize)
    
    if operation == 'grow' or operation == 'contract':    
        num_neurons = int(sum(community_sizes))
        spikes = np.zeros((num_neurons,total_duration))
        master_spike = np.zeros((1,total_duration))
        master = homogeneous_poisson_process(rate = spiking_rates[0]*Hz, 
                                             t_start = 0.0*ms, 
                                             t_stop = total_duration*ms, 
                                             as_array = True) 
        for i,e in enumerate(master):    
            master_spike[0][int(e)] = 1
        for i in range(community_sizes[0]):
            spikes[i] = jitter(master_spike, k)
        
        comms = []
        for i in range(1,layers):
            comms.append([homogeneous_poisson_process(rate = spiking_rates[i]*Hz, 
                                                      t_start = 0.0*ms, 
                                                      t_stop = i*binsize*ms, 
                                                      as_array = True) for j in range(community_sizes[i])])
        neuron_count = community_sizes[0]
        for i,e in enumerate(comms):
            for j,f in enumerate(e):
                for k,m in enumerate(f):
                    spikes[neuron_count+j][int(m)] = 1
            neuron_count = neuron_count + len(e)
        
        neuron_count = community_sizes[0]
        for i in range(1, len(comms)+1):
            for j in range(neuron_count, neuron_count + community_sizes[i]):
                for k in np.nonzero(spikes[0][(i*binsize):])[0]:
                    jitt = random.randint(-5,5)
                    try:spikes[j,(i*binsize)+k+jitt] = 1
                    except:spikes[j,k] = 1
            neuron_count = neuron_count + community_sizes[i]
        if operation =='contract':
            spikes = np.flip(spikes,1)
        
    if operation == 'merge' or operation == 'transient':
        num_neurons = int(sum(community_sizes[0]))
        
        spikes = np.zeros((num_neurons,total_duration))
        
        for s in range(layers):
            neuron_count = 0
            for i,e in enumerate(community_sizes[s]):
                initial_master = homogeneous_poisson_process(rate = spiking_rates[s][i]*Hz,
                                                             t_start = s*(binsize)*ms, 
                                                             t_stop = (s+1)*binsize*ms, 
                                                             as_array = True)
                master_spikes = np.zeros((1,total_duration))
    
                for j,f in enumerate(initial_master):
                    master_spikes[0][int(f)] = 1

                for j in range(e):
                    spikes[neuron_count+j][int(s*binsize):int((s+1)*binsize)] = jitter(
                        master_spikes[:,int(s*binsize):int((s+1)*binsize)], k)
                neuron_count = neuron_count + e
        
            
    if spy:
        fig,ax = plt.subplots(1,1,figsize=(20,10))
        ax.imshow(spikes, origin = 'lower', interpolation='nearest', aspect='auto', 
                  extent = [0,total_duration,0,num_neurons])
        ax.set_title('Spike Trains generated via Poisson Process for %d synthetic neurons'%num_neurons, 
                     fontsize = 30)
        ax.set_xlabel('TIME (in Miliseconds)', fontsize = 20)
        ax.set_xticks([j*binsize for j in range(int(total_duration/binsize)+1)])
        ax.set_yticks([i*10 for i in range(int(num_neurons/10)+1)])
        ax.set_ylabel('Neuron ID', fontsize = 25)
        ax.set_xlabel('Time (Frames)', fontsize = 20)
        ax.tick_params(axis = 'both', labelsize = 20)
            
    return(spikes)

def community_consensus_iterative(C):
    """
    Function finding the consensus on the given set of partitions. See the paper:
        
    'Robust detection of dynamic community structure in networks', Danielle S. Bassett, 
    Mason A. Porter, Nicholas F. Wymbs, Scott T. Grafton, Jean M. Carlson et al.
    
    We apply Leiden algorithm to maximize modularity.
        
    Parameters
    ---------------
    C: array
       Matrix of size ``parameter_space x (length * size)`` where each row is the community assignment of the corresponding
       parameters.
            
    Returns
    ------------
    partition: Leidenalg object
       See https://leidenalg.readthedocs.io/en/stable/
            
    """
        
    npart,m  = C.shape 
    C_rand3 = np.zeros((C.shape)) #permuted version of C
    X = np.zeros((m,m)) #Nodal association matrix for C
    X_rand3 = X # Random nodal association matrix for C_rand3

        # randomly permute rows of C
    for i in range(npart):
        C_rand3[i,:] = C[i,np.random.permutation(m)]
        for k in range(m):
            for p in range(m):
                if int(C[i,k]) == int(C[i,p]): X[p,k] = X[p,k] + 1 #(i,j) is the # of times node i and j are assigned in the same comm
                if int(C_rand3[i,k]) == int(C_rand3[i,p]): X_rand3[p,k] = X_rand3[p,k] + 1 #(i,j) is the # of times node i and j are expected to be assigned in the same comm by chance
        #thresholding
        #keep only associated assignments that occur more often than expected in the random data

    X_new3 = np.zeros((m,m))
    X_new3[X>(np.max(np.triu(X_rand3,1)))/2] = X[X>(np.max(np.triu(X_rand3,1)))/2]
        
        ##turn thresholded nodal association matrix into igraph
    edge_list = []
    weight_list = []
    for k,e in enumerate(np.transpose(np.nonzero(X_new3))):
        i,j = e[0], e[1]
        pair = (i,j)
        edge_list.append(pair)
        weight_list.append(X_new3[i][j])
        
    G = ig.Graph()
    G.add_vertices(m)
    G.add_edges(edge_list)
    G.es['weight'] = weight_list
    G.vs['id'] = list(range(m))
        
    optimiser = la.Optimiser()
    partition = la.ModularityVertexPartition(G, weights = 'weight')
    diff = optimiser.optimise_partition(partition, n_iterations = -1)
        
    return(partition)

def consensus_display(partition, n, t):
    """
    Helper function to visualize the consensus from ``community_consensus_iterative``.
    
    Parameters
    ------------
    partition: Leidenalg object
       See https://leidenalg.readthedocs.io/en/stable/
    n: int
        Number of neurons.
    t: int
        Number of layers.
    Returns
    ------------
    comms: array
        Community membership array of size ``n x t``.
    cmap: matplotlib object
        Colormap used to plot the membership information.
    color: list
        List of strings encoding the colors of the communities.
    """
    membership = [[] for i in range(partition._len)]
    for i,m in enumerate(partition._membership):
        time = floor(i/n)
        node_id = i%n
        membership[m].append((node_id,time))

    number_of_colors = len(membership)

    comms = np.zeros((n,t))

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

    for i, l in enumerate(membership):
        for j,k in enumerate(l):
            comms[k[0]][k[1]] = i

    cmap = mpl.colors.ListedColormap(color)
    
    return(comms, cmap, color)

def read_csv(path, in_put, subject, roi):
    """
    Function for reading .csv files containing spike complexity, features or raw calcium time series.
        
    Parameters
    -------------
    path : str
        path of the file
    in_put : str
        Type of the input i.e. spike complexity, features or time series. Will be appended to the path.
    subject : str
        Subject ID. Will be appended to the path.
    roi : str
        Subdirectory containing ROI files.
        
    Returns
    -----------
    spikes : array
        Numpy array containing the input.
        
    """
        
    spike = open( path + in_put + subject + "_spikes_complexity.csv", "r")
    reader_spike = csv.reader(spike)
    n = read_roi(path, roi, subject)
    spikes = np.zeros((n,8000)) # roi x time
    
    for i,line in enumerate(reader_spike):
        for j in range(len(line)):
            spikes[i][j]=line[j]
            
    return(spikes)

def load_obj(path, name):
    """
    Function that loads the pickled objects.
        
    Parameters
    -------------
    path : str
        Path directory for the object.
    name : str
        name of the .pkl object.
            
    Returns
    -----------
    pickled : .pkl object
        Object to be returned.
    """
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
            
def read_roi(path, roi, subject_roi):
    """
    Helper function for reading .roi files. This function is necessary for reading out the .csv files.
     
    Parameters
    -------------
    path : str
        Path of the object.
    roi : str
        Name of the ROI file that will be appended to the path.
    subject_roi : str
        Name of the subject.
        
    Returns
    ---------
    n : int
        Number of ROIs for the given subject.
    """
    roi = read_roi_zip(glob(path+roi+subject_roi +'.zip')[0])
    n = len(roi)
    for i, R in enumerate(roi):
        x = roi[R]['x']
        y = roi[R]['y']
return(n)

# In[ ]:




