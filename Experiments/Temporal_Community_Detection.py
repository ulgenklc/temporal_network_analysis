#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from scipy.spatial.distance import jensenshannon
from math import floor
#import leidenalg as la
#import igraph as ig
#from infomap import Infomap, MultilayerNode


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
        t = self.length
        n = self.size
        aggragated = np.zeros((n,n))
        
        for i,c in enumerate(self.list_adjacency):
            aggragated = aggragated + c
            
        if normalized: return (aggragated/t)
        else: return (aggragated)
    
    def binarize(self, array):
        n,t = array.shape
        binary_spikes = np.zeros(array.shape)
        for i in range(n):
            for j in range(t):
                if array[i][j] == 0: pass
                else: binary_spikes[i][j] = 1
        return(binary_spikes)
    
    def threshold(self, array, thresh):
        n,t = array.shape
        thresholded_array = np.copy(array)
        for i in range(n):
            for j in range(t):
                if array[i][j] < thresh: thresholded_array[i][j] = 0
                else: pass
        return(thresholded_array)
    
    def bin_time_series(self, array, gaussian = True, **kwargs):
        #input: nxt matrix 
        #returns: binned time series i.e. l x n x binsize
        
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
    
    def average_degree(self,layer):
        
        average_degree = 0
        
        for i in range(self.size):
            average_degree = average_degree + len(self.neighbors(i,layer))
        
        return(average_degree/(2*self.size))
    
    def create_igraph(self):
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
    
    def infomap(self, inter_edge, threshold, update_method = None, **kwargs):
        '''
        Infomap helper function. 
        '''
        im = Infomap("--two-level --directed --silent")
            ######### Make Network
            ## add intra edges
        thresholded_adjacency = []
        for l in range(self.length):
            thresholded_adjacency.append(self.threshold(self.list_adjacency[l], thresh = threshold))
            for n1,e in enumerate(thresholded_adjacency[l]):## list of length 2 corresponding to the adjacency matrices in each layer
                for n2,w in enumerate(e):
                    s = MultilayerNode(layer_id = l, node_id = n1)
                    t = MultilayerNode(layer_id = l, node_id = n2)
                    im.add_multilayer_link(s, t, w)
                    im.add_multilayer_link(t, s, w)
                
        ## add inter edges
        if update_method == 'local' or update_method == 'global': 
        
            updated_interlayer = self.update_interlayer(kwargs['spikes'], 0, inter_edge, 0.1, update_method) 
        
            for l in range(self.length-1):
                for k in range(self.size):# number of nodes which is 60 in the multilayer network
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
                    
        elif update_method == None:
            for l in range(self.length-1):
                for k in range(self.size):# number of nodes which is 60 in the multilayer network
                    s = MultilayerNode(layer_id = l, node_id = k)
                    t = MultilayerNode(layer_id = l+1, node_id = k)
                    im.add_multilayer_link(s, t, inter_edge)
                    im.add_multilayer_link(t, s, inter_edge)
        
        im.run()
        return(im)
    
    def membership(self, interslice_partition): ## returns the community assignments from the leiden algorithm as
        ##                                       tuple (n,t) n is the node id t is the layer that node is in
        n = self.size
        membership = [[] for i in range(interslice_partition._len)]
        for i,m in enumerate(interslice_partition._membership):
            time = floor(i/n)
            node_id = i%n
            membership[m].append((node_id,time))
        return(membership, len(membership))
    
    def community(self, membership, ax):
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
    
    def raster_plot(self, spikes, ax, color = None, **kwargs):#plots the raster plot of the spike activity on a 
        # given axis 'comm_assignment' and 'color' arguments are the outputs of the function 'community' 
        # and they display the community assignment of the spiking activity if provided. if not, raster 
        # ais going to be plotted blue
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
        #function graphing the edge trajcetories of the temporal
        ## network. Tresh is for thresholding the paths that are strongere than the given value.
        ## if node_id is None, function is going to graph all of the nodes's trajectories.
        ## community argument is for indicating the community assignment
        ## of the nodes if exists, if not pass along None.
        ## edge_color
        ## pv == pass a list of pv cell indices or None --dashes the pv cells
        
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
        return(interlayer_indices, interlayer_weights)
    
    def neighborhood_flow(self, layer, node, interlayer_indices, interlayer_weights, thresh):
        length = int(min(len(interlayer_weights['%d,%d'%(layer,node)]),len(interlayer_weights['%d,%d'%(layer+1,node)]))*thresh)
        w = 1-jensenshannon(interlayer_weights['%d,%d'%(layer,node)][:length],interlayer_weights['%d,%d'%(layer+1,node)][:length])**2
        nbr = interlayer_indices['%d,%d'%(layer,node)][:length]
        return(w,nbr)
        
        
    def update_interlayer(self, spikes, X, omega_global, percentage, method):
        
        ## all three methods in this function assumes the diagonal coupling
        ## i.e. output is the list(of length layers -1) of lists (each of length number of neuorns)
        ## corresponding to a node's interlayer coupling strength with it's future self.
        binned_spikes = self.bin_time_series(spikes, gaussian = False)
        sp = np.nonzero(binned_spikes)
        
        layers ,num_neurons, t = self.length, self.size, self.windowsize
        
        count_spikes = np.zeros((layers, num_neurons))
        interlayer = np.ones((layers-1, num_neurons))
    
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
    
    def community_consensus_iterative(self, C):
        ## function finding the consensus of a given set of partitions. refer to the paper:
        ## 'Robust detection of dynamic community structure in networks', Danielle S. Bassett, 
        ## Mason A. Porter, Nicholas F. Wymbs, Scott T. Grafton, Jean M. Carlson et al.
        
        
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
        Main Function to run community detection.
        
        Parameters
        ===========
        method: str
            Either MMM or infomap indicating the community detection method
        update_method: str
            Interlayer edges will be processed based on one of the three methods, either 
            local, global or neigborhood see `infomap`.
        consensus: bool
            Statistically significant partitions will be found from a given set of partitions.
        interlayers: 1-D array like
            A range of values for setting the interlayer edges of the network.
        resolutions: 1-D array like
            A range of values for the resolution parameters.
        thresholds: 1-D array like
            A range of values to threshold the network.
        spikes: 2-D array
            Initial array containing the spikes.
        
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
                else: inter_edge = e
                
                for j,f in enumerate(kwargs['resolutions']):
                    parts, inter_parts = self.leiden(igraphs, inter_edge, f)
                    
                    C[i*grid+j,:] = inter_parts.membership
                    comm_labels, comm_size  = self.membership(inter_parts)
                    membership_labels.append(comm_labels)
                    
                membership_partitions['interlayer=%.3f'%e] = membership_labels
            
        elif method == 'infomap':
            grid = len(kwargs['interlayers'])
            membership_partitions = {}
            C = np.zeros((grid*grid, self.size*self.length))
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
                    
                    C[i*grid+j,:] = [node[2] for node in np.sort(ordered_nodes, order = ['layer', 'nodeid'])]
        
                membership_partitions['interlayer=%.3f'%interlayer] = inter_membership
        
        if consensus: 
            return(self.membership(self.community_consensus_iterative(C))[0], C)
        else: 
            return(membership_partitions, C)


# In[ ]:




