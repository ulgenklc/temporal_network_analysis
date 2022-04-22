#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import random
from scipy.ndimage import gaussian_filter1d
from math import floor

from elephant.spike_train_generation import homogeneous_poisson_process
import elephant.conversion as conv
import neo as n
import quantities as pq
from quantities import Hz, s, ms

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score

import leidenalg as la
import igraph as ig

# In[ ]:


def normalized_cross_corr(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    x_cov_std = np.nanmax(np.sqrt(np.correlate(x - x_mean, x - x_mean, 'full')))
    y_cov_std = np.nanmax(np.sqrt(np.correlate(y - y_mean, y - y_mean, 'full')))

    normalization = x_cov_std * y_cov_std
        

    unnormalized_correlation = np.correlate(x - x_mean, y - y_mean, 'full')
    
    corr_array = unnormalized_correlation/normalization

    return(corr_array)

def max_norm_cross_corr(x1, x2):
    
    correlation= normalized_cross_corr(x1, x2)
    
    lag = abs(correlation).argmax() - len(x1)+1
    
    max_corr = max(abs(correlation))
    
    return(max_corr, lag)

def cross_correlation_matrix(data):
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


# In[ ]:


def bin_time_series(array, binsize, gaussian = True, **kwargs):
        #input: nxt matrix 
        #returns: binned time series i.e. l x n x binsize
        
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
    n,t = array.shape
    thresholded_array = np.copy(array)
    for i in range(n):
        for j in range(t):
            if array[i][j] < thresh: thresholded_array[i][j] = 0
            else: pass
    return(thresholded_array)

def gaussian_filter(array,sigma):
    #sigma=0.25==gaussian kernel with length 3
    #sigma=0.5==gaussian kernel with length 5
    #sigma=1==gaussian kernel with length 9
    return(gaussian_filter1d(array,sigma))

def jitter(spike, k):
    #jittering the given spike train
    jittered = np.zeros(spike.shape)
    for i in np.nonzero(spike)[1]:
        jitt = random.randint(-k,k)
        try:jittered[0,i+jitt] = 1
        except:jittered[0,i] = 1
    return(jittered)

def spike_count(spikes, ax, num_bins = None, t_min = None, t_max = None):
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
    repeated = find_repeated(l)
    k = []
    for r in repeated:
        first = l.index(r)
        reversed_l = l[::-1]
        last = len(l) - reversed_l.index(r)
        k.append((first,last))
    return(k)

def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


# In[ ]:


def generate_ground_truth(comm_sizes, method = 'scattered', pad = False, community_operation = 'grow'):
    ##genertaes community labels according to two methods one in which the rest of the network except the planted communities
    # are scattered i.e. they all have their own community or they are all in one community, integrated.
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
        return(truth_labels)
                
    elif community_operation == 'simple_grow':
        layers = len(comm_sizes)
        truth_labels = []
        for l in range(layers):
            for i,c in enumerate(comm_sizes):
                if l == 0:
                    label = i
                elif l<i:
                    label = i
                else:
                    label = 0
                truth_labels = truth_labels + [label for j in range(c)]
            
        if pad:
            l1 = truth_labels[:sum(comm_sizes)]
            l2 = [0 for i in range(sum(comm_sizes))]
            truth_labels = l1 + truth_labels +l2
        return(truth_labels)
    
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
                
        return(truth_labels)
            
                
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
            
        return(truth_labels)
     
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
                    if O/abs(c[1]-c[0]) >= 1/2 or O/abs(p[1]-p[0])>= 1/2:
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
        
        if method == 'scattered':
            return(truth_labels)
        elif method == 'integrated':
            new_truth_labels = []
            if pad:
                layers = layers + 2
            for i in range(layers):
                for j,e in enumerate(truth_labels[i*num_neurons:(i+1)*num_neurons]):
                    if truth_labels[i*num_neurons:(i+1)*num_neurons].count(e) == 1:
                        new_truth_labels.append(0)
                    else:
                        new_truth_labels.append(e)
            return(new_truth_labels)

def information_recovery(pred_labels, comm_size, truth, interlayers, other_parameter, com_op):
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
    if community_operation == 'grow' or community_operation == 'contract':
        n = sum(comm_sizes)
        layers = len(comm_sizes)
        l = layers + 2
    
        scattered_truth = generate_ground_truth(comm_sizes, 
                                                method = 'integrated', 
                                                pad = True, 
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
                                                 pad = True, 
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
    
    elif community_operation == 'merge' or community_operation == 'transient' or community_operation == 'simple_grow':
        try:
            n = sum(comm_sizes[0])
        except: n = sum(comm_sizes)
        
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


# In[ ]:


def create_time_series(operation, community_sizes, spiking_rates, spy = True, windowsize = 1000, k = 5):
    
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
        
    elif operation == 'merge' or operation == 'transient':
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
        
    elif operation == 'simple_grow':
        num_neurons = int(sum(community_sizes))
        
        spikes = np.zeros((num_neurons,total_duration))
        comms = [[] for i in range(layers)]
        rates = [[] for i in range(layers)]
        for i,c in enumerate(community_sizes):
            for l in range(layers-i):
                if l == 0:
                    size = c
                    rate = spiking_rates[i]
                elif i == 0:
                    size = sum(community_sizes[:(l+1)])
                    rate = spiking_rates[0]
                else:
                    size = community_sizes[i+l]
                    rate = spiking_rates[i+l]
                comms[l].append(size)
                rates[l].append(rate)
        
        for s in range(layers):
            neuron_count = 0
            for i,e in enumerate(comms[s]):
                initial_master = homogeneous_poisson_process(rate = rates[s][i]*Hz,
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

def consensus_display(partition, n, t):
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


# In[ ]:





# In[ ]:




