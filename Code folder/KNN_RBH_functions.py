import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors as sk_KNN

#=====================================================================
#=====================================================================

# Liste of the functions defined in this file :
#   - KNN_clustering_1
#   - KNN_clustering_2
#   - RBH_filtering_1
#   - RBH_filtering_2
#   - apply_version
#   - dictionary_intersection_1
#   - dictionary_intersection_2
#   - cleanGraph
#   - saveNetwork

#=====================================================================
#=====================================================================

def KNN_clustering_1(df , n_points=None , n_neighbors=2 , algo='auto') :

    """
    Applies a classic K-Nearest Neighbors algorithm to a Pandas dataframe.

    input1 : A Pandas dataframe.
    input2 : An Integer value (default value = None).
        If customized to N, the function will only apply the chosen algorithm to the N first rows from the dataframe.
        If not, the function will look at all rows.
    input3 : A Integer value (default value = 2).
        If set to N, the KNN algorithm will search for the N nearest neighbors for eacch gene in the dataframe.
    input4 : A string value (default value = 'auto').
        Accepted values are 'auto', 'ball_tree', 'kd_tree' and 'brute'.

    output1 : A neighborhood dictionary with genes as keys and list of genes as values.
    """
    
    if n_points==None : n_points = len(df)
    L_genes = list(df['ID_REF'][0:n_points])
    
    L_colonnes = list(df.keys())[1::]
    
    L_values = []
    for col in L_colonnes : L_values.append(list(df[col][0:n_points]))
    L_values = np.transpose(L_values)
    
    model = sk_KNN(n_neighbors=n_neighbors , algorithm=algo)
    KNN = model.fit(L_values)
    neighbors = model.kneighbors()[1]

    neighbor_dict = {}
    for i,gene in enumerate(L_genes):
        neighbor_dict[gene] = []
        for v_idx in neighbors[i] : neighbor_dict[gene].append(L_genes[v_idx])
    
    return neighbor_dict

#=====================================================================

def KNN_clustering_2(df , n_points=None , factor=1 , algo='auto') :

    """
    Applies a dynamic K-Nearest Neighbors algorithm to a Pandas dataframe.
    For each gene in the dataframe, all other genes are sorted by ascending distance.
    A distance threshold is calculated for each gene and filters out all genes with a higher distance.

    input1 : A Pandas dataframe.
    input2 : An Integer value (default value = None).
        If customized to N, the function will only apply the chosen algorithm to the N first rows from the dataframe.
        If not, the function will look at all rows.
    input3 : A Float value (default value = 1).
        Used for the dynamic threshold calculation.
    input4 : A string value (default value = 'auto').
        Accepted values are 'auto', 'ball_tree', 'kd_tree' and 'brute'.

    output1 : A neighborhood dictionary with genes as keys and list of tuples as values.
        Each tuple contains a gene and it's distance to the key gene.
    """
    
    if n_points==None : n_points = len(df)
    L_genes = list(df['ID_REF'][0:n_points])
    
    L_colonnes = list(df.keys())[1::]
    
    L_values = []
    for col in L_colonnes : L_values.append(list(df[col][0:n_points]))
    L_values = np.transpose(L_values)
    
    model = sk_KNN(n_neighbors=n_points-1 , algorithm=algo)
    KNN = model.fit(L_values)
    all_distances , all_neighbors = model.kneighbors()
    
    Distances , Neighbors = [] , []
    for i,L_dist in enumerate(all_distances):
        L_dist = list(L_dist)
        L_neigh = list(all_neighbors[i])
#        print(i,L_genes[i])
#        print(len(L_neigh),L_neigh)
        
        mean = np.mean(L_dist)
        std = np.std(L_dist)
        threshold = mean-(factor*std)
#        print(threshold)
        
        L_del = []
        for j,d in enumerate(L_dist):
            if d > threshold : L_del.append(j)
        for idx in sorted(L_del , reverse=True) : 
            dump = L_dist.pop(idx)
            dump = L_neigh.pop(idx)
#        print(len(L_neigh),L_neigh)
#        for j,n in enumerate(L_neigh): print(n,L_genes[n],L_dist[j])
        Distances.append(L_dist)
        Neighbors.append(L_neigh)
#        print("-----------------")

    neighbor_dict = {}
    for i,gene in enumerate(L_genes):
        neighbor_dict[gene] = []
        L_neig = Neighbors[i]
        L_dist = Distances[i]
        for j,v_idx in enumerate(L_neig) :
            data = (L_genes[v_idx] , L_dist[j])
            neighbor_dict[gene].append(data)
    
    return neighbor_dict

#=====================================================================

def RBH_filtering_1(neighbor_dict):

    """
    Generates a Reciprocal Best Hit neighborhood dictionary from a simple neighborhood dictionary.
    Both dictionary consist of genes as keys and list of genes as values.

    input1 : A simple neighborhood dictionary.
        Each gene is associated to a list of other genes that it considers it's neighbors.

    output1 : A Reciprocal Best Hit neighborhood dictionary.
        Each gene is associated to a list of other genes who themselves consider the key gene their neighbor.
    """
    
    RBH_dict = {}
    for n1,voisins in neighbor_dict.items():
        RBH_dict[n1] = []
        for n2 in voisins :
            if n1 in neighbor_dict[n2] : RBH_dict[n1].append(n2)

    return RBH_dict
    

#=====================================================================

def RBH_filtering_2(neighbor_dict):

    """
    Generates a Reciprocal Best Hit neighborhood dictionary from a simple neighborhood dictionary.
    Both dictionary consist of genes as keys and list of tuples as values.
    Each tuple contains a gene and it's distance to the key gene.

    input1 : A simple neighborhood dictionary.
        Each gene is associated to a list of other genes that it considers it's neighbors.

    output1 : A Reciprocal Best Hit neighborhood dictionary.
        Each gene is associated to a list of other genes who themselves consider the key gene their neighbor.
    """

    RBH_dict = {}
    for n1,voisins in neighbor_dict.items():
        RBH_dict[n1] = []
        for (n2,dist) in voisins :
            for couple in neighbor_dict[n2] :
                if couple[0] == n1 :
#                    print(n1,n2,dist,couple)
                    RBH_dict[n1].append((n2,dist))
                    break

    return RBH_dict
    
#=====================================================================

def apply_version(df , n_points=None , method='KNN_1' , n_neighbors=2 , factor=1 , algo='auto' , RBH=False):

    """
    Selects the KNN algorithm version and the Reciprocal Best Hit filtering as required.

    input1 : A Pandas dataframe.
    input2 : An Integer value (default value = None).
        If customized to N, the function will only apply the chosen algorithm to the N first rows from the dataframe.
        If not, the function will look at all rows.
    input3 : A string value (default value = KNN_1).
        Accepted values are 'KNN_1' or 'KNN_2'.
        Used to select a version of the KNN algorithm.
    input4 : An Integer value (default value = 2).
        Used only if input3 is set to 'KNN_1'.
    input5 : A Float value (default value = 1).
        Used only if input3 is set to 'KNN_2'.
    input6 : A string value (default value = 'auto').
        Accepted values are 'auto', 'ball_tree', 'kd_tree' and 'brute'.
    input7 : A Boolean-like value (default value = False).
        If set to True, the function will call the RBH filtering adapted to the input3 value.

    output1 : A simple neighborhood dictionary.
        Returned only if input7 is set to False.
    output2 : A Reciprocal Best Hit neighborhood dictionary.
        Returned only if input7 is set to True.
    """

    if method == 'KNN_1':
        neighbor_dict = KNN_clustering_1(df , n_points=n_points , n_neighbors=n_neighbors , algo='auto')
        if RBH : RBH_dict = RBH_filtering_1(neighbor_dict)
        
    elif method == 'KNN_2':
        neighbor_dict = KNN_clustering_2(df , n_points=n_points , factor=factor , algo='auto')
        if RBH : RBH_dict = RBH_filtering_2(neighbor_dict)


    if not RBH : return neighbor_dict
    else : return RBH_dict

#=====================================================================

def dictionary_intersection_1(dict_list):

    """
    Intersects the contents of several simple neighborhood dictionaries.
    All provided dictionaries consist of genes as keys and list of genes as values.

    input1 : A list of simple neighborhood dictionaries.
        In all dictionaries, each gene is associated to a list of other genes that it considers it's neighbors.

    output1: An intersected simple neighborhood dictionary.
        Each gene is associated to a list of other genes that it considers it's neighbors in all provided dictionaries.
    """

    Inter_dict = {}
    
    for n1,v_1 in dict_list[0].items():
        Inter_dict[n1] = v_1 # Get the neighbors from the first dictionary
        for neighbor_dict in dict_list[1::]: # For each subsequent dictionary
            v = neighbor_dict[n1] # Get the neighbors
            Inter_dict[n1] = list(set(Inter_dict[n1]) & set(v)) # Keep only the common neighbors
        if len(Inter_dict[n1]) != 0 : # If some neighbors are common to all datasets, we keep them
#            print(f"{n1} : {len(Inter_dict[n1])}")
            continue
        else : del(Inter_dict[n1]) # If there is no common neighbors, we delete the current gene key

    return Inter_dict
    
#=====================================================================

def dictionary_intersection_2(dict_list):

    """
    Intersects the contents of several Reciprocal Best Hit neighborhood dictionaries.
    All provided dictionaries consist of genes as keys and list of tuples as values.
    Each tuple contains a gene and it's distance to the key gene.

    input1 : A list of Reciprocal Best Hit neighborhood dictionaries.
        In all dictionaries, each gene is associated to a list of other genes that it considers it's neighbors.

    output1: An intersected simple neighborhood dictionary.
        Each gene is associated to a list of other genes that it considers it's neighbors in all provided dictionaries.
    """

    Inter_dict = {}
    
    for n1,v_1 in dict_list[0].items():  
        Inter_dict[n1] = []
        for (n2,dist) in v_1 : Inter_dict[n1].append(n2) # Get ONLY the neighbors from the first dictionary
        for neighbor_dict in dict_list[1::]: # For each subsequent dictionary
            v , neig = neighbor_dict[n1] , []
            for (n2,dist) in v : neig.append(n2) # Get ONLY the neighbors
            Inter_dict[n1] = list(set(Inter_dict[n1]) & set(neig)) # Keep only the common neighbors
        
        if len(Inter_dict[n1]) != 0 : # If some neighbors are common to all datasets
            for i,n2 in enumerate(Inter_dict[n1]): # For each kept neighbor, we calculate it's average distance
                avg_dist = 0
                for j,neighbor_dict in enumerate(dict_list) :
                    for (n2_temp,dist) in neighbor_dict[n1] :
                        if n2_temp == n2 :
                            to_rem = (n2_temp,dist)
                            avg_dist += dist
                            break
                    neighbor_dict[n1].remove(to_rem) # Removing the neighbor to speed up the next search. The removed neighbor isn't used anymore past this point
                avg_dist = avg_dist/len(dict_list)
                Inter_dict[n1][i] = (n2,avg_dist)
        
        else : del(Inter_dict[n1]) # If there is no common neighbors, we delete the current gene key
        

    return Inter_dict

#=====================================================================

def cleanGraph(graph , anchor_lists=[]):

    """
    Removes from a graph any non-special node that doesn't have any neighbor of interest (i.e. anchor neighbors).

    input1 : A Networkx graph.
    input2 : A list of lists of special nodes (anchors).

    output1 : The filtered network.
    """

    L_delete = []
    for node in graph.nodes :
        cand = True # We assume the current node is initialy not a special node.

        # We look for the current node in all anchor lists
        for i,L_anchors in enumerate(anchor_lists):
            if node in L_anchors :
                cand = False
                break
        # If the current node is indeed not a special node, we search if it has any anchor neighbor.
        if cand :
            out = True # We assume the current node has initialy no special neighbors.
            
            L_voisins = graph.neighbors(node)
            for voisin in L_voisins : 
                for L_anchors in anchor_lists :
                    if voisin in L_anchors :
                        out = False
                        break
                if out == False : break # We stop searching when the first special neighbor is found.
                
            if out : L_delete.append(node)
        
    for node in L_delete : graph.remove_node(node)

    return graph

#=====================================================================

def saveNetwork(graph , method='KNN_1' , anchor_lists=[] , color_list=[] , file_name="KNN-RBH_Network.txt"):

    """
    Writes a text file with the content of a network.

    input1 : A Networkx graph.
    input2 : A string value (default value = 'KNN_1').
        Accepted values are 'KNN_1' or 'KNN_2'.
        If set to 'KNN_1', the edges are represented only by their respective two connected nodes.
        If set to 'KNN_2', the edges' nodes are followed by the edges' attributes.
    input3 : A list of lists of special nodes (anchors).
    input4 : A list of colors in rgb format (i.e. tuples of 3 float values, each between 0 and 1).
    input5 : A name for the resulting text file.

    output : None.
    """

    L_nodes = list(graph.nodes)
    L_edges = list(graph.edges)

    with open(file_name,'w') as file :
        for i,node in enumerate(L_nodes) :
            cand = True
            for j,L_anchors in enumerate(anchor_lists):
                if node in L_anchors :
                    file.write(f">{node}:{color_list[j]}\n")
                    cand = False
                    break
            if cand : file.write(f">{node}:(0, 0, 0)\n")
        for i,(n1,n2) in enumerate(L_edges) :
            file.write(f"{n1} ; {n2}")
            if method == 'KNN_2' :
                L_attr = list(graph[n1][n2].keys())
                for attr in L_attr : file.write(f" ; {attr}:{graph[n1][n2][attr]}")
            file.write('\n')

#=====================================================================
