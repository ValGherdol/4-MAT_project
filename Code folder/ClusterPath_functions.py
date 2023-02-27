from sklearn.cluster import KMeans as sk_KMeans
import pandas as pd
import time

#=====================================================================
#=====================================================================

# Liste of the functions defined in this file :
#   - time_count
#   - KMeans_1D
#   - clusterPathing
#   - commonPath
#   - regroupPaths
#   - concatenatePath
#   - savePaths
#   - deviationPathing
#   - saveDeviantPaths
#   - saveNetwork

#=====================================================================
#=====================================================================

def time_count(secondes) :
    """
    Receives a duration in seconds and gives back the equivalent in hours, minutes and seconds.
    
    input1 : A numeral value (in seconds)
    
    output1 : A string with hours, minutes and seconds (hours and minutes only if there is at least one of them)
    """
    heures,minutes = 0,0
    temps = f""
    if secondes > 3600 :
        heures = int(secondes // 3600)
        secondes -= heures * 3600
        temps += f"{heures}h"
    if secondes > 60 :
        minutes = int(secondes // 60)
        secondes -= minutes * 60
        temps += f"{minutes}m"
    temps += f"{round(secondes,2)}s"
    return temps

#=====================================================================

def KMeans_1D(df , col_num , n_points=None , n_clusters=3 , random_state=0):
    """
    Applies a KMeans algorithm on a single given column of a dataframe and return the result.
    The clusters are sorted by ascending order of their centers' values.

    input1 : A Pandas dataframe who's first column must be named 'ID_REF'.
    input2 : An Integer value indicating which column will be analyzed.
        The function follows an index-based logic starting from the first non-'ID_REF' column.
        In other words, the dataframe's second column (who's original index is 1 in the dataframe) is called by a value of 0 since it's the first non-'ID_REF' column.
    input3 : An Integer value (default value = None).
        If customized to N, the function will only apply the algorithm to the N first values from the studied column.
        If not, the function will look at all values.
    input4 : An Integer value (default value = 3).
    input5 : An Integer value (default value = 0).
    
    output1 : A dictionary where the keys are the genes' names and the values are the id number of the cluster the gene has been put in.
    """
    
    if n_points==None : n_points = len(df)
    
    L_names = list(df['ID_REF'])
    L_colonnes = list(df.keys())[1::]
    
    L_values = list(df[L_colonnes[col_num]][0:n_points])
    for i,value in enumerate(L_values) : L_values[i] = [value]
    
    # Cluster calculation
    model = sk_KMeans(n_clusters=n_clusters , random_state=random_state)
    KM = model.fit(L_values)
    labels = KM.labels_
    centers = KM.cluster_centers_
    
    # Cluster sorting based on ascending values of centroids
    L_centers = []
    for c in centers : L_centers.append(c[0])
    L_sorted_centers = sorted(L_centers)
    L_ordered_centers = []
    for c in L_sorted_centers :
        idx = L_centers.index(c)
        L_ordered_centers.append((idx,c))
    
    # Gene indices sorting by common cluster
    dico_clusters = {}
    for i in range(n_clusters) : dico_clusters[i] = 0
    for i in list(dico_clusters.keys()) :
        (idx,center) = L_ordered_centers[i]
        dico_clusters[i] = [center,[],[]]
        for j,lab in enumerate(labels) :
            if lab == idx : 
                dico_clusters[i][1].append(L_names[j])
                dico_clusters[i][2].append(L_values[j][0])
#        for k,v in dico_clusters.items(): print(k,type(v))
#    print("---------------")
    
    # Associate each gene name to the gene's cluster
    dico_points = {}
    for name in L_names :
        for k,v in dico_clusters.items():
            if name in v[1] : 
                dico_points[name] = k
                break
    return dico_points

#=====================================================================

def clusterPathing(dict_list):

    """
    Generates for each gene a cluster path in the form of an Integer vector.

    input1 : A list of dictionaries with gene names as keys and Integers as values.

    output1 : A dictionary with gene names as keys and vectors of Integers as values.
    """

    D_pathPerGene = {}
    for dico in dict_list :
        for gene,cluster_num in dico.items():
            if gene not in D_pathPerGene.keys() : D_pathPerGene[gene] = [cluster_num]
            else : D_pathPerGene[gene].append(cluster_num)

    return D_pathPerGene

#=====================================================================

def commonPath(D_pathPerGene):

    """
    Regroups all genes following a same cluster path.

    input1 : A dictionary with gene names as keys and vectors of Integers as values.

    output1 : A dictionary with tuples of Integers as keys and lists of gene names as values.
    """

    D_genePerPath = {}
    for gene,path in D_pathPerGene.items():
        if tuple(path) not in D_genePerPath.keys(): D_genePerPath[tuple(path)] = [gene]
        else : D_genePerPath[tuple(path)].append(gene)

    return D_genePerPath
#=====================================================================

def regroupPaths(LD_genePerPath , globalGenePool):

    """
    Associates to each gene a list of cluster path. Each path comes from a different dataset.

    input1 : A list of dictionaries with tuples of Integers as keys and lists of gene names as values.
    input2 : A list of gene names.

    output1 : A dictionary with gene names as keys and lists of tuples of Integers as values.
    """

    D_allPathsPerGene = {}
    for gene in globalGenePool : # For each gene
        D_allPathsPerGene[gene] = []
        for dico in LD_genePerPath : # For each dictionary of "path - gene list"
            for path,gene_list in dico.items():
                if gene in gene_list :
                    D_allPathsPerGene[gene].append(path) # Recovering the path followed by the current gene
                    break

    return D_allPathsPerGene

#=====================================================================

def concatenatePath(D_allPathsPerGene):

    """
    Regroups all genes following the same groups of cluster paths.

    input1 : A dictionary with gene names as keys and lists of tuples of Integers as values.

    ouptu1 : A dictionary with tuples of tuples of Integers as keys and lists of gene names as values.
    """

    D_genePerConcatenatedPath = {}
    for gene,L_paths in D_allPathsPerGene.items():
        if len(L_paths) == 1 : concat_Path = L_paths[0]
        else : concat_Path = tuple(L_paths)
        if concat_Path not in D_genePerConcatenatedPath.keys(): D_genePerConcatenatedPath[concat_Path] = [gene]
        else : D_genePerConcatenatedPath[concat_Path].append(gene)

    return D_genePerConcatenatedPath

#=====================================================================

def savePaths(D_genePerConcatenatedPath , anchor_lists=[] , label_list=[] , file_name="ListOfPaths.txt"):

    """
    Writes a list of concatenated paths and their respective set of genes in a text file.
    For each concatenated path is indicated how many genes follow it and how many of those genes are anchor genes from each type.

    input1 : A dictionary with tuples of tuples of Integers as keys and lists of gene names as values.
    input2 : A list of anchor genes.
    input3 : A list of anchor labels.
    input4 : A name for the text file.

    output1 : None.
    """

    with open(file_name,'w') as file :
        for path in sorted(D_genePerConcatenatedPath, key=lambda path: len(D_genePerConcatenatedPath[path]), reverse=True):
            L_gene = D_genePerConcatenatedPath[path]
            line = f">Path:{path} ; {len(L_gene)} total genes"
            L_count = [0 for i in range(len(anchor_lists))]
            for gene in L_gene :
                for i,L_anchors in enumerate(anchor_lists):
                    if gene in L_anchors : L_count[i] += 1
            for i,c in enumerate(L_count) : line += f" ; {c} {label_list[i]} genes"
            line += '\n'  
            file.write(line)
            file.write(' '.join(L_gene)+'\n')

#=====================================================================

def deviationPathing(D_allPathsPerGene , anchor_list=None , Max_deviation=0 , path_lengths=[] , search_tps=False):

    """
    Regroups couples of genes that follow paths with an allowed number of differences.

    input1 : A dictionary with gene names as keys and lists of tuples of Integers as values.
    input2 : An Integer or list of Integers.
    input3 : A list of Integers.

    ouptu1 : A dictionary with the following content :
        - The keys are tuples containing :
            - Either tuple of Integers or a tuple of several tuples of Integers
            - A Float value.
        - The values are couples of gene names.
    """

    start = time.time()

    if anchor_list == None : Nb_gene_to_process = len(list(D_allPathsPerGene.keys()))
    else : Nb_gene_to_process = len(anchor_list)

    D_genePerDeviatedPath = {}
    L_processed_anchors = []
    for i,gene_1 in enumerate(list(D_allPathsPerGene.keys())): # For each gene
        if anchor_list == None : L_other_gene = list(D_allPathsPerGene.keys())[i+1::]
        elif gene_1 in anchor_list :
            L_other_gene = list(D_allPathsPerGene.keys())
            L_processed_anchors.append(gene_1)
        else : continue
        
        t_node = time.time()
        L_path_1 = D_allPathsPerGene[gene_1]
        for gene_2 in L_other_gene : # For each othe gene
            if gene_2 in L_processed_anchors : continue
            
            L_path_2 = D_allPathsPerGene[gene_2]
            # Case 1 : There is no sub-path (only one dataset provided)
            if len(L_path_1) == 1 : 
                d = 0 # deviation counter
                concat_path = []
                for j,p1 in enumerate(L_path_1[0]):
                    p2 = L_path_2[0][j]
                    if p1 == p2 : concat_path.append(str(p1))
                    else : # Cluster IDs are arranged by ascending order
                        d += 1
                        if p1 < p2 : concat_path.append(f"{p1}/{p2}")
                        else : concat_path.append(f"{p2}/{p1}")
                        
                    
                if d <= Max_deviation : # If the max number of allowed deviations is not exceeded
                    concat_path = tuple(concat_path)
                    match_ratio = (len(L_path_1[0])-d)/len(L_path_1[0])
                    key = (concat_path,match_ratio)
                    if key not in D_genePerDeviatedPath.keys(): D_genePerDeviatedPath[key] = [(gene_1,gene_2)]
                    else : D_genePerDeviatedPath[key].append((gene_1,gene_2))
                    
            # Case 2 : The path is composed of sub-paths (several datasets provided)
            else :
                n_tp = sum(path_lengths) # The total number of timepoints across all datasets
                # Case 2a : A max number of allowed deviations has been provided for each dataset
                if type(Max_deviation) == list :
                    d = [0 for sub_path in L_path_1] # deviation counter (one for each dataset)
                    concat_path = []
                    out = False
                    for j,subpath_1 in enumerate(L_path_1) :
                        subpath_2 = L_path_2[j]
                        sub_concat = []
                        for k,p1 in enumerate(subpath_1):
                            p2 = subpath_2[k]
                            if p1 == p2 : sub_concat.append(str(p1))
                            else : # Cluster IDs are arranged by ascending order
                                d[j] += 1
                                if p1 < p2 : sub_concat.append(f"{p1}/{p2}")
                                else : sub_concat.append(f"{p2}/{p1}")
                        
                        if d[j] <= Max_deviation[j] : concat_path.append(tuple(sub_concat))
                        else : # If the max number of allowed deviations is exceeded for one dataset, the current couple is droped.
                            out = True
                            break
                    if out : continue
                    concat_path = tuple(concat_path)
                    match_ratio = (n_tp-sum(d))/n_tp
                    key = (concat_path,match_ratio)
                    if key not in D_genePerDeviatedPath.keys(): D_genePerDeviatedPath[key] = [(gene_1,gene_2)]
                    else : D_genePerDeviatedPath[key].append((gene_1,gene_2))
                
                # Case 2b : A single global max number of allowed deviations has been provided for all datasets
                else :
                    d = 0 # deviation counter
                    concat_path = []
                    out = False
                    for j,subpath_1 in enumerate(L_path_1) :
                        subpath_2 = L_path_2[j]
                        sub_concat = []
                        for k,p1 in enumerate(subpath_1):
                            p2 = subpath_2[k]
                            if p1 == p2 : sub_concat.append(str(p1))
                            else : # Cluster IDs are arranged by ascending order
                                d += 1
                                if p1 < p2 : sub_concat.append(f"{p1}/{p2}")
                                else : sub_concat.append(f"{p2}/{p1}")
                                
                        if d <= Max_deviation : concat_path.append(tuple(sub_concat))
                        else : # If the max number of allowed deviations is exceeded for one dataset, the current couple is droped.
                            out = True
                            break
                    if out : continue
                    concat_path = tuple(concat_path)
                    match_ratio = (n_tp-d)/n_tp
                    key = (concat_path,match_ratio)
                    if key not in D_genePerDeviatedPath.keys(): D_genePerDeviatedPath[key] = [(gene_1,gene_2)]
                    else : D_genePerDeviatedPath[key].append((gene_1,gene_2))
        if search_tps :
            if anchor_list == None :
                if (i+1)%1000 == 0 : print(f"Gene {gene_1} processed in {time_count(time.time()-t_node)} ({i+1}). Time since launch : {time_count(time.time()-start)}")
                else : print(f"Gene {gene_1} processed in {time_count(time.time()-t_node)} ({i+1}/{Nb_gene_to_process})")
            else :
                L = len(L_processed_anchors)
                if L%1000 == 0 : print(f"Gene {gene_1} processed in {time_count(time.time()-t_node)} ({L}). Time since launch : {time_count(time.time()-start)}")
                else : print(f"Gene {gene_1} processed in {time_count(time.time()-t_node)} ({L}/{Nb_gene_to_process})")
    return D_genePerDeviatedPath

#=====================================================================

def saveDeviantPaths(D_genePerDeviatedPath , file_name="ListOfDeviantPaths.txt"):
    """
    Writes a list of concatenated paths and their respective set of genes in a text file.
    For each path is indicated it's match ratio (i.e. the number of non-deviant timepoints) and how many couples of genes follow it.

    input1 : A dictionary with the following content :
        - The keys are tuples containing :
            - Either tuple of Integers or a tuple of several tuples of Integers
            - A string following a '[x]/[y]' template where :
                - [x] is the number of identical timepoints for two genes following the path
                - [y] in the total number of timepoints in the path
        - The values are couples of gene names.
    input2 : A name for the text file.

    output1 : None.
    """

    with open(file_name,'w') as file :
        for key in sorted(D_genePerDeviatedPath, key=lambda key: len(D_genePerDeviatedPath[key]), reverse=True):
            (path,match_ratio) = key
            L_couples = D_genePerDeviatedPath[(path,match_ratio)]
            file.write(f">Path:{path} ; Ratio:{match_ratio} ; {len(L_couples)} total couples\n")
            line = []
            for couple in L_couples : line.append(str(couple))
            file.write(' '.join(line)+'\n')

#=====================================================================

def saveNetwork(graph , anchor_lists=[] , color_list=[] , file_name="ClusterPath_Network.txt"):

    """
    Writes a text file with the content of a network.

    input1 : A Networkx graph.
    input2 : A list of lists of special nodes (anchors).
    input3 : A list of colors in rgb format (i.e. tuples of 3 float values, each between 0 and 1).
    input4 : A name for the resulting text file.

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
            L_attr = list(graph[n1][n2].keys())
            for attr in L_attr : file.write(f" ; {attr}:{graph[n1][n2][attr]}")
            file.write('\n')
    
#=====================================================================
