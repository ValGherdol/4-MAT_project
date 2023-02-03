
### Imports
#=====================================================================================================================================================

import time
from collections import Counter
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from scipy.spatial import distance
import networkx as nx
import pickle
import Steps_functions as fnc
import NPC_functions_2_Val as NPC
import warnings
warnings.filterwarnings("ignore")

### Arguments loading
#=====================================================================================================================================================

# ARGUMENTS ARE TO BE GIVEN IN THE FOLLOWING ORDER :
#   - The setting file
#   - The datasets

arguments = sys.argv[1::]

D_args = {}
with open(arguments[0],'r') as file :
    for line in file :
        if line[0] in ['>','\n'] : continue
#        print(line[0:-1])
        arg = line[0:-1].split(' : ')
#        print(arg[1],type(arg))
        D_args[arg[0]] = arg[1]
#print(D_args)

### Initial data arguments ---------------------------------------------------------------
if D_args['Anchor Genes lists'] == 'None' : D_args['Anchor Genes lists'] = []
else : D_args['Anchor Genes lists'] = D_args['Anchor Genes lists'].split(',')

if D_args['Anchor Genes labels'] == 'default' :
    D_args['Anchor Genes labels'] = []
    for i in range(len(D_args['Anchor Genes lists'])) : D_args['Anchor Genes labels'].append(f"Anchor-{i+1}")
else : D_args['Anchor Genes labels'] = D_args['Anchor Genes labels'].split(',')

if D_args['Anchor colors'] == 'default' : D_args['Anchor colors'] = [(0,0,0) for i in range(len(D_args['Anchor Genes lists']))]
else :
    D_args['Anchor colors'] = D_args['Anchor colors'].split('/')
    for i,rgb in enumerate(D_args['Anchor colors']):
        D_args['Anchor colors'][i] = rgb.split(',')
        for j,c in enumerate(D_args['Anchor colors'][i]) : D_args['Anchor colors'][i][j] = float(c)
        D_args['Anchor colors'][i] = tuple(D_args['Anchor colors'][i])

D_args['Steps list'] = D_args['Steps list'].split(',')
for i,a in enumerate(D_args['Steps list']): D_args['Steps list'][i] = int(a)

D_args['Consensus step'] = bool(int(D_args['Consensus step']))

if D_args['Linkage dictionary'] == 'default' : D_assoc = {}
else :
    with open(D_args['Linkage dictionary'],'rb') as file : D_assoc = pickle.load(file)

### Step 1 arguments ---------------------------------------------------------------------
D_args['Anchor centered'] = bool(int(D_args['Anchor centered']))

if D_args['Pearson principal threshold'] == 'default' : D_args['Pearson threshold'] = 0.5
elif ',' in D_args['Pearson principal threshold'] :
    D_args['Pearson principal threshold'] = D_args['Pearson principal threshold'].split(',')
    for i,a in enumerate(D_args['Pearson principal threshold']): D_args['Pearson principal threshold'][i] = float(a)
else : D_args['Pearson principal threshold'] = float(D_args['Pearson principal threshold'])

D_args['Pearson dynamic threshold'] = bool(int(D_args['Pearson dynamic threshold']))

if D_args['Dynamic threshold factor'] == 'default' : D_args['Dynamic threshold factor'] = 1
else : D_args['Dynamic threshold factor'] = float(D_args['Dynamic threshold factor'])

if D_args['Neighborhood threshold'] == 'default' : D_args['Neighborhood threshold'] = None
else : D_args['Neighborhood threshold'] = int(D_args['Neighborhood threshold'])

if D_args['Minimum neighborhood P-CEIN'] != 'default' : 
    D_args['Minimum neighborhood P-CEIN'] = D_args['Minimum neighborhood P-CEIN'].split(',')
    for i,a in enumerate(D_args['Minimum neighborhood P-CEIN']): D_args['Minimum neighborhood P-CEIN'][i] = int(a)

D_args['Research time announcement'] = bool(int(D_args['Research time announcement']))

D_args['Global time announcement'] = bool(int(D_args['Global time announcement']))

D_args['New Network'] = bool(int(D_args['New Network']))

### Step 2 arguments ---------------------------------------------------------------------
D_args['RBH sub-step'] = bool(int(D_args['RBH sub-step']))

if D_args['Number of neighbors'] == 'default' : D_args['Number of neighbors'] = 2
else : D_args['Number of neighbors'] = int(D_args['Number of neighbors'])

if D_args['Threshold factor KNN_2'] == 'default' : D_args['Threshold factor KNN_2'] = 1
elif ',' in D_args['Threshold factor KNN_2'] :
    D_args['Threshold factor KNN_2'] = D_args['Threshold factor KNN_2'].split(',')
    for i,a in enumerate(D_args['Threshold factor KNN_2']): D_args['Threshold factor KNN_2'][i] = float(a)
else : D_args['Threshold factor KNN_2'] = float(D_args['Threshold factor KNN_2'])

### Step 3 arguments ---------------------------------------------------------------------

if D_args['Threshold factor NPC'] == 'default' : D_args['Threshold factor NPC'] = 1
elif ',' in D_args['Threshold factor NPC'] :
    D_args['Threshold factor NPC'] = D_args['Threshold factor NPC'].split(',')
    for i,a in enumerate(D_args['Threshold factor NPC']): D_args['Threshold factor NPC'][i] = float(a)
else : D_args['Threshold factor NPC'] = float(D_args['Threshold factor NPC'])

D_args['Minimum neighborhood NPC'] = D_args['Minimum neighborhood NPC'].split(',')
for i,a in enumerate(D_args['Minimum neighborhood NPC']): D_args['Minimum neighborhood NPC'][i] = int(a)

if D_args['Minimum redundancy'] == 'default' : D_args['Minimum redundancy'] = len(arguments[1::])
else : D_args['Minimum redundancy'] = int(D_args['Minimum redundancy'])

### Step 4 arguments ---------------------------------------------------------------------
if D_args['Number of clusters'] == 'default' : D_args['Number of clusters'] = 3
else : D_args['Number of clusters'] = int(D_args['Number of clusters'])

### Consensus arguments ------------------------------------------------------------------
if D_args['Consensus levels list'] == 'default' : D_args['Consensus levels list'] = [1,2,3,4]
else :
    D_args['Consensus levels list'] = D_args['Consensus levels list'].split(',')
    for i,a in enumerate(D_args['Consensus levels list']): D_args['Consensus levels list'][i] = int(a)

#print(D_args)

### Data loading
#=====================================================================================================================================================

# Creation of the log file. All major events will be written in it.
Log_file = "RunLog.txt"
with open(Log_file,'a') as file :file.write(">Initial Data\n")

# Global gene pool
Pool_file = D_args['Global gene pool']
L_pool = []
with open(Pool_file,'r') as file :
    for line in file : L_pool.append(line[0:-1])
print(f"Pool de gènes : {len(L_pool)}")
with open(Log_file,'a') as file : file.write(f"Global gene pool : {len(L_pool)}\n")

# Anchor genes lists (if provided)
L_L_All_anchors = [] # List of all complete anchor genes lists
L_L_anchors = [] # List of anchor gens lists whose anchors are in the global gene pool
for i,anchor_file in enumerate(D_args['Anchor Genes lists']) :
    L_L_All_anchors.append([])
    L_L_anchors.append([])
    with open(anchor_file,'r') as file :
        for line in file : L_L_All_anchors[i].append(line[0:-1])
for gene in L_pool :
    for i,L_All_anchors in enumerate(L_L_All_anchors) :
        if gene in L_All_anchors : L_L_anchors[i].append(gene)

L_Final_Anchors = [] # List of all unique anchors
for L_anchors in L_L_anchors : L_Final_Anchors += L_anchors
if L_Final_Anchors != [] : # If anchor genes are provided
    with open(Log_file,'a') as file : file.write(f"Total anchor pool : {len(L_Final_Anchors)}\n")

L_labels = D_args['Anchor Genes labels']

L_colors = D_args['Anchor colors']

for i,L_anchors in enumerate(L_L_anchors) :
    print(f"{L_labels[i]} Genes : {len(L_anchors)} / {len(L_L_All_anchors[i])}")
    with open(Log_file,'a') as file : file.write(f"{L_labels[i]} genes : {len(L_anchors)} / {len(L_L_All_anchors[i])}\n")
print("\n--------------------------- \n")

# Datasets to analyse
L_DF = []
for arg in arguments[1::] : L_DF.append(pd.read_csv(arg))
with open(Log_file,'a') as file :
    file.write(f"Number of datasets : {len(L_DF)}\n")
    file.write(f"-------------------------------------------------------\n")
for i,df in enumerate(L_DF) :
    if len(df) != len(L_pool) :
        print(f"ERROR : Global pool length and dataset-{i+1} length don't match. Program terminating.")
        with open(Log_file,'a') as file : file.write(f"ERROR : Global pool length and dataset-{i+1} length don't match.\n")
        sys.exit()

# Steps to run
L_etapes = D_args['Steps list']
D_etapes_names = {1:'P-CEIN',2:'KNN-RBH',3:'NPC',4:'ClusterPath'}

# Linkage dictionary initialization
consensus = D_args['Consensus step']
if D_assoc == {} :
    Result_dict_file = "Result_0_dictionary.pickle"
    if D_args['Anchor Genes lists'] != [] : # if anchors are provided
        for n in L_pool :
            for i,L_anchors in enumerate(L_L_anchors) :
                if n in L_anchors :
                    D_assoc[(n,L_labels[i])] = {}
                    break
    else : # If no anchors are provided
        for n in L_pool : D_assoc[(n,'Candidate')] = {}
else : Result_dict_file = D_args['Linkage dictionary']

start = time.time()
### 1 : Pearson based Co-Expression Intersected Network (P-CEIN)
#=====================================================================================================================================================

if 1 in L_etapes :
    etape = D_etapes_names[1]
    step_time = time.time()
    with open(Log_file,'a') as file : file.write(">Step 1 : Pearson based Co-Expression Intersected Network\n")
    print(f"Analyse {etape}")
    
    ### New graph building (or preexisting graph loading)
    create = D_args['New Network']
    
    file_name = f"Result_1a_{etape}_graph.txt" # Ici car utilisé pour le chargement si create==False
    
    if create : # Build a new network
        if D_args['Anchor centered'] : a_L = L_Final_Anchors
        else : a_L = None
        P_t = D_args['Pearson principal threshold']
        N_t = D_args['Neighborhood threshold']
        S_tps = D_args['Research time announcement']
        G_tps = D_args['Global time announcement']
        with open(Log_file,'a') as file : file.write(f"Building of a new network with a {P_t} threshold.\n")
        RICEP = fnc.ThresholdedCrossNetworkConstancy_5(dataframe_list=L_DF , gene_list=L_pool , anchor_list=a_L ,
                                                       threshold_Pearson=P_t , threshold_neighbours=N_t ,
                                                       pass_attr=[] , search_tps=S_tps , global_tps=G_tps ,
                                                       ret=True)

        # Initial new graph saving
        L_nodes = list(RICEP.nodes)
        L_edges = list(RICEP.edges)

        with open(file_name,'w') as file :
            for i,node in enumerate(L_nodes) :
                cand = True
                for j,L_anchors in enumerate(L_L_anchors):
                    if node in L_anchors :
                        file.write(f">{node}:{L_colors[j]}\n")
                        cand = False
                        break
                if cand : file.write(f">{node}:(0,0,0)\n")
            for i,(n1,n2) in enumerate(L_edges) : 
                file.write(f"{n1} ; {n2}")
                L_attr = list(RICEP[n1][n2].keys())
                for attr in L_attr : file.write(f" ; {attr}:{RICEP[n1][n2][attr]}")
                file.write('\n')
        
        with open(Log_file,'a') as file : file.write(f"Full network built and saved : {len(RICEP.nodes)} genes and {len(RICEP.edges)} associations.\n")

        # Initial new graph's general data saving
        df_data = fnc.save_PearsonGraph_data_2(RICEP , L_anchors_classes=L_labels , L_anchors_list=L_L_anchors , dataframe_file=f"Result_1a_{etape}_data.csv")
        del(df_data)
        with open(Log_file,'a') as file : file.write(f"Full network data saved.\n")

    else : # Load a preexisting network
        RICEP = fnc.load_PearsonGraph(file_name,annonce=10000)
        with open(Log_file,'a') as file : file.write(f"Network loaded.\n")

    ### Dynamic Pearson threshold calculation (if asked for)
    if D_args['Pearson dynamic threshold'] :
        print("\nFiltrage des arêtes via second palier dynamique")
        L_corr = []
        factor = D_args['Dynamic threshold factor']
        for (n1,n2) in RICEP.edges: L_corr.append(RICEP[n1][n2]['weight'])
        
        mean = np.mean(L_corr)
        std = np.std(L_corr)
        dynamic = round(mean + (factor * std),2)
        print(f"Palier dynamique de Pearson calculé à {dynamic} (valeure absolue)")
        with open(Log_file,'a') as file : file.write(f"Dynamic secondary Pearson threshold calculated : {dynamic}.\n")

        # Deletion of edges with weight lower than the dynamic threshold
        L_delete = []
        for (n1,n2) in RICEP.edges :
            if RICEP[n1][n2]['weight'] < dynamic : L_delete.append((n1,n2))
        for (n1,n2) in L_delete : RICEP.remove_edge(n1,n2)
        
        # Deletion of points with no edges
        L_delete = []
        for n1 in RICEP.nodes :
            if RICEP[n1] == {} : L_delete.append(n1)
        for n1 in L_delete : RICEP.remove_node(n1)
        
        print(f"Nombre de points restant : {len(RICEP.nodes)}\nNombre d'arêtes restantes : {len(RICEP.edges)}")
        print("----------------------------")

        # Dynamically filtered graph saving
        L_nodes = list(RICEP.nodes)
        L_edges = list(RICEP.edges)

        with open(f"Result_1b_{etape}_dynamic_pearson_graph.txt",'w') as file :
            for i,node in enumerate(L_nodes) :
                cand = True
                for j,L_anchors in enumerate(L_L_anchors):
                    if node in L_anchors :
                        file.write(f">{node}:{L_colors[j]}\n")
                        cand = False
                        break
                if cand : file.write(f">{node}:(0,0,0)\n")
            for i,(n1,n2) in enumerate(L_edges) : 
                file.write(f"{n1} ; {n2}")
                L_attr = list(RICEP[n1][n2].keys())
                for attr in L_attr : file.write(f" ; {attr}:{RICEP[n1][n2][attr]}")
                file.write('\n')
        with open(Log_file,'a') as file : file.write(f"Dynamic Pearson network saved : {len(RICEP.nodes)} genes and {len(RICEP.edges)} associations.\n")

        # Dynamically filtered graph's general data saving
        df_data = fnc.save_PearsonGraph_data_2(RICEP , L_anchors_classes=L_labels , L_anchors_list=L_L_anchors , dataframe_file=f"Result_1b_{etape}_dynamic_pearson_data.csv")
        del(df_data)
        with open(Log_file,'a') as file : file.write(f"Dynamic Pearson network data saved.\n")
    
    ### Minimum-anchor-neighborhood filtering (if asked for)
    if D_args['Minimum neighborhood P-CEIN'] != 'default' :
        print("\nFiltrage des candidats reliés à trop peu de gènes d'intérêt")
        
        L_limite = D_args['Minimum neighborhood P-CEIN']
        L_delete = []
        for n1 in RICEP.nodes :
            cand = True
            for i,L_anchors in enumerate(L_L_anchors):
                if n1 in L_anchors :
                    cand = False
                    break
            if cand :
                N_voisins = [0 for i in range(len(L_L_anchors))]
                for n2 in RICEP[n1] :
                    for i,L_anchors in enumerate(L_L_anchors):
                        if n2 in L_anchors :
                            N_voisins[i] += 1
                            break
                out = True
                for i,limite in enumerate(L_limite):
                    if N_voisins[i]>=limite :
                        out = False
                        break
                if out : L_delete.append(n1)

        for n in L_delete : RICEP.remove_node(n)
        print(f"Nombre de points restant : {len(RICEP.nodes)}\nNombre d'arêtes restantes : {len(RICEP.edges)}")
        print("----------------------------")

        # Anchor-neighborhood filtered graph saving
        L_nodes = list(RICEP.nodes)
        L_edges = list(RICEP.edges)

        with open(f"Result_1c_{etape}_relevant_neighborhood_graph.txt",'w') as file :
            for i,node in enumerate(L_nodes) :
                cand = True
                for j,L_anchors in enumerate(L_L_anchors):
                    if node in L_anchors :
                        file.write(f">{node}:{L_colors[j]}\n")
                        cand = False
                        break
                if cand : file.write(f">{node}:(0,0,0)\n")
            for i,(n1,n2) in enumerate(L_edges) : 
                file.write(f"{n1} ; {n2}")
                L_attr = list(RICEP[n1][n2].keys())
                for attr in L_attr : file.write(f" ; {attr}:{RICEP[n1][n2][attr]}")
                file.write('\n')
        with open(Log_file,'a') as file : file.write(f"Relevant neighborhood network saved : {len(RICEP.nodes)} genes and {len(RICEP.edges)} associations.\n")

        # Anchor-neighborhood filtered graph's general data saving
        df_data = fnc.save_PearsonGraph_data_2(RICEP , L_anchors_classes=L_labels , L_anchors_list=L_L_anchors , dataframe_file=f"Result_1c_{etape}_relevant_neighborhood_data.csv")
        del(df_data)
        with open(Log_file,'a') as file : file.write(f"Relevant neighborhood network data saved.\n")
    
    ### Saving of P-CEIN associations in Linkage dictionary for consensus
    ### For each anchor, we save it's neighbors and their respective correlation values
    if D_args['Anchor Genes lists'] != [] : # If anchors are provided
        for n1 in L_pool :
            for i,L_anchors in enumerate(L_L_anchors):
                if n1 in L_anchors :
                    D_assoc[(n1,L_labels[i])][etape] = []
#                    print(f"{n1} ({L_labels[i]}) : {len(RICEP[n1])}")
                    if n1 in RICEP.nodes :
                        for n2 in RICEP[n1] : D_assoc[(n1,L_labels[i])][etape].append((n2,RICEP[n1][n2]['weight']*RICEP[n1][n2]['sign']))
                        break
#                    print("----------")
    else : # If anchors are not provided (unfocused analysis)
        for n1 in L_pool :
            D_assoc[(n1,'Candidate')][etape] = []
            if n1 in RICEP.nodes :
                for n2 in RICEP[n1] : D_assoc[(n1,'Candidate')][etape].append((n2,RICEP[n1][n2]['weight']*RICEP[n1][n2]['sign']))
    del(RICEP)
    with open(Result_dict_file,'wb') as file : pickle.dump(D_assoc,file)

#    for k,v in D_assoc.items():print(k,v)

    print(f"Etape Co-Expression par Pearson terminée. Temps de l'étape : {fnc.time_count(time.time()-step_time)}")
    with open(Log_file,'a') as file : file.write(f"End of Step 1. Time of execution : {fnc.time_count(time.time()-step_time)}\n")
    print(f"Temps depuis lancement : {fnc.time_count(time.time()-start)}")
    with open(Log_file,'a') as file :
        file.write(f"Time since launch : {fnc.time_count(time.time()-start)}\n")
        file.write(f"-------------------------------------------------------\n")
    print("\n--------------------------- \n")

### 2 : K-Nearest Neighbors enhanced with Reciprocal Best Hit (KNN-RBH)
#=====================================================================================================================================================

if 2 in L_etapes :
    etape = D_etapes_names[2]
    step_time = time.time()
    with open(Log_file,'a') as file : file.write(">Step 2 : K-Nearest Neighbors enhanced with Reciprocal Best Hit\n")
    
    ### KNN-RBH neighborhood calculation for each dataset
    L_RBH = []
    version = D_args['KNN version']
    rbh = D_args['RBH sub-step']
    neigh = D_args['Number of neighbors']
    factor = D_args['Threshold factor KNN_2']
    if type(factor) != list : # If a single dynamic factor is provided
        for i,df in enumerate(L_DF) :
            print(f"Analyse {etape} pour dataset {i+1}")
            dump, RBH = fnc.Cluster_filtering_2(df , n_points=None , method=version , n_neighbors=neigh , factor=factor , RBH=rbh)
            L_RBH.append(RBH)
            print(f"Time since step's begining : {fnc.time_count(time.time()-step_time)}")
            print('-----')
    else : # If different dynamic factors are provided
        for i,df in enumerate(L_DF) :
            print(f"Analyse {etape} pour dataset {i+1}")
            dump, RBH = fnc.Cluster_filtering_2(df , n_points=None , method=version , n_neighbors=neigh , factor=factor[i] , RBH=rbh)
            L_RBH.append(RBH)
            print(f"Time since step's begining : {fnc.time_count(time.time()-step_time)}")
            print('-----')
    del(dump)
        
    with open(Log_file,'a') as file : file.write(f"K-Neighborhoods calculated.\n")

    ### KNN-RBH dictionaries intersection
    RBH_inter = {}
    if version == 'KNN_1' : # KNN version 1 : only the neighbors are known
        for n1,v_1 in L_RBH[0].items():
            RBH_inter[n1] = v_1 # Get the neighbors from the first dataset
            for rbh in L_RBH[1::]: # For each subsequent dataset
                v = rbh[n1] # Get the neighbors
                RBH_inter[n1] = list(set(RBH_inter[n1]) & set(v)) # Keep only the common neighbors
            if len(RBH_inter[n1]) != 0 : # If some neighbors are common to all datasets, we keep them
#                print(f"{n1} : {len(RBH_inter[n1])}")
                continue
            else : del(RBH_inter[n1]) # If there is no common neighbors, we delete the current gene key

    else : # KNN version 2 : each neighbor of a gene knows the distance between them
        for n1,v_1 in L_RBH[0].items():
#            print(n1)
#            print(v_1)   
            RBH_inter[n1] = []
            for (n2,dist) in v_1 : RBH_inter[n1].append(n2) # Get ONLY the neighbors from the first dataset
#            print(f"{n1} : {len(RBH_inter[n1])}")            
            for rbh in L_RBH[1::]: # For each subsequent dataset
                v , neig = rbh[n1] , []
#                print(v)
                for (n2,dist) in v : neig.append(n2) # Get ONLY the neighbors
                RBH_inter[n1] = list(set(RBH_inter[n1]) & set(neig)) # Keep only the common neighbors
#            print(f"{n1} RBH : {RBH_inter[n1]}")
            if len(RBH_inter[n1]) != 0 : # If some neighbors are common to all datasets
#                print(f"{n1} RBH : {RBH_inter[n1]}")
                for i,n2 in enumerate(RBH_inter[n1]): # For each kept neighbor, we calculate it's average distance
                    avg_dist = 0
                    for j,rbh in enumerate(L_RBH) :
                        for (n2_temp,dist) in rbh[n1] :
                            if n2_temp == n2 :
#                                print(n2,dist)
                                sup = (n2_temp,dist)
                                avg_dist += dist
                                break
                        rbh[n1].remove(sup) # Removing the neighbor to speed up the next search. The removed neighbor isn't used anymore past this point
                    avg_dist = avg_dist/len(L_RBH)
                    RBH_inter[n1][i] = (n2,avg_dist)
#                print(f"{n1} RBH : {RBH_inter[n1]}")
            else : del(RBH_inter[n1]) # If there is no common neighbors, we delete the current gene key
#            print("--------")
#    print(len(RBH_inter))
    with open(Log_file,'a') as file : file.write(f"K-Neighborhoods intersected.\n")

    ### Initial network building
    if version == 'KNN_1' : # KNN version 1 : only the neighbors are known
        I = nx.Graph()
        for n1,voisins in RBH_inter.items():
            for n2 in voisins :
                if not I.has_edge(n1,n2) : I.add_edge(n1,n2)
        print(f"Nombre initial de points : {len(I.nodes)}")
        print(f"Nombre initial d'arêtes : {len(I.edges)}")
        print("---------------")

    else : # KNN version 2 : each neighbor of a gene knows the distance between them
        I = nx.Graph()
        for n1,voisins in RBH_inter.items():
            for (n2,dist) in voisins :
                if not I.has_edge(n1,n2) : I.add_edge(n1,n2,weight=dist)
        print(f"Nombre initial de points : {len(I.nodes)}")
        print(f"Nombre initial d'arêtes : {len(I.edges)}")
        print("---------------")

    with open(Log_file,'a') as file : file.write(f"Network built : {len(I.nodes)} genes and {len(I.edges)} associations.\n")
        

    ### Deletion of candidate genes with no edges to anchor genes
    ### WARNING : Only if anchor genes are provided !!!
    if D_args['Anchor Genes lists'] != []:
        L_del = []
        for node in I.nodes :
            ancre = False
            for L_anchors in L_L_anchors :
                if node in L_anchors :
                    ancre = True
                    break
            if not ancre :
                L_voisins = I.neighbors(node)
                candidat = False
                for voisin in L_voisins :
                    for L_anchors in L_L_anchors :
                        if voisin in L_anchors : 
                            candidat = True
                            break
                    if candidat : break
                if not candidat : L_del.append(node)
        print(f"Nombre de non candidats : {len(L_del)}")
        
        for node in L_del : I.remove_node(node)
        with open(Log_file,'a') as file : file.write(f"Network filtered : {len(I.nodes)} genes and {len(I.edges)} associations.\n")

    CN_map = []
    for node in I.nodes :
        ancre = False
        for i,L_anchors in enumerate(L_L_anchors) :
            if node in L_anchors :
                CN_map.append(L_colors[i])
                ancre = True
        if not ancre : CN_map.append((0,0,0))

    graph_file = f"Result_2_{etape}_Graph.txt"
    with open(graph_file,'w') as file :
        for i,node in enumerate(list(I.nodes)) : file.write(f">{node}:{CN_map[i]}\n")
        if version == 'KNN_1' : # KNN version 1 : only the neighbors are known
            for i,(n1,n2) in enumerate(list(I.edges)) : file.write(f"{n1} ; {n2}\n")
        else : # KNN version 2 : each neighbor of a gene knows the distance between them
            for i,(n1,n2) in enumerate(list(I.edges)) : file.write(f"{n1} ; {n2} ; weight:{I[n1][n2]['weight']}\n")
    with open(Log_file,'a') as file : file.write(f"Network saved.\n")

    if D_args['Anchor Genes lists'] != [] :
        print(f"Nombre filtré de points : {len(I.nodes)}")
        print(f"Nombre filtré d'arêtes : {len(I.edges)}")
#    fnc.drawNetwork(I , n_color=CN_map , e_color=(0.75,0.75,0.75) , n_size=50 , mode='spring')

    ### Saving of KNN-RBH associations in Linkage dictionary for consensus
    ### For each anchor, we save it's neighbors
    if D_args['Anchor Genes lists'] != [] : # If anchors are provided
        for gene,assoc in D_assoc.items():
            D_assoc[gene][etape] = []
            n1 = gene[0]
#            print(n1,assoc)
            if n1 in I.nodes : 
                if version == 'KNN_1' : D_assoc[gene][etape] = list(I[n1])
                else :
                    for n2 in list(I[n1]) : D_assoc[gene][etape].append((n2,I[n1][n2]['weight']))
    else : # If anchors are not provided
        for gene,assoc in D_assoc.items():
            D_assoc[gene][etape] = []
            n1 = gene[0]
#            print(n1,assoc)
            if n1 in I.nodes : 
                if version == 'KNN_1' : D_assoc[gene][etape] = list(I[n1])
                else :
                    for n2 in list(I[n1]) : D_assoc[gene][etape].append((n2,I[n1][n2]['weight']))
    del(I)
    with open(Result_dict_file,'wb') as file : pickle.dump(D_assoc,file)
    
    print(f"Etape KNN-RBH terminée. Temps de l'étape : {fnc.time_count(time.time()-step_time)}")
    with open(Log_file,'a') as file : file.write(f"End of Step 2. Time of execution : {fnc.time_count(time.time()-step_time)}\n")
    print(f"Temps depuis lancement : {fnc.time_count(time.time()-start)}")
    with open(Log_file,'a') as file :
        file.write(f"Time since launch : {fnc.time_count(time.time()-start)}\n")
        file.write(f"-------------------------------------------------------\n")
    print("\n--------------------------- \n")

### 3 : Network Properties Closeness (NPC)
#=====================================================================================================================================================

if 3 in L_etapes :
    etape = D_etapes_names[3]
    step_time = time.time()

    with open(Log_file,'a') as file : file.write(">Step 3 : Network Properties Closeness\n")
    
    ### Datasets alternative loading
    L_DF_alt = []
    for arg in arguments[1::] : L_DF_alt.append(pd.read_csv(arg, index_col=0, header=0))

    ### Recovery of anchor gene indices in each dataset
    L_GeneIdx , L_IndexValue = [],[]
    for df in L_DF_alt : 
        L_idx = df.index.to_list()
        L_GeneIdx.append(L_idx)
        L_idxval = []
        for gene in L_idx : L_idxval.append(L_idx.index(gene))
        L_IndexValue.append(L_idxval)
        
    L_L_AnchorIdx = [[] for i in range(len(L_L_anchors))]
    for i,df in enumerate(L_DF_alt) :
        L_anchorsidx = [[] for j in range(len(L_L_anchors))]
        for j,L_anchors in enumerate(L_L_anchors) :
            for gene in L_anchors : L_anchorsidx[j].append(L_GeneIdx[i].index(gene))
            L_L_AnchorIdx[j].append(L_anchorsidx[j])

    L_L_dict_AnchorIdx = [[] for i in range(len(L_L_anchors))]
    for i,df in enumerate(L_DF_alt) :
        for j,L_AnchorIdx in enumerate(L_L_AnchorIdx):
            d = dict(zip(L_AnchorIdx[i],L_L_anchors[j]))
            L_L_dict_AnchorIdx[j].append(d)

    ### Datasets normalizaton
    min_max_scaler = preprocessing.MinMaxScaler() ### Fixed normalization
    L_DF_Norm = []
    for df in L_DF_alt : L_DF_Norm.append(min_max_scaler.fit_transform(df))

    ### Neighborhoods calculation : one seach for each dataset and each anchor genes list
    list_labels = L_labels
    L_L_Ranks = [[] for i in range(len(list_labels))]

    version = D_args['Filter version']
    T_factor = D_args['Threshold factor NPC']
    for a,df_norm in enumerate(L_DF_Norm) : # For each dataset
        
        X_normalized = df_norm
        df_aux = L_DF[a]
        geneIndex = L_GeneIdx[a]

        NN = len(L_pool)-1 # Number of general neighbors for each gene = Number of "other genes" from each gene's POV.
            
        distance = 'euclidean'
        D = metrics.pairwise_distances(X_normalized, metric=distance) # Pairwise distance Matrix
        print(f"Pairwise distances calculated : {D.shape}")

        idx = np.argsort(D) # Sort the pairwise distances and retrieve the indices of origin

        nbrsTestIdx = np.zeros((len(X_normalized), NN), dtype=int)
        nbrsTestDist = np.zeros((len(X_normalized), NN))

        print(f"Sorting the pairwise distances and keeping the {NN} closest for each gene")
        for i in range(len(X_normalized)):
            nbrsTestIdx[i] = idx[i][:NN]
            for j in range(NN) :
                nbrsTestDist[i][j] = D[int(nbrsTestIdx[i][j])][j]
            if (i+1)%1000 == 0 : print(f"Distances sorted for {i+1} genes out of {len(X_normalized)}")

        del(D)
            
        distancesKNN = nbrsTestDist
        indicesKNN = nbrsTestIdx

        if type(T_factor) != list : factor = T_factor
        else : factor = T_factor[a]
        
        for b,L_AnchorIdx in enumerate(L_L_AnchorIdx) : # For each anchor genes list
            print(f"Anayse {etape} sur Dataset {a+1} avec liste {list_labels[b]}")

            specialGenesIndex = L_AnchorIdx[a] # List of anchor gene index for current dataset
            MinNB = D_args['Minimum neighborhood NPC'][b] # Minumum number of neighbors of interest

            # Initial network building for property analysis
            G, adj = NPC.createGraph_V(indicesKNN, distancesKNN, specialGenesIndex, geneIndex)
            print(f"Graph créé ({len(G.nodes)} nodes & {len(G.edges)} edges) : {fnc.time_count(time.time()-step_time)}")

            # Deletion of edges with too high distances
            G = NPC.cleanGraphDistance_V(G, distancesKNN, factor=factor, version='Bilateral')#np.mean(distancesKNN)+2*(np.std(distancesKNN)))
            print(f"Graph nettoyé Dist ({len(G.nodes)} nodes & {len(G.edges)} edges) : {fnc.time_count(time.time()-step_time)}")

            # Deletion of nodes with with too few anchor neighbors (and their remaining edges)
            G = NPC.cleanGraph(G, MinNB)
            print(f"Graph nettoyé Nodes ({len(G.nodes)} nodes & {len(G.edges)} edges) : {fnc.time_count(time.time()-step_time)}")
            
            # Ranking matrix filling
            rankMat = NPC.rankingNodes_V(G, specialGenesIndex) # The idea is to rank genes and give the top 10
            print(f"Matrice de classement remplie ({len(rankMat)}) : {fnc.time_count(time.time()-step_time)}")
                    
            #-------------------------------------------
            to_remove = []

            for i in range(0, len(rankMat)):
                if rankMat[i][6] == 0: to_remove.append(rankMat[i])

            for i in range(0, len(to_remove)) : rankMat.remove(to_remove[i])
#            print(rankMat)
#            print(to_remove)
                
            #-------------------------------------------
            for i in range(0, len(rankMat)): rankMat[i][0] = df_aux.loc[int(rankMat[i][0]), 'ID_REF']  ## --> convertion of the gene number to the gene name.

            for i in range(0, len(rankMat)):
                for j in range(0, len(rankMat[i][8])):
                    #print(rankMat[i][8][j])
                    rankMat[i][8][j] = df_aux.loc[int(rankMat[i][8][j]), 'ID_REF']  ## --> convertion of the gene number to the gene name.
#            print(len(rankMat))
                
            #-------------------------------------------
            rankMat.sort(key=lambda k: (  k[1],  k[2], k[3], k[4], k[5], -k[6], -k[7]), reverse=True) ### k[1] -> k[5] descending order, k[6] & k[7] ascending order
#            print(rankMat)
                
            #-------------------------------------------
            rankMat_str = copy.deepcopy(rankMat)
            for line in rankMat_str :
                for i,x in enumerate(line) :
                    if i == 8 : continue
                    if type(x) != str : line[i]=str(x)
#            print(rankMat_str)
                
            #-------------------------------------------
            interet = ["Dorm" , "Germ"][b]
            file_name = f"Result_3a_{etape}_{interet}_for_Dataset_{a+1}_min{MinNB}.txt"
#            print(file_name)
            L_L_Ranks[b].append(file_name)
            headers=['ID_REF', '#SpNeighbors', '#TotalNeighbors', '#Cliques', '#Spcliques', '#MaximalClique', 'AvgDistance', 'Specific Genes']
            with open(file_name,'w') as file :
                file.write('\t'.join(headers)+'\n')
                for line in rankMat_str :
                    SpecialNg = line.pop(8)
                    file.write('\t'.join(line[0:-1])+'\t')
                    file.write(','.join(SpecialNg)+'\n')
        print("----------------------------")
    
    with open(Log_file,'a') as file : file.write(f"All datasets analyzed.\n")
    
    ### Regrouping of the calculation files names in a single text file
    lists_file_name = "Result_3t_CandidatLists_file.txt"
    with open(lists_file_name,'w') as file :
        for i,label in enumerate(L_labels):
            file.write(f">{label} files :\n")
            for name in L_L_Ranks[i] : file.write(name+'\n')

    ### Neighborhoods intersection :
    ### All kept candidates are linked to at least one anchor gene in chosen minimum number of datasets.
    ### From a dataset to another, the linked anchor gene may be diferrent.


    # Recovery of value of minimum number of datasets where a given candidate has to be found in to be kept
    # Si by oversight, there is less datasets than the provided minimum value, the value is set to the total number of datasets
    Num_dataset = D_args['Minimum redundancy']
    if Num_dataset > len(L_DF) : Num_dataset = len(L_DF)
    with open(Log_file,'a') as file : file.write(f"Minimum dataset redundancy for candidates : {Num_dataset}/{len(L_DF)}.\n")

    # Regrouping of the result tables sorted by anchor type
    with open(lists_file_name,'r') as file :
        L_classe , L_tabs = [],[]
        for line in file :
            if line[0]=='>':
                classe = line[1:-1].split()[0]
                L_classe.append(classe)
                L_tabs.append([])
            else : L_tabs[-1].append(pd.read_csv(line[0:-1],delimiter='\t'))
    with open(Log_file,'a') as file : file.write(f"Neighborhoods intersected.\n")

    # Regrouping of the candidates lists sorted by anchor type
    L_Rank_Idx = []
    for tab in L_tabs : # Pour chaque groupe de tableaux
        L_Rank_Idx.append([])
        for t in tab : L_Rank_Idx[-1].append(list(t['ID_REF']))

    # Candidates lists intersection
    L_Cands = []
    for L_ranks in L_Rank_Idx : # For each anchor type
        dico = {} # Dictionary where each candidate is associated to the total number of datasets it's been found in
        for rank in L_ranks : # For each candidates list
            for name in rank :
                if name not in dico.keys(): dico[name] = 1
                else : dico[name] += 1

        # Deletion of all candidates found in less datasets than the required minimum
        L_delete = []
        for k,v in dico.items():
            if v < Num_dataset : L_delete.append(k)
        for name in L_delete : del(dico[name])
        L_Cands.append(dico)

    Inter_all = L_Cands[0]
    for inter in L_Cands[1::]: Inter_all = list(set(Inter_all) & set(inter))

    for i,label in enumerate(L_labels):
        print(f"Candidats inter {label} : {len(L_Cands[i])}\n{L_Cands[i]}")
        with open(Log_file,'a') as file : file.write(f"Inter {label} candidates : {len(L_Cands[i])}.\n")
        print("\n----------------------\n")
    print(f"Candidats inter Multi-Ancres : {len(Inter_all)}\n{Inter_all}")
    with open(Log_file,'a') as file : file.write(f"Inter multi-label candidates : {len(Inter_all)}.\n")
    print("\n----------------------\n")

    # Looking for anchor genes among candidate genes
    print("Gènes d'intérêt eux-mêmes candidats :")
    for i,inter in enumerate(L_Cands) :
        lab_1 = L_labels[i]
        for gene in inter :
            for j,L_anchor in enumerate(L_L_anchors) :
                lab_2 = L_labels[j]
                if gene in L_anchor : print(f"{gene}: {lab_2} candidat chez {lab_1}")
    print("\n----------------------\n")

    ### Neighborhood vectors dictionary

    # Vector of links of interest per dataset for each candidate
    L_dico_V = []
    for i,dico in enumerate(L_Cands) :
        D_voisin = {}
        for gene,value in dico.items() : 
            D_voisin[gene] = []
            for j,tab in enumerate(L_tabs[i]) :
                L_names = list(tab['ID_REF'])
                if gene in L_names : v = int(tab.loc[tab['ID_REF'] == gene]["#SpNeighbors"])
                else : v = 0
                D_voisin[gene].append(v)
#        for k,v in D_voisin.items(): print(k,v)"
#        print("------------------------")
        L_dico_V.append(D_voisin)

    # Dictionary of neighbors sorted by occurence in different datasets
    L_dico_N = []
    for i,dico in enumerate(L_Cands) :
        D_temp = {}
        for gene,value in dico.items() :
            D_temp[gene] = {"total":0}
            for j in range(len(L_DF)): D_temp[gene][j+1] = []
            L_voisinage = []
            inter_voisinage = []
            for tab in L_tabs[i]:
                L_names = list(tab['ID_REF'])
                if gene in L_names : 
                    L_voisinage.append(list(tab.loc[tab['ID_REF']==gene]["Specific Genes"])[0].split(','))
                    inter_voisinage += L_voisinage[-1]
            inter_voisinage = set(inter_voisinage)
            D_temp[gene]["total"] = len(inter_voisinage)
            for n in inter_voisinage :
                p = 0
                for voisinage in L_voisinage :
                    if n in voisinage : p += 1
                D_temp[gene][p].append(n)
        L_dico_N.append(D_temp)

    ### Candidate genes saving
    file_name = f"Result_3b_{etape}_CandidateGenes.txt"
    headers = ['ID_REF']
    for label in L_labels : headers.append(f"#{label}_Neighbors")
    headers.append("#UniqueNeighbors")
    for i in range(len(L_DF)) : headers.append(f"{i+1}-Neighborhood")
#    print(headers)

    with open(file_name,'w') as file :
        file.write('\t'.join(headers)+'\n')
        for gene in L_pool :
            match = False
            line = [0 for i in range(len(headers))]
            line[0] = gene
            for i,inter in enumerate(L_Cands) :
                head_1 = f"#{L_labels[i]}_Neighbors"
                if gene in list(inter.keys()) :
#                    print(gene,L_labels[i])
                    match = True
                    line[headers.index(head_1)] = f"{L_dico_V[i][gene]}"
                    dico_N = L_dico_N[i]
                    for k,v in dico_N[gene].items():
#                        print(gene , k , v)
                        if k=="total" : line[headers.index("#UniqueNeighbors")] += v
                        else :
                            head_2 = f"{k}-Neighborhood"
#                            print(head_2,line[headers.index(head_2)])
#                            print(f"{head_2} (pré): {line[headers.index(head_2)]}")
                            if line[headers.index(head_2)] == 0 :
#                                print("vide")
                                if v == [] : line[headers.index(head_2)] = 'None'
                                else : line[headers.index(head_2)] = ','.join(v)
                            else :
#                                print("non vide")
                                line[headers.index(head_2)] += '/'
                                if v == [] : line[headers.index(head_2)] = 'None'
                                else : line[headers.index(head_2)] += ','.join(v)
#                            print(f"{head_2} (post): {line[headers.index(head_2)]}")
#                            print(line)
#                            print("- - - - - - - - -")
                else : line[headers.index(head_1)] = 'None'
#            print(line)
            for i,a in enumerate(line) : line[i] = str(a)
            if match == True :
#                print(gene,match)
#                print('\t'.join(line))
                file.write('\t'.join(line)+'\n')
#                print("-------------------------")

    with open(Log_file,'a') as file : file.write(f"Candidate genes data saved.\n")

    ### Building the neighborhood network and saving it
    G = nx.Graph()
    for i,inter in enumerate(L_Cands) :
        for n1 in list(inter.keys()) :
            G.add_node(n1)
            for k,v in L_dico_N[i][n1].items():
                if type(v) == list :
                    for n2 in v :
                        if (n2 not in G.nodes) : G.add_node(n2)
                        attr = {"weight":k}
                        G.add_edge(n1,n2)
                        G[n1][n2].update(attr)

    CN_map = []
    for node in G.nodes :
        ancre = False
        for i,L_anchors in enumerate(L_L_anchors) :
            if node in L_anchors :
                CN_map.append(L_colors[i])
                ancre = True
                break
        if not ancre : CN_map.append((0,0,0))

    WE_map = []
    for (n1,n2) in G.edges : WE_map.append(G[n1][n2]['weight'])

    CE_map = []
    L_weights = []
    base = 1/len(L_DF)
    for i in range(len(L_DF)): L_weights.append(base*(i+1))
    for (n1,n2) in G.edges :
        w = G[n1][n2]['weight']
        CE_map.append(L_weights[w-1])
#alt        CE_map.append((L_weights[w-1],L_weights[w-1],L_weights[w-1]))

    graph_file = f"Result_3c_{etape}_graph.txt"
    with open(graph_file,'w') as file :
        for i,node in enumerate(list(G.nodes)) : file.write(f">{node}:{CN_map[i]}\n")
        for i,(n1,n2) in enumerate(list(G.edges)) : file.write(f"{n1} {n2} weight:{WE_map[i]}\n")

    with open(Log_file,'a') as file : file.write(f"Network built and saved : {len(G.nodes)} genes and {len(G.edges)} associations.\n")

    print(f"Nombre de points : {len(G.nodes)}")
    print(f"Nombre d'arêtes : {len(G.edges)}")

    ### Saving of NPC associations in Linkage dictionary for consensus
    ### For each anchor, we create as many sub-lists as there are datasets. The anchor's neighbors (i.e. the candidates) are
    ###     then sorted according to the number of datasets they found in. The more datasets find a given candidate, the more
    ###     toward the last sub-list the candidate is put in.
    for gene,assoc in D_assoc.items():
        D_assoc[gene][etape] = [[] for i in range(len(L_DF))]
        n1 = gene[0]
        if n1 not in G.nodes : continue
        else : 
#            print(gene)
            for n2 in G[n1] : 
                w = G[n1][n2]['weight']
                D_assoc[gene][etape][w-1].append(n2)
#                print(n2,G[n1][n2])
    with open(Result_dict_file,'wb') as file : pickle.dump(D_assoc,file)
    del(G)
    
    print(f"Etape Candidature par Voisinage d'Intérêt terminée. Temps de l'étape : {fnc.time_count(time.time()-step_time)}")
    with open(Log_file,'a') as file : file.write(f"End of Step 3. Time of execution : {fnc.time_count(time.time()-step_time)}\n")
    print(f"Temps depuis lancement : {fnc.time_count(time.time()-start)}")
    with open(Log_file,'a') as file :
        file.write(f"Time since launch : {fnc.time_count(time.time()-start)}\n")
        file.write(f"-------------------------------------------------------\n")
    print("\n--------------------------- \n")

### 4 : Cluster Path
#=====================================================================================================================================================

if 4 in L_etapes :
    etape = D_etapes_names[4]
    print(f"Analyse {etape}\n")
    
    step_time = time.time()

    with open(Log_file,'a') as file : file.write(">Step 4 : Cluster Path\n")

    nb_clust = D_args['Number of clusters']

    LD_vecteurs = []
    for df in L_DF : # For each dataset
        L_colonnes = list(df.keys())[1::]
        n_tp = len(L_colonnes)
        
        ### Clustering calculation
        L_dico = []
        for i in range(n_tp):
            L_dico.append(fnc.KM1D_v3(df , i , n_clusters=nb_clust))
        
        ### Cluster pathing
        D_points = {}
        for dico in L_dico :
            for k,v in dico.items():
                if k not in D_points.keys():D_points[k] = [v]
                else : D_points[k].append(v)

        ### Regrouping genes by common cluster path
        D_vecteurs = {}
        for k,v in D_points.items():
            if tuple(v) not in D_vecteurs.keys(): D_vecteurs[tuple(v)] = [k]
            else : D_vecteurs[tuple(v)].append(k)
        LD_vecteurs.append(D_vecteurs)

    with open(Log_file,'a') as file : file.write(f"All datasets analyzed.\n")

    D_names = {}
    for name in L_pool : # For each gene
        D_names[name] = []
        for dico in LD_vecteurs : # For each dictionary "path - genes list"
            for k,v in dico.items():
                if name in v :
                    D_names[name].append(k) # Recovering the path followed by the current gene
                    break

    D_groupes = {}
    for k,v in D_names.items():
        if tuple(v) not in D_groupes.keys(): D_groupes[tuple(v)] = [k]
        else : D_groupes[tuple(v)].append(k)

    file_name = f"Result_4a_{etape}_list.txt"

#    print(f"Nombre de combinaisons de chemins : {len(D_groupes)}")
    with open(file_name,'w') as file :
        for k in sorted(D_groupes, key=lambda k: len(D_groupes[k]), reverse=True):
            v = D_groupes[k]
#            print(k,v)
            line = f">Path:{k} ; {len(v)} total genes"
            L_count = [0 for i in range(len(L_L_anchors))]
            for name in v :
                for i,L_anchors in enumerate(L_L_anchors):
                    if name in L_anchors : L_count[i] += 1
            for i,c in enumerate(L_count) : line += f" ; {c} {L_labels[i]} genes"
            line += '\n'  
#            print(line[0:-1])
            file.write(line)
            file.write(' '.join(v)+'\n')

    with open(Log_file,'a') as file : file.write(f"Common path gene lists saved : {len(D_groupes)} paths found.\n")

    # Building the neighborhood network and saving it
    G = nx.Graph()
    for k in sorted(D_groupes, key=lambda k: len(D_groupes[k]), reverse=True):
        groupe = D_groupes[k]
        for i,n1 in enumerate(groupe) :
            if n1 not in list(G.nodes) : G.add_node(n1)
            for j,n2 in enumerate(groupe[i+1::]):
                if n2 not in list(G.nodes) : G.add_node(n2)
                G.add_edge(n1,n2)

    CN_map = []
    for node in G.nodes :
        ancre = False
        for i,L_anchors in enumerate(L_L_anchors) :
            if node in L_anchors :
                CN_map.append(L_colors[i])
                ancre = True
                break
        if not ancre : CN_map.append((0,0,0))

    graph_file = f"Result_4b_{etape}_graph.txt"
    with open(graph_file,'w') as file :
        for i,node in enumerate(list(G.nodes)) : file.write(f">{node}:{CN_map[i]}\n")
        for i,(n1,n2) in enumerate(list(G.edges)) : file.write(f"{n1} {n2}\n")

    with open(Log_file,'a') as file : file.write(f"Network built and saved : {len(G.nodes)} genes and {len(G.edges)} associations.\n")

    print(f"Nombre de points : {len(G.nodes)}")
    print(f"Nombre d'arêtes : {len(G.edges)}\n")
    
    ### Saving of Cluster Path associations in Linkage dictionary for consensus
    ### For each anchor, we save it's neighbors
    if D_args['Anchor Genes lists'] != [] : # If there are anchors
        for gene,assoc in D_assoc.items():D_assoc[gene][etape] = []

        for path,L_names in D_groupes.items():
            for n1 in L_names :
                ancre = False
                for i,L_anchors in enumerate(L_L_anchors):
                    if n1 in L_anchors :
                        cle = (n1,L_labels[i])
                        ancre = True
                if not ancre : continue
                for n2 in L_names :
                    if n1==n2 : continue
                    D_assoc[cle][etape].append(n2)
                    
    else : # If there are no anchors
        for gene,assoc in D_assoc.items():D_assoc[gene][etape] = []

        for path,L_names in D_groupes.items():
            for n1 in L_names :
                for n2 in L_names :
                    if n1==n2 : continue
                    D_assoc[(n1,'Candidate')][etape].append(n2)
                    
    del(G)           
    with open(Result_dict_file,'wb') as file : pickle.dump(D_assoc,file)
    
    print(f"Etape Chemin de Clusters terminée. Temps de l'étape : {fnc.time_count(time.time()-step_time)}")
    with open(Log_file,'a') as file : file.write(f"End of Step 4. Time of execution : {fnc.time_count(time.time()-step_time)}\n")
    print(f"Temps depuis lancement : {fnc.time_count(time.time()-start)}")
    with open(Log_file,'a') as file :
        file.write(f"Time since launch : {fnc.time_count(time.time()-start)}\n")
        file.write(f"-------------------------------------------------------\n")
    print("\n--------------------------- \n")
    
### END : Methods Consensus
#=====================================================================================================================================================

if consensus :
    print(f"Consensus\n")
    step_time = time.time()

    with open(Log_file,'a') as file : file.write(">Final Step : Methods Consensus\n")
    
    L_niv = D_args['Consensus levels list'] # Liste des niveaux à retenir pour le fichier text du consensus
    
    ### Création du dictionnaire de répartition des candidats par récurrence d'association
    D_repartition = {'ID_REF':[],'Class':[],'#Unique':[]}
    a = list(D_assoc.keys())[0]
    Num_methods = 0
    L_n_method = []
    for i in range(len(list(D_assoc[a].keys()))): 
        if i == 0 :
            D_repartition[f"#{i+1}-Method"] = []
            L_n_method.append(f"#{i+1}-Method")
        else :
            D_repartition[f"#{i+1}-Methods"] = []
            L_n_method.append(f"#{i+1}-Methods")
        Num_methods += 1
#    print("Nombre de méthodes :",Num_methods)

    L_methods = []
    for methode in list(D_assoc[a].keys()):
        L_methods.append(methode)
        if methode == D_etapes_names[1]: # Pearson Co-Epxression : sort Positive and Negative values
            D_repartition[f"{methode} Pos"] = []
            D_repartition[f"{methode} Neg"] = []
        elif methode == D_etapes_names[3]: # Network Properties : sort association by dataset redundancy
            for i in range(len(D_assoc[a][methode])): D_repartition[f"{methode}-{i+1}"] = []
        else: # KNN-RBH or Cluster Path : just get the associated genes
            D_repartition[methode] = []
#    print(D_repartition)

    D_noms = {}
    D_vecteurs = {}
    for gene,voisinages in D_assoc.items():
        # Récupération des données du gène courant
        ref,clas = gene
        D_repartition['ID_REF'].append(ref)
        D_repartition['Class'].append(clas)
        D_noms[ref] = {'class':clas}
        D_vecteurs[ref] = {}
        for i in reversed(range(Num_methods)) : D_noms[ref][i+1] = []
#        print(gene)
#        print(D_noms[ref])
        
        # Décompte des voisins uniques
        L_neig = []
        L_PCEIN_pos , L_PCEIN_neg = [],[]
        L_KNN = []
        for methode,L_voisins in voisinages.items():
#            print(methode,len(L_voisins))
            if methode != D_etapes_names[3] : # Pearson Co-Expression, KNN-RBH or Cluster Path
                if methode == D_etapes_names[4] : L_neig += L_voisins # Cluster Path
                else : # Pearson Co-Expression, KNN-RBH
                    for (n,c) in L_voisins :
                        L_neig.append(n)
                        if methode == D_etapes_names[1] : # Pearson Co-Expression
#                            print(n,c)
                            if c > 0 : L_PCEIN_pos.append(n)
                            else : L_PCEIN_neg.append(n)
                        else : L_KNN.append(n) # KNN-RBH
            else : # Network Properties
                for i,rank in enumerate(L_voisins) : L_neig += rank
        Set_neig = set(L_neig)
        Diff = len(L_neig)-len(Set_neig)
#        print(f"Union voisinages = {len(L_neig)} | Intersection Voisinage = {len(Set_neig)} | Difference = {Diff}")
        D_repartition['#Unique'].append(len(Set_neig))
        
        # Tri des voisins (nombre de méthodes, localisation)
#        sous_D = {'#1-Method':0,'#2-Methods':0,'#3-Methods':0,'#4-Methods':0,
#                  'RICEP':0,'KNN-RBH':0,'NPC-1':0,'NPC-2':0,'NPC-3':0,'CP':0}

        sous_D = {}
        for m in L_n_method : sous_D[m] = 0
#        print(sous_D)
        for m in L_methods :
            if m == D_etapes_names[1] : # Pearson Co-Expression
                sous_D[f"{m} Pos"] = 0
                sous_D[f"{m} Neg"] = 0
            elif m == D_etapes_names[3] : # Network Properties
                for n in range(len(voisinages[m])) : sous_D[f"{m}-{n+1}"] = 0
            else : sous_D[m] = 0
#            print(m,sous_D)
#        print(sous_D)
            
        for name in Set_neig :
            
            D_vecteurs[ref][name] = ['0','0','0','0','0']
            N_meth = 0
            if D_etapes_names[1] in L_methods : # Pearson Co-Expression
                if name in L_PCEIN_pos :
                    N_meth += 1
                    sous_D[f"{D_etapes_names[1]} Pos"] += 1
                    D_vecteurs[ref][name][0] = '1'
                elif name in L_PCEIN_neg :
                    N_meth += 1
                    sous_D[f"{D_etapes_names[1]} Neg"] += 1
                    D_vecteurs[ref][name][1] = '1'
            if (D_etapes_names[2] in L_methods) and (name in L_KNN) : # KNN-RBH
                N_meth += 1
                sous_D[D_etapes_names[2]] += 1
                D_vecteurs[ref][name][2] = '1'
            if D_etapes_names[3] in L_methods : # Network Properties
                for n in range(len(voisinages[D_etapes_names[3]])):
                    if name in voisinages[D_etapes_names[3]][n] :
                        N_meth += 1
                        sous_D[f"{D_etapes_names[3]}-{n+1}"] += 1
                        D_vecteurs[ref][name][3] = '1'
                        break
            if (D_etapes_names[4] in L_methods) and (name in voisinages[D_etapes_names[4]]) : # Cluster Path
                N_meth += 1
                sous_D[D_etapes_names[4]] += 1
                D_vecteurs[ref][name][4] = '1'
            
            if N_meth == 1 : sous_D['#1-Method'] += 1
            if N_meth == 2 : sous_D['#2-Methods'] += 1
            if N_meth == 3 : sous_D['#3-Methods'] += 1
            if N_meth == 4 : sous_D['#4-Methods'] += 1
            if N_meth in L_niv : D_noms[ref][N_meth].append(name)
#        print(sous_D)
        for k,v in sous_D.items(): D_repartition[k].append(v)
#        print("-------------")

    ### Sauvegarde des données chiffrées par gène ancre
    tab_file = "Result_END_consensus.csv"
    DF_Repartition = pd.DataFrame.from_dict(D_repartition)
    L_sorting = ['#4-Methods','#3-Methods','#2-Methods','#1-Method']
    L_delete = []
    for s in L_sorting :
        if s not in list(D_repartition.keys()): L_delete.append(s)
    for s in L_delete : L_sorting.remove(s)
    asc = tuple([0 for i in range(len(L_sorting))])
    sorted_DF = DF_Repartition.sort_values(L_sorting , ascending = asc)
    sorted_DF.to_csv(tab_file , index=False)

    L_names = list(sorted_DF['ID_REF'])
    for interet in L_names :
        D_meth = D_noms[interet]
        classe = D_meth['class']
#        print(interet,classe)
        for meth,L_names in D_meth.items():
            if meth == 'class' : continue
#            print(meth,L_names)
#        print("-------------------------")

    ### Sauvegarde des listes de candidats par gène ancre
    file_name_1 = "Result_END_GenesOfInterest_ListByAnchors.txt"
    with open(file_name_1,'w') as file:
        L_names = list(sorted_DF['ID_REF'])
        D_n = {}
        for niv in reversed(L_niv) : D_n[niv] = 0
        for anchor in L_names :
            D_meth = D_noms[anchor]
            classe = D_meth['class']
            line = f">{anchor} ({classe})\n"
            file.write(line)
            for niv,L_noms in D_meth.items():
                if niv == 'class' or niv not in L_niv : continue
                line = f"Niv.{niv} :"
                if len(L_noms)==0 : line += ' None\n'
                else :
                    D_n[niv] += len(L_noms)
                    for name in L_noms :
                        ancre = False
                        for i,L_anchors in enumerate(L_L_anchors):
                            lab = L_labels[i]
                            if name in L_anchors :
                                line += f" {name}({lab[0]})"
                                ancre = True
                                break
                        if not ancre : line += f" {name}"
                    line += '\n'
                file.write(line)
        file.write("----------------------------------------------------------------------\n")
        for niv,num in D_n.items(): file.write(f"> Nombre total de Niv.{niv} : {num}\n")

    with open(Log_file,'a') as file : file.write(f"Anchor-ordered results saved.\n")

    ### Sauvegarde des associations candidat-ancre
    file_name_2 = "Result_END_GenesOfInterest_ListByCandidates.txt"
    with open(file_name_2,'w') as file :
        if D_args['Anchor Genes lists'] != [] : # S'il y a des ancres
            line = '\t'.join(['Candidate','Anchor','Anchor Function','Consensus Level',f"{D_etapes_names[1]} Pos",f"{D_etapes_names[1]} Neg",D_etapes_names[2],D_etapes_names[3],D_etapes_names[4]])+'\n'
        else : # S'il y a des ancres
            line = '\t'.join(['Candidate 1','Candidate 2','Consensus Level',f"{D_etapes_names[1]} Pos",f"{D_etapes_names[1]} Neg",D_etapes_names[2],D_etapes_names[3],D_etapes_names[4]])+'\n'
        file.write(line)
        L_names = list(sorted_DF['ID_REF'])
        D_n = {}
        for niv in reversed(L_niv) : D_n[niv] = 0
        for anchor in L_names :
            D_meth = D_noms[anchor]
            classe = D_meth['class']
            for niv,L_noms in D_meth.items():
                if (niv == 'class') or (niv not in L_niv) or (len(L_noms)==0) : continue
                for name in L_noms :
                    line = ''
                    vecteur = D_vecteurs[anchor][name]
                    ancre = False
                    for i,L_anchors in enumerate(L_L_anchors) :
                        if name in L_anchors :
                            line += f"{name}({L_labels[i][0]})"
                            ancre = True
                            break
                    if not ancre : line += f"{name}"
                    if D_args['Anchor Genes lists'] != [] : # S'il y a des ancres
                        line += '\t'+'\t'.join([anchor,classe,str(niv)])+'\t'+'\t'.join(vecteur)+'\n'
                    else :
                        line += '\t'+'\t'.join([anchor,str(niv)])+'\t'+'\t'.join(vecteur)+'\n'
                    file.write(line)

    with open(Log_file,'a') as file : file.write(f"Candidate-ordered results saved.\n")

    ### Sauvegarde du réseau final
    file_name_3 = "Result_END_graph.txt"

    #  Préparation des arguments : seul le niveau de consensus et les méthodes dont des résultats existent sont prises en compte
    D_attr = {"Consensus":0}
    a = list(D_assoc.keys())[0]
    L_steps = list(D_assoc[a].keys())
#    print(L_steps)
    for s in L_steps : D_attr[s] = "NO"
#    print(D_attr)

    G = nx.Graph()
    D_anchors = {}
    L_anchors , L_candidates = [] , []
    for gene,L_meth in D_assoc.items(): # Récupération et tri des ancres
        n1,category = gene
        G.add_node(n1)
        L_anchors.append(n1)

        if category not in list(D_anchors.keys()): D_anchors[category] = [n1]
        else : D_anchors[category].append(n1)

    for gene,L_meth in D_assoc.items(): # Récupération des candidats
        n1,category = gene
        for methode,voisins in L_meth.items():
            if methode in [D_etapes_names[1] , D_etapes_names[2]]: # Pearson Co-Expression or KNN-RBH
                for v in voisins :
                    n2,c = v[0],round(v[1],4)
                    if (n2 not in L_candidates) and (n2 not in L_anchors) : L_candidates.append(n2)
                    if (n1,n2) not in G.edges :
                        G.add_edge(n1 , n2)
                        G[n1][n2].update(copy.deepcopy(D_attr))
                    if G[n1][n2][methode] == 'NO':
                        G[n1][n2][methode] = c
                        G[n1][n2]['Consensus'] += 1
                    
            elif methode == D_etapes_names[3] : # Network Properties (Gabriel)
                for rank,L_v in enumerate(voisins) :
                    for n2 in L_v :
                        if (n2 not in L_candidates) and (n2 not in L_anchors) : L_candidates.append(n2)
                        if (n1,n2) not in G.edges :
                            G.add_edge(n1 , n2)
                            G[n1][n2].update(copy.deepcopy(D_attr))
                        if G[n1][n2][D_etapes_names[3]] == 'NO':
                            G[n1][n2][D_etapes_names[3]] = rank+1 # Le +1 est nécessaire car le 'rank' part de 0.
                            G[n1][n2]['Consensus'] += 1
            else : # Cluster Path
                for n2 in voisins :
                    if (n1,n2) not in G.edges :
                        G.add_edge(n1 , n2)
                        G[n1][n2].update(copy.deepcopy(D_attr))
                    if G[n1][n2][D_etapes_names[4]] == 'NO':
                        G[n1][n2][D_etapes_names[4]] = 'YES'
                        G[n1][n2]['Consensus'] += 1

    if D_args['Anchor Genes lists'] != [] : # S'il y a des ancres
        print(f"Nombre de points : {len(G.nodes)} / {len(L_anchors)} ancres / {len(L_candidates)} candidats")
    else : # S'il n'y a pas d'ancres
        print(f"Nombre de points : {len(G.nodes)}")
    print(f"Nombre d'arêtes : {len(G.edges)}")
    print("----------------------------")
    
    with open(Log_file,'a') as file : file.write(f"Consensus network built : {len(G.nodes)} genes and {len(G.edges)} associations.\n")

    CN_map , CE_map , WE_map = [],[],[]
    if D_args['Anchor Genes lists'] != [] :
        for n in G.nodes :
            cand = True
            for k,v in D_anchors.items():
                if n in v :
                    idx = L_labels.index(k)
                    CN_map.append(L_colors[idx])
                    cand = False
                    break
            if cand : CN_map.append((0,0,0))
    else :
        for n in G.nodes : CN_map.append((0,0,0))
            
    base = 1/len(L_steps)
    for (n1,n2) in G.edges :
        w = G[n1][n2]['Consensus']
        CE_map.append((1-(base*w),1-(base*w),1-(base*w)))
        WE_map.append(w)

    save_file = "Result_END_graph.txt"

    with open(save_file,'w') as file :
        for node in sorted(L_candidates) : file.write(f">{node}:(0, 0, 0)\n")
        if D_args['Anchor Genes lists'] != [] : # S'il y a des ancres
            for k,v in D_anchors.items() :
                idx = L_labels.index(k)
                for node in sorted(v) : file.write(f">{node}:{str(L_colors[idx])}\n")
        else : # S'il n'y a pas d'ancres
            for k,v in D_anchors.items() :
                for node in sorted(v) : file.write(f">{node}:(0, 0, 0)\n")
        
        for (n1,n2) in G.edges :
            D_attr = G[n1][n2]
            line = f"{n1} ; {n2}"
            for k,v in D_attr.items(): line += f" ; {k}:{str(v)}"
#            print(line)
            file.write(line+'\n')

    with open(Log_file,'a') as file : file.write(f"Consensus network saved.\n")
        
    print(f"Consensus des méthodes terminée. Temps de l'étape : {fnc.time_count(time.time()-step_time)}")
    with open(Log_file,'a') as file : file.write(f"End of Final Step. Time of execution : {fnc.time_count(time.time()-step_time)}\n")
    print(f"Temps depuis lancement : {fnc.time_count(time.time()-start)}")
    with open(Log_file,'a') as file : file.write(f"Time since launch : {fnc.time_count(time.time()-start)}\n")

