
### Imports
#=====================================================================================================================================================

import time
import sys
import os
import copy
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from scipy.spatial import distance
import networkx as nx
import pickle
import Setting_functions as Set_fnc
import PCEIN_functions as PCEIN
import KNN_RBH_functions as KNN_RBH
import ClusterPath_functions as CP
import Consensus_functions as Cons_fnc
import NPC_functions as NPC
import warnings
warnings.filterwarnings("ignore")

### Arguments loading
#=====================================================================================================================================================

# ARGUMENTS ARE TO BE GIVEN IN THE FOLLOWING ORDER :
#   - The setting file
#   - The datasets

arguments = sys.argv[1::]

D_args , D_assoc , folder_name = Set_fnc.settings(arguments[0] , len(arguments[1::]))
#print(D_args)

### Data loading
#=====================================================================================================================================================

# Creation of the log file. All major events will be written in it -----------------------
Log_file = f"{folder_name}RunLog.txt"
with open(Log_file,'a') as file :file.write(">Initial Data\n")

# Global gene pool -----------------------------------------------------------------------
L_pool = Set_fnc.globalGenePool(D_args['Global gene pool'])
print(f"Pool de gènes : {len(L_pool)}")
with open(Log_file,'a') as file : file.write(f"Global gene pool : {len(L_pool)}\n")

# Anchor genes lists (if provided) -------------------------------------------------------
L_L_All_anchors , L_L_anchors , L_Final_anchors = Set_fnc.anchorGenePools(D_args['Anchor Genes lists'] , L_pool)

if L_Final_anchors != [] : # If anchor genes are provided
    with open(Log_file,'a') as file : file.write(f"Total anchor pool : {len(L_Final_anchors)}\n")

L_labels = D_args['Anchor Genes labels']

L_colors = D_args['Anchor colors']

for i,L_anchors in enumerate(L_L_anchors) :
    print(f"{L_labels[i]} Genes : {len(L_anchors)} / {len(L_L_All_anchors[i])}")
    with open(Log_file,'a') as file : file.write(f"{L_labels[i]} genes : {len(L_anchors)} / {len(L_L_All_anchors[i])}\n")
print("\n--------------------------- \n")

# Datasets to analyse --------------------------------------------------------------------
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

# Steps to run ---------------------------------------------------------------------------
L_steps = D_args['Steps list']
D_steps_names = {1:'P-CEIN',2:'KNN-RBH',3:'NPC',4:'ClusterPath'}
consensus = D_args['Consensus step']

# Linkage dictionary initialization ------------------------------------------------------
D_assoc , Result_dict_file = Set_fnc.linkageDictionary(D_assoc , D_args , folder_name , L_pool , L_L_anchors , L_labels)


start = time.time()
### 1 : Pearson based Co-Expression Intersected Network (P-CEIN)
#=====================================================================================================================================================

if 1 in L_steps :
    etape = D_steps_names[1]
    step_time = time.time()
    with open(Log_file,'a') as file : file.write(">Step 1 : Pearson based Co-Expression Intersected Network\n")
    print(f"Analyse {etape}")
    
    ### New graph building (or preexisting graph loading)
    create = D_args['New Network']
    
    file_name_1a = f"{folder_name}Result_1a_{etape}_graph.txt" # Initialized here because used for the network's loading if create==False

    if create == False : # Load a preexisting network
        RICEP = PCEIN.loadPearsonCoExpressionIntersectedNetwork(file_name_1a,annonce=10000)
        with open(Log_file,'a') as file : file.write(f"Network loaded.\n")
    
    else : # Build a new network
        if D_args['Anchor centered'] : a_L = L_Final_anchors
        else : a_L = None
        P_t = D_args['Pearson principal threshold']
        S_tps = D_args['Research time announcement']

        L_DF_3TP_min = [] # Any dataset with less than 3 timepoints will not be used
        for df in L_DF :
            n_col = len(df.keys())
            if n_col >= 4 : # 3 timepoints + the 'ID_REF' column = minimum of 4 columns
                L_DF_3TP_min.append(df)
        print(f"Number of datasets with 3 or more timepoints : {len(L_DF_3TP_min)}/{len(L_DF)}")
        
        with open(Log_file,'a') as file : file.write(f"Building of a new network with a {P_t} threshold.\n")
        RICEP = PCEIN.buildPearsonCoExpressionIntersectedNetwork(dataframe_list=L_DF_3TP_min , gene_list=L_pool ,
                                                                 anchor_list=a_L ,threshold_Pearson=P_t , search_tps=S_tps)
        print(f"Initial network composition : {len(RICEP.nodes)} nodes and {len(RICEP.edges)} edges")
        print("----------------------------")

        # Initial new graph saving
        L_nodes = list(RICEP.nodes)
        L_edges = list(RICEP.edges)
        PCEIN.saveNetwork(RICEP , anchor_lists=L_L_anchors , color_list=L_colors , file_name=file_name_1a)
        with open(Log_file,'a') as file : file.write(f"Full network built and saved : {len(RICEP.nodes)} genes and {len(RICEP.edges)} associations.\n")

        # Initial new graph's general data saving
        df_data = PCEIN.saveNetworkData(RICEP , L_anchors_labels=L_labels , L_anchors_list=L_L_anchors , dataframe_file=f"{folder_name}Result_1a_{etape}_data.csv")
        del(df_data)
        with open(Log_file,'a') as file : file.write(f"Full network data saved.\n")

    ### Dynamic Pearson threshold calculation (if asked for)
    if D_args['Pearson dynamic threshold'] :
        print("\nEdge filtering via secondary dynamic threshold")

        RICEP , dynamic = PCEIN.dynamic_filtering(RICEP , factor=D_args['Dynamic threshold factor'])

        print(f"Dynamic Pearson threshold calculated : {dynamic} (absolute value)")
        with open(Log_file,'a') as file : file.write(f"Dynamic secondary Pearson threshold calculated : {dynamic}.\n")
        
        print(f"Dynamically filtered network composition : {len(RICEP.nodes)} nodes and {len(RICEP.edges)} edges")
        print("----------------------------")

        # Dynamically filtered graph saving
        file_name_1b = f"{folder_name}Result_1b_{etape}_dynamic_graph.txt"
        L_nodes = list(RICEP.nodes)
        L_edges = list(RICEP.edges)
        PCEIN.saveNetwork(RICEP , anchor_lists=L_L_anchors , color_list=L_colors , file_name=file_name_1b)
        with open(Log_file,'a') as file : file.write(f"Dynamic Pearson network saved : {len(RICEP.nodes)} genes and {len(RICEP.edges)} associations.\n")

        # Dynamically filtered graph's general data saving
        df_data = PCEIN.saveNetworkData(RICEP , L_anchors_labels=L_labels , L_anchors_list=L_L_anchors , dataframe_file=f"{folder_name}Result_1b_{etape}_dynamic_data.csv")
        del(df_data)
        with open(Log_file,'a') as file : file.write(f"Dynamic Pearson network data saved.\n")
    
    ### Minimum-anchor-neighborhood filtering (if asked for)
    if D_args['Minimum neighborhood P-CEIN'] != 'default' :
        print("\nCandidates filtering based on their quantity of anchor neighbors.")
        
        RICEP = PCEIN.neighborhood_filtering(RICEP , anchor_lists=L_L_anchors , limite_list=D_args['Minimum neighborhood P-CEIN'])
        print(f"Neighbor filtered network composition : {len(RICEP.nodes)} nodes and {len(RICEP.edges)} edges")
        print("----------------------------")

        # Anchor-neighborhood filtered graph saving
        file_name_1c = f"{folder_name}Result_1c_{etape}_relevant_neighborhood_graph.txt"
        L_nodes = list(RICEP.nodes)
        L_edges = list(RICEP.edges)
        PCEIN.saveNetwork(RICEP , anchor_lists=L_L_anchors , color_list=L_colors , file_name=file_name_1c)
        with open(Log_file,'a') as file : file.write(f"Relevant neighborhood network saved : {len(RICEP.nodes)} genes and {len(RICEP.edges)} associations.\n")

        # Anchor-neighborhood filtered graph's general data saving
        df_data = PCEIN.saveNetworkData(RICEP , L_anchors_classes=L_labels , L_anchors_list=L_L_anchors , dataframe_file=f"{folder_name}Result_1c_{etape}_relevant_neighborhood_data.csv")
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

    print(f"End of Pearson Co-Expression step. Run time : {Set_fnc.time_count(time.time()-step_time)}")
    with open(Log_file,'a') as file : file.write(f"End of Step 1. Run time : {Set_fnc.time_count(time.time()-step_time)}\n")
    print(f"Time since launch : {Set_fnc.time_count(time.time()-start)}")
    with open(Log_file,'a') as file :
        file.write(f"Time since launch : {Set_fnc.time_count(time.time()-start)}\n")
        file.write(f"-------------------------------------------------------\n")
    print("\n--------------------------- \n")

### 2 : K-Nearest Neighbors enhanced with Reciprocal Best Hit (KNN-RBH)
#=====================================================================================================================================================

if 2 in L_steps :
    etape = D_steps_names[2]
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
            RBH = KNN_RBH.apply_version(df , n_points=None , method=version , n_neighbors=neigh , factor=factor , RBH=rbh)
            L_RBH.append(RBH)
            print(f"Time since step's beginning : {Set_fnc.time_count(time.time()-step_time)}")
            print('-----')
    else : # If different dynamic factors are provided
        for i,df in enumerate(L_DF) :
            print(f"Analyse {etape} pour dataset {i+1}")
            RBH = KNN_RBH.apply_version(df , n_points=None , method=version , n_neighbors=neigh , factor=factor[i] , RBH=rbh)
            L_RBH.append(RBH)
            print(f"Time since step's beginning : {Set_fnc.time_count(time.time()-step_time)}")
            print('-----')
        
    with open(Log_file,'a') as file : file.write(f"K-Neighborhoods calculated.\n")

    ### KNN-RBH dictionaries intersection
    if version == 'KNN_1' : RBH_inter = KNN_RBH.dictionary_intersection_1(L_RBH)
    elif version == 'KNN_2' : RBH_inter = KNN_RBH.dictionary_intersection_2(L_RBH)
        
#    print(len(RBH_inter))
    with open(Log_file,'a') as file : file.write(f"K-Neighborhoods intersected.\n")

    ### Initial network building
    if version == 'KNN_1' : # KNN version 1 : only the neighbors are known
        I = nx.Graph()
        for n1,voisins in RBH_inter.items():
            for n2 in voisins :
                if not I.has_edge(n1,n2) : I.add_edge(n1,n2)

    else : # KNN version 2 : each neighbor of a gene knows the distance between them
        I = nx.Graph()
        for n1,voisins in RBH_inter.items():
            for (n2,dist) in voisins :
                if not I.has_edge(n1,n2) : I.add_edge(n1,n2,weight=dist)
    print(f"Initial network composition : {len(I.nodes)} nodes and {len(I.edges)} edges")
    print("---------------")

    with open(Log_file,'a') as file : file.write(f"Initial network built : {len(I.nodes)} genes and {len(I.edges)} associations.\n")
        

    ### Deletion of candidate genes with no edges to anchor genes
    ### WARNING : Only if anchor genes are provided !!!
    if D_args['Anchor Genes lists'] != []:

        I = KNN_RBH.cleanGraph(I , anchor_lists=L_L_anchors)

        print(f"Filtered network composition : {len(I.nodes)} nodes and {len(I.edges)} edges")
        print("---------------")
        with open(Log_file,'a') as file : file.write(f"Network filtered : {len(I.nodes)} genes and {len(I.edges)} associations.\n")

    ### Graph saving
    file_name_2 = f"{folder_name}Result_2_{etape}_Graph.txt"
    KNN_RBH.saveNetwork(I , method=version , anchor_lists=L_L_anchors , color_list=L_colors , file_name=file_name_2)
    with open(Log_file,'a') as file : file.write(f"Network saved.\n")

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
    
    print(f"End of Reciprocal KNN step. Run time : {Set_fnc.time_count(time.time()-step_time)}")
    with open(Log_file,'a') as file : file.write(f"End of Step 2. Time of execution : {Set_fnc.time_count(time.time()-step_time)}\n")
    print(f"Time since launch : {Set_fnc.time_count(time.time()-start)}")
    with open(Log_file,'a') as file :
        file.write(f"Time since launch : {Set_fnc.time_count(time.time()-start)}\n")
        file.write(f"-------------------------------------------------------\n")
    print("\n--------------------------- \n")

### 3 : Network Properties Closeness (NPC)
#=====================================================================================================================================================

if 3 in L_steps :
    etape = D_steps_names[3]
    step_time = time.time()

    with open(Log_file,'a') as file : file.write(">Step 3 : Network Properties Closeness\n")
    
    ### Datasets alternative loading (with names as indices)
    L_DF_alt = []
    for arg in arguments[1::] : L_DF_alt.append(pd.read_csv(arg, index_col=0, header=0))

    ### Recovery of anchor gene indices in each dataset
    L_GeneIdx , L_L_AnchorIdx = NPC.indices_recovery(L_DF_alt , anchor_lists=L_L_anchors)

    ### Datasets normalizaton
    L_DF_Norm = []
    for df in L_DF_alt : L_DF_Norm.append(NPC.overall_normalize(df))

    ### Neighborhoods calculation : one seach for each dataset and each anchor genes list
    list_labels = L_labels
    L_L_Ranks = [[] for i in range(len(list_labels))]

    version = D_args['Distance filter version']
    T_factor = D_args['Distance threshold factor NPC']
    for a,df_norm in enumerate(L_DF_Norm) : # For each normalized dataset
        print(f"{etape} analysis on Dataset {a+1}")
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
        print("-----")
        del(D)
        
        distancesKNN = nbrsTestDist
        indicesKNN = nbrsTestIdx

        if type(T_factor) != list : factor = T_factor
        else : factor = T_factor[a]
        
        for b,L_AnchorIdx in enumerate(L_L_AnchorIdx) : # For each anchor genes list
            print(f"{list_labels[b]} list analysis")

            specialGenesIndex = L_AnchorIdx[a] # List of anchor gene index for current dataset
            MinNB = D_args['Minimum neighborhood NPC'][b] # Minumum number of neighbors of interest

            # Initial network building for property analysis
            G, adj = NPC.createGraph(indicesKNN, distancesKNN, specialGenesIndex, geneIndex)
            print(f"Network built ({len(G.nodes)} nodes & {len(G.edges)} edges)")

            # Deletion of edges with too high distances
            G = NPC.cleanGraphDistance(G, distancesKNN, factor=factor, version='Bilateral')#np.mean(distancesKNN)+2*(np.std(distancesKNN)))
            print(f"Graph nettoyé Dist ({len(G.nodes)} nodes & {len(G.edges)} edges)")

            # Deletion of nodes with with too few anchor neighbors (and their remaining edges)
            if MinNB == 'Dynamic' :
                G , MinNB = NPC.dynamicCleanGraph(G, specialGenesIndex)
                with open(Log_file,'a') as file : file.write(f"Dataset {a+1} dynamic minimum of {list_labels[b]} neighbors : {MinNB}.\n")
            else : G = NPC.cleanGraph(G, MinNB)
            print(f"Graph nettoyé Nodes ({len(G.nodes)} nodes & {len(G.edges)} edges) : {Set_fnc.time_count(time.time()-step_time)}")
            
            # Ranking matrix filling
            rankMat = NPC.rankingNodes(G, specialGenesIndex)
            print(f"Matrice de classement remplie ({len(rankMat)}) : {Set_fnc.time_count(time.time()-step_time)}")
            del(G)
            #-------------------------------------------
            to_remove = []

            for i in range(0, len(rankMat)):
                if rankMat[i][6] == 0 : to_remove.append(rankMat[i]) # Remove nodes with meanDistance = 0

            for i in range(0, len(to_remove)) : rankMat.remove(to_remove[i])
#            print(rankMat)
#            print(to_remove)
                
            #-------------------------------------------
            for i in range(0, len(rankMat)): rankMat[i][0] = df_aux.loc[int(rankMat[i][0]), 'ID_REF']  ## --> convertion of the gene number to the gene name.

            for i in range(0, len(rankMat)):
                for j in range(0, len(rankMat[i][8])):
                    #print(rankMat[i][7][j])
                    rankMat[i][8][j] = df_aux.loc[int(rankMat[i][8][j]), 'ID_REF']  ## --> convertion of the gene number to the gene name.
#            print(len(rankMat))
                
            #-------------------------------------------
            rankMat.sort(key=lambda k: (  k[1], k[2], k[3], k[4], -k[5], -k[6], -k[7]), reverse=True) ### k[1] -> k[4] descending order, k[5] -> k[7] ascending order
#            print(rankMat)
                
            #-------------------------------------------
            rankMat_str = copy.deepcopy(rankMat)
            for line in rankMat_str :
                for i,x in enumerate(line) :
                    if i == 8 : continue # listSpecialNg is not converted to str
                    if type(x) != str : line[i]=str(x)
#                print(line)
#            print(rankMat_str)
                
            #-------------------------------------------
            label = D_args['Anchor Genes labels'][b]
            file_name_3a = f"{folder_name}Result_3a_{etape}_{label}_for_Dataset_{a+1}_min{MinNB}.txt"
            L_L_Ranks[b].append(file_name_3a)
            NPC.saveRankedNodes(rankMat_str , file_name = file_name_3a)
            print("-----")
        print("----------------------------")
    
    with open(Log_file,'a') as file : file.write(f"All datasets analyzed.\n")
    
    ### Regrouping of the calculation files names in a single text file
    lists_file_name = f"{folder_name}Result_3t_{etape}_CandidateLists_file.txt"
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

    L_tabs , L_Cands , Inter_all = NPC.sortCandidates(lists_file_name , Min_redundancy = Num_dataset)

    with open(Log_file,'a') as file : file.write(f"Neighborhoods intersected.\n")

    for i,label in enumerate(L_labels):
        print(f"Inter {label} candidates : {len(L_Cands[i])}")
#        print(f"{L_Cands[i]}")
        with open(Log_file,'a') as file : file.write(f"Inter {label} candidates : {len(L_Cands[i])}.\n")
        print("\n----------------------\n")
    print(f"Candidats inter Multi-Ancres : {len(Inter_all)}")
#    print(f"{Inter_all}")
    with open(Log_file,'a') as file : file.write(f"Inter multi-label candidates : {len(Inter_all)}.\n")
    print("\n----------------------\n")

    """
    # Looking for anchor genes among candidate genes
    print("Gènes d'intérêt eux-mêmes candidats :")
    for i,inter in enumerate(L_Cands) :
        lab_1 = L_labels[i]
        for gene in inter :
            for j,L_anchor in enumerate(L_L_anchors) :
                lab_2 = L_labels[j]
                if gene in L_anchor : print(f"{gene}: {lab_2} candidat chez {lab_1}")
    print("\n----------------------\n")
    """
    
    ### Neighborhood vectors dictionary
    L_dico_V = NPC.candidateVectorsOfInterest(L_Cands , L_tabs) # Vector of links of interest per dataset for each candidate
    L_dico_N = NPC.sortSpecialNeighbors(L_Cands , L_tabs) # Dictionary of neighbors of each anchor gene sorted by occurence in different datasets

    ### Candidate genes saving
    file_name_3b = f"{folder_name}Result_3b_{etape}_CandidateGenes.txt"
    NPC.saveFinalCandidates(L_Cands , L_dico_N , L_dico_V , L_DF , L_pool , L_labels , file_name=file_name_3b)

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
                        G.add_edge(n1,n2,weight=k)
    print(f"Final network composition : {len(G.nodes)} nodes and {len(G.edges)} edges")
    print("---------------")
    
    graph_file_3c = f"{folder_name}Result_3c_{etape}_graph.txt"
    NPC.saveNetwork(G , anchor_lists=L_L_anchors , color_list=L_colors , file_name=graph_file_3c)
    with open(Log_file,'a') as file : file.write(f"Network built and saved : {len(G.nodes)} genes and {len(G.edges)} associations.\n")

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
    
    print(f"End of Network Properties Closeness step. Run time : {Set_fnc.time_count(time.time()-step_time)}")
    with open(Log_file,'a') as file : file.write(f"End of Step 3. Time of execution : {Set_fnc.time_count(time.time()-step_time)}\n")
    print(f"Time since launch : {Set_fnc.time_count(time.time()-start)}")
    with open(Log_file,'a') as file :
        file.write(f"Time since launch : {Set_fnc.time_count(time.time()-start)}\n")
        file.write(f"-------------------------------------------------------\n")
    print("\n--------------------------- \n")

### 4 : Cluster Path
#=====================================================================================================================================================

if 4 in L_steps :
    etape = D_steps_names[4]
    print(f"Analyse {etape}\n")
    
    step_time = time.time()

    with open(Log_file,'a') as file : file.write(">Step 4 : Cluster Path\n")

    nb_clust = D_args['Number of clusters']

    LD_commonPath = []
    for df in L_DF : # For each dataset
        L_colonnes = list(df.keys())[1::]
        n_tp = len(L_colonnes)
        
        # Clustering calculation
        L_dico = []
        for i in range(n_tp): L_dico.append(CP.KMeans_1D(df , i , n_clusters=nb_clust))
        
        # Cluster pathing
        D_genePath = CP.clusterPathing(L_dico)

        # Regrouping genes by common cluster path
        D_commonPath = CP.commonPath(D_genePath)
        LD_commonPath.append(D_commonPath)

    with open(Log_file,'a') as file : file.write(f"All datasets analyzed.\n")

    # Regrouping the paths from all datasets for each gene
    D_geneRegroupedPaths = CP.regroupPaths(LD_commonPath , L_pool)

    # Regrouping genes by common concatenated cluster path
    D_groupes = CP.concatenatePath(D_geneRegroupedPaths)

    # Saving the paths and the genes that follow them
    file_name_4a = f"{folder_name}Result_4a_{etape}_list.txt"
    CP.savePaths(D_groupes , anchor_lists=L_L_anchors , label_list=L_labels , file_name=file_name_4a)
    with open(Log_file,'a') as file : file.write(f"Common path gene lists saved : {len(D_groupes)} paths found.\n")

    # Building the neighborhood network and saving it
    G = nx.Graph()
    for path in sorted(D_groupes, key=lambda path: len(D_groupes[path]), reverse=True):
        groupe = D_groupes[path]
        for i,n1 in enumerate(groupe) :
            if n1 not in list(G.nodes) : G.add_node(n1)
            for j,n2 in enumerate(groupe[i+1::]):
                if n2 not in list(G.nodes) : G.add_node(n2)
                G.add_edge(n1,n2,path=path)
    print(f"Final network composition : {len(G.nodes)} nodes and {len(G.edges)} edges")
    print("---------------")

    graph_file_4b = f"{folder_name}Result_4b_{etape}_graph.txt"
    CP.saveNetwork(G , anchor_lists=L_L_anchors , color_list=L_colors , file_name=graph_file_4b)

    with open(Log_file,'a') as file : file.write(f"Network built and saved : {len(G.nodes)} genes and {len(G.edges)} associations.\n")
    
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
    
    print(f"End of Cluster Path step. Run time : {Set_fnc.time_count(time.time()-step_time)}")
    with open(Log_file,'a') as file : file.write(f"End of Step 4. Time of execution : {Set_fnc.time_count(time.time()-step_time)}\n")
    print(f"Time since launch : {Set_fnc.time_count(time.time()-start)}")
    with open(Log_file,'a') as file :
        file.write(f"Time since launch : {Set_fnc.time_count(time.time()-start)}\n")
        file.write(f"-------------------------------------------------------\n")
    print("\n--------------------------- \n")
    
### END : Methods Consensus
#=====================================================================================================================================================

if consensus :
    print(f"Consensus\n")
    step_time = time.time()

    with open(Log_file,'a') as file : file.write(">Final Step : Methods Consensus\n")
    
    L_niv = D_args['Consensus levels list'] # List of consensus level to keep for the text files

    ### Creation of the candidate sorting dictionary (sorted by redundancy of association)
    D_repartition , D_noms , D_vecteurs = Cons_fnc.assoc_redundancy(D_assoc , D_steps_names , L_niv)
#    print(D_repartition)

    ### Saving the numeral data for each anchor gene : the data are sorted by descening values of successive high consensus level
    tab_file = f"{folder_name}Result_END_consensus.csv"
    sorted_DF = Cons_fnc.saveAnchorConsensusData(D_repartition , file_name=tab_file)

    ### Saving the lists of candidates associated to each anchor genes : candidates are sorted by their consensus level
    file_name_END_1 = f"{folder_name}Result_END_GenesOfInterest_ListByAnchors.txt"
    Cons_fnc.saveResultByAnchor(sorted_DF , L_niv , D_noms , anchor_lists=L_L_anchors ,
                                label_list=L_labels , file_name=file_name_END_1)

    with open(Log_file,'a') as file : file.write(f"Anchor-ordered results saved.\n")

    ### Saving the data of all candidate-anchor associations 
    file_name_END_2 = f"{folder_name}Result_END_GenesOfInterest_ListByCandidates.txt"
    Cons_fnc.saveResultByCandidates(sorted_DF , L_niv , D_steps_names , D_noms , D_vecteurs ,
                                    anchor_lists=L_L_anchors , label_list=L_labels , file_name=file_name_END_2)

    with open(Log_file,'a') as file : file.write(f"Candidate-ordered results saved.\n")

    ### Building and Saving the final network
    G , L_anchors , L_candidates , D_anchors = Cons_fnc.buildConsensusNetwork(D_assoc , D_steps_names)

    if L_L_anchors != [] : # If anchors have been provided
        print(f"Total genes : {len(G.nodes)} / {len(L_anchors)} anchors / {len(L_candidates)} candidates")
    else : # If no anchor has been provided
        print(f"Total genes : {len(G.nodes)}")
    print(f"Total edges : {len(G.edges)}")
    print("----------------------------")
    with open(Log_file,'a') as file : file.write(f"Consensus network built : {len(G.nodes)} genes and {len(G.edges)} associations.\n")

    graph_file_END = f"{folder_name}Result_END_graph.txt"
    Cons_fnc.saveNetwork(G , L_candidates , D_anchors , anchor_lists=L_L_anchors ,
                         label_list=L_labels , color_list=L_colors , file_name=graph_file_END)

    with open(Log_file,'a') as file : file.write(f"Consensus network saved.\n")
    print(f"End of Consensus step. Run time : {Set_fnc.time_count(time.time()-step_time)}")
    with open(Log_file,'a') as file : file.write(f"End of Final Step. Time of execution : {Set_fnc.time_count(time.time()-step_time)}\n")
    print(f"Time since launch : {Set_fnc.time_count(time.time()-start)}")
    with open(Log_file,'a') as file : file.write(f"Time since launch : {Set_fnc.time_count(time.time()-start)}\n")

