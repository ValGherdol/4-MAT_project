import time
import numpy as np
import pandas as pd
import networkx as nx

#=====================================================================
#=====================================================================

# List of the functions defined in this file :
#   - buildPearsonCoExpressionIntersectedNetwork
#   - saveNetwork
#   - saveNetworkData
#   - loadPearsonCoExpressionIntersectedNetwork
#   - dynamic_filtering
#   - neighborhood_filtering

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

def buildPearsonCoExpressionIntersectedNetwork(dataframe_list , gene_list=None , anchor_list=None ,
                                               threshold_Pearson=0.5 , search_tps=False):
    """
    From a list of at least one dataframe, generates a network of coexpressed data using Pearson's Correlation Coefficient (PCC).
    For each couple of genes in the dataframes, the genes are linked by an edge if their PCC is higher or equal to a threshold in absolute value.
    If several dataframes are provided, all of them must give a PCC that checks the threshold and be of the same sign for the genes to be linked.
    
    input1 : A list of Pandas dataframe.
        All of them must contain a column called 'ID_REF' and these columns must contain the same list of genes but necessary in the same order.
    input2 : A list of genes (default value = None).
        If provided, the function will only look at these genes.
        If not, it will look at all genes from the first dataframe.
    input3 : A list of anchor genes (default value = None).
        If provided, the function will only calculate the PCCs for couples with at least one of these genes.
        If not, it will calculate the PCCs for all couples.
    input4 : A numeral value (default value = 0.5) that filters out any PCC with a lower absolute value.
        Any value lower or equal to 0 will make the function keep all calculated PCCs.
        Any value higher or equal to 1 will make the function filter out all calculated PCCs.
    input5 : A Boolean value (default value = False).
        If set to True, the function will write on the console the time passed to analyze each gene, as well as the time passed since the function's call every 1000 genes.

    output1 : The co-expression network.
    """

    start = time.time()
    
    ### Récupération de l'entièreté des points en cas de liste d'étude non fournie
    if gene_list == None : gene_list = list(dataframe_list[0]["ID_REF"])
#    print(gene_list)
    
    ### Récupération des vecteurs de valeurs de chaque points dans les datasets et ordonnancement unique
    L_L_values = []
    for df in dataframe_list :
        L_colonnes = list(df.keys())[1::]
        L_values = []
        for n in gene_list :
            row = df[df['ID_REF']==n]
            values = []
            for col in L_colonnes : values.append(float(row[col]))
            L_values.append(values)
        L_L_values.append(L_values)
        
    print(f"Data loaded.")
    print(f"Time since launch : {time_count(time.time()-start)}")
    print("------------------------------")
    
    G = nx.Graph()

    ### Génération initiale du réseau + récupération des corrélations moyennes retenues si palier dynamique demandé

    if anchor_list == None : # S'il n'y a pas de liste de points d'intérêt <=> on calcule les corrélations de tous les couples possibles.
        for i,n1 in enumerate(gene_list): # Pour chaque point de la liste générale
            if not G.has_node(n1): G.add_node(n1)
            t_node = time.time()
            for j,n2 in enumerate(gene_list[i+1::]): # Pour chaque point suivant de la liste générale
                pearson = 0
                skip = False
                for k,L_values in enumerate(L_L_values) : # Pour chaque datasets
                    Lv_1 = L_values[i]
                    Lv_2 = L_values[i+j+1]
    #                print(n1,Lv_1)
    #                print(n2,Lv_2)
                    p = np.corrcoef(Lv_1,Lv_2)[0][1]
                    if type(threshold_Pearson) == float : # S'il y a un unique palier de sélection à prendre en compte pour tous les datasets
    #                    print(p,'/',threshold_Pearson)
                        if abs(p) < threshold_Pearson : # Si une corrélation est sous le palier, on oublie l'arête
                            skip = True
                            break
                        elif pearson == 0 : pearson += p # Si c'est la première corrélation valide calculée, on la retient
                        
                        elif np.sign(p) != np.sign(pearson) : # Si l'une des corrélation valide suivante n'est pas dans le même sens, on oublie l'arête
                            skip = True
                            break
                        else : pearson += p # Sinon, on l'ajoute à/aux corrélation(s) retenue(s)
                    elif type(threshold_Pearson) == list : # S'il y a différents paliers de sélection à prendre en compte d'un dataset à l'autre
    #                    print(p,'/',threshold_Pearson[k])
                        if abs(p) < threshold_Pearson[k] : # Si une corrélation est sous le palier de son dataset, on oublie l'arête
                            skip = True
                            break
                        elif pearson == 0 : pearson += p # Si c'est la première corrélation valide calculée, on la retient
                        
                        elif np.sign(p) != np.sign(pearson) : # Si l'une des corrélation valide suivante n'est pas dans le même sens, on oublie l'arête
                            skip = True
                            break
                        else : pearson += p # Sinon, on l'ajoute à/aux corrélation(s) retenue(s)
                if skip == True : 
    #                print()
                    continue
    #            print('OK :',pearson)
                if pearson != 0 :
                    pearson = pearson/len(dataframe_list)
    #                print("pearson moyen :",pearson)
                    G.add_edge(n1,n2)
                    L_attr = {'weight':abs(pearson) , 'sign':np.sign(pearson)}
                    if np.sign(pearson) == 1 : L_attr['color'] = (0 , 0.5 , 0)
                    elif np.sign(pearson) == -1 : L_attr['color'] = (1 , 0 , 0)
                    G[n1][n2].update(L_attr)
    #            print("\n--------------------\n")
            if search_tps :
                if (i+1)%1000 == 0 : print(f"Gene {n1} processed in {time_count(time.time()-t_node)} ({i+1}). Time since launch : {time_count(time.time()-start)}")
                else : print(f"Gene {n1} processed in {time_count(time.time()-t_node)} ({i+1}/{len(gene_list)})")

    else : # S'il y a une liste de points d'intérêt <=> on ne calcule que les corrélations des couples qui incluent au moins un point d'intérêt
        for i,n1 in enumerate(anchor_list): # Pour chaque point de la liste d'intérêt
            if not G.has_node(n1): G.add_node(n1)
            t_node = time.time()
            for j,n2 in enumerate(gene_list): # Pour chaque point de la liste générale
                if n1 == n2 : continue # Un point ne s'analyse pas lui-même
                if (n1,n2) in G.edges : continue # On ignore les couples à deux points d'intérêt déjà retenus
                pearson = 0
                skip = False
                for k,L_values in enumerate(L_L_values) : # Pour chaque datasets
                    Lv_1 = L_values[gene_list.index(n1)]
                    Lv_2 = L_values[gene_list.index(n2)]
    #                print(n1,Lv_1)
    #                print(n2,Lv_2)
                    p = np.corrcoef(Lv_1,Lv_2)[0][1]
                    if type(threshold_Pearson) == float : # S'il y a un unique palier de sélection à prendre en compte pour tous les datasets
    #                    print(p,'/',threshold_Pearson)
                        if abs(p) < threshold_Pearson : # Si une corrélation est sous le palier, on oublie l'arête
                            skip = True
                            break
                        elif pearson == 0 : pearson += p # Si c'est la première corrélation valide calculée, on la retient
                        
                        elif np.sign(p) != np.sign(pearson) : # Si l'une des corrélation valide suivante n'est pas dans le même sens, on oublie l'arête
                            skip = True
                            break
                        else : pearson += p # Sinon, on l'ajoute à/aux corrélation(s) retenue(s)
                    elif type(threshold_Pearson) == list : # S'il y a différents paliers de sélection à prendre en compte d'un dataset à l'autre
    #                    print(p,'/',threshold_Pearson[k])
                        if abs(p) < threshold_Pearson[k] : # Si une corrélation est sous le palier de son dataset, on oublie l'arête
                            skip = True
                            break
                        elif pearson == 0 : pearson += p # Si c'est la première corrélation valide calculée, on la retient
                        
                        elif np.sign(p) != np.sign(pearson) : # Si l'une des corrélation valide suivante n'est pas dans le même sens, on oublie l'arête
                            skip = True
                            break
                        else : pearson += p # Sinon, on l'ajoute à/aux corrélation(s) retenue(s)
                if skip == True : 
    #                print()
                    continue
    #            print('OK :',pearson)
                if pearson != 0 :
                    pearson = pearson/len(dataframe_list)
    #                print("pearson moyen :",pearson)
                    G.add_edge(n1,n2)
                    L_attr = {'weight':abs(pearson) , 'sign':np.sign(pearson)}
                    if np.sign(pearson) == 1 : L_attr['color'] = (0 , 0.5 , 0)
                    elif np.sign(pearson) == -1 : L_attr['color'] = (1 , 0 , 0)
                    G[n1][n2].update(L_attr)
    #            print("\n--------------------\n")
            if search_tps :
                if (i+1)%1000 == 0 : print(f"Gene {n1} processed in {time_count(time.time()-t_node)} ({i+1}). Time since launch : {time_count(time.time()-start)}")
                else : print(f"Gene {n1} processed in {time_count(time.time()-t_node)} ({i+1}/{len(anchor_list)})")

    return G
    
#=====================================================================

def saveNetwork(graph , anchor_lists=[] , color_list=[] , file_name="PCEIN_Network.txt") :

    """
    Writes a text file with the content of a network.

    input1 : A co-expression network in the form of a Networkx graph.
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

def saveNetworkData(graph , L_anchors_labels=[] , L_anchors_list=[] , dataframe_file='Data_PearsonGraph.csv' , annonce=None) :
    """
    Saves the general data of each node from a co-expression network.

    input1 : A co-expression network in the form of a Networkx graph.
    input2 : A list of gene labels (dafault value = an empty list).
        If provided, the function will label some genes with the corresponding value and the rest of the genes with the word 'Candidate'.
        If not, all genes will be labeled 'Candidate'.
    input3 : A list of lists of genes (default value = an empty list).
        If provided, the function will associate the label from input2 to the genes contained in the corresponding list.
        THERE MUST BE AS MANY LISTS OF GENES AS THERE ARE LABELS. IF A GENE IS IN SEVERAL LISTS, IT WILL ALWAYS BE GIVEN THE LABEL OF THE FIRST LIST ITS ENCOUNTERED IN.
    input4 : A name for the created file (default value = 'Data_PearsonGraph.csv').
        If customized, make sure your file hase a csv extension.
    input5 : A Integer value (default value = None).
        If customized, the function will write on the console the number of genes saved every time this number is a multiple of the input value.
        
    output1 : a Pandas dataframe with the following data for each gene :
        - the gene's name (column 'ID_REF')
        - the gene's label (column 'Label')
        - the gene's degree, a.k.a it's number of neighbors linked by an edge (column 'Nb_Edges')
        - the gene's number of positive correlations (column 'Nb_Positives')
        - the gene's number of negative correlations (column 'Nb_Negatives')
        - the gene's average Pearson's Correlation Coefficient in absolute value (column 'Pearson_Mean')
        - the gene's standard deviation of PCCs in absolute value (column 'Pearson_Std')
        - the gene's maximum PCC in absolute value (column 'Pearson_Max')
        - the gene's minimum PCC in absolute value (column 'Pearson_Min')
    """

    L_nodes = list(graph.nodes)
    Data = {'ID_REF':[] , 'Label':[] ,
            'Nb_Anchor_Neighbors':[],'Nb_Edges':[] ,
            'Nb_Positives':[] , 'Nb_Negatives':[] ,
            'Pearson_Mean':[] , 'Pearson_Std' :[] ,
            'Pearson_Max':[] , 'Pearson_Min':[]}
    for i,n1 in enumerate(L_nodes):
        LN_anchors = [0 for i in range(len(L_anchors_labels))]
        L_Pearson = []
        nb_pos,nb_neg = 0,0
        for n2,attr in graph[n1].items():
            p = list(attr.values())[0]
            sens = list(attr.values())[1]
            if sens > 0 : nb_pos += 1
            else : nb_neg += 1
            L_Pearson.append(float(p))
            for j,L_anchors in enumerate(L_anchors_list) :
                if n2 in L_anchors :
                    LN_anchors[j] += 1
                    break
        n = len(L_Pearson)
        if n == 0 : m,s,P_max,P_min = 0,0,0,0
        else :
            m = np.mean(np.array(L_Pearson))
            s = np.std(np.array(L_Pearson))
            P_max = max(L_Pearson)
            P_min = min(L_Pearson)

        Data['ID_REF'].append(n1)
        Data['Nb_Anchor_Neighbors'].append(LN_anchors)
        Data['Nb_Edges'].append(n)
        Data['Nb_Positives'].append(nb_pos)
        Data['Nb_Negatives'].append(nb_neg)
        Data['Pearson_Mean'].append(m)
        Data['Pearson_Std'].append(s)
        Data['Pearson_Max'].append(P_max)
        Data['Pearson_Min'].append(P_min)

        cand = True
        for j,L_anchors in enumerate(L_anchors_list) :
            if n1 in L_anchors :
                Data['Label'].append(L_anchors_labels[j])
                cand = False
                break
        if cand : Data['Label'].append('Candidate')
        
        if annonce != None :
            if i%annonce == 0 : print(f"{i} points analysés")
    
    df_graph = pd.DataFrame(Data)
    
    df_graph.to_csv(dataframe_file , index=False)
    
    return df_graph

#=====================================================================

def loadPearsonCoExpressionIntersectedNetwork(file_name , annonce=None):
    """
    Load a Networkx graph from a text file.
    The file must be organized as follow :
        - The nodes are indicated at the beginning of the file. Each line starts with a '>' followed by the node's name and it's color in an RGB format.
        - The edges are indicated after the nodes. Each line indicates two genes' names and the attributes of the edges.

    input1 : a text file that contains a graph.
    input2 : an Integer value (default value = None).
        If customized, the function will write on the console the number of edges loaded every time this number is a multiple of the input value.

    output1 : the loaded graph
    """
    
    graph = nx.Graph()
    
    with open(file_name,'r') as file :
        for line in file :
#            print(line[0:-1])
            if line[0]=='>': 
                graph.add_node(line[1:-1])
                if annonce != None :
                    i = len(graph.nodes)
                    if i%annonce == 0 and i != 0 : print(f"{i} points ont été chargées")
                continue
            
            data = line[0:-1].split(" ; ")
#            print(data)
            name_1,name_2 = data.pop(0),data.pop(0)
            L_keys,L_vals = [],[]
#            print(data)
            for d in data :
#                print(d,'\n')
                key,val = d.split(':')
                L_keys.append(key)
                try : L_vals.append(float(val))
                except ValueError : L_vals.append(val)
            L_attr = dict(zip(L_keys,L_vals))
            
            if 'color' not in L_keys :
                if L_attr['sign'] < 0 : L_attr['color'] = (1 , 0 , 0)
                else : L_attr['color'] = (0 , 0.5 , 0)
            
            graph.add_edge(name_1,name_2)
            graph[name_1][name_2].update(L_attr)
            
            if annonce != None :
                j = len(graph.edges)
                if j%annonce == 0 and j != 0 : print(f"{j} arêtes ont été chargées")
    
    print(f"Graph à {len(graph.nodes)} points et {len(graph.edges)} arêtes chargé")
    return graph

#=====================================================================

def dynamic_filtering(graph , factor=1):

    """
    Filters a graph via a dynamic threshold calculated through the edges' weight.
    Any edge with a weigth lower than the calculated threshold is removed.
    Once all filtered edges have been removed, all nodes with no edges are removes as well.

    input1 : A co-expression network in the form of a Networkx graph.
    input2 : A Float value (default value = 1).
        Used for the dynamic threshold calcuation.

    output1 : The filtered network.
    output2 : The value of the dynamic threshold.
    """

    L_corr = []
    for (n1,n2) in graph.edges: L_corr.append(graph[n1][n2]['weight'])

    # Dynamic threshold calculation
    mean = np.mean(L_corr)
    std = np.std(L_corr)
    dynamic = round(mean + (factor * std),2)

    # Deletion of edges with weight lower than the dynamic threshold
    L_delete = []
    for (n1,n2) in graph.edges :
        if graph[n1][n2]['weight'] < dynamic : L_delete.append((n1,n2))
    for (n1,n2) in L_delete : graph.remove_edge(n1,n2)
        
    # Deletion of points with no edges
    L_delete = []
    for n1 in graph.nodes :
        if graph[n1] == {} : L_delete.append(n1)
    for n1 in L_delete : graph.remove_node(n1)

    return graph , dynamic

#=====================================================================

def neighborhood_filtering(graph , anchor_lists=[] , limite_list=[]):

    """
    Removes from a graph any non-special node that doesn't have enough neighbors of interest (i.e. anchor neighbors).

    input1 : A co-expression network in the form of a Networkx graph.
    input2 : A list of lists of special nodes (anchors).
    input3 : A list of Integer values (one value for each list of special nodes provided in input2).

    output1 : The filtered network.
    """

    L_delete = []
    for n1 in graph.nodes :
        cand = True # We assume the current node is initialy not a special node.

        # We look for the current node in all anchor lists
        for i,L_anchors in enumerate(anchor_lists):
            if n1 in L_anchors :
                cand = False
                break

        # If the current node is indeed not a special node, we count how many neighbors from each anchor list it has.
        if cand : 
            N_voisins = [0 for i in range(len(anchor_lists))]
            
            for n2 in graph[n1] : # For each neighbor, we look for it in all anchor lists.
                for i,L_anchors in enumerate(anchor_lists):
                    if n2 in L_anchors :
                        N_voisins[i] += 1
                        break
                    
            out = True # We assume the current node has initialy not enough special neighbors.
            for i,limite in enumerate(limite_list):
                if N_voisins[i]>=limite : # If the current node has the minimum number of neighbors required in at least one anchor list, we keep it.
                    out = False
                    break
            if out : L_delete.append(n1)

    for n in L_delete : RICEP.remove_node(n)

    return graph

#=====================================================================
