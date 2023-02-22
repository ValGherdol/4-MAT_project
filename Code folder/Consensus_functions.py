import copy
import pandas as pd
import networkx as nx

#=====================================================================
#=====================================================================

# Liste of the functions defined in this file :
#   - assoc_redundancy
#   - saveAnchorConsensusData
#   - saveResultByAnchor
#   - saveResultByCandidates
#   - buildConsensusNetwork
#   - saveNetwork

#=====================================================================
#=====================================================================

def assoc_redundancy(association_dict , step_dict , level_list):

    """
    Sums up for each anchor gene the numeral values of the consensus of the methods.

    input1 : The linkage dictionary with anchor genes as keys and sub-dictionaries as values.
        All sub-dictionaries have the method names as keys and lists of genes as values.
    input2 : A dictionary with method names as keys and Integers as values.
    input3 : A list of Integers

    output1 : A dictionary with the following content :
        - An 'ID_REF' key with a list of anchor gene names as value.
        - A 'Class' key with a list of anchor labels as value.
        - A '#Unique' key with a list of Integers as value.
            Indicates the number of unique candidates across all methods.
        - Four '#k-Method(s)' keys (with k going from 1 to 4) with lists of Integers as values.
            Indicate the number of unique candidates found by exactly k methods.
        - 5 '[method name]' keys with lists of Integers as values.
            Indicates the number of unique candidates found by the corresponding method.
            NOTE : The Pearson method is split in two keys (one labeled 'Pos', the other 'Neg') to separate the positive and negative correlations.
    output2 : A dictionary with anchor gene names as keys and sub-dictionaries as values.
        Each sub-dictionary has the following content :
            - A 'class' key with an anchor label as value.
            - Four 'k' keys (with k going from 1 to 4) with list gene names as values.
                Indicate the number of unique candidates found by exactly k methods.
    output3 : A dictionary with anchor names as keys and sub-dictionaries as values.
        Each sub-dictionary has candidate names as keys and vector of Integers as values.
        The vectors indicates which method(s) has found the association between the anchor key and the candidate key.
    """

    D_redundancy = {'ID_REF':[],'Class':[],'#Unique':[]}
    a = list(association_dict.keys())[0] # First key in the Result dictionary (i.e. a gene)
    Num_methods = 0 # Number of methods used to fill the Result dictionary
    L_n_method = [] # List of the numbers of methods
    for i in range(len(list(association_dict[a].keys()))):
        if i == 0 :
            D_redundancy[f"#{i+1}-Method"] = []
            L_n_method.append(f"#{i+1}-Method")
        else :
            D_redundancy[f"#{i+1}-Methods"] = []
            L_n_method.append(f"#{i+1}-Methods")
        Num_methods += 1

    L_methods = [] # List of methods used to fill the Result dictionary
    for methode in list(association_dict[a].keys()):
        L_methods.append(methode)
        if methode == step_dict[1]: # Pearson Co-Epxression : sort Positive and Negative values
            D_redundancy[f"{methode} Pos"] = []
            D_redundancy[f"{methode} Neg"] = []
        elif methode == step_dict[3]: # Network Properties : sort association by dataset redundancy
            for i in range(len(association_dict[a][methode])): D_redundancy[f"{methode}-{i+1}"] = []
        else: # KNN-RBH or Cluster Path : just get the associated genes
            D_redundancy[methode] = []

    D_noms = {}
    D_vecteurs = {}
    for gene,voisinages in association_dict.items(): # For each gene that act as a key in the Result dictionary (a.k.a anchor genes)
        # Get the current gene's data
        ref,clas = gene
        D_redundancy['ID_REF'].append(ref)
        D_redundancy['Class'].append(clas)
        D_noms[ref] = {'class':clas}
        D_vecteurs[ref] = {}
        for i in reversed(range(Num_methods)) : D_noms[ref][i+1] = []
        
        # Cound the unique associated genes (a.k.a neighbors)
        L_neig = []
        L_PCEIN_pos , L_PCEIN_neg = [],[]
        L_KNN = []
        for methode,L_voisins in voisinages.items():
            if methode != step_dict[3] : # Pearson Co-Expression, KNN-RBH or Cluster Path
                if methode == step_dict[4] : L_neig += L_voisins # Cluster Path
                else : # Pearson Co-Expression, KNN-RBH
                    for (n,c) in L_voisins :
                        L_neig.append(n)
                        if methode == step_dict[1] : # Pearson Co-Expression
                            if c > 0 : L_PCEIN_pos.append(n)
                            else : L_PCEIN_neg.append(n)
                        else : L_KNN.append(n) # KNN-RBH
            else : # Network Properties
                for i,rank in enumerate(L_voisins) : L_neig += rank
        Set_neig = set(L_neig)
        Diff = len(L_neig)-len(Set_neig)
        D_redundancy['#Unique'].append(len(Set_neig))

        
        # Neighbors sorting (number of methods, localisation)
        sous_D = {}
        for m in L_n_method : sous_D[m] = 0
        for m in L_methods :
            if m == step_dict[1] : # Pearson Co-Expression
                sous_D[f"{m} Pos"] = 0
                sous_D[f"{m} Neg"] = 0
            elif m == step_dict[3] : # Network Properties
                for n in range(len(voisinages[m])) : sous_D[f"{m}-{n+1}"] = 0
            else : sous_D[m] = 0
            
        for name in Set_neig :
            
            D_vecteurs[ref][name] = ['0','0','0','0','0']
            N_meth = 0
            if step_dict[1] in L_methods : # Pearson Co-Expression
                if name in L_PCEIN_pos :
                    N_meth += 1
                    sous_D[f"{step_dict[1]} Pos"] += 1
                    D_vecteurs[ref][name][0] = '1'
                elif name in L_PCEIN_neg :
                    N_meth += 1
                    sous_D[f"{step_dict[1]} Neg"] += 1
                    D_vecteurs[ref][name][1] = '1'
            if (step_dict[2] in L_methods) and (name in L_KNN) : # KNN-RBH
                N_meth += 1
                sous_D[step_dict[2]] += 1
                D_vecteurs[ref][name][2] = '1'
            if step_dict[3] in L_methods : # Network Properties
                for n in range(len(voisinages[step_dict[3]])):
                    if name in voisinages[step_dict[3]][n] :
                        N_meth += 1
                        sous_D[f"{step_dict[3]}-{n+1}"] += 1
                        D_vecteurs[ref][name][3] = '1'
                        break
            if (step_dict[4] in L_methods) and (name in voisinages[step_dict[4]]) : # Cluster Path
                N_meth += 1
                sous_D[step_dict[4]] += 1
                D_vecteurs[ref][name][4] = '1'
            
            if N_meth == 1 : sous_D['#1-Method'] += 1
            if N_meth == 2 : sous_D['#2-Methods'] += 1
            if N_meth == 3 : sous_D['#3-Methods'] += 1
            if N_meth == 4 : sous_D['#4-Methods'] += 1
            if N_meth in level_list : D_noms[ref][N_meth].append(name)
#        print(sous_D)
        for k,v in sous_D.items(): D_redundancy[k].append(v)
#        print("-------------")

    return D_redundancy , D_noms , D_vecteurs

#=====================================================================

def saveAnchorConsensusData(D_redundancy , file_name="AnchorConsensus.csv"):

    """
    """

    DF_redundancy = pd.DataFrame.from_dict(D_redundancy)
    L_sorting = ['#4-Methods','#3-Methods','#2-Methods','#1-Method']
    L_delete = []
    for s in L_sorting :
        if s not in list(D_redundancy.keys()): L_delete.append(s)
    for s in L_delete : L_sorting.remove(s)
    asc = tuple([0 for i in range(len(L_sorting))])
    sorted_DF = DF_redundancy.sort_values(L_sorting , ascending = asc)
    sorted_DF.to_csv(file_name , index=False)

    return sorted_DF
#=====================================================================

def saveResultByAnchor(consensus_df , level_list , D_names , anchor_lists=[] , label_list=[] , file_name="ResultByAnchors.txt"):

    """
    """
    
    with open(file_name,'w') as file:
        L_names = list(consensus_df['ID_REF'])
        D_n = {}
        for niv in reversed(level_list) : D_n[niv] = 0
        for anchor in L_names :
            D_meth = D_names[anchor]
            classe = D_meth['class']
            line = f">{anchor} ({classe})\n"
            file.write(line)
            for niv,L_noms in D_meth.items():
                if niv == 'class' or niv not in level_list : continue
                line = f"Lv.{niv} :"
                if len(L_noms)==0 : line += ' None\n'
                else :
                    D_n[niv] += len(L_noms)
                    for name in L_noms :
                        ancre = False
                        for i,L_anchors in enumerate(anchor_lists):
                            lab = label_list[i]
                            if name in L_anchors :
                                line += f" {name}({lab[0]})"
                                ancre = True
                                break
                        if not ancre : line += f" {name}"
                    line += '\n'
                file.write(line)
        file.write("----------------------------------------------------------------------\n")
        for niv,num in D_n.items(): file.write(f"> Total number of Lv.{niv} : {num}\n")

#=====================================================================

def saveResultByCandidates(consensus_df , level_list , D_steps_names , D_names , D_vectors ,
                           anchor_lists=[] , label_list=[] , file_name="ResultByCandidates.txt"):

    """
    """

    with open(file_name,'w') as file :
        if anchor_lists != [] :line = '\t'.join(['Candidate','Anchor','Anchor Function','Consensus Level',
                                                 f"{D_steps_names[1]} Pos",f"{D_steps_names[1]} Neg",
                                                 D_steps_names[2],D_steps_names[3],D_steps_names[4]])+'\n'
        else : line = '\t'.join(['Candidate 1','Candidate 2','Consensus Level',
                                 f"{D_steps_names[1]} Pos",f"{D_steps_names[1]} Neg",
                                 D_steps_names[2],D_steps_names[3],D_steps_names[4]])+'\n'
        file.write(line)
        L_names = list(consensus_df['ID_REF'])
        D_n = {}
        for niv in reversed(level_list) : D_n[niv] = 0
        for anchor in L_names :
            D_meth = D_names[anchor]
            classe = D_meth['class']
            for niv,L_noms in D_meth.items():
                if (niv == 'class') or (niv not in level_list) or (len(L_noms)==0) : continue
                for name in L_noms :
                    line = ''
                    vecteur = D_vectors[anchor][name]
                    ancre = False
                    for i,L_anchors in enumerate(anchor_lists) :
                        if name in L_anchors :
                            line += f"{name}({label_list[i][0]})"
                            ancre = True
                            break
                    if not ancre : line += f"{name}"
                    
                    if anchor_lists != [] : line += '\t'+'\t'.join([anchor,classe,str(niv)])+'\t'+'\t'.join(vecteur)+'\n'
                    else : line += '\t'+'\t'.join([anchor,str(niv)])+'\t'+'\t'.join(vecteur)+'\n'
                    file.write(line)

#=====================================================================
def buildConsensusNetwork(association_dict , D_steps_names):

    """
    """

    # Preparing the edges' attributes : consensus level and the result from each method
    D_attr = {"Consensus":0}
    a = list(association_dict.keys())[0]
    L_steps = list(association_dict[a].keys())
#    print(L_steps)
    for s in L_steps : D_attr[s] = "NO"
#    print(D_attr)

    # Building the network
    G = nx.Graph()
    D_anchors = {}
    L_anchors , L_candidates = [] , []
    for gene,L_meth in association_dict.items(): # Processing the anchor genes first
        n1,category = gene
        G.add_node(n1) # Anchor gene added to the network
        L_anchors.append(n1) # Saving the gene in an general anchor-exclusive list

        # Saving the gene in a anchor-exclusive dictionary where anchors are sorted by anchor categories (a.k.a labels)
        if category not in list(D_anchors.keys()): D_anchors[category] = [n1]
        else : D_anchors[category].append(n1)

    for gene,L_meth in association_dict.items(): # Processing the candidates for each anchor gene
        n1,category = gene
        for methode,voisins in L_meth.items(): # For each method
            if methode in [D_steps_names[1] , D_steps_names[2]]: # Pearson Co-Expression or KNN-RBH : each element of 'voisins' is a tuple of a candidate and a float value
                for v in voisins :
                    n2,c = v[0],round(v[1],4)
                    if (n2 not in L_candidates) and (n2 not in L_anchors) : # If the candidate hasn't been encountered yet and isn't an anchor gene either, it is saved in a candidate-exclusive list
                        L_candidates.append(n2)
                    if (n1,n2) not in G.edges : # If the current anchor and candidate genes are not already linked, an edge is created with the initial state of the attributes
                        G.add_edge(n1 , n2)
                        G[n1][n2].update(copy.deepcopy(D_attr))
                    
                    if G[n1][n2][methode] == 'NO': # Since the candidate is associated to the current anchor, the attributes are updated if not already
                        G[n1][n2][methode] = c
                        G[n1][n2]['Consensus'] += 1
                    
            elif methode == D_steps_names[3] : # Network Properties (Gabriel) : 'voisins' is composed of several lists of candidates.
                for rank,L_v in enumerate(voisins) : # For each sub-list of candidates with the same dataset-redundancy rank
                    for n2 in L_v :
                        if (n2 not in L_candidates) and (n2 not in L_anchors) : # If the candidate hasn't been encountered yet and isn't an anchor gene either, it is saved in a candidate-exclusive list
                            L_candidates.append(n2)
                        if (n1,n2) not in G.edges : # If the current anchor and candidate genes are not already linked, an edge is created with the initial state of the attributes
                            G.add_edge(n1 , n2)
                            G[n1][n2].update(copy.deepcopy(D_attr))
                        
                        if G[n1][n2][D_steps_names[3]] == 'NO': # Since the candidate is associated to the current anchor, the attributes are updated if not already
                            G[n1][n2][D_steps_names[3]] = rank+1 # The +1 is necessary, as the 'rank' variables start from 0
                            G[n1][n2]['Consensus'] += 1
            else : # Cluster Path : 'voisins' is just a list of candidates
                for n2 in voisins :
                    if (n2 not in L_candidates) and (n2 not in L_anchors) : # If the candidate hasn't been encountered yet and isn't an anchor gene either, it is saved in a candidate-exclusive list
                            L_candidates.append(n2)
                    if (n1,n2) not in G.edges : # If the candidate hasn't been encountered yet and isn't an anchor gene either, it is saved in a candidate-exclusive list
                        G.add_edge(n1 , n2)
                        G[n1][n2].update(copy.deepcopy(D_attr))
                    
                    if G[n1][n2][D_steps_names[4]] == 'NO': # Since the candidate is associated to the current anchor, the attributes are updated if not already
                        G[n1][n2][D_steps_names[4]] = 'YES'
                        G[n1][n2]['Consensus'] += 1

    return G , L_anchors , L_candidates , D_anchors
    
#=====================================================================

def saveNetwork(graph , candidate_list , anchor_dict , anchor_lists=[] , label_list=[] , color_list=[] , file_name="Consensus_Network.txt"):

    """
    Writes a text file with the content of a network.

    input1 : A co-expression network in the form of a Networkx graph.
    input2 : A list of lists of special nodes (anchors).
    input3 : A list of colors in rgb format (i.e. tuples of 3 float values, each between 0 and 1).
    input4 : A name for the resulting text file.

    output : None.
    """

    with open(file_name,'w') as file :
        for node in sorted(candidate_list) : file.write(f">{node}:(0, 0, 0)\n")
        if anchor_lists != [] :
            for k,v in anchor_dict.items() :
                idx = label_list.index(k)
                for node in sorted(v) : file.write(f">{node}:{str(color_list[idx])}\n")
        else :
            for k,v in anchor_dict.items() :
                for node in sorted(v) : file.write(f">{node}:(0, 0, 0)\n")
        
        for (n1,n2) in graph.edges :
            D_attr = graph[n1][n2]
            line = f"{n1} ; {n2}"
            for k,v in D_attr.items(): line += f" ; {k}:{str(v)}"
            file.write(line+'\n')

#=====================================================================
