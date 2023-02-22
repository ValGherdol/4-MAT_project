import copy
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import clique
import time
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

#=====================================================================
#=====================================================================

# Liste of the functions defined in this file :
#   - indices_recovery
#   - overall_normalize
#   - createGraph
#   - cleanGraphDistance
#   - cleanGraph
#   - dynamicCleanGraph
#   - countNeighbors
#   - listNeighbors
#   - getCliquesInfo
#   - getEdgesData
#   - rankingNodes
#   - saveRankedNodes
#   - sortCandidates
#   - candidateVectorsOfInterest
#   - sortSpecialNeighbors
#   - saveFinalCandidates
#   - saveNetwork

#=====================================================================
#=====================================================================

def indices_recovery(alt_dataframe_list , anchor_lists=[]):
  
  """
  Converts all genes' names into indices.

  input1 : A list of Pandas dataframes with the name column acting as the index column.

  output1 : A list of sub-lists of indices for all genes.
    Each sub-list is associated to a dataframe.
  output2 : A list of sub-lists of sub-sub-lists of indices for annchor genes.
    Each sub-list is associated to a dataframe. Each sub-sub-list regroups anchor genes of the same label.
    In all sub-lists, sub-sub-lists of a same anchor label are in the same position.
  """

  L_GeneIdx = []
  for df in alt_dataframe_list :
    L_idx = df.index.to_list()
    L_GeneIdx.append(L_idx)
    L_idxval = []
    for gene in L_idx :L_idxval.append(L_idx.index(gene))
        
  L_L_AnchorIdx = [[] for i in range(len(anchor_lists))]
  for i,df in enumerate(alt_dataframe_list) :
    L_anchorsidx = [[] for j in range(len(anchor_lists))]
    for j,L_anchors in enumerate(anchor_lists) :
      for gene in L_anchors : L_anchorsidx[j].append(L_GeneIdx[i].index(gene))
      L_L_AnchorIdx[j].append(L_anchorsidx[j])

  return L_GeneIdx , L_L_AnchorIdx
#=====================================================================

def overall_normalize(dataframe):
    """
    Normalizes the values from a Pandas dataframe based on the highest and lowest values of the whole dataframe.

    input1 : A Pandas dataframe with value columns only (the usual 'ID_REF' column that contains the genes' names is used as the index column).

    output1 : A matrix filled with the normalized values.
    """

    L_vector = []
    for col in list(dataframe.keys()): L_vector.append(list(dataframe[col]))
    L_vector = np.transpose(L_vector)

    mat = copy.deepcopy(L_vector)
    
    N_max = np.max(mat)
    N_min = np.min(mat)
    N_diff = N_max-N_min
    for i,row in enumerate(mat):
        for j,value in enumerate(mat[i]):
            mat[i][j] = (mat[i][j]-N_min)/N_diff
    
    return mat

#=====================================================================

def createGraph(indicesKNN, distanceKNN, specialGenesIndex, geneIndex):
  """
  create a graph network based on KNN distances
  input1 indicesKNN: matrix with indices of k closest neighbors
  input2 distanceKNN: matrix with distances of k closest neighbors
  input3 specialGenesIndex: identifier of special genes (red)
  input4 geneIndex: identifier of all genes (blue)
  output1 G: graph networkx
  output2 adj: list of blue nodes
  """
  adj = []
  G = nx.Graph()

  for i in specialGenesIndex : G.add_node(str(i), label=str(geneIndex[i]), color="red") # Add all special genes first
  
  for i in specialGenesIndex :
    lKNN = indicesKNN[i, :] #taking all neigbors of special gene i
    dKNN = distanceKNN[i, :] #taking all distances  --> previously taking indicesKNN
    for j in range(1,len(lKNN)): #ignore first position because it's itself
      if lKNN[j] not in G :
        G.add_node(str(lKNN[j]), label=str(geneIndex[lKNN[j]]), color="blue") # Add non-special genes that are neighbors to special genes 
        adj.append(lKNN[j])      
      G.add_edge(str(i), str(lKNN[j]), weight=float(dKNN[j]))

  return G, adj

#=====================================================================

def cleanGraphDistance(G, distancesKNN, factor=1, version='Unilateral') :
  """
  Remove nodes with distance less than threshold
  input1 G: graph networkx
  input2 indicesKNN
  input3 distanceKNN
  output1 G: graph networkx (updated)
  """
  to_remove = []
  max_diff = 0
  for a,b,attrs in G.edges(data=True):
    th_a = np.mean(distancesKNN[int(a)]) - factor * np.std(distancesKNN[int(a)])
    th_b = np.mean(distancesKNN[int(b)]) - factor * np.std(distancesKNN[int(b)])
    
#    print(f"Values for node {a} : {np.mean(distancesKNN[int(a)])} + {factor * np.std(distancesKNN[int(a)])} = {th_a}")
#    print(f"Values for node {b} : {np.mean(distancesKNN[int(b)])} + {factor * np.std(distancesKNN[int(b)])} = {th_b}")

#    diff = abs(th_a-th_b)
#    if diff > max_diff : max_diff = diff
#    print(f"Maximum threshold difference : {max_diff}")
#    print("--------------")

    ### Unilateral version :
    ### Edge is removed if both threshold don't accept it.
    ### It is kept if at least one threshold accepts it.
    if version == 'Unilateral' :
      if attrs["weight"] >= th_a and attrs["weight"] >= th_b : to_remove.append((a,b))

    ### Bilateral version :
    ### Edge is removed if at least one threshold doesn't accept it.
    if version == 'Bilateral' :
      if attrs["weight"] >= th_a or attrs["weight"] >= th_b : to_remove.append((a,b))

  G.remove_edges_from(to_remove)

  return G

#=====================================================================

def cleanGraph(G, nbNeighbors):
  """
  Remove nodes with less than nbNeighbors
  input1 G: graph networkx
  input2 nbNeighbors: number of minimum neighbors
  output1 G: graph networkx (updated)
  """
  toBeRemoved = []
  for node in G:
    if len(list(G.neighbors(node))) < nbNeighbors : toBeRemoved.append(node)
  G.remove_nodes_from(toBeRemoved)
  return G

#=====================================================================

def dynamicCleanGraph(G, specialNodes):
  """
  Calculates a minimal number of special neighbors and remove nodes
  with less than this number.
  input1 G: graph networkx
  input2 specialNodes: set of special nodes
  output1 G: graph networkx (updated)
  output1 minNeighbors: calculated minimum number of special neighbors
  """
  L_nbNeighbors = []
  for node in G:
    if int(node) in specialNodes : continue
    L_nbNeighbors.append(len(list(G.neighbors(node))))
  print(f"Non special nodes : {len(L_nbNeighbors)}")
  Avg_N = np.mean(L_nbNeighbors)
  Std_N = np.std(L_nbNeighbors)
  print(f"Average neighborhood : {Avg_N}")
  print(f"Standard neighborhood deviation: {Std_N}")

  minNeighbors = int(Avg_N + Std_N)

  print(f"Dynamic minimum neighborhood : {minNeighbors}")
  
  toBeRemoved = []
  for node in G:
    if node in specialNodes : continue
    if len(list(G.neighbors(node))) < minNeighbors : toBeRemoved.append(node)
  G.remove_nodes_from(toBeRemoved)
  
  return G , minNeighbors

#=====================================================================

def countNeighbors(G, node, specialNg):
  """
  count Neighbors of a given node
  input1 G: graph networkx
  input2 node: a node of G
  input3 specialNg: set of special nodes
  output1 numNg: number of neighbors under conditions
  output2 totalNg: total number of neighbors (without restrictions)
  """  
  neighbors = list(G.neighbors(node))
  
  count = 0
  for i in neighbors:
    if int(i) in specialNg:
      count = count + 1
  return count, len(neighbors)

#=====================================================================

def listNeighbors(G, node, specialNg):
  neighbors = list(G.neighbors(node))
  
  list_special = []
  for i in neighbors:
    if int(i) in specialNg:
      list_special.append(i)
  return list_special

#=====================================================================

def getCliquesInfo(node, cliques, specialNg, minSizeClique=3):
  """
  Get clique properties of a given node
  input1 node: a node of G
  input2 cliques: generator object of all max cliques in the graph for the input1 node
  input3 specialNg: set of special nodes
  output1 nbCliques: number of cliques contaning the node
  output2 sizeMaxCliques: size of largest clique contaning the node
  output3 specialCliques: number of special nodes in the clique
  """
  nbCliques = 0; largestClique = 0; specialScore = 0

  # Since we go through an iterator, we don't know in advance the number of cliques we'll analyze.
  # As such, we use a infinite while loop that will end when the iterator is empty.
  # Also, since the iterator is only aboit cliques that contain the node given as input1, no test is
  # required to make sure the node is in the cliques.
  while 1 :
    try :
      clique = next(cliques)
      if len(clique) >= minSizeClique:
        #print (f"clique=={clique}")
        nbCliques = nbCliques + 1 
        if len(clique) > largestClique : largestClique = len(clique)
        for j in clique:
          if int(j) in specialNg : specialScore += 1
#        print(f"{node} : current clique {nbCliques} of size {len(clique)} / cumulative special score {specialScore}")
#      if nbCliques%10000 == 0 : print(f"Node {node} : {nbCliques} analyzed")
    except StopIteration : break
    
  return nbCliques, largestClique, specialScore

#=====================================================================

def getEdgesData(edgesData, node, specialNg):
  """
  Compute the max, mean and min distances of a node
  input1 edgesData: a list of tuple (node1, node2, {'weight': edge cost})
  input2 node: a node of Graph
  input3 specialNg: set of special nodes
  output1 mean distance: the average distance of node and all its neighbors
  output2 mean distance special nodes: the average distance of a node and its special neighbors
  output3 max distance: the maximum distance of node and all its neighbors
  output4 max distance special nodes: the max distance of a node and its special neighbors
  output5 min distance: the minimum distance of node and all its neighbors
  output6 min distance special nodes: the minimum distance of a node and its special neighbors
  """
  dist = 0; count = 0; distSpecialNg = 0; countSp = 0
  maxDist = 0; maxDistSpecial = 0; minDist = 1e6; minDistSpecial = 1e6
  for ed in edgesData:
    n1, n2, dicWeigth = ed
    if node==n1 or node==n2 :
      dist = dist + dicWeigth['weight']
      count = count + 1
      if dicWeigth['weight'] > maxDist : maxDist = dicWeigth['weight']
      if dicWeigth['weight'] < minDist : minDist = dicWeigth['weight']
      
      if (node==n1 or node==n2) and (int(n1) in specialNg) or (int(n2) in specialNg):
        distSpecialNg = distSpecialNg + dicWeigth['weight']
        countSp = countSp + 1
        if dicWeigth['weight'] > maxDistSpecial : maxDistSpecial = dicWeigth['weight']
        if dicWeigth['weight'] < minDistSpecial : minDistSpecial = dicWeigth['weight']

  return (dist/(count)), (distSpecialNg/(countSp)), maxDist, maxDistSpecial, minDist, minDistSpecial

#=====================================================================

def rankingNodes(G, specialNodes):
  """
  rank all nodes in a graph
  input1 G: graph networkx
  input2 specialNodes: set of special nodes
  output1 rankMat: numpy array containind nodes and its properties
  nbSpecialNg --> number of neighbors under conditions
  nbTotalNg --> total number of neighbors (without restrictions)
  nbCliques --> number of cliques contaning the node
  largestClique --> size of largest clique contaning the node
  specialCliques --> number of special nodes in the clique
  meanDistance --> the average distance of node and all its neighbors
  dictionary_index[str(node)]      
  """
  edgesData = G.edges.data()
  
  rankMat = []
  count = 0
  for i,node in enumerate(G):
    if int(node) not in specialNodes:
      start = time.time()
#      print(f"Ranking for node {node}")
      cliques = nx.find_cliques(G,nodes=[node]) # All max cliques for the current node
      nbSpecialNg, nbTotalNg = countNeighbors(G, node, specialNodes)
      listSpecialNg = listNeighbors(G, node, specialNodes)
#      print(f"{node} : {len(listSpecialNg)}")
      nbCliques, largestClique, specialScore =  getCliquesInfo(node, cliques, specialNodes)
#      print(f"{node} : {nbCliques} cliques / score of {specialScore} / largest clique of size {largestClique} / {nbSpecialNg} special neigbors / {nbTotalNg} total neighbors")
      meanDistance, meanDistanceSp, maxDistance, maxDistanceSp, minDistance, minDistanceSp = getEdgesData(edgesData, node, specialNodes)
#      print(f"Avg dist : {meanDistance} / Avg Special dist {meanDistanceSp}")
      mat = []
      mat.append(node); mat.append(nbSpecialNg); mat.append(specialScore); mat.append(nbCliques); mat.append(largestClique); mat.append(minDistance); mat.append(meanDistance); mat.append(maxDistance); mat.append(listSpecialNg)
#      mat.append(node); mat.append(nbSpecialNg); mat.append(nbTotalNg); mat.append(specialCliques); mat.append(nbCliques); mat.append(largestClique); mat.append(meanDistance); mat.append(meanDistanceSp); mat.append(listSpecialNg)
      rankMat.append(mat)
      count +=1
#      print(f"Node {node} ({count}/{len(G.nodes)-len(specialNodes)}) ranked in {round(time.time()-start,3)} seconds : {nbCliques} cliques / {len(G[node])} neighbors")
      if (count)%1000 == 0 : print(f"Remplissage matrice : {count} points traités sur {len(G.nodes)-len(specialNodes)}")
#      print("-------------")

  return rankMat

#=====================================================================

def saveRankedNodes(stringed_rankMat , file_name="Rank Matrix.txt"):
  
  """
  Writes the matrix of ranked non-special nodes in a text file.

  input1 : A rank matrix with string values.
  input2 : A string value as name for the text file.

  output1 : None.
  """

  headers=['ID_REF', '#SpNeighbors', 'Score', '#Cliques', '#MaximalClique', 'MinDistance', 'AvgDistance', 'MaxDistance', 'Specific Genes']

  with open(file_name,'w') as file :
    file.write('\t'.join(headers)+'\n')
    for line in stringed_rankMat :
      SpecialNg = line.pop(8)
      file.write('\t'.join(line)+'\t')
      file.write(','.join(SpecialNg)+'\n')

#=====================================================================

def sortCandidates(candidate_files_list , Min_redundancy=1):

  """
  Intersects several ranking matrices to sort redundant candidates.

  input1 : A text file which contains names of rank matrices text files sorted by type anchor.
  input2 : An Integer value (default value = 1).

  output1 : A list of sub-lists of ranking matrices.
    Each sub-list contains ranking matrices for a specific type of anchors.
  output2 : A list of dictionaries with candidate genes as keys and and Integer as values.
    In each dictionary, each gene is associated to the number of dataset it has been found in.
    There is one dictionary per type of anchors.
  output3 : A list of candidates who show up in all sub-lists.
  """

  # Regrouping of the result tables sorted by anchor type
  with open(candidate_files_list,'r') as file :
    L_classe , L_tabs = [],[]
    for line in file :
      if line[0]=='>':
        classe = line[1:-1].split()[0]
        L_classe.append(classe)
        L_tabs.append([])
      else : L_tabs[-1].append(pd.read_csv(line[0:-1],delimiter='\t'))

  # Regrouping of the candidates lists sorted by anchor type
  L_Rank_Idx = []
  for tab in L_tabs : # For each group of tabs
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
      if v < Min_redundancy : L_delete.append(k)
    for name in L_delete : del(dico[name])
    L_Cands.append(dico)

  # Intersection of all candidates to see if some are linked to all types of anchors
  Inter_all = L_Cands[0]
  for inter in L_Cands[1::]: Inter_all = list(set(Inter_all) & set(inter))

  return L_tabs , L_Cands , Inter_all
  
#=====================================================================

def candidateVectorsOfInterest(candidates_dicts , rankMat_list):

  """
  For each candidate genes, search for the number of special neighbors it has in a given ranking matrix.
  Once passed through all ranking matrices of a same anchor type, regroups all retrieved values in a vector.

  input1 : A list of dictionaries with candidate genes as keys and Integers as values.
  input2 : A list of sub-lists of ranking matrices, each sub-list associated to a specific type of anchors.

  output1: A list of dictionaries with candidate genes as keys and vectors of Integers as values.
    Each dictionary is associated to a specific type of anchor.
  """

  L_dico_V = []
  for i,dico in enumerate(candidates_dicts) :
    D_voisin = {}
    for gene,value in dico.items() :
      D_voisin[gene] = []
      for j,tab in enumerate(rankMat_list[i]) :
        L_names = list(tab['ID_REF'])
        if gene in L_names : v = int(tab.loc[tab['ID_REF'] == gene]["#SpNeighbors"])
        else : v = 0
        D_voisin[gene].append(v)
    L_dico_V.append(D_voisin)

  return L_dico_V

#=====================================================================

def sortSpecialNeighbors(candidates_dicts , rankMat_list):

  """
  For each candidate genes, search for all it's special neighbors and sort them based on the number of datasets they are associated in.

  input1 : A list of dictionaries with candidate genes as keys and Integers as values.
  input2 : A list of sub-lists of ranking matrices, each sub-list associated to a specific type of anchors.

  output1: A list of dictionaries with candidate genes as keys and sub_dictionaries as values.
    Each dictionary is associated to a specific type of anchor.
    Each sub-dictionary contains the following key-value associations :
      - 'total' : Integer = the total number of unique special neighbors for the current candidate gene.
      - Integer (p) : list = the list of special neighbors associated to the current candidate gene in p ranking matrices.
          (p goes from 1 to the number of datasets provided for the analysis)
  """

  L_dico_N = []
  for i,dico in enumerate(candidates_dicts) :
    D_temp = {}
    
    for gene,value in dico.items() :
      D_temp[gene] = {"total":0}
      
      for j in range(len(rankMat_list[i])): D_temp[gene][j+1] = []
      L_voisinage = [] # List of sub-lists of neighbors in each ranking matrices
      inter_voisinage = [] # List of unique neighbors across all ranking matrices

      for tab in rankMat_list[i]:
        L_names = list(tab['ID_REF'])
        if gene in L_names :
          L_voisinage.append(list(tab.loc[tab['ID_REF']==gene]["Specific Genes"])[0].split(','))
          inter_voisinage += L_voisinage[-1]
      inter_voisinage = set(inter_voisinage)
      D_temp[gene]["total"] = len(inter_voisinage)

      for n in inter_voisinage : # For each unique special neighbor, count how many ranking matrices associates it to the current candidate.
        p = 0
        for voisinage in L_voisinage :
          if n in voisinage : p += 1
        D_temp[gene][p].append(n)
    L_dico_N.append(D_temp)

  return L_dico_N

#=====================================================================

def saveFinalCandidates(CR_dict_list , AN_dict_list , CV_dict_list ,
                        dataset_list , globalGenePool , anchorLabels ,
                        file_name="FinalCandidateGenes.txt"):

  """

  input1 : A list of dictionaries with gene names as keys and redundancy Integer values.
  input2 : A list of dictionaries with anchor gene names as keys and list of associated candidate gene names as values.
  input3 : A list of dictionaries with candidate gene names as keys and list of Integer vectors as values.
  input4 : A list of Pandas dataframes.
  input5 : A list of all genes.
  input6 : A list of labels for anchor genes.
  input7 : A name for the written file.

  output1 : None.
  """

  headers = ['ID_REF']
  for label in anchorLabels : headers.append(f"#{label}_Neighbors")
  headers.append("#UniqueNeighbors")
  for i in range(len(dataset_list)) : headers.append(f"{i+1}-Neighborhood")

  with open(file_name,'w') as file :
    file.write('\t'.join(headers)+'\n')
    for gene in globalGenePool :
      match = False
      line = [0 for i in range(len(headers))]
      line[0] = gene
      for i,inter in enumerate(CR_dict_list) :
        head_1 = f"#{anchorLabels[i]}_Neighbors"
        if gene in list(inter.keys()) :
          match = True
          line[headers.index(head_1)] = f"{CV_dict_list[i][gene]}"
          dico_N = AN_dict_list[i]
          for k,v in dico_N[gene].items():
            if k=="total" : line[headers.index("#UniqueNeighbors")] += v
            else :
              head_2 = f"{k}-Neighborhood"
              if line[headers.index(head_2)] == 0 :
                if v == [] : line[headers.index(head_2)] = 'None'
                else : line[headers.index(head_2)] = ','.join(v)
              else :
                line[headers.index(head_2)] += '/'
                if v == [] : line[headers.index(head_2)] = 'None'
                else : line[headers.index(head_2)] += ','.join(v)
        else : line[headers.index(head_1)] = 'None'
      
      for i,a in enumerate(line) : line[i] = str(a)
      if match == True : file.write('\t'.join(line)+'\n')

#=====================================================================

def saveNetwork(graph , anchor_lists=[] , color_list=[] , file_name="NPC_Network.txt"):

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
#=====================================================================
