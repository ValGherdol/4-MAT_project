import numpy as np
import networkx as nx
from networkx.algorithms.approximation import clique
import time

import warnings
warnings.filterwarnings("ignore")

#=====================================================================
#=====================================================================


#=====================================================================
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

  for i in specialGenesIndex:
    G.add_node(str(i), label=str(geneIndex[i]), color="red")
    lKNN = indicesKNN[i, :] #taking all neigbors of special gene i
    dKNN = distanceKNN[i, :] #taking all distances  --> previously taking indicesKNN
    for j in range(1,len(lKNN)): #ignore first position because it's itself
      if lKNN[j] not in G:
        if lKNN[j] not in specialGenesIndex: #and (dKNN[j] < 0.1):
          G.add_node(str(lKNN[j]), label=str(geneIndex[lKNN[j]]), color="blue")
          adj.append(lKNN[j])
        else:
          G.add_node(str(lKNN[j]), label=str(geneIndex[lKNN[j]]), color="red")        
      G.add_edge(str(i), str(lKNN[j]), weight=float(dKNN[j]))

  return G, adj

#=====================================================================

def completeGraph(G, adj, indicesKNN, distanceKNN):
  """
  complete the Graph by adding edges between not special nodes (blue)
  input1 G: graph networkx
  input2 adj: list contaning all G nodes 
  input3 indicesKNN: matrix with indices of k closest neighbors
  input4 distanceKNN: matrix with distances of k closest neighbors
  output1 G: graph networkx (updated)
  """
  for j in adj:
    lKNN = indicesKNN[j, :] #taking all neigbors of special gene i
    dKNN = distanceKNN[j, :] #taking all distances
    for k in range(1, len(lKNN)):
      if (str(lKNN[k]) in G) and (dKNN[k] < 0.5):
        G.add_edge(str(j), str(lKNN[k]), weight=float(dKNN[k]))

  return G

#=====================================================================

def completeGraphMin(G, adj, indicesKNN, distanceKNN):
    
  """
  complete the Graph by adding edges between not special nodes (blue) already present in the graph
  input1 G: graph networkx
  input2 adj: list contaning all blue nodes 
  input3 indicesKNN: matrix with indices of k closest neighbors
  input4 distanceKNN: matrix with distances of k closest neighbors
  output1 G: graph networkx (updated)
  """
  for j in G:
    if j in adj:
      lKNN = indicesKNN[j, :] #taking all neigbors of gene j
      dKNN = distanceKNN[j, :] #taking all distances
      for k in range(1, len(lKNN)):
        if str(lKNN[k]) in G: #and (dKNN[k] < 0.1):
          G.add_edge(str(j), str(lKNN[k]), weight=float(dKNN[k]))
          print("entrou")

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

def cleanGraphDistance(G, distancesKNN):
  """
  Remove nodes with distance less than threshold
  input1 G: graph networkx
  input2 indicesKNN
  input3 distanceKNN
  output1 G: graph networkx (updated)
  """
  to_remove = []
  for a,b,attrs in G.edges(data=True):
    #print( np.mean(distancesKNN[int(a)]), np.std(distancesKNN[int(a)]))
    threshold = np.mean(distancesKNN[int(a)]) + np.std(distancesKNN[int(a)])
    #print(distancesKNN[int(a)])
    #print(threshold)
    if attrs["weight"] >= threshold and a != b: to_remove.append((a,b))
    
    #print(a)
    #print(len(distancesKNN))
    #print(distancesKNN)
    #print(distancesKNN[int(a)])
    
    #print(threshold)
    #if attrs["weight"] >= threshold and a != b:
  #threshold = np.mean(distancesKNN[int(a)]) 
  #for nodes in G.nodes():
    #threshold = np.mean(distancesKNN[int(nodes)]) 
    #print(threshold)
    #to_remove = [(a,b) for a, b, attrs in G.edges(data=True) if (attrs["weight"] >= threshold) and (a != b)]
  #print(to_remove)
  G.remove_edges_from(to_remove)

  return G

#=====================================================================

def getBlueNodes(G, specialGenesIndex):
  rnodes = []
  for node in G.nodes:
    if int(node) not in specialGenesIndex:
      rnodes.append(int(node))
  return rnodes

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

def getEdgesData(edgesData, node, specialNg):
  """
  Compute the mean distance of a node
  input1 edgesData: a list of tuple (node1, node2, {'weight': edge cost})
  input2 node: a node of Graph
  input3 specialNg: set of special nodes
  output1 mean distance: the average distance of node and all its neighbors
  output2 mean distance special nodes: the average distance of a node and its special neighbors (TODO)
  """
  dist = 0; count = 0; distSpecialNg = 0; countSp = 0
  for ed in edgesData:
    n1, n2, dicWeigth = ed
    if node==n1 or node==n2:
      dist = dist + dicWeigth['weight']
      count = count + 1
      if (node==n1 or node==n2) and (int(n1) in specialNg) or (int(n2) in specialNg):
        distSpecialNg = distSpecialNg + dicWeigth['weight']
        countSp = countSp + 1

  return (dist/(count)), (distSpecialNg/(countSp))

#=====================================================================

def getCliquesInfo(cliques, node, specialNg, minSizeClique=3):
  """
  Get clique properties of a given node
  input1 cliques: list of all cliques in a graph
  input2 node: a node of G
  input3 specialNg: set of special nodes
  output1 nbCliques: number of cliques contaning the node
  output2 sizeMaxCliques: size of largest clique contaning the node
  output3 specialCliques: number of special nodes in the clique
  """
  nbCliques = 0; largestClique = 0; specialCliques = 0
  #print (cliques)
  for clique in cliques:
    if len(clique) >= minSizeClique:
      #print ("clique== ",clique)
      if node in clique:
        nbCliques = nbCliques + 1
        if len(clique) > largestClique:
          largestClique = len(clique)
        for j in specialNg:
          if str(j) in clique:
            specialCliques = specialCliques + 1
  return nbCliques, largestClique, specialCliques

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

  cliques = nx.find_cliques(G)
  #cliques = nx.enumerate_all_cliques(G)
  lCliques = list(cliques)
  print(f"{len(lCliques)} cliques have been found")
  edgesData = G.edges.data()
  print("Edges data retrieved")
  
  rankMat = []
  count = 0
  for i,node in enumerate(G):
    print(i,node)
    if int(node) not in specialNodes:   
      #print ("node==", node)
      nbSpecialNg, nbTotalNg = countNeighbors(G, node, specialNodes)
      listSpecialNg = listNeighbors(G, node, specialNodes)
      nbCliques, largestClique, specialCliques =  getCliquesInfo(lCliques, node, specialNodes)
      meanDistance, meanDistanceSp = getEdgesData(edgesData, node, specialNodes)
      mat = []
      mat.append(node); mat.append(nbSpecialNg); mat.append(nbTotalNg); mat.append(specialCliques); mat.append(nbCliques); mat.append(largestClique); mat.append(meanDistance); mat.append(meanDistanceSp); mat.append(listSpecialNg)
      rankMat.append(mat)
      count +=1
    if (i+1)%1000 == 0 : print(f"Remplissage matrice : {i+1} points traités sur {len(G)}")

  return rankMat

#=====================================================================

def rankingNodes2(G, specialNodes):
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
  cliques = nx.find_cliques(G)
  lCliques = list(cliques)
  edgesData = G.edges.data() 
  rankMat = []
  count = 0
  for i,node in enumerate(G):
    if int(node) not in specialNodes:   
      nbSpecialNg, nbTotalNg = countNeighbors(G, node, specialNodes)
      nbCliques, largestClique, specialCliques =  getCliquesInfo(lCliques, node, specialNodes)
      meanDistance, meanDistanceSp = getEdgesData(edgesData, node, specialNodes)
      mat = []
      mat.append(nbSpecialNg); mat.append(specialCliques); mat.append(nbCliques); mat.append(largestClique); mat.append(meanDistance)
      rankMat.append(mat)
      count +=1
    if (i+1)%1000 == 0 : print(f"Remplissage matrice : {i+1} points traités sur {len(G)}")

  return rankMat

#=====================================================================

def createGraph_V(indicesKNN, distanceKNN, specialGenesIndex, geneIndex):
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

def cleanGraphDistance_V(G, distancesKNN, factor=1, version='Unilateral') :
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

def dynamicCleanGraph_V(G, specialNodes):
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

def getEdgesData_V(edgesData, node, specialNg):
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

def getCliquesInfo_V(node, cliques, specialNg, minSizeClique=3):
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

def rankingNodes_V(G, specialNodes):
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
      nbCliques, largestClique, specialScore =  getCliquesInfo_V(node, cliques, specialNodes)
#      print(f"{node} : {nbCliques} cliques / score of {specialScore} / largest clique of size {largestClique} / {nbSpecialNg} special neigbors / {nbTotalNg} total neighbors")
      meanDistance, meanDistanceSp, maxDistance, maxDistanceSp, minDistance, minDistanceSp = getEdgesData_V(edgesData, node, specialNodes)
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
