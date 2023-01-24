import networkx as nx
from networkx.algorithms.approximation import clique

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
    input4 geneIndex: identifier of all genes (red)
    output1 G: graph networkx
    output2 adj: list of blue nodes
    """
    adj = []
    G = nx.Graph()

  
    for i in specialGenesIndex:
        G.add_node(str(i), label=str(geneIndex[i]), color="red")
        lKNN = indicesKNN[i, :] #taking all neigbors of special gene i
        dKNN = distanceKNN[i, :] #taking all distances  --> previously taking indicesKNN
        #print(lKNN)
        #print(lKNN, dKNN)
        for j in range(1,len(lKNN)): #ignore first position it's itself
            if lKNN[j] not in G:
                if lKNN[j] not in specialGenesIndex: #and (dKNN[j] < 0.1):
                    #print(dKNN[j])
                    G.add_node(str(lKNN[j]), label = str(geneIndex[lKNN[j]]), color="blue")
                    adj.append(lKNN[j])
                else: G.add_node(str(lKNN[j]), label = str(geneIndex[lKNN[j]]), color="red")        
            G.add_edge(str(i), str(lKNN[j]), weight=float(dKNN[j]))
    #for i in geneIndexValue:
        #G.add_node(str(i), label=str(geneIndex[i]), color="blue")


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
        #print(j, dKNN)
        for k in range(1, len(lKNN)):
            if (str(lKNN[k]) in G) and (dKNN[k] < 0.5):
                G.add_edge(str(j), str(lKNN[k]), weight=float(dKNN[k]))
                #print("entrou")

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
        #print (" NB ", node, len(list(G.neighbors(node))))
        if len(list(G.neighbors(node))) < nbNeighbors:
            #print("entrou")
            toBeRemoved.append(node)
    #print (" toBeRemoved" , toBeRemoved)
    G.remove_nodes_from(toBeRemoved)
    #print (G.edges.data())
    return G

#=====================================================================

def cleanGraphDistance(G, threshold):
    """
    Remove nodes with distance less than threshold
    input1 G: graph networkx
    input2 indicesKNN
    input3 distanceKNN
    output1 G: graph networkx (updated)
    """
    #print(G)
    to_remove = []
    to_remove = [(a,b) for a, b, attrs in G.edges(data=True) if (attrs["weight"] >= threshold) and (a != b)]
    #print(to_remove)
    G.remove_edges_from(to_remove)

    return G

#=====================================================================

def getBlueNodes(G, specialGenesIndex):
    rnodes = []
    #print (len(G.nodes))
    for node in G.nodes:
        #print ("d", node, type(node), str(node))
        if int(node) not in specialGenesIndex : rnodes.append(int(node))
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
        if int(i) in specialNg : count = count + 1
    return count, len(neighbors)

#=====================================================================

def listNeighbors(G, node, specialNg):
    neighbors = list(G.neighbors(node))
    
    list_special = []
    for i in neighbors:
        if int(i) in specialNg : list_special.append(i)
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
    #print(edgesData, node, specialNg)
    dist = 0; count = 0; distSpecialNg = 0; countSp = 0
    #print(edgesData)
    for ed in edgesData:
        n1, n2, dicWeigth = ed
        #print(n1, n2, dicWeigth)
        if node==n1 or node==n2:
            #print (node, n1, n2, dicWeigth)
            dist = dist + dicWeigth['weight']
            count = count + 1
            if (node==n1 or node==n2) and (int(n1) in specialNg) or (int(n2) in specialNg):
                #print ("special node distance ->", node, n1, n2, dicWeigth)
                distSpecialNg = distSpecialNg + dicWeigth['weight']
                countSp = countSp + 1
    #if count == 0:# Verify why I have count zero       
    #  return 100000
  
    #print(dist, count, distSpecialNg, countSp)
    #print(node)
    #print(dist, count, distSpecialNg, countSp)
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
                if len(clique) > largestClique : largestClique = len(clique)
                for j in specialNg:
                    if str(j) in clique : specialCliques = specialCliques + 1
    return nbCliques, largestClique, specialCliques

#=====================================================================

def rankingNodes(G, specialNodes):
    """
    rank all nodes in a graph
    input1 G: graph networkx
    input2 specialNodes: set of special nodes
    output1 rankMat: numpy array containind nodes and its properties
    """  
    cliques = nx.find_cliques(G)
    #cliques = nx.enumerate_all_cliques(G)
    lCliques = list(cliques)
    edgesData = G.edges.data() 
    #print(edgesData)
    #rankMat = np.zeros((len(G.nodes), 7))
    rankMat = []
    count = 0
    for node in G:
        if int(node) not in specialNodes:   
            #print ("node==", node)
###-------------------------------------------------------------------------------------------------
#            nbSpecialNg, nbTotalNg = countNeighbors(G, node, specialGenesIndex)
#            listSpecialNg = listNeighbors(G, node, specialGenesIndex)
#            nbCliques, largestClique, specialCliques =  getCliquesInfo(lCliques, node, specialGenesIndex)
            nbSpecialNg, nbTotalNg = countNeighbors(G, node, specialNodes)
            listSpecialNg = listNeighbors(G, node, specialNodes)
            nbCliques, largestClique, specialCliques =  getCliquesInfo(lCliques, node, specialNodes)
###-------------------------------------------------------------------------------------------------
            #print(edgesData, node, specialGenesIndex)
            #print (node, "nbSpecialNg=", nbSpecialNg, "nbTotalNg=", nbTotalNg, "nbCliques=", nbCliques, "largestClique=", largestClique, "specialCliques=", specialCliques)
###-------------------------------------------------------------------------------------------------
#            meanDistance, meanDistanceSp = getEdgesData(edgesData, node, specialGenesIndex)
            meanDistance, meanDistanceSp = getEdgesData(edgesData, node, specialNodes)
###-------------------------------------------------------------------------------------------------
            #print (node, "nbSpecialNg=", nbSpecialNg, "nbTotalNg=", nbTotalNg, "nbCliques=", nbCliques, "largestClique=", largestClique, "specialCliques=", specialCliques, "meanDistance=", meanDistance)
            #rankMat[count][0] = node; rankMat[count][1] = nbSpecialNg; rankMat[count][2] = nbTotalNg; rankMat[count][3] = nbCliques; rankMat[count][4] = largestClique; rankMat[count][5] = specialCliques; rankMat[count][6] = meanDistance; 
            mat = []
            #nbSpecialNg --> number of neighbors under conditions
            #nbTotalNg --> total number of neighbors (without restrictions)
            #nbCliques --> number of cliques contaning the node
            #largestClique --> size of largest clique contaning the node
            #specialCliques --> number of special nodes in the clique
            #meanDistance --> the average distance of node and all its neighbors
            #dictionary_index[str(node)]
            mat.append(node); mat.append(nbSpecialNg); mat.append(nbTotalNg); mat.append(specialCliques); mat.append(nbCliques); mat.append(largestClique); mat.append(meanDistance); mat.append(meanDistanceSp); mat.append(listSpecialNg)
            rankMat.append(mat)
            count +=1

    return rankMat
