import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA as sk_PCA
from sklearn.manifold import TSNE as sk_TSNE
from sklearn.cluster import KMeans as sk_KMeans
from sklearn.cluster import SpectralClustering as sk_Spectral
from sklearn.cluster import AffinityPropagation as sk_AffiPropa
from sklearn.cluster import AgglomerativeClustering as sk_Agglo
from sklearn.neighbors import NearestNeighbors as sk_KNN
from scipy.spatial import distance
import itertools
import networkx as nx

#=====================================================================
#=====================================================================

# Liste des fonctions présentes ici :
#   - time_count
#   - PearsonGraph
#   - PearsonGraph_2
#   - drawPearson
#   - save_PearsonGraph
#   - save_PearsonGraph_2
#   - save_PearsonGraph_data
#   - save_PearsonGraph_data_2
#   - load_PearsonGraph
#   - load_PearsonGraph_2
#   - update_PearsonGraph
#   - searchInGraph
#   - searchInGraphRestricted
#   - searchInMultiGraph
#   - searchInMultiGraphRestricted
#   - searchInMultiGraphUniforme
#   - searchInMultiGraphUniformeRestricted
#   - CommonRelationAnalysis_2
#   - CrossNetwork
#   - CrossNetworkConstancy
#   - ThresholdedCrossNetworkConstancy
#   - ThresholdedCrossNetworkConstancy_2
#   - ThresholdedCrossNetworkConstancy_3
#   - ThresholdedCrossNetworkConstancy_4
#   - ThresholdedCrossNetworkConstancy_5
#   - showCrossNetwork
#   - showCrossNetwork_SubGraph
#   - showPrincipalCrossNetwork_SubGraph
#   - savePrincipalCrossNetwork_SubGraph
#   - showExclusiveCrossNetwork_SubGraph
#   - saveExclusiveCrossNetwork_SubGraph
#   - load_CrossNetwork_SubGraph
#   - drawNetwork
#---------------------------------------
#   - over_expression
#   - under_expression
#   - higher_epression
#---------------------------------------
#   - KMeans_clustering_1
#   - Spectral_clustering_1
#   - AffinityPropagation_1
#   - Agglomerative_clustering_1
#   - KNN_clustering_1
#   - KNN_clustering_2
#   - Cluster_filtering_1
#   - Cluster_filtering_2
#   - KM1D_v1
#   - KM1D_v2
#   - KM1D_v3
#   - SortData
#   - Cluster_Amplification
#   - Expression_plot
#   - PCA
#   - T_SNE
#   - T_SNE_2


#=====================================================================
#=====================================================================

def time_count(secondes) :
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

def PearsonGraph(df , n=None , node_col='ID_REF' , threshold=0.5):
    # Fonction calculant un graph entre points selon leurs coeffiscients de corrélation de Pearson (CCP).
    # Arguments :
    #   - Un dataframe
    #   - Un nombre de lignes à traiter (défaut = None ; si défaut, le corps de la fonction considère l'ensemble du dataframe)
    #   - Le nom de la colonne indiquant les noms des points dans le dataframe (défaut = 'ID_REF')
    #   - Le palier de sélection du CCP (défaut = 0.5)
    # Rendus :
    #   - Le graph des points étudiés.
    #       Un point est représenté par son nom composé de son indice dans le dataframe et du gène qu'il représente.
    #       Une arrête est représentée par :
    #           - Les noms des deux points qu'elle connecte.
    #           - Son poids, égal à la valeur absolue du CCP des points connectés.
    #           - Le signe du CCP, égal à {1 ou -1}.
    #           - Sa couleur : rouge si le CCP est négatif, vert s'il est positif.
    #   - Le temps que le calcul a mis à se faire.
    
    if n == None : n = len(df)
    G = nx.Graph()
    start = time.time()
    
    ### Ajout des noeuds
    L_nodes = list(df[node_col][0:n])
    L_idx = df[0:n].index.values.tolist()
    
    ### Récupération des colonnes de valeurs
    L_L_values = []
    for i,c in enumerate(list(df.keys())):
        if c != node_col : L_L_values.append(list(df[c][0:n]))
    L_L_values = np.array([np.array(L) for L in L_L_values])
#    print(L_L_values)
    
    ### Calcul des coeffiscients de correlation de Pearson : si le 
    ### coeffiscient est supérieur à un certain palier (en valeur absolue),
    ### une arrête est créée entre les noeuds concernés.
    for i,n1 in enumerate(L_nodes):
        t_node = time.time()
        name_1 = (L_idx[i],n1)
        Lv_1 = list(L_L_values[:,i])
        if not G.has_node(name_1) : G.add_node(name_1)
        for j,n2 in enumerate(L_nodes[i+1::]):
            name_2 = (L_idx[j+i+1],n2)
            Lv_2 = list(L_L_values[:,i+1+j])
            pearson = np.corrcoef(Lv_1,Lv_2)[0][1]
            if abs(pearson) > threshold :
                s = np.sign(pearson)
                G.add_edge(name_1,name_2,weight=abs(pearson),sign=s)
                if s < 0 : G[name_1][name_2].update({'color':'red'}) # Rouge = corrélation négative
                else : G[name_1][name_2].update({'color':'limegreen'})  # Vert = corrélation positive
#            print(name_1,Lv_1)
#            print(name_2,Lv_2)
#            print(name_1,name_2,pearson)
#            print("-------------")
        print(f"Noeud {name_1} traité en {time_count(time.time()-t_node)} ({i+1})")
    
    ### Calcul des positions des noeuds :
    ### - la coordonnée x correspond à la moyenne des valeurs d'expressions du noeud
    ### - la coordonnée y correspond au degré du noeud = son nombre de voisins/d'arrêtes
    D_pos = {}
    for i,n1 in enumerate(L_nodes):
        name = (L_idx[i],n1)
        Lv = list(L_L_values[:,i])
        x_pos = np.mean(Lv)
        y_pos = G.degree[name]
        D_pos[name] = {'pos':[x_pos,y_pos]}
#        print(f"({i},{n1}) :{x_pos}")
#        print(f"({i},{n1}) :{G.degree[(i,n1)]}")
#    print(D_pos)
    nx.set_node_attributes(G,D_pos)
    end = time.time()-start
    
    return G , end

#=====================================================================

def PearsonGraph_2(df , n=None , node_col='ID_REF' , threshold=0.5 , unique=False):
    # Fonction calculant un graph entre points selon leurs coeffiscients de corrélation de Pearson (CCP).
    # Version 2 de la fonction précédente, donnant le choix entre deux modèles différents pour les noms des points.
    # Arguments :
    #   - Un dataframe.
    #   - Un nombre de lignes à traiter (défaut = None ; si défaut, le corps de la fonction considère l'ensemble du dataframe). [int]
    #   - Le nom de la colonne indiquant les noms des points dans le dataframe (défaut = 'ID_REF'). [string]
    #   - Le palier de sélection du CCP (défaut = 0.5). [int or float]
    #   - L'indication que les noms de points sont uniques (défaut = False). [boolean]
    #       - si False, les points sont nommés par un 2-uplet contenant leurs noms et indices dans le dataframe.
    #       - si True, les points sont nommés via leurs seuls noms tels qu'indiqués dans la colonne 'node_col'.
    # Rendus :
    #   - Le graph des points étudiés.
    #       Un point est représenté par son nom tel que choisi par l'indicateur 'unique'.
    #       Une arrête est représentée par :
    #           - Les noms des deux points qu'elle connecte.
    #           - Son poids, égal à la valeur absolue du CCP des points connectés.
    #           - Le signe du CCP, égal à {-1 ou 1}.
    #           - Sa couleur : rouge si le CCP est négatif, vert s'il est positif.
    #   - Le temps que le calcul a mis à se faire.
    
    if n == None : n = len(df)
    G = nx.Graph()
    start = time.time()
    
    ### Ajout des noeuds
    L_nodes = list(df[node_col][0:n])
    L_idx = df[0:n].index.values.tolist()
    
    ### Récupération des colonnes de valeurs
    L_L_values = []
    for i,c in enumerate(list(df.keys())):
        if c != node_col : L_L_values.append(list(df[c][0:n]))
    L_L_values = np.transpose(L_L_values)
#    print(L_L_values)
    
    ### Calcul des coeffiscients de correlation de Pearson : si le 
    ### coeffiscient est supérieur à un certain palier (en valeur absolue),
    ### une arrête est créée entre les noeuds concernés.
    for i,n1 in enumerate(L_nodes):
        t_node = time.time()
        if unique : name_1 = n1
        else : name_1 = (L_idx[i],n1)
        Lv_1 = list(L_L_values[i])
        if not G.has_node(name_1) : G.add_node(name_1)
        for j,n2 in enumerate(L_nodes[i+1::]):
            if unique : name_2 = n2
            else : name_2 = (L_idx[j+i+1],n2)
            Lv_2 = list(L_L_values[i+1+j])
            pearson = np.corrcoef(Lv_1,Lv_2)[0][1]
            if abs(pearson) > threshold :
                s = np.sign(pearson)
                G.add_edge(name_1,name_2,weight=abs(pearson),sign=s)
                if s < 0 : G[name_1][name_2].update({'color':'red'}) # Rouge = corrélation négative
                else : G[name_1][name_2].update({'color':'limegreen'})  # Vert = corrélation positive
#            print(name_1,Lv_1)
#            print(name_2,Lv_2)
#            print(name_1,name_2,pearson)
#            print("-------------")
        print(f"Noeud {name_1} traité en {time_count(time.time()-t_node)} ({i+1})")
    
    ### Calcul des positions des noeuds :
    ### - la coordonnée x correspond à la moyenne des valeurs d'expressions du noeud
    ### - la coordonnée y correspond au degré du noeud = son nombre de voisins/d'arrêtes
    D_pos = {}
    for i,n1 in enumerate(L_nodes):
        if unique : name = n1
        else : name = (L_idx[i],n1)
        Lv = list(L_L_values[i])
        x_pos = np.mean(Lv)
        y_pos = G.degree[name]
        D_pos[name] = {'pos':[x_pos,y_pos]}
#        print(f"({i},{n1}) :{x_pos}")
#        print(f"({i},{n1}) :{G.degree[(i,n1)]}")
#    print(D_pos)
    nx.set_node_attributes(G,D_pos)
    end = time.time()-start
    
    return G , end

#=====================================================================

def drawPearson(G , n_color='black' , labels=False , n_size=100 , e_width=1 , label_size=12 , mode=None):
    # Fonction dessinant un graph.
    # Arguments :
    #   - Un graph.
    #   - La couleur des points du graph (défaut = 'blue'). [string or array-like]
    #   - Le choix de l'affichage des noms des points (défaut = Pas d'affichage). [boolean]
    #   - La taille des points (défaut = 100). [int or float]
    #   - L'épaisseur des arrêtes entre les points (défaut = 1). [int or float]
    #   - Le positionnement des points dans le dessin (défaut = None). [None or string]
    #       - si None, les points sont positionnés selon les coordonnées indiqués dans leurs attributs).
    #       - si spécifié, doit apparaitre dans la liste suivante : 'circular' , 'random' , 'shell', 'spectral' , 'spring'.
    # Rendus (mutuellement exclusifs) :
    #   - None ; Affichage du dessin du graph.
    #   - La valeur 0 si l'indicateur 'mode' renseigné n'est pas pris en charge.

    L_mode = [None , 'circular' , 'random' , 'shell', 'spectral' , 'spring']

    if mode not in L_mode :
        print("Mode demandé non pris en charge")
        return 0

    C_map = []
    for edge in G.edges: C_map.append(G.get_edge_data(edge[0],edge[1])['color'])
        
    D_pos = nx.get_node_attributes(G,'pos')
    
    if mode == None : nx.draw(G , pos = D_pos , node_color=n_color , edge_color=C_map , 
                              with_labels=labels , node_size = n_size , width=e_width , font_size=label_size)
    if mode == 'circular' : nx.draw_circular(G , node_color=n_color , edge_color=C_map , 
                                             with_labels=labels , node_size = n_size ,
                                             width=e_width , font_size=label_size)
    if mode == 'random' : nx.draw_random(G , node_color=n_color , edge_color=C_map , 
                                         with_labels=labels , node_size = n_size ,
                                         width=e_width , font_size=label_size)
    if mode == 'shell' : nx.draw_shell(G , node_color=n_color , edge_color=C_map , 
                                       with_labels=labels , node_size = n_size ,
                                       width=e_width , font_size=label_size)
    if mode == 'spectral' : nx.draw_spectral(G , node_color=n_color , edge_color=C_map , 
                                             with_labels=labels , node_size = n_size ,
                                             width=e_width , font_size=label_size)
    if mode == 'spring' : nx.draw_spring(G , node_color=n_color , edge_color=C_map , 
                                         with_labels=labels , node_size = n_size ,
                                         width=e_width , font_size=label_size)

#=====================================================================

def save_PearsonGraph(graph , file_name="PearsonGraph.txt" , pass_attr=[] , annonce=None):
    # Fonction générant un fichier texte contenant la liste des arrêtes d'un graph avec leurs attributs.
    # Arguments :
    #   - Un graph.
    #   - Le nom du fichier généré (défaut = PearsonGraph.txt). [string]
    #   - La liste des attributs à ne pas sauvegarder (défaut = liste vide).
    #   - L'indicateur de demande d'affichage de progression (défaut = None).
    #       - si None, pas d'affichage.
    #       - si spécifié, affiche le nombre de points enregistrés à chaque modulo de la valeur spécifiée.
    # Rendu : None

    L_nodes = list(graph.nodes)

    for i,n1 in enumerate(L_nodes):
        for j,n2 in enumerate(L_nodes[i+1::]):
            if graph.has_edge(n1,n2):
                L_attr = list(graph[n1][n2].keys())
                for a in pass_attr : L_attr.remove(a)
                break
        break

    with open(file_name,'w') as file :
        for i,n1 in enumerate(L_nodes):
            for j,n2 in enumerate(L_nodes[i+1::]):
                if graph.has_edge(n1,n2): 
                    file.write(f"{n1} ; {n2}")
                    for attr in L_attr : file.write(f" ; {attr}:{graph[n1][n2][attr]}")
                    file.write('\n')
            if annonce != None :
                if (i%annonce) == 0 and i>0 : print(f"{i} points enregistrés")

#=====================================================================

def save_PearsonGraph_2(graph , file_name="PearsonGraph.txt" , pass_attr=[] , annonce=None):
    # Version 2 de la fonction précédante, celle-ci génére un fichier texte contenant la liste des points
    # d'un graph et celle de ses arêtes avec leurs attributs.
    # Arguments :
    #   - Un graph.
    #   - Le nom du fichier généré (défaut = PearsonGraph.txt). [string]
    #   - La liste des attributs à ne pas sauvegarder (défaut = liste vide).
    #   - L'indicateur de demande d'affichage de progression (défaut = None).
    #       - si None, pas d'affichage.
    #       - si spécifié, affiche le nombre de points et d'arêtes enregistrés à chaque modulo de la valeur spécifiée.
    # Rendu : None

    L_nodes = list(graph.nodes)
    L_edges = list(graph.edges)

    for i,(n1,n2) in enumerate(L_edges) :
        L_attr = list(graph[n1][n2].keys())
        for a in pass_attr : del(graph[n1][n2][a])
    
    with open(file_name,'w') as file :
        for i,node in enumerate(L_nodes) : file.write(f">{node}\n")
        if annonce != None :
            if (i%annonce) == 0 and i>0 : print(f"{i} points enregistrés")
        for i,(n1,n2) in enumerate(L_edges) : 
            file.write(f"{n1} ; {n2}")
            for attr in L_attr : file.write(f" ; {attr}:{graph[n1][n2][attr]}")
            file.write('\n')
            if annonce != None :
                if (i%annonce) == 0 and i>0 : print(f"{i} arêtes enregistrées")

#=====================================================================

def save_PearsonGraph_data(graph , dataframe_file='Data_PearsonGraph.csv' , annonce=None) :
    # Fonction générant un dataframe des données générales d'un graph et l'enregistre dans un fichier csv.
    # Arguments :
    #   - Un graph.
    #   - Le nom du fichier généré (défaut = Data_PearsonGraph.csv). [string]
    #   - L'indicateur de demande d'affichage de progression (défaut = None).
    #       - si None, pas d'affichage.
    #       - si spécifié, affiche le nombre de points analysés à chaque modulo de la valeur spécifiée.
    # Rendu :
    #   - Le dataframe généré, avec les colonnes suivantes :
    #       - 'ID_REF' : le nom des points du graph.
    #       - 'Nb_Edges' : le degré des points.
    #       - 'Nb_Positives' : le nombre de voisins à corrélation positive.
    #       - 'Nb_Negatives' : le nombre de voisins à corrélation négative.
    #       - 'Pearson_Mean' : la moyenne des poids des arrêtes.
    #       - 'Pearson_Std' : l'écart-type des poids des arrêtes.
    #       - 'Pearson_Max' : le poids maximum parmi les arrêtes.
    #       - 'Pearson_Min' : le poids minimum parmi les arrêtes.

    L_nodes = list(graph.nodes)
    Data = {'ID_REF':[] , 'Nb_Edges':[] ,
            'Nb_Positives':[] , 'Nb_Negatives':[] ,
            'Pearson_Mean':[] , 'Pearson_Std' :[] ,
            'Pearson_Max':[] , 'Pearson_Min':[]}
    for i,n1 in enumerate(L_nodes):
        L_Pearson = []
        nb_pos,nb_neg = 0,0
        for n2,attr in graph[n1].items():
            p = list(attr.values())[0]
            sens = list(attr.values())[1]
            if sens > 0 : nb_pos += 1
            else : nb_neg += 1
            L_Pearson.append(float(p))
        n = len(L_Pearson)
        m = np.mean(np.array(L_Pearson))
        s = np.std(np.array(L_Pearson))
        P_max = max(L_Pearson)
        P_min = min(L_Pearson)

        Data['ID_REF'].append(n1)
        Data['Nb_Edges'].append(n)
        Data['Nb_Positives'].append(nb_pos)
        Data['Nb_Negatives'].append(nb_neg)
        Data['Pearson_Mean'].append(m)
        Data['Pearson_Std'].append(s)
        Data['Pearson_Max'].append(P_max)
        Data['Pearson_Min'].append(P_min)
        
        if annonce != None :
            if i%annonce == 0 : print(f"{i} points analysés")
    
    df_graph = pd.DataFrame(Data)
    
    df_graph.to_csv(dataframe_file , index=False)
    
    return df_graph

#=====================================================================

def save_PearsonGraph_data_2(graph , L_anchors_classes=[] , L_anchors_list=[] , dataframe_file='Data_PearsonGraph.csv' , annonce=None) :
    # Fonction générant un dataframe des données générales d'un graph et l'enregistre dans un fichier csv.
    # Arguments :
    #   - Un graph.
    #   - La liste des labels des points d'intérêt.
    #   - La liste des listes de points d'intérêt.
    #   - Le nom du fichier généré (défaut = Data_PearsonGraph.csv). [string]
    #   - L'indicateur de demande d'affichage de progression (défaut = None).
    #       - si None, pas d'affichage.
    #       - si spécifié, affiche le nombre de points analysés à chaque modulo de la valeur spécifiée.
    # Rendu :
    #   - Le dataframe généré, avec les colonnes suivantes :
    #       - 'ID_REF' : le nom des points du graph.
    #       - 'Class' : la classe des points, qui sont soit des Ancre (points d'intérêt avec un label fourni), soit des Candidats.
    #       - 'Nb_Edges' : le degré des points.
    #       - 'Nb_Positives' : le nombre de voisins à corrélation positive.
    #       - 'Nb_Negatives' : le nombre de voisins à corrélation négative.
    #       - 'Pearson_Mean' : la moyenne des poids des arrêtes.
    #       - 'Pearson_Std' : l'écart-type des poids des arrêtes.
    #       - 'Pearson_Max' : le poids maximum parmi les arrêtes.
    #       - 'Pearson_Min' : le poids minimum parmi les arrêtes.

    L_nodes = list(graph.nodes)
    Data = {'ID_REF':[] , 'Class':[] ,
            'Nb_Anchor_Neighbors':[],'Nb_Edges':[] ,
            'Nb_Positives':[] , 'Nb_Negatives':[] ,
            'Pearson_Mean':[] , 'Pearson_Std' :[] ,
            'Pearson_Max':[] , 'Pearson_Min':[]}
    for i,n1 in enumerate(L_nodes):
        LN_anchors = [0 for i in range(len(L_anchors_classes))]
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
                Data['Class'].append(L_anchors_classes[j])
                cand = False
                break
        if cand : Data['Class'].append('Candidate')
        
        if annonce != None :
            if i%annonce == 0 : print(f"{i} points analysés")
    
    df_graph = pd.DataFrame(Data)
    
    df_graph.to_csv(dataframe_file , index=False)
    
    return df_graph

#=====================================================================

def load_PearsonGraph(file_name , annonce=None):
    # Fonction chargeant un graph à partir d'un fichier texte.
    # Arguments :
    #   - Un fichier texte dont chaque ligne représente une arrête du graph en indiquant :
    #       - Le nom des deux points reliés par l'arrête
    #       - Une suite d'attributs représentés des duos ['nom d'attribut':'valeur']
    #       Modèle : Point_1 Point_2 Attribut_1:Valeur_1 Attribut_2:Valeur_2
    #   - L'indicateur de demande d'affichage de progression (défaut = Pas d'affichage). [None ou Int]
    # Une fois le graph chargé, affichage d'un message indiquant le nombre de points qu'il contient.
    # Rendu : Le graph contenu dans le fichier texte.
    
    graph = nx.Graph()
    
    with open(file_name,'r') as file :
        for line in file :
#            print(line[0:-1])
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
                i = len(graph.edges)
                if i%annonce == 0 : print(f"{i} arêtes ont été chargées")
    
    print(f"Chargement d'un graph à {len(graph.nodes)} points terminée")
    return graph

#=====================================================================

def load_PearsonGraph_2(file_name , annonce=None):
    # Version 2 de la fonction précédante, celle-ci charge un graph à partir d'un fichier texte
    # contenant les listes de points et d'arêtes d'un graph.
    # Arguments :
    #   - Un fichier texte dont chaque ligne représente soit :
    #       - Un point du graph précédé du caractère '>'
    #       - Une arête du graph, composée :
    #           - Du nom des deux points reliés par l'arrête
    #           - D'une suite d'attributs représentés par des duos ['nom d'attribut':'valeur']
    #           Modèle : Point_1 ; Point_2 ; Attribut_1:Valeur_1 ; Attribut_2:Valeur_2
    #   - L'indicateur de demande d'affichage de progression (défaut = Pas d'affichage). [None ou Int]
    # Une fois le graph chargé, affichage d'un message indiquant le nombre de points et d'arêtes qu'il contient.
    # Rendu : Le graph contenu dans le fichier texte.
    
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

def update_PearsonGraph(file_name , df , update_name='UpdatedPearsonGraph.txt' , node_col='ID_REF' , threshold=0.5):
    # Fonction ajoutant à un graph déjà existant de nouveaux points et arrêtes.
    # Arguments :
    #   - Un fichier texte dont chaque ligne représente une arrête du graph en indiquant :
    #       - Le nom des deux points reliés par l'arrête
    #       - Une suite d'attributs représentés des duos ['nom d'attribut':'valeur']
    #       Modèle : Point_1 Point_2 Attribut_1:Valeur_1 Attribut_2:Valeur_2
    #   - Un dataframe contenant les points à ajouter au graph.
    #       - IMPORTANT : Doit contenir les points déjà existants afin de calculer les CCP entre anciens et nouveaux points.
    #   - Un nom de fichier texte pour le graph enrichi (défaut = 'UpdatedPearsonGraph.txt'). [string]
    #   - Le nom de la colonne indiquant les noms des points dans le dataframe (défaut = 'ID_REF'). [string]
    #   - Le palier de sélection du CCP (défaut = 0.5). [int or float]
    # Les arrêtes déjà existantes sont systématiquement ajoutées au nouveau fichier. Les nouvelles arrêtes sont ajoutées
    #   au fur et à mesure du calcul de leurs CCP.
    # Rendu : None
    
    ### Récupération des noms des points
    L_nodes = list(df[node_col])
    
    ### Récupération des colonnes de valeurs
    L_L_values = []
    for i,c in enumerate(list(df.keys())):
        if c != node_col : L_L_values.append(list(df[c]))
    L_L_values = np.array([np.array(L) for L in L_L_values])
    
    ### Récupération des arrêtes déjà existantes dans le graph
    L_arretes = []
    with open(update_name,'w') as u_file :
        with open(file_name,'r') as c_file :
            for line in c_file : 
                u_file.write(line)
                L_arretes.append(tuple(line.split()[0:2]))
    
    ### Calcul des coeffiscients de correlation de Pearson : si le 
    ### coeffiscient est supérieur à un certain palier (en valeur absolue),
    ### une arrête est créée entre les noeuds concernés et une ligne est
    ### ajoutée au fichier du graph mis à jour.
    for i,n1 in enumerate(L_nodes):
        t_node = time.time()
        name_1 = n1
        Lv_1 = list(L_L_values[:,i])
        if not G.has_node(name_1) : G.add_node(name_1)
        for j,n2 in enumerate(L_nodes[i+1::]):
            name_2 = n2
            if (name_1,name_2) in L_arretes : continue
            Lv_2 = list(L_L_values[:,i+1+j])
            pearson = np.corrcoef(Lv_1,Lv_2)[0][1]
            if abs(pearson) > threshold :
                s = np.sign(pearson)
                new_line = f"{name_1} {name_2} weight:{abs(pearson)} sign:{s}\n"
                with open(update_name,'a') as u_file : u_file.write(new_line)
        print(f"Update du point {name_1} traité en {time_count(time.time()-t_node)} ({i+1})")

#=====================================================================

def searchInGraph(graph_file , L_gene , temps=False , annonce=False) :
    # Fonction cherchant l'ensemble des voisins de plusieurs points dans un graph.
    # Arguments :
    #   - Un fichier texte dont chaque ligne représente une arrête du graph en indiquant :
    #       - Le nom des deux points reliés par l'arrête
    #       - Une suite d'attributs représentés des duos ['nom d'attribut':'valeur']
    #       Modèle : Point_1 Point_2 Attribut_1:Valeur_1 Attribut_2:Valeur_2
    #   - La liste des noms des points dont les voisins doivent être recherchés (1 point minimum). [Liste de string]
    #   - Le choix de l'affichage du temps de recherche (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage du nombre de voisins trouvés pour chaque point renseigné (défaut = Pas d'affichage) [boolean]
    # Rendu : Un triple dictionnaire où à chaque point renseigné est associé un double dictionnaire où à chaque voisin du point renseigné est associé le dictionnaire des attributs de l'arrête qui les relie.
    
    start = time.time()
    
    D_dico = {}
    for gene in sorted(L_gene) : D_dico[gene] = {} # On force l'ordre alphabétique des points d'ancrage dans le dictionnaire
    
    with open(graph_file,'r') as file :
        for line in file :
            data = line[0:-1].split()
            n1,n2 = data[0:2]
            if (n1 not in L_gene) and (n2 not in L_gene) : continue # Si aucun des points lus n'est à étudier
            if n1 > n2 : n1,n2 = n2,n1 # on force les deux points à suivre l'ordre alphabétique
            
            L_keys,L_values = [],[]
            for attr in data[2::]:
                d = attr.split(':')
                L_keys.append(d[0])
                try : L_values.append(float(d[1]))
                except ValueError : L_values.append(d[1])
            D_attr = dict(zip(L_keys,L_values))
            
            if n1 in L_gene : D_dico[n1][n2] = D_attr
            if n2 in L_gene : D_dico[n2][n1] = D_attr
    
    if temps : print(f"Temps de recherche : {time_count(time.time()-start)}")
    
    L_delete = []
    for gene,dico in D_dico.items():
        if len(dico)==0 : 
            print(f"Le point {gene} n'a aucun voisin dans le graph proposé ou n'y existe pas")
            L_delete.append(gene)
        else : 
            if annonce : print(f"Il y a {len(dico)} points reliés au point {gene} dans le graph proposé")
    for gene in L_delete : del(D_dico[gene])
    
    return D_dico

#=====================================================================

def searchInGraphRestricted(graph_file , L_gene , temps=False , annonce=False) :
    # Fonction cherchant parmi plusieurs points d'un graph lesquels sont voisins entre eux.
    # Arguments :
    #   - Un fichier texte dont chaque ligne représente une arrête du graph. [Liste de string]
    #   - La liste des noms des points dont les voisins doivent être recherchés (2 points minimum). [Liste de string]
    #   - Le choix de l'affichage du temps de recherche (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage du nombre de voisins trouvés pour chaque point renseigné (défaut = Pas d'affichage) [boolean]
    # Rendu : Un triple dictionnaire où à chaque point renseigné est associé un double dictionnaire où à chaque voisin du point renseigné est associé le dictionnaire des attributs de l'arrête qui les relie.
    
    start = time.time()
    
    D_dico = {}
    for gene in sorted(L_gene) : D_dico[gene] = {} # On force l'ordre alphabétique des points d'ancrage dans le dictionnaire

    attributes = []
    
    with open(graph_file,'r') as file :
        for line in file :
            data = line[0:-1].split()
            n1,n2 = data[0:2]
            if (n1 not in L_gene) or (n2 not in L_gene) : continue # Si au moins l'un des points lus n'est à étudier
            if n1 > n2 : n1,n2 = n2,n1 # on force les deux points à suivre l'ordre alphabétique
            L_values = []
            for attr in data[2::]:
                d = attr.split(':')
                if d[0] not in attributes : attributes.append(d[0])
                try : L_values.append(float(d[1]))
                except ValueError : L_values.append(d[1])
            
            D_dico[n1][n2] = [tuple(L_values)]
            D_dico[n2][n1] = [tuple(L_values)]
    
    if temps : print(f"Temps de recherche : {time_count(time.time()-start)}")
    
    L_delete = []
    for gene,dico in D_dico.items():
        if len(dico)==0 : L_delete.append(gene)
        else : 
            if annonce : print(f"Il y a {len(dico)} points reliés au point {gene}")
    for gene in L_delete : del(D_dico[gene])
    
    return D_dico , attributes

#=====================================================================

def searchInMultiGraph(graph_file_list , L_gene , temps=False , annonce=False) :
    # Fonction cherchant les voisins de N points dans M graphs. Il suffit qu'un couple de points soient voisins dans un seul graph pour être retenu.
    # Arguments :
    #   - Une liste de fichiers textes dont chaque ligne représente une arrête du graph. [Liste de string]
    #   - La liste des noms des points dont les voisins doivent être recherchés (1 point minimum). [Liste de string]
    #   - Le choix de l'affichage du temps de recherche (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage du nombre de voisins trouvés pour chaque point renseigné (défaut = Pas d'affichage). [boolean]
    # Rendu : 
    #   - Un double dictionnaire où à chaque point renseigné est associé un dictionnaire associant chacun de ses voisins 
    #       à la liste des valeurs d'attributs de l'arrête qui les relie dans chaque graph.
    #       NOTA BENE : le double dictionnaire est non redondant. Si un point p1 compte parmi ses voisins un point p2,
    #                   ce même point p2 ne compte pas le point p1 parmi ses propre voisins.
    #   - Une liste des noms des attributs des arrêtes. (Ce rendu est notamment utilisé par les fonctions CrossNetwork).
    
    D_dico = {}
    for gene in sorted(L_gene) : D_dico[gene] = {} # On force l'ordre alphabétique des points d'ancrage dans le dictionnaire
    
    attributes = []
    
    for i,graph_file in enumerate(graph_file_list) :
        start = time.time()
    
        with open(graph_file,'r') as file :
            for line in file :
                data = line[0:-1].split()
                n1,n2 = data[0:2]
                if (n1 not in L_gene) and (n2 not in L_gene) : continue # Si aucun des points lus n'est à étudier
                if n1 > n2 : n1,n2 = n2,n1 # on force les deux points à suivre l'ordre alphabétique
                L_values = []
                for attr in data[2::]:
                    d = attr.split(':')
                    if d[0] not in attributes : attributes.append(d[0])
                    try : L_values.append(float(d[1]))
                    except ValueError : L_values.append(d[1])
                
                if n2 not in D_dico[n1].keys() : D_dico[n1][n2] = [tuple(L_values)]
                else : D_dico[n1][n2].append(tuple(L_values))
    
        if temps : print(f"Temps de recherche dans le graph {i+1} : {time_count(time.time()-start)}")
        
    L_delete = []
    for gene,dico in D_dico.items():
        if len(dico)==0 : 
            print(f"Le point {gene} n'existe pas dans le graph proposé")
            L_delete.append(gene)
        else : 
            if annonce : print(f"Il y a {len(dico)} points reliés au point {gene}")
    for gene in L_delete : del(D_dico[gene])
    
    return D_dico , attributes

#=====================================================================

def searchInMultiGraphRestricted(graph_file_list , L_gene , temps=False , annonce=False) :
    # Fonction cherchant parmi N points lesquels sont voisins entre eux dans M graphs. Il suffit qu'un couple de points soient voisins dans un seul graph pour être retenu.
    # Arguments :
    #   - Une liste de fichiers textes dont chaque ligne représente une arrête du graph (2 points minimum). [Liste de string]
    #   - La liste des noms des points dont les voisins doivent être recherchés. [Liste de string]
    #   - Le choix de l'affichage du temps de recherche (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage du nombre de voisins trouvés pour chaque point renseigné (défaut = Pas d'affichage) [boolean]
    # Rendu : Un triple dictionnaire où à chaque point renseigné est associé un double dictionnaire où à chaque voisin du point renseigné est associé le dictionnaire des attributs de l'arrête qui les relie.
    
    D_dico = {}
    for gene in sorted(L_gene) : D_dico[gene] = {} # On force l'ordre alphabétique des points d'ancrage dans le dictionnaire
    
    attributes = []
    
    for i,graph_file in enumerate(graph_file_list) :
        start = time.time()
    
        with open(graph_file,'r') as file :
            for line in file :
                data = line[0:-1].split()
                n1,n2 = data[0:2]
                if (n1 not in L_gene) or (n2 not in L_gene) : continue # Si au moins l'un des points lus n'est à étudier
                if n1 > n2 : n1,n2 = n2,n1 # on force les deux points à suivre l'ordre alphabétique 
                L_values = []
                for attr in data[2::]:
                    d = attr.split(':')
                    if d[0] not in attributes : attributes.append(d[0])
                    try : L_values.append(float(d[1]))
                    except ValueError : L_values.append(d[1])

                if n2 not in D_dico[n1].keys() : D_dico[n1][n2] = [tuple(L_values)]
                else : D_dico[n1][n2].append(tuple(L_values))
                
                if n1 not in D_dico[n2].keys() : D_dico[n2][n1] = [tuple(L_values)]
                else : D_dico[n2][n1].append(tuple(L_values))

        if temps : print(f"Temps de recherche dans le graph {i+1} : {time_count(time.time()-start)}")
    
    L_delete = []
    for gene,dico in D_dico.items():
        if len(dico)==0 : L_delete.append(gene)
        else : 
            if annonce : print(f"Il y a {len(dico)} points reliés au point {gene}")
    for gene in L_delete : del(D_dico[gene])
    
    return D_dico , attributes

#=====================================================================

def searchInMultiGraphUniforme(graph_file_list , L_gene , temps=False , annonce=False) :
    # Fonction cherchant les voisins de N points dans M graphs. Un couple de points n'est retenu que s'ils sont voisins dans tous les M graphs.
    # Arguments :
    #   - Une liste de fichiers textes dont chaque ligne représente une arrête du graph. [Liste de string]
    #   - La liste des noms des points dont les voisins doivent être recherchés. [Liste de string]
    #   - Le choix de l'affichage du temps de recherche (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage du nombre de voisins trouvés pour chaque point renseigné (défaut = Pas d'affichage). [boolean]
    # Rendu : 
    #   - Un double dictionnaire où à chaque point renseigné est associé un dictionnaire associant chacun de ses voisins 
    #       à la liste des valeurs d'attributs de l'arrête qui les relie dans chaque graph.
    #       NOTA BENE : le double dictionnaire est non redondant. Si un point p1 compte parmi ses voisins un point p2,
    #                   ce même point p2 ne compte pas le point p1 parmi ses propre voisins.
    #   - Une liste des noms des attributs des arrêtes. (Ce rendu est notamment utilisé par les fonctions CrossNetwork).
    
    D_dico = {}
    for gene in sorted(L_gene) : D_dico[gene] = {} # On force l'ordre alphabétique des points d'ancrage dans le dictionnaire
    
    attributes = []
    
    for i,graph_file in enumerate(graph_file_list) :
        start = time.time()
        L_clés = list(D_dico.keys()) # Liste des points "ancrage" dans le dictionnaire de voisins (utilisés pour la non-redondance)
    
        with open(graph_file,'r') as file :
            for line in file :
                data = line[0:-1].split()
                n1,n2 = data[0:2]
                if (n1 not in L_gene) and (n2 not in L_gene) : continue # Si aucun des points lus n'est à étudier
                if n1 > n2 : n1,n2 = n2,n1 # on force les deux points à suivre l'ordre alphabétique 
                if n1 not in L_clés : continue # Si le 1er point n'est plus un point d'ancrage (vraiment utile à partir du 2ème graph étudié)
                L_values = []
                for attr in data[2::]:
                    d = attr.split(':')
                    if d[0] not in attributes : attributes.append(d[0])
                    try : L_values.append(float(d[1]))
                    except ValueError : L_values.append(d[1])
                
                if n2 not in D_dico[n1].keys() : D_dico[n1][n2] = [tuple(L_values)]
                else : D_dico[n1][n2].append(tuple(L_values))
        
        if temps : print(f"Temps de recherche dans le graph {i+1} : {time_count(time.time()-start)}")
        
        ### Élagage du dictionnaire en deux étapes, intra et extra ancrage.
        L_del_clés = []
        for gene,dico in D_dico.items():
            L_del_voisins = []
            ### Étape intra ancrage : à partir du 2ème graph analysé, tout point d'ancrage dont un voisin présente moins
            ### d'arrêtes que de graphs analysés (ce qui revient à dire que cette arrête n'est pas systématique) perd ce voisin.
            if i > 0 :
                for voisin,L_edges in dico.items():
                    if len(L_edges) < i+1 : L_del_voisins.append(voisin)
                for voisin in L_del_voisins : del(dico[voisin])
            ### Etape extra ancrage : tout point d'ancrage n'étant associé à aucun voisin est supprimé du dictionnaire.
            if len(dico)==0 : L_del_clés.append(gene)
        for gene in L_del_clés : del(D_dico[gene])
    
    return D_dico , attributes

#=====================================================================

def searchInMultiGraphUniformeRestricted(graph_file_list , L_gene , temps=False , annonce=False , count_line=None) :
    # Fonction cherchant parmi N points lesquels sont voisins entre eux dans M graphs. Un couple de points n'est retenu que s'ils sont voisins dans tous les M graphs.
    #   - Une liste de fichiers textes dont chaque ligne représente une arrête du graph. [Liste de String]
    #   - La liste des noms des points dont les voisins doivent être recherchés. [Liste de String]
    #   - Le choix de l'affichage du temps de recherche (défaut = Pas d'affichage). [Boolean]
    #   - Le choix de l'affichage du nombre de voisins trouvés pour chaque point renseigné (défaut = Pas d'affichage) [Boolean]
    #   - Le choix de l'affichage de la progression de lecture des graphs (défaut = Pas d'affichage) [None ou Int]
    # Rendu : Un triple dictionnaire où à chaque point renseigné est associé un double dictionnaire où à chaque voisin du point renseigné est associé le dictionnaire des attributs de l'arrête qui les relie.
    
    D_dico = {}
    for gene in sorted(L_gene) : D_dico[gene] = {} # On force l'ordre alphabétique des points d'ancrage dans le dictionnaire
    
    attributes = []
    
    for i,graph_file in enumerate(graph_file_list) :
        start = time.time()
        L_clés = list(D_dico.keys()) # Liste des points "ancrage" dans le dictionnaire de voisins (utilisés pour la non-redondance)

        t,r = 0,0
        with open(graph_file,'r') as file :
            for line in file :
                t += 1
                
                data = line[0:-1].split()
                n1,n2 = data[0:2]
                if (n1 not in L_gene) or (n2 not in L_gene) : continue # Si aucun des points lus n'est à étudier
                if n1 > n2 : n1,n2 = n2,n1 # on force les deux points à suivre l'ordre alphabétique 
                if n1 not in L_clés : continue # Si le 1er point n'est plus un point d'ancrage (vraiment utile à partir du 2ème graph étudié)
                L_values = []
                for attr in data[2::]:
                    d = attr.split(':')
                    if d[0] not in attributes : attributes.append(d[0])
                    try : L_values.append(float(d[1]))
                    except ValueError : L_values.append(d[1])

                r += 1
                if n2 not in D_dico[n1].keys() : D_dico[n1][n2] = [tuple(L_values)]
                else : D_dico[n1][n2].append(tuple(L_values))

                if count_line != None :
                    if (t%count_line == 0) : print(f"Graph {i} : {t} arrêtes analysées dont {r} retenues ({time_count(time.time()-start)})")
        
        ### Élagage du dictionnaire en deux étapes, intra et extra ancrage.
        L_del_clés = []
        for gene,dico in D_dico.items():
            L_del_voisins = []
            ### Étape intra ancrage : à partir du 2ème graph analysé, tout point d'ancrage dont un voisin présente moins
            ### d'arrêtes que de graphs analysés (ce qui revient à dire que cette arrête n'est pas systématique) perd ce voisin.
            if i > 0 :
                for voisin,L_edges in dico.items():
                    if len(L_edges) < i+1 : L_del_voisins.append(voisin)
                for voisin in L_del_voisins : del(dico[voisin])
            ### Etape extra ancrage : tout point d'ancrage n'étant associé à aucun voisin est supprimé du dictionnaire.
            if len(dico)==0 : L_del_clés.append(gene)
        for gene in L_del_clés : del(D_dico[gene])

        if temps : print(f"Temps de recherche dans le graph {i+1} : {time_count(time.time()-start)}\n")
    
    if annonce : 
        for gene,dico in D_dico.items() : print(f"Il y a {len(dico)} points reliés au point {gene}")
    
    return D_dico , attributes

#=====================================================================

def CommonRelationAnalysis_2(graph_file_list , gene_list , search_tps=False , search_anc=False , global_tps=False):
    
    start = time.time()
    D_croisement = {'ID_REF':[]}

    for gene in gene_list : D_croisement['ID_REF'].append(gene)

    ### Liste des dictionnaires de sous-dictionnaires de voisins des points étudiés pour chaque graph.
    ### danns chaque dictionnaire, chaque point courant est associé au sous-dictionnaire de ses voisins.
    ### Chaque sous-dictionnaire indique pour chaque voisin sa relation au point associé.
    L_D_dico_G = []
    for file in graph_file_list : L_D_dico_G.append(searchInGraph(file , gene_list , temps=search_tps , annonce=search_anc))

    ### Récupération des listes de voisins de chaque point étudié dans chaque graph
    ### puis intersection de ces listes pour avoir la liste des voisins communs.
    D_L_voisins = {}
    for gene in gene_list : 
        D_L_voisins[gene] = []
    for D_dico_G in L_D_dico_G : 
        for gene,dico in D_dico_G.items() : D_L_voisins[gene].append(list(dico.keys()))
    D_L_com = {}
    for gene in gene_list :
        D_L_com[gene] = D_L_voisins[gene][0]
        for L_voisins in D_L_voisins[gene][1::]: D_L_com[gene] = list(set(D_L_com[gene]) & set(L_voisins))
    
    ### Ajout d'une clé 'nombre de voisins communs' au dictionnaire de croisement
    key_voisin = f"Nb_voisins_communs {len(graph_file_list)}_graphs"
    D_croisement[key_voisin] = [len(D_L_com[gene]) for gene in gene_list]
    
    ### Création du dictionnaire de signes : chaque clé est un produit cartésien entre 
    ### les valeurs (-1,1) de longueur égale au nombre de graph.
    ### Ex pour 2 graphs : (-1,-1) / (-1,1) / (1,-1) / (1,1) <=> arrangement miroir
    keys_signs = list(itertools.product((-1,1),repeat=len(graph_file_list)))
    values_signs = [0 for i in range(len(keys_signs))]
    dico_signs = dict(zip(keys_signs,values_signs))
    
    ### Création du dictionnaire de scores : chaque clé est un 2-uplet dont le 1er
    ### élément représente le nombre de graphs présentants une corrélation négative
    ### et le 2ème élément représente le nombre de graphs une corrélation positive.
    ### Ex pour 3 graphs : (0,3) / (1,2) / (2,1) / (3,0) <=> arrangement miroir
    keys_scores = [(j,len(graph_file_list)-j) for j in range(len(graph_file_list)+1)]
    values_scores = [0 for i in range(len(keys_scores))]
    dico_scores = dict(zip(keys_scores,values_scores))
    
    ### Ajout des arrangements de signs et de scores aux clés du dictionnaire de croisement
    for key in list(dico_signs.keys()) : D_croisement[f"Signes {key}"] = []
    for key in list(dico_scores.keys()) : D_croisement[f"Score {key}"] = []
    
    ### On recherche pour chaque point le signe de sa corrélation avec chacun de ses voisins 
    ### communs dans chaque graph puis on incrémente la clé correspondante à l'ordre de
    ### ces signes.
    for gene,L_com in D_L_com.items() :
        for voisin in L_com :
            L_signes = []
            for D_dico_G in L_D_dico_G:L_signes.append(D_dico_G[gene][voisin]['sign'])
            key = tuple(L_signes)
            dico_signs[key] += 1
        for key,val in dico_signs.items(): D_croisement[f"Signes {key}"].append(val)
        for k1,v1 in dico_signs.items():
            S_sign = sum(k1)
            for k2,v2 in dico_scores.items():
                S_score = (k2[1]-k2[0])
                if S_sign == S_score : dico_scores[k2] += v1
        for key,val in dico_scores.items(): D_croisement[f"Score {key}"].append(val)
        dico_signs.update( (k,0) for k in dico_signs)
        dico_scores.update( (k,0) for k in dico_scores)
    
    if global_tps : print(f"Temps d'analyse de {len(gene_list)} points parmi {len(graph_file_list)} graphs : {time_count(time.time()-start)}")
    
    return pd.DataFrame(D_croisement)

#=====================================================================

def CrossNetwork(graph_file_list , gene_list ,
                 cross_network_file="CrossNetwork.txt" ,
                 search_tps=False , search_anc=False ,
                 searche_cpt=None , global_tps=False):
    # Fonction qui recherche dans N graphs les voisins de M points et génère un nouveau graph en ne gardant 
    # une arrête entre deux points parmi les M que si ces deux points sont reliés dans tous les graphs étudiés.
    # Arguments :
    #   - Une liste de fichiers textes dont chaque ligne représente une arrête d'un graph. [Liste de string]
    #   - La liste des noms des points dont les voisins doivent être recherchés. [Liste de string]
    #   - Le nom du fichier texte où sera sauvegardé le graph d'intersection résultant (défaut = "CrossNetwork.txt"). [string]
    #   - Le choix de l'affichage du temps de recherche pour chaque graph (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage du nombre de voisins trouvés pour chaque point renseigné (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage de la progression de lecture des graphs (défaut = Pas d'affichage). [None ou Int]
    #   - Le choix de l'affichage du temps de construction total du graph d'intersection (défaut = Pas d'affichage). [boolean]
    # Une fois le graph sauvegardé dans un nouveau fichier texte, affichage du nombre d'arrêtes retenus.
    # Rendu : None.
    
    start = time.time()
    
    ### Double dictionnaire de voisins des points étudiés pour chaque graph.
    ### Chaque point courant est associé au sous-dictionnaire de ses voisins.
    ### Chaque sous-dictionnaire indique pour chaque voisin la liste des relation au point courant.
    D_search , attributes = searchInMultiGraphUniformeRestricted(graph_file_list , gene_list , temps=search_tps , annonce=search_anc , count_line=search_cpt)
    
    if global_tps : print(f"Les {len(graph_file_list)} graphs analysés. Temps écoulé depuis lancement : {time_count(time.time()-start)}")
    
    ### Ecriture du réseau dans un fichier texte.
    ### Chaque ligne représente une arrête en indiquant :
    ###     - Les noms des deux points reliés par l'arrête
    ###     - La valeur absolue moyenne des corrélation de Pearson entre les deux points
    ###     - La moyenne des signes des corrélation de Pearson entre les deux points
    file = open(cross_network_file,'w')
    n = 0
    for n1,voisins in D_search.items():
        for n2,L_edges in voisins.items():
            val = [0 for i in range(len(attributes))]
            cross_edge = dict(zip(attributes,val))
            for edge in L_edges :
                for a in range(len(attributes)) : cross_edge[attributes[a]] += edge[a]
            line = f"{n1} {n2}"
            for k,v in cross_edge.items() :
                cross_edge[k] = cross_edge[k]/len(L_edges)
                line += f" {k}:{cross_edge[k]}"
            line += '\n'
            file.write(line)
            n += 1
    file.close()

    if global_tps : print(f"Le réseau d'intersection a été calculé. Temps écoulé depuis lancement : {time_count(time.time()-start)}")
    
    print(f"{n} arrêtes communes ont été retenues")
#=====================================================================

def CrossNetworkConstancy(graph_file_list , gene_list ,
                          cross_network_file="CrossNetwork.txt" ,
                          search_tps=False , search_anc=False ,
                          search_cpt=None , global_tps=False):
    # Fonction qui recherche dans N graphs les voisins de M points et génère un nouveau graph en ne gardant 
    # une arrête entre deux points parmi les M que si ces deux points sont reliés dans tous les graphs étudiés
    # et tous ces liens présente le même signe de corrélation.
    # Arguments :
    #   - Une liste de fichiers textes dont chaque ligne représente une arrête d'un graph. [Liste de string]
    #   - La liste des noms des points dont les voisins doivent être recherchés. [Liste de string]
    #   - Le nom du fichier texte où sera sauvegardé le graph d'intersection résultant (défaut = "CrossNetwork.txt"). [string]
    #   - Le choix de l'affichage du temps de recherche pour chaque graph (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage du nombre de voisins trouvés pour chaque point renseigné (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage de la progression de lecture des graphs (défaut = Pas d'affichage). [None ou Int]
    #   - Le choix de l'affichage du temps de construction total du graph d'intersection (défaut = Pas d'affichage). [boolean]
    # Une fois le graph sauvegardé dans un nouveau fichier texte, affichage du nombre d'arrêtes retenus.
    # Rendu : None.
    
    start = time.time()
    
    ### Double dictionnaire de voisins des points étudiés pour chaque graph.
    ### Chaque point courant est associé au sous-dictionnaire de ses voisins.
    ### Chaque sous-dictionnaire indique pour chaque voisin la liste des relation au point courant.
    D_search , attributes = searchInMultiGraphUniformeRestricted(graph_file_list , gene_list , temps=search_tps , annonce=search_anc , count_line=search_cpt)
    
    if global_tps : print(f"Les {len(graph_file_list)} graphs analysés. Temps écoulé depuis lancement : {time_count(time.time()-start)}")
    
    ### Ecriture du réseau dans un fichier texte.
    ### Chaque ligne représente une arrête en indiquant :
    ###     - Les noms des deux points reliés par l'arrête
    ###     - La valeur absolue moyenne des corrélation de Pearson entre les deux points
    ###     - La moyenne des signes des corrélation de Pearson entre les deux points
    file = open(cross_network_file,'w')
    n = 0
    for n1,voisins in D_search.items():
        for n2,L_edges in voisins.items():
            val = [0 for i in range(len(attributes))]
            cross_edge = dict(zip(attributes,val))
            for edge in L_edges :
                for a in range(len(attributes)) : cross_edge[attributes[a]] += edge[a]
            line = f"{n1} {n2}"
            for k,v in cross_edge.items() :
                cross_edge[k] = cross_edge[k]/len(L_edges)
                line += f" {k}:{cross_edge[k]}"
            if abs(cross_edge['sign']) != 1 : continue
            line += '\n'
            file.write(line)
            n += 1
    file.close()

    if global_tps : print(f"Le réseau d'intersection a été calculé. Temps écoulé depuis lancement : {time_count(time.time()-start)}")
    
    print(f"{n} arrêtes communes ont été retenues")

#=====================================================================

def ThresholdedCrossNetworkConstancy(graph_file_list , gene_list , threshold=None , 
                                     cross_network_file="CrossNetwork.txt" , pass_attr=[] , 
                                     search_tps=False , search_anc=False ,
                                     search_cpt=None , global_tps=False):
    # Fonction qui recherche dans N graphs les voisins de M points et génère un nouveau graph en ne gardant 
    # une arrête entre deux points parmi les M que si ces deux points sont reliés dans tous les graphs étudiés
    # et tous ces liens présente le même signe de corrélation. Un palier T délimite le nombre maximal de voisins qu'un
    # point peut avoir, les voisins choisis sont ceux présentant le poids le plus important en valeur absolue.
    # Arguments :
    #   - Une liste de fichiers textes dont chaque ligne représente une arrête d'un graph. [Liste de string]
    #   - La liste des noms des points dont les voisins doivent être recherchés. [Liste de string]
    #   - La valeur du palier délimitant le nombre paximum de voisins par points du graph d'intersection (défaut = Pas de palier). [None ou Int]
    #   - Le nom du fichier texte où sera sauvegardé le graph d'intersection résultant (défaut = "CrossNetwork.txt"). [string]
    #   - Le choix de l'affichage du temps de recherche pour chaque graph (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage du nombre de voisins trouvés pour chaque point renseigné par graph (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage de la progression de lecture des graphs (défaut = Pas d'affichage). [None ou Int]
    #   - Le choix de l'affichage du temps de construction total du graph d'intersection (défaut = Pas d'affichage). [boolean]
    # Une fois le graph sauvegardé dans un nouveau fichier texte, affichage du nombre d'arrêtes retenus.
    # Rendu : None.
    
    start = time.time()
    
    ### Double dictionnaire de voisins des points étudiés pour chaque graph.
    ### Chaque point courant est associé au sous-dictionnaire de ses voisins.
    ### Chaque sous-dictionnaire indique pour chaque voisin la liste des relation au point courant.
    D_search , attributes = searchInMultiGraphUniformeRestricted(graph_file_list , gene_list , temps=search_tps , annonce=search_anc , count_line=search_cpt)
    
    if global_tps : print(f"Les {len(graph_file_list)} graphs analysés. Temps écoulé depuis lancement : {time_count(time.time()-start)}")
    
    ### Pour chaque point d'ancrage, filtrage OUT des voisins ne présentant pas systématiquement le même signe de corrélation.
    ### Si par ce filtrage, un point d'ancrage se retrouve sans voisins, il est supprimé du dictionnaire.
    ### On en profite pour calculer le poids moyen de l'arrête entre les points d'ancrage et leurs voisins retenus.
    L_delete_1 = []
    n = 0
    for n1,voisins in D_search.items():
        L_delete_2 = []
        for n2,L_edges in voisins.items():
            L_weight , L_sign = [],[]
            for edge in L_edges :
                L_weight.append(edge[0])
                L_sign.append(edge[1])
            if abs(np.mean(L_sign)) != 1 : L_delete_2.append(n2)
            else : voisins[n2] = (np.mean(L_weight),np.mean(L_sign))
        for n2 in L_delete_2 : del(voisins[n2])
        n += len(L_delete_1)
        if len(voisins) == 0 : L_delete_1.append(n1)
    for n1 in L_delete_1 : del(D_search[n1])
    
    if global_tps : print(f"{n} arrêtes à signe non constant ont été supprimées. Temps écoulé depuis lancement : {time_count(time.time()-start)}")
        
    ### Si un palier T est indiqué, pour chaque point d'ancrage possédant un nombre de voisins supérieur 
    ### au palier T, tri des voisins par ordre décroissant de poids absolu et conservation des T-premiers uniquement.
    if threshold != None :
        G = nx.Graph()
        for n1 in list(D_search.keys()):
            voisins = D_search.pop(n1)
            for n2 in list(voisins.keys()):
                w,s = voisins.pop(n2)
                G.add_edge(n1,n2)
                L_attr = {'weight':w , 'sign':s}
                if s == 1 : L_attr['color'] = (0 , 0.5 , 0)
                elif s == -1 : L_attr['color'] = (1 , 0 , 0)
                G[n1][n2].update(L_attr)
        a,b = 0,0
        for n1 in G.nodes :
            if len(G[n1]) <= threshold : continue
            a += 1
            L_voisins , L_attributes = list(G[n1].keys()) , list(G[n1].values())
            L_weight = []
            for D_attributes in L_attributes : L_weight.append(D_attributes['weight'])
            
            Sorted_weight = sorted(L_weight , reverse=True)
            
            Index_weight = []
            for weight in Sorted_weight : Index_weight.append((weight , L_weight.index(weight) , L_voisins[L_weight.index(weight)]))
         
            for i,v in enumerate(Index_weight[threshold::]) :
                G.remove_edge(n1,v[2])
                b += 1
        
        if global_tps : print(f"{b} arrêtes réparties sur {a} points d'ancrage ont été supprimées par le palier. Temps écoulé depuis lancement : {time_count(time.time()-start)}")
    
    ### Ecriture du réseau dans un fichier texte.
    ### Chaque ligne représente une arrête en indiquant :
    ###     - Les noms des deux points reliés par l'arrête
    ###     - La valeur absolue moyenne des corrélation de Pearson entre les deux points
    ###     - Le signe de la corrélation de Pearson entre les deux points

    save_PearsonGraph(G , cross_network_file , pass_attr=pass_attr)
    
    if global_tps : print(f"Le réseau d'intersection a été calculé. Temps écoulé depuis lancement : {time_count(time.time()-start)}")

    n , m = len(G.edges) , len(G.nodes)
    del(G)
    
    print(f"Un total de {n} arêtes reliant {m} points ont été retenues")
    
#=====================================================================

def ThresholdedCrossNetworkConstancy_2(dataframe_list , gene_list , threshold_Pearson=0.5 , 
                                       threshold_neighbours=None , cross_network_file="CrossNetwork.txt" , 
                                       pass_attr=[] , search_tps=False , global_tps=False):
    # Version 2 de la fonction précédente, cette fonction génère un graphe d'intersection à partir de N datasets directement.
    # Parmi les M points renseignés, une arête entre deux points est générée si ces deux points montrent une corrélation
    # significative et de même signe dans chacun des N datasets. L'arête prend alors pour poids la valeur moyenne de la corrélation
    # et pour signe celui commun à toutes les corrélations.
    # Un palier T peut délimiter le nombre maximal de voisins qu'un point peut avoir, les voisins choisis sont ceux présentant le poids
    # le plus important en valeur absolue.
    # Arguments :
    #   - Une liste de dataframes dont chaque ligne représente un gène et son vecteur de valeurs. [Liste de pandas.DataFrame]
    #       !!! TOUS LES DATAFRAMES DOIVENT CONTENIR LA MÊME LISTE DE GENES (mais pas nécessairement dans le même ordre) !!!
    #   - La liste des noms des points dont les voisins doivent être recherchés. [Liste de string]
    #       !!! TOUS LES POINTS FOURNIS DOIVENT ÊTRE TROUVABLE DANS TOUS LES DATAFRAMES !!!
    #   - Le palier de sélection au-dessus duquel on conserve un coefficient de corrélation de Pearson (défaut = 0.5). [Float]
    #   - Le nombre maximal de voisins qu'un point peut avoir (défaut = None). [None ou Int]
    #       Si None, tous les points conservent l'ensemble de leurs voisins.
    #       Si Int, les voisins de chaque points sont ordonnés par corrélation absolue décroissante et seules les T plus grandes sont conservées.
    #       !!! LA CONSERVATION D'UNE ARÊTE EST IMPLICITEMENT RÉCIPROQUE : Un point n2 peut faire parti des T plus proche voisins d'un point n1
    #           mais ce même point n1 peut ne pas faire parti des T plus proches voisins du point n2 !!!
    #       !!! La sélection des arêtes à supprimer dépend de l'ordre dans lequel les points ont été ajoutés dans le graph. Il est tout à fait
    #           possible qu'un point n2 possédant de base moins de T voisins perdent une arête avec un point n1 si ce dernier a plus de T voisins
    #           et que n2 ne se trouve pas dans les T premiers, tout comme il est possible qu'un point n3 possédant initialement plus de T voisins
    #           passe finalement sous ce palier si des points analysés avant lui se voient retirer leur lien avec n3 si celui-ci ne pas parti de leurs
    #           propres T plus proches voisins !!!
    #   - Le nom du fichier texte où sera sauvegardé le graph d'intersection résultant (défaut = "CrossNetwork.txt"). [string]
    #   - La liste des attributs d'arêtes à ne pas écrire dans le fichier de sauvegarde du graph (défaut = []) [Liste de string]
    #   - Le choix de l'affichage du nombre de voisins trouvés pour chaque point renseigné (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage du temps de construction total du graph d'intersection (défaut = Pas d'affichage). [boolean]
    # Une fois le graph sauvegardé dans un nouveau fichier texte, affichage du nombre d'arrêtes retenus.
    # Rendu : None.
    
    start = time.time()
    
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
        
    if global_tps : 
        print(f"Chargement des données terminé.")
        print(f"Temps écoulé depuis lancement : {time_count(time.time()-start)}")
        print("------------------------------")
        
    G = nx.Graph()

    ### Génération initiale du réseau.
    for i,n1 in enumerate(gene_list):
        t_node = time.time()
        for j,n2 in enumerate(gene_list[i+1::]):
            pearson = 0
            skip = False
            for L_values in L_L_values :
                Lv_1 = L_values[i]
                Lv_2 = L_values[i+j+1]
#                print(n1,Lv_1)
#                print(n2,Lv_2)
                p = np.corrcoef(Lv_1,Lv_2)[0][1]
#                print(p)
                if abs(p) < threshold_Pearson : # Si une corrélation est sous le palier, on oublie l'arête
                    skip = True
                    break
                elif pearson == 0 : pearson += p # Si c'est la première corrélation calculée, on la retient
                    
                elif np.sign(p) != np.sign(pearson) : # Si l'une des corrélation suivante n'est pas dans le même sens, on oublie l'arête
                    skip = True
                    break
                else : pearson += p # Sinon, on l'ajoute à/aux corrélation(s) retenue(s)
#            print(pearson)
            if skip == True : continue
            if pearson != 0 :
                pearson = pearson/len(dataframe_list)
#                print("pearson moyen :",pearson)
                G.add_edge(n1,n2)
                L_attr = {'weight':abs(pearson) , 'sign':np.sign(pearson)}
                if np.sign(pearson) == 1 : L_attr['color'] = (0 , 0.5 , 0)
                elif np.sign(pearson) == -1 : L_attr['color'] = (1 , 0 , 0)
                G[n1][n2].update(L_attr)
#            print("\n--------------------\n")
        if search_tps : print(f"Noeud {n1} traité en {time_count(time.time()-t_node)} ({i+1})")
            
    if global_tps : 
        print(f"Intersection terminée. {len(G.edges)} arêtes ont été retenues.")
        print(f"Temps écoulé depuis lancement : {time_count(time.time()-start)}")
        print("------------------------------")

    ### Séléction des voisins les plus corrélés si demandé.
    if threshold_neighbours != None :
        a,b = 0,0
        for n1 in G.nodes :
#            print(n1,len(G[n1]))
            if len(G[n1]) <= threshold_neighbours : continue
            a += 1
            L_voisins , L_attributes = list(G[n1].keys()) , list(G[n1].values())
            L_weight = []
            for D_attributes in L_attributes : L_weight.append(D_attributes['weight'])

            Sorted_weight = sorted(L_weight , reverse=True)

            Index_weight = []
            for weight in Sorted_weight : Index_weight.append((weight , L_weight.index(weight) , L_voisins[L_weight.index(weight)]))
#            print(n1,Index_weight[threshold_neighbours][0])
            for i,v in enumerate(Index_weight[threshold_neighbours::]) :
                G.remove_edge(n1,v[2])
                b += 1
            
        if global_tps : 
            print(f"Filtrage des voisins terminée. {a} points ont perdu {b} arêtes.")
            print(f"Temps écoulé depuis lancement : {time_count(time.time()-start)}")
            print("------------------------------")
    
    ### Ecriture du réseau dans un fichier texte.
    ### Chaque ligne représente une arrête en indiquant :
    ###     - Les noms des deux points reliés par l'arrête
    ###     - La valeur absolue moyenne des corrélation de Pearson entre les deux points
    ###     - Le signe de la corrélation de Pearson entre les deux points

    save_PearsonGraph(G , cross_network_file , pass_attr=pass_attr)
    
    if global_tps : print(f"Le réseau d'intersection a été calculé. Temps écoulé depuis lancement : {time_count(time.time()-start)}")

    n , m = len(G.edges) , len(G.nodes)
    del(G)
    
    print(f"Un total de {n} arêtes reliant {m} points ont été retenues")

#=====================================================================

def ThresholdedCrossNetworkConstancy_3(dataframe_list , gene_list=None , threshold_Pearson=0.5 , 
                                       threshold_neighbours=None , cross_network_file="CrossNetwork.txt" , 
                                       pass_attr=[] , search_tps=False , global_tps=False , ret=False):
    # Version 3 de la fonction précédente, cette fonction génère un graphe d'intersection à partir de N datasets directement.
    # Parmi les M points renseignés, une arête entre deux points est générée si ces deux points montrent une corrélation
    # significative et de même signe dans chacun des N datasets. L'arête prend alors pour poids la valeur moyenne de la corrélation
    # et pour signe celui commun à toutes les corrélations.
    # Un palier T peut délimiter le nombre maximal de voisins qu'un point peut avoir, les voisins choisis sont ceux présentant le poids
    # le plus important en valeur absolue.
    # Arguments :
    #   - Une liste de dataframes dont chaque ligne représente un gène et son vecteur de valeurs. [Liste de pandas.DataFrame]
    #       !!! TOUS LES DATAFRAMES DOIVENT CONTENIR LA MÊME LISTE DE GENES (mais pas nécessairement dans le même ordre) !!!
    #   - La liste des noms des points dont les voisins doivent être recherchés. (défaut = None). [None ou Liste de string]
    #       !!! TOUS LES POINTS FOURNIS DOIVENT ÊTRE TROUVABLE DANS TOUS LES DATAFRAMES !!!
    #       Si défaut, la fonction prend en compte l'entièreté des points présents dans le premier dataframe de la liste fournie.
    #   - Le ou les palier(s) de sélection au-dessus duquel/desquels on conserve un coefficient de corrélation de Pearson (défaut = 0.5). [Float ou liste de Floats]
    #   - Le nombre maximal de voisins qu'un point peut avoir (défaut = None). [None ou Int]
    #       Si None, tous les points conservent l'ensemble de leurs voisins.
    #       Si Int, les voisins de chaque points sont ordonnés par corrélation absolue décroissante et seules les T plus grandes sont conservées.
    #       !!! LA CONSERVATION D'UNE ARÊTE EST IMPLICITEMENT RÉCIPROQUE : Un point n2 peut faire parti des T plus proche voisins d'un point n1
    #           mais ce même point n1 peut ne pas faire parti des T plus proches voisins du point n2 !!!
    #       !!! La sélection des arêtes à supprimer dépend de l'ordre dans lequel les points ont été ajoutés dans le graph. Il est tout à fait
    #           possible qu'un point n2 possédant de base moins de T voisins perdent une arête avec un point n1 si ce dernier a plus de T voisins
    #           et que n2 ne se trouve pas dans les T premiers, tout comme il est possible qu'un point n3 possédant initialement plus de T voisins
    #           passe finalement sous ce palier si des points analysés avant lui se voient retirer leur lien avec n3 si celui-ci ne pas parti de leurs
    #           propres T plus proches voisins !!!
    #   - Le nom du fichier texte où sera sauvegardé le graph d'intersection résultant (défaut = "CrossNetwork.txt"). [string]
    #   - La liste des attributs d'arêtes à ne pas écrire dans le fichier de sauvegarde du graph (défaut = []) [Liste de string]
    #   - Le choix de l'affichage du nombre de voisins trouvés pour chaque point renseigné (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage du temps de construction total du graph d'intersection (défaut = Pas d'affichage). [boolean]
    #   - Le choix de retourné le graph construit (défaut = Pas de retour). [booléen]
    # Une fois le graph sauvegardé dans un nouveau fichier texte, affichage du nombre d'arrêtes retenus.
    # Rendu : None ou le graph.
    
    start = time.time()
    
    ### Récupération de l'entièreté des points en cas de liste d'étude non fournie
    if gene_list == None : gene_list = list(dataframe_list[0]["ID_REF"])
    
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
        
    if global_tps : 
        print(f"Chargement des données terminé.")
        print(f"Temps écoulé depuis lancement : {time_count(time.time()-start)}")
        print("------------------------------")
    
    G = nx.Graph()

    ### Génération initiale du réseau
    for i,n1 in enumerate(gene_list): # Pour chaque point de la liste d'étude
        t_node = time.time()
        for j,n2 in enumerate(gene_list[i+1::]): # Pour chaque point suivant de la liste d'étude
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
            if (i+1)%1000 == 0 : print(f"Noeud {n1} traité en {time_count(time.time()-t_node)} ({i+1}). Temps depuis lancement : {time_count(time.time()-start)}")
            else : print(f"Noeud {n1} traité en {time_count(time.time()-t_node)} ({i+1})")
            
    if global_tps : 
        print(f"Intersection terminée. {len(G.edges)} arêtes ont été retenues.")
        print(f"Temps écoulé depuis lancement : {time_count(time.time()-start)}")
        print("------------------------------")

    ### Séléction des voisins les plus corrélés si demandé.
    if threshold_neighbours != None :
        a,b = 0,0
        for n1 in G.nodes :
#            print(n1,len(G[n1]))
            if len(G[n1]) <= threshold_neighbours : continue
            a += 1
            L_voisins , L_attributes = list(G[n1].keys()) , list(G[n1].values())
            L_weight = []
            for D_attributes in L_attributes : L_weight.append(D_attributes['weight'])

            Sorted_weight = sorted(L_weight , reverse=True)

            Index_weight = []
            for weight in Sorted_weight : Index_weight.append((weight , L_weight.index(weight) , L_voisins[L_weight.index(weight)]))
#            print(n1,Index_weight[threshold_neighbours][0])
            for i,v in enumerate(Index_weight[threshold_neighbours::]) :
                G.remove_edge(n1,v[2])
                b += 1
            
        if global_tps : 
            print(f"Filtrage des voisins terminée. {a} points ont perdu {b} arêtes.")
            print(f"Temps écoulé depuis lancement : {time_count(time.time()-start)}")
            print("------------------------------")
    
    ### Ecriture du réseau dans un fichier texte.
    ### Chaque ligne représente une arrête en indiquant :
    ###     - Les noms des deux points reliés par l'arrête
    ###     - La valeur absolue moyenne des corrélation de Pearson entre les deux points
    ###     - Le signe de la corrélation de Pearson entre les deux points

    save_PearsonGraph(G , cross_network_file , pass_attr=pass_attr)
    
    if global_tps : print(f"Le réseau d'intersection a été calculé. Temps écoulé depuis lancement : {time_count(time.time()-start)}")

    n , m = len(G.edges) , len(G.nodes)
    
    print(f"Un total de {n} arêtes reliant {m} points ont été retenues")

    if ret == True : return G

#=====================================================================

def ThresholdedCrossNetworkConstancy_4(dataframe_list , gene_list=None , threshold_Pearson=0.5 , 
                                       threshold_neighbours=None , pass_attr=[] , search_tps=False ,
                                       global_tps=False , ret=False):
    # Version 4 de la fonction précédente, cette fonction génère un graphe d'intersection à partir de N datasets directement.
    # Parmi les M points renseignés, une arête entre deux points est générée si ces deux points montrent une corrélation
    # significative et de même signe dans chacun des N datasets. L'arête prend alors pour poids la valeur moyenne de la corrélation
    # et pour signe celui commun à toutes les corrélations.
    # Un palier T peut délimiter le nombre maximal de voisins qu'un point peut avoir, les voisins choisis sont ceux présentant le poids
    # le plus important en valeur absolue.
    # Arguments :
    #   - Une liste de dataframes dont chaque ligne représente un gène et son vecteur de valeurs. [Liste de pandas.DataFrame]
    #       !!! TOUS LES DATAFRAMES DOIVENT CONTENIR LA MÊME LISTE DE GENES (mais pas nécessairement dans le même ordre) !!!
    #   - La liste des noms des points dont les voisins doivent être recherchés. (défaut = None). [None ou Liste de string]
    #       !!! TOUS LES POINTS FOURNIS DOIVENT ÊTRE TROUVABLE DANS TOUS LES DATAFRAMES !!!
    #       Si défaut, la fonction prend en compte l'entièreté des points présents dans le premier dataframe de la liste fournie.
    #   - Le ou les palier(s) de sélection au-dessus duquel/desquels on conserve un coefficient de corrélation de Pearson (défaut = 0.5). [Float ou liste de Floats]
    #   - Le nombre maximal de voisins qu'un point peut avoir (défaut = None). [None ou Int]
    #       Si None, tous les points conservent l'ensemble de leurs voisins.
    #       Si Int, les voisins de chaque points sont ordonnés par corrélation absolue décroissante et seules les T plus grandes sont conservées.
    #       !!! LA CONSERVATION D'UNE ARÊTE EST IMPLICITEMENT RÉCIPROQUE : Un point n2 peut faire parti des T plus proche voisins d'un point n1
    #           mais ce même point n1 peut ne pas faire parti des T plus proches voisins du point n2. Par conséquent, il faut que les deux points
    #           se trouvent parmi les plus T corrélés de l'autre pour que l'arête soit conservée. !!!
    #   - Le nom du fichier texte où sera sauvegardé le graph d'intersection résultant (défaut = "CrossNetwork.txt"). [string]
    #   - La liste des attributs d'arêtes à ne pas écrire dans le fichier de sauvegarde du graph (défaut = []) [Liste de string]
    #   - Le choix de l'affichage du nombre de voisins trouvés pour chaque point renseigné (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage du temps de construction total du graph d'intersection (défaut = Pas d'affichage). [boolean]
    #   - Le choix de retourné le graph construit (défaut = Pas de retour). [booléen]
    # Une fois le graph sauvegardé dans un nouveau fichier texte, affichage du nombre d'arrêtes retenus.
    # Rendu : None ou le graph.
    
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
        
    if global_tps : 
        print(f"Chargement des données terminé.")
        print(f"Temps écoulé depuis lancement : {time_count(time.time()-start)}")
        print("------------------------------")
    
    G = nx.Graph()

    ### Génération initiale du réseau
    for i,n1 in enumerate(gene_list): # Pour chaque point de la liste d'étude
        if not G.has_node(n1): G.add_node(n1)
        t_node = time.time()
        for j,n2 in enumerate(gene_list[i+1::]): # Pour chaque point suivant de la liste d'étude
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
            if (i+1)%1000 == 0 : print(f"Noeud {n1} traité en {time_count(time.time()-t_node)} ({i+1}). Temps depuis lancement : {time_count(time.time()-start)}")
            else : print(f"Noeud {n1} traité en {time_count(time.time()-t_node)} ({i+1})")
            
    if global_tps : 
        print(f"Intersection terminée. {len(G.edges)} arêtes ont été retenues.")
        print(f"Temps écoulé depuis lancement : {time_count(time.time()-start)}")
        print("------------------------------")

    ### Séléction des voisins les plus corrélés si demandé.
    if threshold_neighbours != None :
        a = 0
        Set_delete = [] # Liste des arêtes à supprimer une fois qu'elles auront toutes été rassemblées.
        for n1 in gene_list :
#            print(n1,len(G[n1]))
            if len(G[n1]) <= threshold_neighbours : continue
            L_voisins , L_attributes = list(G[n1].keys()) , list(G[n1].values())
            L_weight = []
            for D_attributes in L_attributes : L_weight.append(D_attributes['weight'])

            Sorted_weight = sorted(L_weight , reverse=True)

            Index_weight = []
            for weight in Sorted_weight : Index_weight.append((weight , L_weight.index(weight) , L_voisins[L_weight.index(weight)]))
#            print(n1,Index_weight[threshold_neighbours][0])
            for i,v in enumerate(Index_weight[threshold_neighbours::]) :
#                print(v)
                if (v[2],n1) not in Set_delete : 
                    Set_delete.append((n1,v[2]))
                    a += 1
#        print(f"Set length : {len(Set_delete)}")
#        print(Set_delete)
        for n1,n2 in Set_delete : G.remove_edge(n1,n2)
        
        if global_tps : 
            print(f"Filtrage des voisins terminée. {a} arêtes ont été supprimées.")
            print(f"Temps écoulé depuis lancement : {time_count(time.time()-start)}")
            print("------------------------------")
    
    if global_tps : print(f"Le réseau d'intersection a été calculé. Temps écoulé depuis lancement : {time_count(time.time()-start)}")

    n , m = len(G.edges) , len(G.nodes)
    
    print(f"Composition du graph : {m} points et {n} arêtes.")

    if ret == True : return G

#=====================================================================

def ThresholdedCrossNetworkConstancy_5(dataframe_list , gene_list=None , anchor_list=None , threshold_Pearson=0.5 , 
                                       threshold_neighbours=None , pass_attr=[] , search_tps=False ,
                                       global_tps=False , ret=False):
    # Version 5 de la fonction précédente, cette fonction génère un graphe d'intersection à partir de N datasets directement.
    # On lui renseigne une liste de M points présents dans les N datasets ainsi qu'une liste de X points d'intérêt présents dans les M.
    # Pour chaque couple contenant au moins un point Xi, une arête est générée si le couple montre une corrélation significative et de même signe
    # dans chacun des N datasets. L'arête prend alors pour poids la valeur moyenne de la corrélation et pour signe celui commun à toutes les corrélations.
    # Une fois toutes les corrélations calculées, il est possible d'évaluer un palier dynamique qui va exclure toutes les corrélations ne le passant pas.
    # Un palier T peut délimiter le nombre maximal de voisins qu'un point peut avoir, les voisins choisis sont ceux présentant le poids le plus important en valeur absolue.
    # Arguments :
    #   - Une liste de dataframes dont chaque ligne représente un gène et son vecteur de valeurs. [Liste de pandas.DataFrame]
    #       !!! TOUS LES DATAFRAMES DOIVENT CONTENIR LA MÊME LISTE DE GENES (mais pas nécessairement dans le même ordre) !!!
    #   - La liste des noms des points dont les corrélations doivent être recherchées. (défaut = None). [None ou Liste de string]
    #       !!! TOUS LES POINTS FOURNIS DOIVENT ÊTRE TROUVABLE DANS TOUS LES DATAFRAMES !!!
    #       Si défaut, la fonction prend en compte l'entièreté des points présents dans le premier dataframe de la liste fournie.
    #   - La liste des noms des points d'intérêt dont au moins un doit être présent dans un couple pour que la corrélation soit calculée. (défaut = None). [None ou Liste de string]
    #       !!! TOUS LES POINTS D'INTERET DOIVENT SE TROUVER DANS LA LISTE PRECEDENTE !!!
    #   - Le ou les palier(s) de sélection principal/principaux au-dessus duquel/desquels on conserve un coefficient de corrélation de Pearson (défaut = 0.5). [Float ou liste de Floats]
    #   - Le choix de calculer dynamiquement un second palier de sélection une fois toutes les corrélations moyennes obtenues. (défaut = False). [boolean]
    #       Si True, le palier dynamique est calculé selon la formule P = moyenne + (facteur * écart-type)
    #   - La valeur du facteur multiplicatif a prendre en compte lors du calcul du palier dynamique. (défaut = 1). [Float]
    #   - Le nombre maximal de voisins qu'un point peut avoir (défaut = None). [None ou Int]
    #       Si None, tous les points conservent l'ensemble de leurs voisins.
    #       Si Int, les voisins de chaque points sont ordonnés par corrélation absolue décroissante et seules les T plus grandes sont conservées.
    #       !!! LA CONSERVATION D'UNE ARÊTE EST IMPLICITEMENT RÉCIPROQUE : Un point n2 peut faire parti des T plus proche voisins d'un point n1
    #           mais ce même point n1 peut ne pas faire parti des T plus proches voisins du point n2. Par conséquent, il faut que les deux points
    #           se trouvent parmi les plus T corrélés de l'autre pour que l'arête soit conservée. !!!
    #   - Le nom du fichier texte où sera sauvegardé le graph d'intersection résultant (défaut = "CrossNetwork.txt"). [string]
    #   - La liste des attributs d'arêtes à ne pas écrire dans le fichier de sauvegarde du graph (défaut = []) [Liste de string]
    #   - Le choix de l'affichage du nombre de voisins trouvés pour chaque point renseigné (défaut = Pas d'affichage). [boolean]
    #   - Le choix de l'affichage du temps de construction total du graph d'intersection (défaut = Pas d'affichage). [boolean]
    #   - Le choix de retourné le graph construit (défaut = Pas de retour). [boolean]
    # Affichage du nombre d'arrêtes retenus.
    # Rendu : None ou le graph.
    
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
        
    if global_tps : 
        print(f"Chargement des données terminé.")
        print(f"Temps écoulé depuis lancement : {time_count(time.time()-start)}")
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
                if (i+1)%1000 == 0 : print(f"Noeud {n1} traité en {time_count(time.time()-t_node)} ({i+1}). Temps depuis lancement : {time_count(time.time()-start)}")
                else : print(f"Noeud {n1} traité en {time_count(time.time()-t_node)} ({i+1}/{len(gene_list)})")

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
                if (i+1)%1000 == 0 : print(f"Noeud {n1} traité en {time_count(time.time()-t_node)} ({i+1}). Temps depuis lancement : {time_count(time.time()-start)}")
                else : print(f"Noeud {n1} traité en {time_count(time.time()-t_node)} ({i+1}/{len(anchor_list)})")

    if global_tps : 
        print(f"Intersection initiale terminée. {len(G.edges)} arêtes ont été retenues.")
        print(f"Temps écoulé depuis lancement : {time_count(time.time()-start)}")
        print("------------------------------")

    ### Séléction des voisins les plus corrélés si demandé.
    if threshold_neighbours != None :
        a = 0
        Set_delete = [] # Liste des arêtes à supprimer une fois qu'elles auront toutes été rassemblées.
        for n1 in gene_list :
#            print(n1,len(G[n1]))
            if len(G[n1]) <= threshold_neighbours : continue
            L_voisins , L_attributes = list(G[n1].keys()) , list(G[n1].values())
            L_weight = []
            for D_attributes in L_attributes : L_weight.append(D_attributes['weight'])

            Sorted_weight = sorted(L_weight , reverse=True)

            Index_weight = []
            for weight in Sorted_weight : Index_weight.append((weight , L_weight.index(weight) , L_voisins[L_weight.index(weight)]))
#            print(n1,Index_weight[threshold_neighbours][0])
            for i,v in enumerate(Index_weight[threshold_neighbours::]) :
#                print(v)
                if (v[2],n1) not in Set_delete : 
                    Set_delete.append((n1,v[2]))
                    a += 1
#        print(f"Set length : {len(Set_delete)}")
#        print(Set_delete)
        for n1,n2 in Set_delete : G.remove_edge(n1,n2)
        
        if global_tps : 
            print(f"Filtrage des voisins terminée. {a} arêtes ont été supprimées.")
            print(f"Temps écoulé depuis lancement : {time_count(time.time()-start)}")
            print("------------------------------")
    
    if global_tps : print(f"Le réseau d'intersection a été calculé. Temps écoulé depuis lancement : {time_count(time.time()-start)}")

    n , m = len(G.edges) , len(G.nodes)
    
    print(f"Composition du graph : {m} points et {n} arêtes.")

    if ret == True : return G
    
#=====================================================================

def showCrossNetwork(network_file , n_color='black' , labels=False , 
                     n_size=20 , e_width=1 , mode=None) :
    # Fonction dessinant un graph d'intersection.
    # Arguments :
    #   - Un fichier texte dont chaque ligne représente une arrête du graph en indiquant :
    #       - Le nom des deux points reliés par l'arrête
    #       - Une suite d'attributs représentés des duos ['nom d'attribut':'valeur']
    #       Modèle : Point_1 Point_2 Attribut_1:Valeur_1 Attribut_2:Valeur_2
    #   - La couleur des points du graph (défaut = 'black'). [string or array-like]
    #   - Le choix de l'affichage des noms des points (défaut = Pas d'affichage). [boolean]
    #   - La taille des points (défaut = 20). [int or float]
    #   - L'épaisseur des arrêtes entre les points (défaut = 1). [int or float]
    #   - La méthode de positionnement des points dans le dessin (défaut = None). [None or string]
    #       - si None, la position (x,y) des points est calculée de la manière suivante :
    #           - x = la moyenne des valeurs de l'attribut 'sign' des arrêtes reliées au noeud
    #           - y = le degré du noeud = son nombre de voisins/d'arrêtes
    #       - si spécifié, doit apparaitre dans la liste suivante : 'circular' , 'random' , 'shell', 'spectral' , 'spring'.
    # La couleur des arrêtes est déterminée selon leur signe moyen :
    #   - Corrélation négative dans l'ensemble des graphs (sign = -1) -> rouge pur
    #   - Corrélation négative dans la majorité des graphs (-1 < sign < 0) -> teinte de rouge clair
    #   - Même nombre de graphs à corrélation négative et positive (sign = 0) -> bleu pur
    #   - Corrélation positive dans la majorité des graphs (0 > sign > 1) -> teinte de vert clair
    #   - Corrélation positive dans l'ensemble des graphs (sign = 1) -> vert foncé
    # Rendus (mutuellement exclusifs) :
    #   - None ; Affichage du dessin du graph.
    #   - La valeur 0 si l'indicateur 'mode' renseigné n'est pas pris en charge.
    
    L_mode = [None , 'circular' , 'random' , 'shell', 'spectral' , 'spring']
    if mode not in L_mode :
        print("Mode demandé non pris en charge")
        return 0
    
    G = nx.Graph()
    
    with open(network_file,'r') as file :
        for line in file :
            data = line[0:-1].split()
            name_1,name_2 = data.pop(0),data.pop(0)
            L_keys,L_vals = [],[]
            for d in data :
                key,val = d.split(':')
                L_keys.append(key)
                try : L_vals.append(float(val))
                except ValueError : L_vals.append(val)
            L_attr = dict(zip(L_keys,L_vals))
            
            ### Choix de la couleur des arrêtes :
            if L_attr['sign'] < 0 : L_attr['color'] = (1 , 1-abs(L_attr['sign']) , 1-abs(L_attr['sign']))
            elif L_attr['sign'] > 0 :
                if L_attr['sign'] == 1 : L_attr['color'] = (0 , 0.5 , 0)
                else : L_attr['color'] = (1-abs(L_attr['sign']) , 1 , 1-abs(L_attr['sign']))
            else : L_attr['color'] = (0 , 0 , 1)
            
            G.add_edge(name_1,name_2)
            G[name_1][name_2].update(L_attr)
            
    C_map = []
    for edge in G.edges: C_map.append(G.get_edge_data(edge[0],edge[1])['color'])
    
    print(f"Nombre de points :{len(G.nodes)}")
    print(f"Nombre d'arrêtes :{len(G.edges)}")
    
    if mode == None :
        ### Calcul des positions des noeuds :
        D_pos = {}
        L_x = []
        for i,node in enumerate(list(G.nodes)):
            L_edge_attr = list(G[node].values())
            L_corr = []
            for edge_attr in L_edge_attr : L_corr.append(edge_attr['sign'])
            x_pos = np.mean(L_corr)
            y_pos = G.degree[node]
            D_pos[node] = (x_pos,y_pos)
            L_x.append(x_pos)
        print(f"{min(L_x)} / {max(L_x)}")
        
        nx.draw(G , pos = D_pos , node_color=n_color , edge_color=C_map , 
                with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'circular' : nx.draw_circular(G , node_color=n_color , edge_color=C_map , 
                                             with_labels=labels , node_size = n_size , width=e_width)
    if mode == 'random' : nx.draw_random(G , node_color=n_color , edge_color=C_map , 
                                         with_labels=labels , node_size = n_size , width=e_width)
    if mode == 'shell' : nx.draw_shell(G , node_color=n_color , edge_color=C_map , 
                                       with_labels=labels , node_size = n_size , width=e_width)
    if mode == 'spectral' : nx.draw_spectral(G , node_color=n_color , edge_color=C_map , 
                                             with_labels=labels , node_size = n_size , width=e_width)
    if mode == 'spring' : nx.draw_spring(G , node_color=n_color , edge_color=C_map , 
                                         with_labels=labels , node_size = n_size , width=e_width)

#=====================================================================

def showCrossNetwork_SubGraph(network_file , node_list , n_color=None , 
                              labels=False , n_size=20 , 
                              e_width=1 , mode=None):
    # Fonction dessinant un sous-graph en n'affichant que les points spécifiés et leurs voisins respectifs directs.
    # Arguments :
    #   - Un fichier texte dont chaque ligne représente une arrête du graph en indiquant :
    #       - Le nom des deux points reliés par l'arrête
    #       - Une suite d'attributs représentés des duos ['nom d'attribut':'valeur']
    #       Modèle : Point_1 Point_2 Attribut_1:Valeur_1 Attribut_2:Valeur_2
    #   - La liste des points d'ancrage du sous-graph. [Liste de points OU Liste de liste de points]
    #   - La couleur des points du graph (défaut = None). [None, ICP OU Liste d'ICP]
    #       - Si None, les points d'ancrage sont colorés en Cyan, le reste en Noir.
    #       - Si ICP(Indicateur Couleur Python), tous les points sont colorés selon cet indicateur.
    #       - Si les points d'ancrage soont répartis en plusieurs sous-listes, la Liste d'ICP doit contenir un ICP pour chaque sous-listes.
    #           Le reste des points sera coloré en Noir.
    #   - Le choix de l'affichage des noms des points (défaut = Pas d'affichage). [boolean]
    #   - La taille des points (défaut = 20). [int or float]
    #   - L'épaisseur des arrêtes entre les points (défaut = 1). [int or float]
    #   - La méthode de positionnement des points dans le dessin (défaut = None). [None or string]
    #       - si None, la position (x,y) des points est calculée de la manière suivante :
    #           - x = la moyenne des valeurs de l'attribut 'sign' des arrêtes reliées au noeud
    #           - y = le degré du noeud = son nombre de voisins/d'arrêtes
    #       - si spécifié, doit apparaitre dans la liste suivante : 'circular' , 'random' , 'shell', 'spectral' , 'spring'.
    # La couleur des arrêtes est déterminée selon leur signe moyen :
    #   - Corrélation négative dans l'ensemble des graphs (sign = -1) -> rouge pur
    #   - Corrélation négative dans la majorité des graphs (-1 < sign < 0) -> teinte de rouge clair
    #   - Même nombre de graphs à corrélation négative et positive (sign = 0) -> bleu pur
    #   - Corrélation positive dans la majorité des graphs (0 > sign > 1) -> teinte de vert clair
    #   - Corrélation positive dans l'ensemble des graphs (sign = 1) -> vert foncé
    # Rendus (mutuellement exclusifs) :
    #   - None ; Affichage du dessin du graph.
    #   - La valeur 0 si l'indicateur 'mode' renseigné n'est pas pris en charge.
    #   - La valeur 1 s'il y a incompatibilité de types entre les arguments node_list et n_color.
    
    L_mode = [None , 'circular' , 'random' , 'shell', 'spectral' , 'spring']
    if mode not in L_mode :
        print("Mode demandé non pris en charge")
        return 0

    if type(n_color) == list :
        if type(node_list[0]) != list :
            print("Incompatibilité entre les points d'ancrages et les couleurs fournies")
            return 1
    if type(node_list[0]) == list :
        if type(n_color) != list :
            print("Incompatibilité entre les points d'ancrages et les couleurs fournies")
            return 1
    
    G = nx.Graph()
    
    with open(network_file,'r') as file :
        for line in file :
            data = line[0:-1].split()
            name_1,name_2 = data.pop(0),data.pop(0)
            
            if type(node_list[0]) != list :
                if (name_1 not in node_list) and (name_2 not in node_list) : continue
            else :
                next_line = True
                for sous_list in node_list :
                    if (name_1 in sous_list) or (name_2 in sous_list) : next_line = False
                if next_line : continue
                
            L_keys,L_vals = [],[]
            for d in data :
                key,val = d.split(':')
                L_keys.append(key)
                try : L_vals.append(float(val))
                except ValueError : L_vals.append(val)
            L_attr = dict(zip(L_keys,L_vals))
            
            ### Choix de la couleur des arrêtes :
            if L_attr['sign'] < 0 : L_attr['color'] = (1 , 1-abs(L_attr['sign']) , 1-abs(L_attr['sign']))
            elif L_attr['sign'] > 0 :
                if L_attr['sign'] == 1 : L_attr['color'] = (0 , 0.5 , 0)
                else : L_attr['color'] = (1-abs(L_attr['sign']) , 1 , 1-abs(L_attr['sign']))
            else : L_attr['color'] = (0 , 0 , 1)
            
            if not G.has_edge(name_1,name_2) :
                G.add_edge(name_1,name_2)
                G[name_1][name_2].update(L_attr)
            
    CE_map = []
    for edge in G.edges : CE_map.append(G.get_edge_data(edge[0],edge[1])['color'])

    CN_map = []
    if type(n_color) == list :
        for node in G.nodes :
            check = False
            for i,sous_list in enumerate(node_list):
                if node in sous_list :
                    CN_map.append(n_color[i])
                    check = True
                    break
            if check == False : CN_map.append((0,0,0))
    else :
        if n_color == None :
            for node in G.nodes :
                if node in node_list : CN_map.append((0,1,1))
                else : CN_map.append((0,0,0))
        else :
            for node in G.nodes : CN_map.append(n_color)
    
    
    print(f"Nombre de points :{len(G.nodes)}")
    print(f"Nombre d'arrêtes :{len(G.edges)}")
    
    if mode == None :
        ### Calcul des positions des noeuds :
        D_pos = {}
        L_x = []
        for i,node in enumerate(list(G.nodes)):
            L_edge_attr = list(G[node].values())
            L_corr = []
            for edge_attr in L_edge_attr : L_corr.append(edge_attr['sign'])
            x_pos = np.mean(L_corr)
            y_pos = G.degree[node]
            D_pos[node] = (x_pos,y_pos)
            L_x.append(x_pos)
        print(f"{min(L_x)} / {max(L_x)}")
        
        nx.draw(G , pos = D_pos , node_color=CN_map , edge_color=CE_map , 
                with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'circular' : nx.draw_circular(G , node_color=CN_map , edge_color=CE_map , 
                                             with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'random' : nx.draw_random(G , node_color=CN_map , edge_color=CE_map , 
                                         with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'shell' : nx.draw_shell(G , node_color=n_color , edge_color=CE_map , 
                                       with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'spectral' : nx.draw_spectral(G , node_color=CN_map , edge_color=CE_map , 
                                             with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'spring' : nx.draw_spring(G , node_color=CN_map , edge_color=CE_map , 
                                         with_labels=labels , node_size = n_size , width=e_width)
#=====================================================================

def showPrincipalCrossNetwork_SubGraph(network_file , node_list , 
                                       n_color=None , labels=False , 
                                       n_size=20 , e_width=1 , mode=None):
    # Fonction dessinant un sous-graph en n'affichant que les points spécifiés et les voisins de ces points si ceux-ci sont liés à au moins deux points spécifiés.
    # Arguments :
    #   - Un fichier texte dont chaque ligne représente une arrête du graph en indiquant :
    #       - Le nom des deux points reliés par l'arrête
    #       - Une suite d'attributs représentés des duos ['nom d'attribut':'valeur']
    #       Modèle : Point_1 Point_2 Attribut_1:Valeur_1 Attribut_2:Valeur_2
    #   - La liste des points d'ancrage du sous-graph. [Liste de points OU Liste de liste de points]
    #   - La couleur des points du graph (défaut = None). [None, ICP OU Liste d'ICP]
    #       - Si None, les points d'ancrage sont colorés en Cyan, le reste en Noir.
    #       - Si ICP(Indicateur Couleur Python), tous les points sont colorés selon cet indicateur.
    #       - Si les points d'ancrage soont répartis en plusieurs sous-listes, la Liste d'ICP doit
    #           contenir un ICP pour chaque sous-listes. Le reste des points sera coloré en Noir.
    #   - Le choix de l'affichage des noms des points (défaut = Pas d'affichage). [boolean]
    #   - La taille des points (défaut = 20). [int or float]
    #   - L'épaisseur des arrêtes entre les points (défaut = 1). [int or float]
    #   - La méthode de positionnement des points dans le dessin (défaut = None). [None or string]
    #       - si None, la position (x,y) des points est calculée de la manière suivante :
    #           - x = la moyenne des valeurs de l'attribut 'sign' des arrêtes reliées au noeud
    #           - y = le degré du noeud = son nombre de voisins/d'arrêtes
    #       - si spécifié, doit apparaitre dans la liste suivante : 'circular' , 'random' , 'shell', 'spectral' , 'spring'.
    # La couleur des arrêtes est déterminée selon leur signe moyen :
    #   - Corrélation négative (sign = -1) -> rouge
    #   - Corrélation positive (sign = 1) -> vert
    # Rendus (mutuellement exclusifs) :
    #   - None ; Affichage du dessin du graph.
    #   - La valeur 0 si l'indicateur 'mode' renseigné n'est pas pris en charge.
    #   - La valeur 1 s'il y a incompatibilité de types entre les arguments node_list et n_color.
    
    ### Vérification validité du mode d'affichage
    L_mode = [None , 'circular' , 'random' , 'shell', 'spectral' , 'spring']
    if mode not in L_mode :
        print("Mode demandé non pris en charge")
        return 0

    ### Vérification compatibilité node_list et n_color
    if type(n_color) == list :
        if type(node_list[0]) != list :
            print("Incompatibilité entre les points d'ancrages et les couleurs fournies")
            return 1
    if type(node_list[0]) == list :
        if type(n_color) != list :
            print("Incompatibilité entre les points d'ancrages et les couleurs fournies")
            return 1
    
    ### Remplissage initial du Graph
    G = nx.Graph()
    with open(network_file,'r') as file :
        for line in file :
            data = line[0:-1].split()
            name_1,name_2 = data.pop(0),data.pop(0)
            
            if type(node_list[0]) != list : 
                if (name_1 not in node_list) and (name_2 not in node_list) : continue
            else :
                skip = True
                for sous_list in node_list :
                    if (name_1 in sous_list) or (name_2 in sous_list) : skip = False
                if skip : continue

            L_keys,L_vals = [],[]
            for d in data :
                key,val = d.split(':')
                L_keys.append(key)
                try : L_vals.append(float(val))
                except ValueError : L_vals.append(val)
            L_attr = dict(zip(L_keys,L_vals))
            
            ### Choix de la couleur des arrêtes :
            if L_attr['sign'] < 0 : L_attr['color'] = (1 , 1-abs(L_attr['sign']) , 1-abs(L_attr['sign']))
            elif L_attr['sign'] > 0 :
                if L_attr['sign'] == 1 : L_attr['color'] = (0 , 0.5 , 0)
                else : L_attr['color'] = (1-abs(L_attr['sign']) , 1 , 1-abs(L_attr['sign']))
            else : L_attr['color'] = (0 , 0 , 1)
            
            if not G.has_edge(name_1,name_2) :
                G.add_edge(name_1,name_2)
                G[name_1][name_2].update(L_attr)
    
    ### Tri des points 'ancres' et 'voisins'  du graph
    if type(node_list[0]) == list : L_ancres = node_list
    else :
        L_ancres = []
        for sous_list in node_list : L_ancres += sous_list
    L_voisins = []
    for node in G.nodes :
        if node not in L_ancres : L_voisins.append(node)
    
    ### Suppression des points 'voisins' qui ne sont reliés qu'à une seule ancre
    L_delete = []
    for voisin in L_voisins :
        if len(G[voisin]) == 1 : L_delete.append(voisin)
    for d in L_delete : G.remove_node(d)
    
    ### Remplissage des liste de couleurs des arrêtes et des points
    CE_map = []
    for edge in G.edges : CE_map.append(G.get_edge_data(edge[0],edge[1])['color'])
        
    CN_map = []
    if type(n_color) == list :
        for node in G.nodes :
            check = False
            for i,sous_list in enumerate(node_list):
                if node in sous_list :
                    CN_map.append(n_color[i])
                    check = True
                    break
            if check == False : CN_map.append((0,0,0))
    else :
        if n_color == None :
            for node in G.nodes :
                if node in node_list : CN_map.append((0,1,1))
                else : CN_map.append((0,0,0))
        else :
            for node in G.nodes : CN_map.append(n_color)
    
    
    print(f"Nombre de points : {len(G.nodes)}")
    print(f"Nombre d'arrêtes : {len(G.edges)}")
    
    if mode == None :
        ### Calcul des positions des noeuds :
        D_pos = {}
        L_x = []
        for i,node in enumerate(list(G.nodes)):
            L_edge_attr = list(G[node].values())
            L_corr = []
            for edge_attr in L_edge_attr : L_corr.append(edge_attr['sign'])
            x_pos = np.mean(L_corr)
            y_pos = G.degree[node]
            D_pos[node] = (x_pos,y_pos)
            L_x.append(x_pos)
        print(f"{min(L_x)} / {max(L_x)}")
        
        nx.draw(G , pos = D_pos , node_color=CN_map , edge_color=CE_map , 
                with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'circular' : nx.draw_circular(G , node_color=CN_map , edge_color=CE_map , 
                                             with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'random' : nx.draw_random(G , node_color=CN_map , edge_color=CE_map , 
                                         with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'shell' : nx.draw_shell(G , node_color=n_color , edge_color=CE_map , 
                                       with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'spectral' : nx.draw_spectral(G , node_color=CN_map , edge_color=CE_map , 
                                             with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'spring' : nx.draw_spring(G , node_color=CN_map , edge_color=CE_map , 
                                         with_labels=labels , node_size = n_size , width=e_width)

#=====================================================================

def savePrincipalCrossNetwork_SubGraph(network_file , node_list , n_color=None , 
                                       save_file="CrossNetwork_PrincipalSubGraph.txt"):
    # Fonction calculant un sous-graph des points spécifiés et leurs voisins respectifs directs et l'enregistre dans un fichier texte.
    # Arguments :
    #   - Un fichier texte dont chaque ligne représente une arrête du graph en indiquant :
    #   - Un nom de fichier texte où sera écrit le sous-graph (défaut)
    #   - La liste des points d'ancrage du sous-graph. [Liste de points OU Liste de liste de points]
    #   - La couleur des points du graph (défaut = None). [None, ICP OU Liste d'ICP]
    #       - Si None, les points d'ancrage sont colorés en Cyan, le reste en Noir.
    #       - Si ICP(Indicateur Couleur Python), tous les points sont colorés selon cet indicateur.
    #       - Si les points d'ancrage soont répartis en plusieurs sous-listes, la Liste d'ICP doit contenir un ICP pour chaque sous-listes.
    #           Le reste des points sera coloré en Noir.
    # La couleur des arrêtes est déterminée selon leur signe moyen :
    #   - Corrélation négative dans l'ensemble des graphs (sign = -1) -> rouge pur
    #   - Corrélation négative dans la majorité des graphs (-1 < sign < 0) -> teinte de rouge clair
    #   - Même nombre de graphs à corrélation négative et positive (sign = 0) -> bleu pur
    #   - Corrélation positive dans la majorité des graphs (0 > sign > 1) -> teinte de vert clair
    #   - Corrélation positive dans l'ensemble des graphs (sign = 1) -> vert foncé
    # Rendus (mutuellement exclusifs) :
    #   - None ; Affichage du dessin du graph.
    #   - La valeur 0 si l'indicateur 'mode' renseigné n'est pas pris en charge.
    #   - La valeur 1 s'il y a incompatibilité de types entre les arguments node_list et n_color.
    

    ### Vérification compatibilité node_list et n_color
    if type(n_color) == list :
        if type(node_list[0]) != list :
            print("Incompatibilité entre les points d'ancrages et les couleurs fournies")
            return 1
    if type(node_list[0]) == list :
        if type(n_color) != list :
            print("Incompatibilité entre les points d'ancrages et les couleurs fournies")
            return 1
    
    ### Remplissage initial du Graph
    G = nx.Graph()
    with open(network_file,'r') as file :
        for line in file :
            data = line[0:-1].split()
            name_1,name_2 = data.pop(0),data.pop(0)
            
            if type(node_list[0]) != list : 
                if (name_1 not in node_list) and (name_2 not in node_list) : continue
            else :
                skip = True
                for sous_list in node_list :
                    if (name_1 in sous_list) or (name_2 in sous_list) : skip = False
                if skip : continue

            L_keys,L_vals = [],[]
            for d in data :
                key,val = d.split(':')
                L_keys.append(key)
                try : L_vals.append(float(val))
                except ValueError : L_vals.append(val)
            L_attr = dict(zip(L_keys,L_vals))
            
            if not G.has_edge(name_1,name_2) :
                G.add_edge(name_1,name_2)
                G[name_1][name_2].update(L_attr)
    
    ### Tri des points 'ancres' et 'voisins'  du graph
    if type(node_list[0]) == list : L_ancres = node_list
    else :
        L_ancres = []
        for sous_list in node_list : L_ancres += sous_list
    L_voisins = []
    for node in G.nodes :
        if node not in L_ancres : L_voisins.append(node)
    
    ### Suppression des points 'voisins' qui ne sont reliés qu'à une seule ancre
    L_delete = []
    for voisin in L_voisins :
        if len(G[voisin]) == 1 : L_delete.append(voisin)
    for d in L_delete : G.remove_node(d)
    
    ### Remplissage de la liste de couleurs des points   
    CN_map = []
    if type(n_color) == list :
        for node in G.nodes :
            check = False
            for i,sous_list in enumerate(node_list):
                if node in sous_list :
                    CN_map.append(n_color[i])
                    check = True
                    break
            if check == False : CN_map.append((0,0,0))
    else :
        if n_color == None :
            for node in G.nodes :
                if node in node_list : CN_map.append((0,1,1))
                else : CN_map.append((0,0,0))
        else :
            for node in G.nodes : CN_map.append(n_color)
    
    ### Ecriture du sous-graph dans un fichier text
    with open(save_file,'w') as file :
        for i,node in enumerate(G.nodes) : file.write(f">{node}:{CN_map[i]}"+'\n')
        for edge in G.edges :
            file.write(f"{edge[0]} {edge[1]}")
            for attr,value in G[edge[0]][edge[1]].items():
                file.write(f" {attr}:{value}")
            file.write('\n')
    
    print(f"Nombre de points :{len(G.nodes)}")
    print(f"Nombre d'arrêtes :{len(G.edges)}")

#=====================================================================

def showExclusiveCrossNetwork_SubGraph(network_file , node_list , 
                                       n_color=None , labels=False , 
                                       n_size=20 , e_width=1 , mode=None):
    # Fonction dessinant un sous-graph en n'affichant que les points spécifiés et leurs éventuels arêtes communes.
    # Arguments :
    #   - Un fichier texte dont chaque ligne représente une arrête du graph en indiquant :
    #       - Le nom des deux points reliés par l'arrête
    #       - Une suite d'attributs représentés des duos ['nom d'attribut':'valeur']
    #       Modèle : Point_1 Point_2 Attribut_1:Valeur_1 Attribut_2:Valeur_2
    #   - La liste des points d'ancrage du sous-graph. [Liste de points OU Liste de liste de points]
    #   - La couleur des points du graph (défaut = None). [None, ICP]
    #       - Si None, les points d'ancrage sont colorés en Noir.
    #       - Si ICP(Indicateur Couleur Python), tous les points sont colorés selon cet indicateur.
    #       - Si les points d'ancrage soont répartis en plusieurs sous-listes, la Liste d'ICP doit
    #           contenir un ICP pour chaque sous-listes. Le reste des points sera coloré en Noir.
    #   - Le choix de l'affichage des noms des points (défaut = Pas d'affichage). [boolean]
    #   - La taille des points (défaut = 20). [int or float]
    #   - L'épaisseur des arrêtes entre les points (défaut = 1). [int or float]
    #   - La méthode de positionnement des points dans le dessin (défaut = None). [None or string]
    #       - si None, la position (x,y) des points est calculée de la manière suivante :
    #           - x = la moyenne des valeurs de l'attribut 'sign' des arrêtes reliées au noeud
    #           - y = le degré du noeud = son nombre de voisins/d'arrêtes
    #       - si spécifié, doit apparaitre dans la liste suivante : 'circular' , 'random' , 'shell', 'spectral' , 'spring'.
    # La couleur des arrêtes est déterminée selon leur signe moyen :
    #   - Corrélation négative dans l'ensemble des graphs (sign = -1) -> rouge pur
    #   - Corrélation négative dans la majorité des graphs (-1 < sign < 0) -> teinte de rouge clair
    #   - Même nombre de graphs à corrélation négative et positive (sign = 0) -> bleu pur
    #   - Corrélation positive dans la majorité des graphs (0 > sign > 1) -> teinte de vert clair
    #   - Corrélation positive dans l'ensemble des graphs (sign = 1) -> vert foncé
    # Rendus (mutuellement exclusifs) :
    #   - None ; Affichage du dessin du graph.
    #   - La valeur 0 si l'indicateur 'mode' renseigné n'est pas pris en charge.
    #   - La valeur 1 s'il y a incompatibilité de types entre les arguments node_list et n_color.
    
    ### Vérification validité du mode d'affichage
    L_mode = [None , 'circular' , 'random' , 'shell', 'spectral' , 'spring']
    if mode not in L_mode :
        print("Mode demandé non pris en charge")
        return 0

    ### Vérification compatibilité node_list et n_color
    if type(n_color) == list :
        if type(node_list[0]) != list :
            print("Incompatibilité entre les points d'ancrages et les couleurs fournies")
            return 1
    if type(node_list[0]) == list :
        if type(n_color) != list :
            print("Incompatibilité entre les points d'ancrages et les couleurs fournies")
            return 1
    
    ### Remplissage initial du Graph
    G = nx.Graph()
    
    # Ajout des points d'intérêt.
    if type(node_list[0]) != list : # Si la liste de points spécifiés est une liste simple.
        for node in node_list : G.add_node(node)
    else : # S'il y a plusieurs sous-listes de points spécifiés.
        for sous_list in node_list :
            for node in sous_list : G.add_node(node)
    
    # Ajout des arêtes entre points d'intérêt.
    with open(network_file,'r') as file :
        for line in file :
            data = line[0:-1].split()
            name_1,name_2 = data.pop(0),data.pop(0)
            
            # Présence des deux points courant parmi les points d'intérêt. Ils doivent y être tous les deux.
            if type(node_list[0]) != list : # Si la liste de points spécifiés est une liste simple.
                if (name_1 not in node_list) or (name_2 not in node_list) : continue
            
            else : # S'il y a plusieurs sous-listes de points spécifiés.
                skip_1,skip_2 = True,True
                for sous_list in node_list :
                    if (name_1 in sous_list) : skip_1 = False
                    if (name_2 in sous_list) : skip_2 = False
                if skip_1 or skip_2 : continue

            L_keys,L_vals = [],[]
            for d in data :
                key,val = d.split(':')
                L_keys.append(key)
                try : L_vals.append(float(val))
                except ValueError : L_vals.append(val)
            L_attr = dict(zip(L_keys,L_vals))
            
            ### Choix de la couleur des arrêtes :
            if L_attr['sign'] < 0 : L_attr['color'] = (1 , 1-abs(L_attr['sign']) , 1-abs(L_attr['sign']))
            elif L_attr['sign'] > 0 :
                if L_attr['sign'] == 1 : L_attr['color'] = (0 , 0.5 , 0)
                else : L_attr['color'] = (1-abs(L_attr['sign']) , 1 , 1-abs(L_attr['sign']))
            else : L_attr['color'] = (0 , 0 , 1)
            
            if not G.has_edge(name_1,name_2) :
                G.add_edge(name_1,name_2)
                G[name_1][name_2].update(L_attr)

    
    ### Remplissage des liste de couleurs des arrêtes et des points
    CE_map = []
    for edge in G.edges : CE_map.append(G.get_edge_data(edge[0],edge[1])['color'])
    
    CN_map = []
    if type(n_color) == list :
        for node in G.nodes :
            check = False
            for i,sous_list in enumerate(node_list):
                if node in sous_list :
                    CN_map.append(n_color[i])
                    check = True
                    break
            if check == False : CN_map.append((0,0,0))
    else :
        if n_color == None :
            for node in G.nodes : CN_map.append((0,0,0))
        else :
            for node in G.nodes : CN_map.append(n_color)
    
    
    print(f"Nombre de points : {len(G.nodes)}")
    print(f"Nombre d'arrêtes : {len(G.edges)}")
    
    if mode == None :
        ### Calcul des positions des noeuds :
        D_pos = {}
        L_x = []
        for i,node in enumerate(list(G.nodes)):
            L_edge_attr = list(G[node].values())
            L_corr = []
            for edge_attr in L_edge_attr : L_corr.append(edge_attr['sign'])
            x_pos = np.mean(L_corr)
            y_pos = G.degree[node]
            D_pos[node] = (x_pos,y_pos)
            L_x.append(x_pos)
        print(f"{min(L_x)} / {max(L_x)}")
        
        nx.draw(G , pos = D_pos , node_color=CN_map , edge_color=CE_map , 
                with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'circular' : nx.draw_circular(G , node_color=CN_map , edge_color=CE_map , 
                                             with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'random' : nx.draw_random(G , node_color=CN_map , edge_color=CE_map , 
                                         with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'shell' : nx.draw_shell(G , node_color=n_color , edge_color=CE_map , 
                                       with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'spectral' : nx.draw_spectral(G , node_color=CN_map , edge_color=CE_map , 
                                             with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'spring' : nx.draw_spring(G , node_color=CN_map , edge_color=CE_map , 
                                         with_labels=labels , node_size = n_size , width=e_width)

#=====================================================================

def saveExclusiveCrossNetwork_SubGraph(network_file , node_list , n_color=None , 
                                       save_file="CrossNetwork_ExclusifSubGraph.txt"):
    # Fonction calculant un sous-graph de points spécifiés et leurs liens respectifs et l'enregistre dans un fichier texte.
    # Arguments :
    #   - Un fichier texte dont chaque ligne représente une arrête du graph en indiquant :
    #   - Un nom de fichier texte où sera écrit le sous-graph (défaut)
    #   - La liste des points d'ancrage du sous-graph. [Liste de points OU Liste de liste de points]
    #   - La couleur des points du graph (défaut = None). [None, ICP OU Liste d'ICP]
    #       - Si None, les points d'ancrage sont colorés en Noir.
    #       - Si ICP(Indicateur Couleur Python), tous les points sont colorés selon cet indicateur.
    #       - Si les points d'ancrage soont répartis en plusieurs sous-listes, la Liste d'ICP doit contenir un ICP pour chaque sous-listes.
    # La couleur des arrêtes est déterminée selon leur signe moyen :
    #   - Corrélation négative dans l'ensemble des graphs (sign = -1) -> rouge pur
    #   - Corrélation négative dans la majorité des graphs (-1 < sign < 0) -> teinte de rouge clair
    #   - Même nombre de graphs à corrélation négative et positive (sign = 0) -> bleu pur
    #   - Corrélation positive dans la majorité des graphs (0 > sign > 1) -> teinte de vert clair
    #   - Corrélation positive dans l'ensemble des graphs (sign = 1) -> vert foncé
    # Rendus (mutuellement exclusifs) :
    #   - None ; Affichage du dessin du graph.
    #   - La valeur 0 si l'indicateur 'mode' renseigné n'est pas pris en charge.
    #   - La valeur 1 s'il y a incompatibilité de types entre les arguments node_list et n_color.
    

    ### Vérification compatibilité node_list et n_color
    if type(n_color) == list :
        if type(node_list[0]) != list :
            print("Incompatibilité entre les points d'ancrages et les couleurs fournies")
            return 1
    if type(node_list[0]) == list :
        if type(n_color) != list :
            print("Incompatibilité entre les points d'ancrages et les couleurs fournies")
            return 1
    
    ### Remplissage initial du Graph
    G = nx.Graph()
    
    # Ajout des points d'intérêt.
    if type(node_list[0]) != list : # Si la liste de points spécifiés est une liste simple.
        for node in node_list : G.add_node(node)
    else : # S'il y a plusieurs sous-listes de points spécifiés.
        for sous_list in node_list :
            for node in sous_list : G.add_node(node)
    
    # Ajout des arêtes entre points d'intérêt.
    with open(network_file,'r') as file :
        for line in file :
            data = line[0:-1].split()
            name_1,name_2 = data.pop(0),data.pop(0)
            
            # Présence des deux points courant parmi les points d'intérêt. Ils doivent y être tous les deux.
            if type(node_list[0]) != list : # Si la liste de points spécifiés est une liste simple.
                if (name_1 not in node_list) or (name_2 not in node_list) : continue
            
            else : # S'il y a plusieurs sous-listes de points spécifiés.
                skip_1,skip_2 = True,True
                for sous_list in node_list :
                    if (name_1 in sous_list) : skip_1 = False
                    if (name_2 in sous_list) : skip_2 = False
                if skip_1 or skip_2 : continue

            L_keys,L_vals = [],[]
            for d in data :
                key,val = d.split(':')
                L_keys.append(key)
                try : L_vals.append(float(val))
                except ValueError : L_vals.append(val)
            L_attr = dict(zip(L_keys,L_vals))
            
            if not G.has_edge(name_1,name_2) :
                G.add_edge(name_1,name_2)
                G[name_1][name_2].update(L_attr)

    
    ### Remplissage de la liste de couleurs des points   
    CN_map = []
    if type(n_color) == list :
        for node in G.nodes :
            check = False
            for i,sous_list in enumerate(node_list):
                if node in sous_list :
                    CN_map.append(n_color[i])
                    check = True
                    break
            if check == False : CN_map.append((0,0,0))
    else :
        if n_color == None :
            for node in G.nodes : CN_map.append((0,0,0))
        else :
            for node in G.nodes : CN_map.append(n_color)
    
    ### Ecriture du sous-graph dans un fichier text
    with open(save_file,'w') as file :
        for i,node in enumerate(G.nodes) : file.write(f">{node}:{CN_map[i]}"+'\n')
        for edge in G.edges :
            file.write(f"{edge[0]} {edge[1]}")
            for attr,value in G[edge[0]][edge[1]].items():
                file.write(f" {attr}:{value}")
            file.write('\n')
    
    print(f"Nombre de points :{len(G.nodes)}")
    print(f"Nombre d'arrêtes :{len(G.edges)}")

#=====================================================================

def load_CrossNetwork_SubGraph(subgraph_file , annonce=False):
    
    G = nx.Graph()
    CN_map , CE_map = [],[]
    with open(subgraph_file,'r') as file :
        for line in file :
            ### Ajout des points et relevé de leur couleur
            if line[0] == '>' :
                node,color = line[1:-1].split(':')
                G.add_node(node)
                CN_map.append(tuple(map(float,color[1:-1].split(', '))))
            ### Ajout des arrêtes et relevé de leurs attributs
            else :
                data = line[0:-1].split()
                name_1,name_2 = data.pop(0),data.pop(0)
                
                L_keys,L_vals = [],[]
                for d in data :
                    key,val = d.split(':')
                    L_keys.append(key)
                    try : L_vals.append(float(val))
                    except ValueError : L_vals.append(val)
                L_attr = dict(zip(L_keys,L_vals))
                
                ### Choix de la couleur des arrêtes :
                if L_attr['sign'] == -1 : L_attr['color'] = (1 , 0 , 0)
                elif L_attr['sign'] == 1 : L_attr['color'] = (0 , 0.5 , 0)
                
                if not G.has_edge(name_1,name_2) :
                    G.add_edge(name_1,name_2)
                    G[name_1][name_2].update(L_attr)
                    CE_map.append(L_attr['color'])
    
    if annonce :
        print(f"Nombre de points : {len(G.nodes)}")
        print(f"Nombre d'arrêtes : {len(G.edges)}")
    
    return G , CN_map , CE_map

#=====================================================================

def drawNetwork(G , n_color='black' , e_color='black' , n_size=20 , 
                e_width=1 , labels=False , mode=None , label_size=12) :
    
    if mode == None :
        ### Calcul des positions des noeuds :
        D_pos = {}
        L_x = []
        for i,node in enumerate(list(G.nodes)):
            L_edge_attr = list(G[node].values())
            L_corr = []
            for edge_attr in L_edge_attr : L_corr.append(edge_attr['sign'])
            x_pos = np.mean(L_corr)
            y_pos = G.degree[node]
            D_pos[node] = (x_pos,y_pos)
            L_x.append(x_pos)
        print(f"{min(L_x)} / {max(L_x)}")
        
        nx.draw(G , pos = D_pos , node_color=CN_map , edge_color=CE_map , 
                with_labels=labels , node_size = n_size , width=e_width)
    
    if mode == 'circular' : nx.draw_circular(G , node_color=n_color , edge_color=e_color , 
                                             with_labels=labels , node_size = n_size ,
                                             width=e_width , font_size=label_size)
    
    if mode == 'random' : nx.draw_random(G , node_color=n_color , edge_color=e_color , 
                                         with_labels=labels , node_size = n_size ,
                                         width=e_width , font_size=label_size)
    
    if mode == 'shell' : nx.draw_shell(G , node_color=n_color , edge_color=e_color , 
                                       with_labels=labels , node_size = n_size ,
                                       width=e_width , font_size=label_size)
    
    if mode == 'spectral' : nx.draw_spectral(G , node_color=n_color , edge_color=e_color , 
                                             with_labels=labels , node_size = n_size ,
                                             width=e_width , font_size=label_size)
    
    if mode == 'spring' : nx.draw_spring(G , node_color=n_color , edge_color=e_color , 
                                         with_labels=labels , node_size = n_size ,
                                         width=e_width , font_size=label_size)
    
#=====================================================================
    
def over_expression(df , n , name_col='ID_REF' , tell=False):
    
    L_gene = list(zip(df.index.to_list(),df[name_col]))
    L_col = list(df.keys()[1::])
    
    ### Filtrage des gènes sur-exprimés dans chaque colonne
    D_over_per_col = {}
    for col in L_col :
        std = np.std(df[col])
        sous_df = df.loc[df[col] > (n*std)]
        L_idx = sous_df.index.to_list()
        L_name = list(sous_df[name_col])
        D_over_per_col[col] = (std,list(zip(L_idx,L_name)))
        if tell == True : print(f"{col} : std = {std} / max = {max(df[col])}")
#        print(f"{col} / {D_over_per_col[col][0]} / {len(D_over_per_col[col][1])}")
    
    ### Filtrage des colonnes où chaque gène est sur-exprimés
    D_over_per_gene = {}
    for gene in L_gene :
        for col,val in D_over_per_col.items():
            if gene in val[1] :
                if gene not in D_over_per_gene.keys(): D_over_per_gene[gene] = [col]
                else : D_over_per_gene[gene].append(col)
#    print(len(D_over_per_gene))

    ### Tri des gènes par nombre de colonnes où ils sont sur-exprimés
    L_keys = [i+1 for i in range(len(L_col))]
    L_list = [[] for i in range(len(L_col))]
    D_tri_over = dict(zip(L_keys,L_list))
    for gene,val in D_over_per_gene.items():
        k = len(val)
        if k in D_tri_over.keys(): D_tri_over[k].append(gene)
        else : D_tri_over[k] = [gene]
#    for key,val in D_tri_over.items(): print(key,len(val))
    
    return D_over_per_col , D_over_per_gene , D_tri_over

#=====================================================================

def under_expression(df , n , name_col='ID_REF' , tell = False):
    
    L_gene = list(zip(df.index.to_list(),df[name_col]))
    L_col = list(df.keys()[1::])
    
    ### Filtrage des gènes sous-exprimés dans chaque colonne
    D_under_per_col = {}
    for col in L_col :
        std = np.std(df[col])
        sous_df = df.loc[df[col] < (-n*std)]
        L_idx = sous_df.index.to_list()
        L_name = list(sous_df[name_col])
        D_under_per_col[col] = (std,list(zip(L_idx,L_name)))
        if tell == True : print(f"{col} : std = {std} / min = {min(df[col])}")
#        print(f"{col} / {D_under_per_col[col][0]} / {len(D_under_per_col[col][1])}")
    
    ### Filtrage des colonnes où chaque gène est sous-exprimés
    D_under_per_gene = {}
    for gene in L_gene :
        for col,val in D_under_per_col.items():
            if gene in val[1] :
                if gene not in D_under_per_gene.keys(): D_under_per_gene[gene] = [col]
                else : D_under_per_gene[gene].append(col)
#    print(len(D_under_per_gene))

    ### Tri des gènes par nombre de colonnes où ils sont sous-exprimés
    L_keys = [i+1 for i in range(len(L_col))]
    L_list = [[] for i in range(len(L_col))]
    D_tri_under = dict(zip(L_keys,L_list))
    for gene,val in D_under_per_gene.items():
        k = len(val)
        if k in D_tri_under.keys(): D_tri_under[k].append(gene)
        else : D_tri_under[k] = [gene]
#    for key,val in D_tri_under.items(): print(key,len(val))

    return D_under_per_col , D_under_per_gene , D_tri_under

#=====================================================================

def higher_expression(df , n , name_col='ID_REF' , tell = False):
    
    L_gene = list(zip(df.index.to_list(),df[name_col]))
    L_col = list(df.keys()[1::])
    
    ### Filtrage des gènes sur-exprimés dans chaque colonne
    D_higher_per_col = {}
    for col in L_col :
        std = np.std(df[col])
        sous_df = df.loc[abs(df[col]) > (n*std)]
        L_idx = sous_df.index.to_list()
        L_name = list(sous_df[name_col])
        D_higher_per_col[col] = (std,list(zip(L_idx,L_name)))
        if tell == True : print(f"{col} : std = {std} / max = {max(df[col])} / min = {min(df[col])}")
#        print(f"{col} / {D_higher_per_col[col][0]} / {len(D_higher_per_col[col][1])}")
    
    ### Filtrage des colonnes où chaque gène est sur-exprimés
    D_higher_per_gene = {}
    for gene in L_gene :
        for col,val in D_higher_per_col.items():
            if gene in val[1] :
                if gene not in D_higher_per_gene.keys(): D_higher_per_gene[gene] = [col]
                else : D_higher_per_gene[gene].append(col)
#    print(len(D_higher_per_gene))

    ### Tri des gènes par nombre de colonnes où ils sont sur-exprimés
    L_keys = [i+1 for i in range(len(L_col))]
    L_list = [[] for i in range(len(L_col))]
    D_tri_higher = dict(zip(L_keys,L_list))
    for gene,val in D_higher_per_gene.items():
        k = len(val)
        if k in D_tri_higher.keys(): D_tri_higher[k].append(gene)
        else : D_tri_higher[k] = [gene]
#    for key,val in D_tri_higher.items(): print(key,len(val))
    

    return D_higher_per_col , D_higher_per_gene , D_tri_higher

#=====================================================================

def KMeans_clustering_1(df , n_points=None , n_clusters=2 , random_state=0):
    
    if n_points==None : n_points = len(df)
    
    L_colonnes = list(df.keys())[1::]
    
    L_values = []
    for col in L_colonnes : L_values.append(list(df[col][0:n_points]))
    L_values = np.transpose(L_values)
    
    model = sk_KMeans(n_clusters=n_clusters , random_state=random_state)
    KM = model.fit(L_values)
    labels = KM.labels_
    centers = KM.cluster_centers_
    
    return KM , labels , centers

#=====================================================================

def Spectral_clustering_1(df , n_points=None , n_clusters=2 , assign_labels='discretize' , random_state=0):
    
    if n_points==None : n_points = len(df)
    
    L_colonnes = list(df.keys())[1::]
    
    L_values = []
    for col in L_colonnes : L_values.append(list(df[col][0:n_points]))
    L_values = np.transpose(L_values)
    
    model = sk_Spectral(n_clusters=n_clusters , assign_labels=assign_labels , random_state=random_state)
    SC = model.fit(L_values)
    labels = SC.labels_
    
    return SC , labels

#=====================================================================

def AffinityPropagation_1(df , n_points=None , random_state=0):
    
    if n_points==None : n_points = len(df)
    
    L_colonnes = list(df.keys())[1::]
    
    L_values = []
    for col in L_colonnes : L_values.append(list(df[col][0:n_points]))
    L_values = np.transpose(L_values)
    
    model = sk_AffiPropa(random_state=random_state)
    AP = model.fit(L_values)
    labels = AP.labels_
    centers_idx = AP.cluster_centers_indices_
    centers_coord = AP.cluster_centers_
    
    return AP , labels , centers_idx , centers_coord

#=====================================================================

def Agglomerative_clustering_1(df , n_points=None , n_clusters=2 , affinity='euclidean' , 
                               linkage='ward' , distance_threshold=None):
    
    if n_points==None : n_points = len(df)
    
    L_colonnes = list(df.keys())[1::]
    
    L_values = []
    for col in L_colonnes : L_values.append(list(df[col][0:n_points]))
    L_values = np.transpose(L_values)
    
    model = sk_Agglo(n_clusters=n_clusters , affinity=affinity , 
                     linkage=linkage , distance_threshold=distance_threshold)
    AC = model.fit(L_values)
    labels = AC.labels_
    
    return SC , labels

#=====================================================================

def KNN_clustering_1(df , n_points=None , n_neighbors=2 , algo='auto') :
    # Fonction appliquant l'algorithme des K-Plus Proches Voisins sur les données d'un dataframe.
    # Arguments :
    #   - Un DataFrame. [pandas.DataFrame]
    #   - Un nombre de lignes à traiter (défaut = None ; si défaut, le corps de la fonction considère l'ensemble du dataframe). [int]
    #   - Le nombre de plus proche voisins à rechercher pour chaque données (défaut = 2). [int]
    #   - La méthode employé par l'algorithme (défaut = 'auto'). [string]
    # Rendu :
    #   - Le modèle entier.
    #   - La double liste de voisinage où chaque sous-liste contient les indices des voisins retenus pour la donnée de l'indice courant.
    
    if n_points==None : n_points = len(df)
    
    L_colonnes = list(df.keys())[1::]
    
    L_values = []
    for col in L_colonnes : L_values.append(list(df[col][0:n_points]))
    L_values = np.transpose(L_values)
    
    model = sk_KNN(n_neighbors=n_neighbors , algorithm=algo)
    KNN = model.fit(L_values)
    neighbors = model.kneighbors()[1]
    
    return KNN , neighbors

#=====================================================================

def KNN_clustering_2(df , n_points=None , factor=1 , algo='auto') :
    # Version alternative de la fontion précédente, où le nombre de voisins retenus est choisi dynamiquement.
    # Pour chaque donnée, l'agorithme calcule la distance des toutes les autres données, et ne retient que celles dont la
    # distance est inférieure à la distance moyenne soustrait d'un écart-type.
    # Arguments :
    #   - Un DataFrame. [pandas.DataFrame]
    #   - Un nombre de lignes à traiter (défaut = None ; si défaut, le corps de la fonction considère l'ensemble du dataframe). [int]
    #   - Un facteur de multiplication à appliquer à l'écart-type lors du calcul du palier de voisinage (défaut = 1). [float]
    #   - La méthode employé par l'algorithme (défaut = 'auto'). [string]
    # Rendu :
    #   - Le modèle entier.
    #   - Les paramètres du modèle.
    #   - La double liste des distances où chaque sous-liste contient la distance correspondante au voisin d'indice égal.
    #   - La double liste de voisinage où chaque sous-liste contient les indices des voisins retenus pour la donnée de l'indice courant.
    
    if n_points==None : n_points = len(df)
    
    L_colonnes = list(df.keys())[1::]
    L_names = list(df['ID_REF'])
    
    L_values = []
    for col in L_colonnes : L_values.append(list(df[col][0:n_points]))
    L_values = np.transpose(L_values)
    
    model = sk_KNN(n_neighbors=n_points-1 , algorithm=algo)
    KNN = model.fit(L_values)
    params = KNN.get_params()
    all_distances , all_neighbors = model.kneighbors()
    
    distances , neighbors = [] , []
    for i,L_dist in enumerate(all_distances):
        L_dist = list(L_dist)
        L_neigh = list(all_neighbors[i])
#        print(i,L_names[i])
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
#        for j,n in enumerate(L_neigh): print(n,L_names[n],L_dist[j])
        distances.append(L_dist)
        neighbors.append(L_neigh)
#        print("-----------------")
    
    return KNN , params , distances , neighbors

#=====================================================================

def Cluster_filtering_1(df , n_points=None , method='KMeans' , n_clusters=2 , 
                        n_neighbors=2 , random_state=0 , affinity='euclidean' , 
                        linkage='ward' , distance_threshold=None , algo='auto' , factor=1 ,
                        RBH=False , plot=False , title=None , xlabel=None , ylabel=None ,
                        title_size = 20 , label_size=20 , tick_size = 20):
    
    if n_points==None : n_points = len(df)
    L_points = list(df['ID_REF'][0:n_points])
    
    if method == 'KMeans' : 
        Labels , Centers = KMeans_clustering_1(df=df , n_points=n_points , 
                                               n_clusters=n_clusters , random_state=random_state)[1::]
    if method == 'Spectral' : 
        Labels = Spectral_clustering_1(df=df , n_points=n_points , n_clusters=n_clusters , 
                                       assign_labels='discretize' , random_state=0)[1]
    if method == 'Affinity' : 
        Labels , Centers_idx , Centers_coord = AffinityPropagation_1(df=df , n_points=n_points ,
                                                                     random_state=random_state)[1::]
    if method == 'Agglomerative' : 
        Labels = Agglomerative_clustering_1(df=df , n_points=n_points , n_clusters=n_clusters , 
                                            affinity=affinity , linkage=linkage , 
                                            distance_threshold=distance_threshold)[1]
    if method == 'KNN_1' :
        Neighbors = KNN_clustering_1(df=df , n_points=n_points , 
                                     n_neighbors=n_neighbors , algo=algo)[1]
    if method == 'KNN_2' :
        Neighbors = KNN_clustering_2(df=df , n_points=n_points , factor=factor , algo=algo)[3]
        print("Clustering KNN ver.2 terminé")
    
    dico = {}
    
    if method in ['KNN_1','KNN_2'] : # association de chaque point à ses voisins
        for i,point in enumerate(L_points):
            dico[point] = []
            for v_idx in Neighbors[i] : dico[point].append(L_points[v_idx])
        if RBH :
            dico_RBH = {}
            for n1,voisins in dico.items():
                dico_RBH[n1] = []
                for n2 in voisins :
                    if n1 in dico[n2] : dico_RBH[n1].append(n2)
            print("Clustering KNN : filtration RBH terminé")
    
    elif method == 'Affinity' : # n_clusters inutile
        for i,cen_idx in enumerate(Centers_idx) : 
            groupe = []
            for j,lab in enumerate(Labels):
                if lab==i : groupe.append(L_points[j])
            center = L_points[cen_idx]
            dico[i] = (center,list(Centers_coord[i]),groupe)
    else :
        for i in range(n_clusters):
            groupe = []
            for j,lab in enumerate(Labels):
                if lab==i : groupe.append(L_points[j])
            if method in 'KMeans' : dico[i] = (list(Centers[i]),groupe)
            if method in ['Spectral','Agglomerative'] : dico[i] = groupe
    
    if plot==True :
        if method in ['KNN_1','KNN_2'] :
            print("No plot available for KNN clustering.")
            if RBH : return dico , dico_RBH
            else : return dico
        x = list(dico.keys())
        y = []
        if method  == 'KMeans' :
            for val in list(dico.values()) : y.append(len(val[1]))
        if method in ['Spectral','Agglomerative'] :
            for val in list(dico.values()) : y.append(len(val))
        if method  == 'Affinity' :
            for val in list(dico.values()) : y.append(len(val[2]))
        plt.bar(x,y)
        if len(dico) < 21 : plt.xticks(np.arange(0,len(dico),step=1),fontsize=tick_size)
        elif len(dico) in range(21,51) : plt.xticks(np.arange(0,len(dico),step=5),fontsize=tick_size)
        elif len(dico) in range(51,101) : plt.xticks(np.arange(0,len(dico),step=10),fontsize=tick_size)
        elif len(dico) in range(101,301) : plt.xticks(np.arange(0,len(dico),step=20),fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        if title != None : plt.title(title,fontsize=title_size)
        if xlabel != None : plt.xlabel(xlabel,fontsize=label_size)
        if ylabel != None : plt.ylabel(ylabel,fontsize=label_size)
        plt.show()

    if method in ['KNN_1','KNN_2'] and RBH : return dico , dico_RBH
    return dico

#=====================================================================

def Cluster_filtering_2(df , n_points=None , method='KMeans' , n_clusters=2 , 
                        n_neighbors=2 , random_state=0 , affinity='euclidean' , 
                        linkage='ward' , distance_threshold=None , algo='auto' , factor=1 ,
                        RBH=False , plot=False , title=None , xlabel=None , ylabel=None ,
                        title_size = 20 , label_size=20 , tick_size = 20):
    ### Variante : dico retourné différent selon version KNN (v1 = juste voisins, v2 = voisins+distances)
    
    if n_points==None : n_points = len(df)
    L_points = list(df['ID_REF'][0:n_points])
    
    if method == 'KMeans' : 
        Labels , Centers = KMeans_clustering_1(df=df , n_points=n_points , 
                                               n_clusters=n_clusters , random_state=random_state)[1::]
    if method == 'Spectral' : 
        Labels = Spectral_clustering_1(df=df , n_points=n_points , n_clusters=n_clusters , 
                                       assign_labels='discretize' , random_state=0)[1]
    if method == 'Affinity' : 
        Labels , Centers_idx , Centers_coord = AffinityPropagation_1(df=df , n_points=n_points ,
                                                                     random_state=random_state)[1::]
    if method == 'Agglomerative' : 
        Labels = Agglomerative_clustering_1(df=df , n_points=n_points , n_clusters=n_clusters , 
                                            affinity=affinity , linkage=linkage , 
                                            distance_threshold=distance_threshold)[1]
    if method == 'KNN_1' :
        Neighbors = KNN_clustering_1(df=df , n_points=n_points , 
                                     n_neighbors=n_neighbors , algo=algo)[1]
    if method == 'KNN_2' :
        Distances, Neighbors = KNN_clustering_2(df=df , n_points=n_points , factor=factor , algo=algo)[2::]
        print("Clustering KNN ver.2 terminé")
    
    dico = {}
    
    if method == 'KNN_1' : # association de chaque point à ses voisins
        for i,point in enumerate(L_points):
            dico[point] = []
            for v_idx in Neighbors[i] : dico[point].append(L_points[v_idx])
        if RBH :
            dico_RBH = {}
            for n1,voisins in dico.items():
                dico_RBH[n1] = []
                for n2 in voisins :
                    if n1 in dico[n2] : dico_RBH[n1].append(n2)
            print("Clustering KNN : filtration RBH terminé")

    if method == 'KNN_2' : # association de chaque point à ses voisins et leurs distances
        for i,point in enumerate(L_points):
            dico[point] = []
            L_neig = Neighbors[i]
            L_dist = Distances[i]
#            print(f"{i} {point} {L_neig} {L_dist}")
            for j,v_idx in enumerate(Neighbors[i]) :
                data = (L_points[v_idx] , L_dist[j])
                dico[point].append(data)
#            print(dico[point])
#            print("-----")
        if RBH :
            dico_RBH = {}
            for n1,voisins in dico.items():
                dico_RBH[n1] = []
                for (n2,dist) in voisins :
                    for couple in dico[n2] :
                        if couple[0] == n1 :
#                            print(n1,n2,dist,couple)
                            dico_RBH[n1].append((n2,dist))
                            break
#                print(f"{n1}\n{dico_RBH[n1]}")
            print("Clustering KNN : filtration RBH terminé")

    elif method == 'Affinity' : # n_clusters inutile
        for i,cen_idx in enumerate(Centers_idx) : 
            groupe = []
            for j,lab in enumerate(Labels):
                if lab==i : groupe.append(L_points[j])
            center = L_points[cen_idx]
            dico[i] = (center,list(Centers_coord[i]),groupe)
    else :
        for i in range(n_clusters):
            groupe = []
            for j,lab in enumerate(Labels):
                if lab==i : groupe.append(L_points[j])
            if method in 'KMeans' : dico[i] = (list(Centers[i]),groupe)
            if method in ['Spectral','Agglomerative'] : dico[i] = groupe
    
    if plot==True :
        if method in ['KNN_1','KNN_2'] :
            print("No plot available for KNN clustering.")
            if RBH : return dico , dico_RBH
            else : return dico
        x = list(dico.keys())
        y = []
        if method  == 'KMeans' :
            for val in list(dico.values()) : y.append(len(val[1]))
        if method in ['Spectral','Agglomerative'] :
            for val in list(dico.values()) : y.append(len(val))
        if method  == 'Affinity' :
            for val in list(dico.values()) : y.append(len(val[2]))
        plt.bar(x,y)
        if len(dico) < 21 : plt.xticks(np.arange(0,len(dico),step=1),fontsize=tick_size)
        elif len(dico) in range(21,51) : plt.xticks(np.arange(0,len(dico),step=5),fontsize=tick_size)
        elif len(dico) in range(51,101) : plt.xticks(np.arange(0,len(dico),step=10),fontsize=tick_size)
        elif len(dico) in range(101,301) : plt.xticks(np.arange(0,len(dico),step=20),fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        if title != None : plt.title(title,fontsize=title_size)
        if xlabel != None : plt.xlabel(xlabel,fontsize=label_size)
        if ylabel != None : plt.ylabel(ylabel,fontsize=label_size)
        plt.show()

    if method in ['KNN_1','KNN_2'] and RBH : return dico , dico_RBH
    return dico

#=====================================================================

def KM1D_v1(df , col_num , n_points=None , n_clusters=3 , random_state=0):
    
    if n_points==None : n_points = len(df)
    
    L_names = list(df['ID_REF'])
    L_colonnes = list(df.keys())[1::]
    
    L_values = list(df[L_colonnes[col_num]][0:n_points])
    for i,value in enumerate(L_values) : L_values[i] = [value]
    
#    for i,names in enumerate(L_names) : print(names,L_values[i])
    

    model = sk_KMeans(n_clusters=n_clusters , random_state=random_state)
    KM = model.fit(L_values)
    labels = KM.labels_
    centers = KM.cluster_centers_
    
    dico = {}
    for i in range(n_clusters):
        dico[i] = [centers[i][0],[],[]]
        for j,lab in enumerate(labels) :
            if lab == i : 
                dico[i][1].append(L_names[j])
                dico[i][2].append(L_values[j][0])
    
    return dico

#=====================================================================

def KM1D_v2(df , col_num , n_points=None , n_clusters=3 , random_state=0 , plot=True):
    
    if n_points==None : n_points = len(df)
    
    L_names = list(df['ID_REF'])
    L_colonnes = list(df.keys())[1::]
    
    L_values = list(df[L_colonnes[col_num]][0:n_points])
    for i,value in enumerate(L_values) : L_values[i] = [value]
    
#    for i,names in enumerate(L_names) : print(names,L_values[i])
    
    # Calcul des clusters
    model = sk_KMeans(n_clusters=n_clusters , random_state=random_state)
    KM = model.fit(L_values)
    labels = KM.labels_
    centers = KM.cluster_centers_
#    print(centers)
    
    # Arrangement des clusters selon valeur croissantes des centroïdes
    L_centers = []
    for c in centers : L_centers.append(c[0])
    L_sorted_centers = sorted(L_centers)
    L_ordered_centers = []
    for c in L_sorted_centers :
        idx = L_centers.index(c)
        L_ordered_centers.append((idx,c))
#    print(L_sorted_centers)
#    print(L_ordered_centers)
#    print("---------------")
    
    # Tri des points par clusters
    dico = {}
    for i in range(n_clusters) : dico[i] = 0
    for i in list(dico.keys()) :
        (idx,center) = L_ordered_centers[i]
        dico[i] = [center,[],[]]
        for j,lab in enumerate(labels) :
            if lab == idx : 
                dico[i][1].append(L_names[j])
                dico[i][2].append(L_values[j][0])
#        for k,v in dico.items(): print(k,type(v))
#    print("---------------")
    
    # Plot des clusters
    if plot :
        for i,val in enumerate(list(dico.values())) :
            print(i,round(val[0],4),len(val[2]))
            x = val[2]
            y = [i for k in range(len(val[2]))]
            plt.scatter(x,y)

    plt.show()
    
    return dico

#=====================================================================

def KM1D_v3(df , col_num , n_points=None , n_clusters=3 , random_state=0):
    
    if n_points==None : n_points = len(df)
    
    L_names = list(df['ID_REF'])
    L_colonnes = list(df.keys())[1::]
    
    L_values = list(df[L_colonnes[col_num]][0:n_points])
    for i,value in enumerate(L_values) : L_values[i] = [value]
    
#    for i,names in enumerate(L_names) : print(names,L_values[i])
    
    # Calcul des clusters
    model = sk_KMeans(n_clusters=n_clusters , random_state=random_state)
    KM = model.fit(L_values)
    labels = KM.labels_
    centers = KM.cluster_centers_
#    print(centers)
    
    # Arrangement des clusters selon valeur croissantes des centroïdes
    L_centers = []
    for c in centers : L_centers.append(c[0])
    L_sorted_centers = sorted(L_centers)
    L_ordered_centers = []
    for c in L_sorted_centers :
        idx = L_centers.index(c)
        L_ordered_centers.append((idx,c))
#    print(L_sorted_centers)
#    print(L_ordered_centers)
#    print("---------------")
    
    # Tri des points par clusters
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
    
    # Suite de clusters par points
    dico_points = {}
    for name in L_names :
        for k,v in dico_clusters.items():
            if name in v[1] : 
                dico_points[name] = k
                break
    return dico_points

#=====================================================================

def SortData(L_df , L_clusters , L_labels=[] , n_points=None , L_outliers=[] , n_loop=0):
    
    if L_labels==[] : L_labels = [f"Cluster {i+1}" for i in range(len(L_clusters))]
    
    L_Dcv , L_Lan , L_Lcv , L_Lcc , L_Lms = [],[],[],[],[]
    if L_outliers == [] : L_outliers = [[] for i in range(len(L_clusters))]
    for k,df in enumerate(L_df):
        
        ### Récupération des données de base
        if n_points==None : n_points = len(df)

        L_names = list(df['ID_REF'])
        L_colonnes = list(df.keys())[1::]

        L_values = []
        for col in L_colonnes : L_values.append(list(df[col][0:n_points]))
        L_values = np.transpose(L_values)

        ### Tri des points Candidats / Clusters
        D_candidates_values = {}
        L_anchors_names = [[] for i in range(len(L_clusters))]
        L_clusters_values = [[] for i in range(len(L_clusters))]

        for i,name in enumerate(L_names) : # Pour chaque gène étudié
            
            # Si le gène courant est un outlier à ne pas prendre en compte : on le passe
            out = False
            for outliers in L_outliers :
                if name in outliers : 
                    out = True
                    break
            if out : continue
            
            candidate = True
            for j,cluster in enumerate(L_clusters) :
                if name in cluster : # Si le gène courant est une ancre de cluster, on retient son vecteur
                    L_anchors_names[j].append(name)
                    L_clusters_values[j].append(L_values[i])
                    candidate = False
                    break
            if candidate : D_candidates_values[i] = L_values[i] # Si le gène courant est un candidat
            
        ### Calcul des centroids des clusters
        L_clusters_centroids = [0 for i in range(len(L_clusters))]
        for i,values in enumerate(L_clusters_values) : L_clusters_centroids[i] = np.average(np.array(values),axis=0)
        
        ### Recherche des ancres outliers
        L_Dist_lists = [[] for i in range(len(L_clusters))] # Listes des distances entre les ancres de cluster et leur centroide
        L_MeanStd = [0 for i in range(len(L_clusters))] 
        for i,L_values in enumerate(L_clusters_values) :
            for v in L_values : L_Dist_lists[i].append(distance.euclidean(v,L_clusters_centroids[i]))
            mean = np.mean(L_Dist_lists[i])
            std = np.std(L_Dist_lists[i])
            L_MeanStd[i] = (mean,std)
            
            for j,v in enumerate(L_Dist_lists[i]):
                if v > (mean+(2*std)) : 
                    L_outliers[i].append(L_anchors_names[i][j])
#        print(L_clusters_centroids)
        L_Dcv.append(D_candidates_values)
        L_Lan.append(L_anchors_names)
        L_Lcv.append(L_clusters_values)
        L_Lcc.append(L_clusters_centroids)
        L_Lms.append(L_MeanStd)
    
    for i,out in enumerate(L_outliers):
        L_outliers[i] = sorted(list(set(out)))
    
    ### Recalcul des clusters en retirant les outliers calculés précédemment
    if n_loop > 0 : L_Dcv , L_Lan , L_Lcv , L_Lcc , L_Lms= SortData(L_df , L_clusters=L_clusters , L_labels=L_labels , 
                                                                    n_points=n_points , L_outliers=L_outliers , 
                                                                    n_loop=n_loop-1)
    
    ### Filtrage des candidats
    
    return L_Dcv , L_Lan , L_Lcv , L_Lcc , L_Lms

#=====================================================================

def Cluster_Amplification(L_df , L_clusters , L_labels=[] , n_points=None , n_loop=0) :
    
    ### Tri des Candidats / Clusters avec suppression des outliers
    L_Dcv , L_Lan , L_Lcv , L_Lcc , L_Lms = SortData(L_df , L_clusters=L_clusters , L_labels=L_labels , 
                                                     n_points=n_points , n_loop=n_loop)

    ### Association des Candidats aux Clusters
    LL_assignments = [[] for i in range(len(L_clusters))]
    L_dico = []
    for k,df in enumerate(L_df):
        print(f"Analyse Cluster Amplification sur dataset {k+1}")
        
        L_names = list(df['ID_REF'])
        L_colonnes = list(df.keys())[1::]
        L_values = []
        for col in L_colonnes : L_values.append(list(df[col][0:n_points]))
        L_values = np.transpose(L_values)
        
        D_candidates_values = L_Dcv[k]
        L_anchors_names = L_Lan[k]
        L_clusters_values = L_Lcv[k]
        L_clusters_centroids = L_Lcc[k]
        L_MeanStd = L_Lms[k]
        
#        print(D_candidates_values)
#        for i,c in enumerate(L_clusters_values) : print(i,c)
#        for i,cen in enumerate(L_clusters_centroids) : print(i,cen)
    
        L_assignments = [[] for i in range(len(L_clusters))]
        
        # Filtrage des candidats trop éloignés du centroide de leur cluster
        for i,v_1 in D_candidates_values.items():
#            print(i,L_names[i],v_1)
            L_mean_dist = [0 for a in range(len(L_clusters))]
            for j,centroid in enumerate(L_clusters_centroids): 
                dist = distance.euclidean(v_1,centroid)
                L_mean_dist[j] = dist
#            print(L_mean_dist)
            nearest_cluster = L_mean_dist.index(min(L_mean_dist))
            Mean,Std = L_MeanStd[nearest_cluster]
            if min(L_mean_dist) > Mean : continue
#            print(nearest_cluster)
            L_assignments[nearest_cluster].append(L_names[i])

        
#        for i,assignment in enumerate(L_assignments): print(f"A{i},{len(assignment)}")

        # Association des candidats à leur ancre de cluster la plus proche
        dico = {}
        for i,assignment in enumerate(L_assignments) :
            LL_assignments[i].append(assignment)
            print(f"Cluster {L_labels[i]} : {len(assignment)} candidats")
            cluster = L_anchors_names[i]
            for candidate in assignment : 
                min_dist = 10000
                nearest = 0
                v_c = L_values[L_names.index(candidate)]
                for anchor in cluster : 
                    v_a = L_values[L_names.index(anchor)]
                    dist = distance.euclidean(v_a,v_c)
                    if dist < min_dist :
                        min_dist = dist
                        nearest = anchor
    #                    print(candidate,nearest,min_dist)
    #            print(candidate,nearest,min_dist)
    #            print("-----")
                dico[candidate] = (nearest,min_dist)
    #        print("\n--------------------------------\n")
        L_dico.append(dico)
#        print(dico)
        print("--------------")
            
    return LL_assignments , L_dico

#=====================================================================

def Expression_plot(df , n_points=None , title='' , selection=[] , selection_colors=['red'] ,
                    out_values=[] , out_colors=[] , selection_linewidth=0.5 , show=True):
    # Fonction traçant et affichant les courbes de valeurs d'un dataframe fourni avec de possibles ordres de priorité.
    # Arguments :
    #   - Un dataframe
    #   - Le nombre de points dont on veut tracer les courbes (défaut = None). [None ou Int]
    #       - Si None, trace les courbes de l'ensemble des lignes du dataframe.
    #       - Si Int, trace autant de courbes que la valeur fournie en partant du début du dataframe.
    #   - Le titre de l'affichage (défaut = ''). [Str]
    #   - Une sélection de lignes prioritaires à rendre facilement reprérable lors de l'affichage (défaut = []). [[], Liste 1D de Str ou Liste 2D de Str]
    #       - Si [], pas de priorité. Trace chaque ligne avec une couleur différente.
    #       - Si Liste 1D de Str, trace d'abord toutes les lignes non prioritaire en gris clair puis toutes les lignes prioritaires selon la couleur fournie dans l'agument 'selection_colors'.
    #       - Si Liste 2D de Str, trace d'abord toutes les lignes non prioritaire en gris clair puis chaque groupe de lignes prioritaires selon les couleurs fournies dans l'agument 'selection_colors'.
    #       - PAR SOUCIS D'OPTIMISATION, cette fonction considère qu'il n'y a pas de doublons d'une même ligne de valeur dans plusieurs groupes de priorité. En cas de présence de doublons, seul le premier rencontré
    #           dans l'ordre de priorité des groupe sera tracé.
    #   - Une liste de couleurs à respecter en cas de sélection de lignes prioritaires (défaut = rouge). [Liste de couleurs]
    #       - S'il n'y a qu'un seul groupe de lignes prioritaires, la couleur rouge est choisie par défaut mais peut être modifiée.
    #       - S'il y a plusieurs groupes de lignes prioritaires, autant de couleurs que de groupes doivent être fournies.
    #   - Une liste de valeurs externes au dataframe (défaut = []). [[], ou Liste 2D de Float)
    #       - Si utilisé, doit obligatoirement recevoir une liste 2D. En cas d'une seule ligne à tracer, il suffit de fournir une liste 2D avec une seule ligne.
    #   - Une liste de couleur à respecter en cas de traçage de valeurs externes (défaut = []). [Liste de couleurs]
    #       - Fournir une couleur par ligne fournie à l'argument 'out_values'.
    #   - L'épaisseur de traçage des lignes (défaut = 0.5). [Float ou Int]
    #   - Le choix de l'affichage direct du graph (défaut=True). [Boolean]
    # Rendus (mutuellement exclusifs) :
    #   - None ; Affichage du dessin du graph.

    ### Récupération des lignes de valeurs du dataframe selon le nombre de ligne souhaité.
    if n_points==None : n_points = len(df)
    L_points = list(df['ID_REF'][0:n_points])
#    print(L_points)
    L_colonnes = list(df.keys())[1::]
    
    L_values = []
    for col in L_colonnes : L_values.append(list(df[col][0:n_points]))

    ### Traçage des lignes de valeur issues du dataframe.
    # Si aucune priorité, traçage par défaut.
    if selection==[]: plt.plot(L_values,linewidth=0.5)

    else :
        L_values = np.transpose(L_values)
        # Si priorité donnée via une liste 1D, création d'un dictionnaire qui retient pour chaque ligne priorisé son indice dans le dataframe.
        # Pour chaque ligne non priorisée du dataframe, elle est tracée lors de sa lecture.
        # Une fois toutes les lignes lues, traçage de celles dont les indices ont été écrit dans le dictionnaire.
        if type(selection[0])!=list:
            dico = {}
            for i,point in enumerate(L_points) :
                if point in selection : dico[point] = i
                if point not in selection : plt.plot(L_values[i],color='lightgray',linewidth=0.1)
            for point,idx in dico.items(): plt.plot(L_values[idx],color=selection_colors[0],linewidth=selection_linewidth)
        # Si priorité donnée via une liste 2D à n lignes, création d'un dictionnaire à n+1 clés (allant de -1 à n-1), chacune associée à une liste initialement vide.
        # Les indices des lignes non priorisées du dataframe sont ajouté à la seule clé négative du dictionnaire.
        # Les indices des lignes priorisées du dataframe sont ajoutés à la clé correspondant à la position de leur groupe de priorité dans la liste 2D.
        # Une fois toutes les lignes lues, traçage de chaque groupe de lignes de valeurs dans l'ordre des clés du dictionnaire.
        else :
            dico = dict(zip([-1]+[i for i in range(len(selection))] , [[] for i in range(len(selection)+1)]))
            for i,point in enumerate(L_points) :
                selected = False
                for j,selec in enumerate(selection):
                    if point in selec : 
                        dico[j].append(i)
                        selected = True
                        break
                if not selected : dico[-1].append(i)
            for idx in dico[-1] : plt.plot(L_values[idx],color='lightgray',linewidth=0.1)
            del(dico[-1])
            for selec,L_idx in dico.items():
                for idx in L_idx : plt.plot(L_values[idx],color=selection_colors[selec],
                                            linewidth=selection_linewidth)
                
    ### Traçage des éventuelles lignes de valeurs externes.
    for i,values in enumerate(out_values) :plt.plot(values,color=out_colors[i],linewidth=out_linewidth)

    ### Ajout de l'éventuel titre et affichage du graphique.
    plt.title(title)

    if show : plt.show()

#=====================================================================

def PCA(df , n_points=None , n_colonnes=None , selection_range=None , n_dim_out=None , svd_solver='auto'):
    
    if n_points==None : n_points = len(df)
    L_points = list(df['ID_REF'])[0:n_points]
    
    if selection_range==None : L_colonnes = list(df.keys())[1::]
    elif type(selection_range) == list :
        if selection_range[1]-selection_range[0]<1:
            print('ERROR : Selection range provived is invalid.')
            return 1
        if selection_range[0] < 1 :
            print('ERROR : Invalid starting column value.')
            return 2
        if selection_range[1] > len(list(df.keys())[1::]) :
            print('ERROR : Invalid ending column value.')
            return 3
        L_colonnes = list(df.keys())[selection_range[0]:selection_range[1]+1]
    elif type(selection_range) == int :
        if selection_range < 1 :
            print('ERROR : Invalid column number.')
            return 5
        if selection_range > len(list(df.keys())[1::]) :
            print('ERROR : Invalid column number.')
            return 6
        L_colonnes = [list(df.keys())[selection_range]]
    
    L_values = []
    for col in L_colonnes : L_values.append(list(df[col][0:n_points]))
    L_values = np.transpose(L_values)
#    for i in range(n_points): print(f"{L_points[i]} : {L_values[i]}")
#    print(L_values)

    pca = sk_PCA(n_components=n_dim_out , svd_solver=svd_solver)
    pca.fit(L_values)
    
    evr = pca.explained_variance_ratio_
    sv = pca.singular_values_
    
    return evr , sv

#=====================================================================

def T_SNE(df , n_points=None , n_colonnes=None , selection_range=None ,
          n_dim_out=2 , perplexity=30 , init='pca'):
    
    if n_points==None : n_points = len(df)
    L_names = list(df['ID_REF'])[0:n_points]
    
    if selection_range==None : L_colonnes = list(df.keys())[1::]
    elif type(selection_range) == list :
        if selection_range[1]-selection_range[0]<1:
            print('ERROR : Selection range provived is invalid.')
            return 1
        if selection_range[0] < 1 :
            print('ERROR : Invalid starting column value.')
            return 2
        if selection_range[1] > len(list(df.keys())[1::]) :
            print('ERROR : Invalid ending column value.')
            return 3
        L_colonnes = list(df.keys())[selection_range[0]:selection_range[1]+1]
    elif type(selection_range) == int :
        if selection_range < 1 :
            print('ERROR : Invalid column number.')
            return 5
        if selection_range > len(list(df.keys())[1::]) :
            print('ERROR : Invalid column number.')
            return 6
        L_colonnes = [list(df.keys())[selection_range]]
    
    L_values = []
    for col in L_colonnes : L_values.append(list(df[col][0:n_points]))
    L_values = np.transpose(L_values)
#    for i in range(n_points): print(f"{L_names[i]} : {L_values[i]}")
#    print(L_values)

    tsne = sk_TSNE(n_components=n_dim_out , perplexity=perplexity , init=init).fit_transform(L_values)

    return tsne

#=====================================================================

def T_SNE_2(df , n_points=None , L_points=[] , n_colonnes=None , selection_range=None ,
          n_dim_out=2 , perplexity=30 , init='pca'):
    
    if n_points==None : n_points = len(df)
    L_names = list(df['ID_REF'])[0:n_points]
    
    if selection_range==None : L_colonnes = list(df.keys())[1::]
    elif type(selection_range) == list :
        if selection_range[1]-selection_range[0]<1:
            print('ERROR : Selection range provived is invalid.')
            return 1
        if selection_range[0] < 1 :
            print('ERROR : Invalid starting column value.')
            return 2
        if selection_range[1] > len(list(df.keys())[1::]) :
            print('ERROR : Invalid ending column value.')
            return 3
        L_colonnes = list(df.keys())[selection_range[0]:selection_range[1]+1]
    elif type(selection_range) == int :
        if selection_range < 1 :
            print('ERROR : Invalid column number.')
            return 5
        if selection_range > len(list(df.keys())[1::]) :
            print('ERROR : Invalid column number.')
            return 6
        L_colonnes = [list(df.keys())[selection_range]]
    
    L_values = []
    for col in L_colonnes : L_values.append(list(df[col][0:n_points]))
    L_values = np.transpose(L_values)
#    for i in range(n_points): print(f"{L_names[i]} : {L_values[i]}")
#    print(L_values)

    if L_points != []:
#        print("-------------------")
        L_idx = []
        L_values_2 = []
        L_names_2 = []
        for i,name in enumerate(L_names):
            if name in L_points : L_idx.append(i)
        for idx in L_idx :
            L_values_2.append(L_values[idx])
            L_names_2.append(L_names[idx])
#        for i in range(len(L_points)): print(f"{L_points[i]} : {L_values_2[i]}")
#        print("-------------------")
        L_values = L_values_2

    tsne = sk_TSNE(n_components=n_dim_out , perplexity=perplexity , init=init).fit_transform(L_values)

    return L_names_2 , tsne
