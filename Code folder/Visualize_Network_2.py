import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

arguments = sys.argv[1::]

network_file = arguments[0]

if len(arguments)>1 : L_query = arguments[1::]

with open(network_file,'r') as file :
    G = nx.Graph()
    n = 0
    if L_query == [] : # If there are no query
        CN_map,CE_map = [],[]
        for line in file :
            if line[0] == '>' : # Adding the nodes
                n += 1
                gene,color = line[1:-1].split(':')
                color = color[1:-1].split(', ')
                for i,v in enumerate(color): color[i] = float(v)
                G.add_node(gene)
                nx.set_node_atttributes(G,{gene:{'color':color}})
                CN_map.append(tuple(color))
                
            else : # Adding the edges
                data = line[0:-1].split(' ; ')
                n1,n2 = data[0:2]
                D_attr = {}
                for d in data[2::] :
                    key,value = d.split(':')
                    if key == 'color' : continue # P-CEIN Network
                    try : D_attr[key] = float(value)
                    except ValueError : D_attr[key] = value
                G.add_edge(n1,n2)
                G[n1][n2].update(D_attr)

                # Determinate the edges' colors
                if 'P-CEIN' in D_attr : # Consensus network
                    if D_attr['P-CEIN'] == 'NO' : CE_map.append((0.8,0.8,0.8))
                    elif D_attr['P-CEIN'] < 0 : CE_map.append((1,0,0))
                    elif D_attr['P-CEIN'] > 0 : CE_map.append((0,1,0))
                elif 'sign' in D_attr : # P-CEIN Network
                    if D_attr['sign'] < 0 : CE_map.append((1,0,0))
                    elif D_attr['sign'] > 0 : CE_map.append((0,1,0))
                else : # KNN-RBH, NPC or Cluster Path network
                    CE_map.append((0.8,0.8,0.8))

    else : # If there are query

        # Regroup the nodes and edges (with their data) in dictionaries
        D_nodes = {}
        D_edges = {}
        for line in file :
            if line[0] == '>' :
                gene,color = line[1:-1].split(':')
                color = color[1:-1].split(', ')
                for i,v in enumerate(color): color[i] = float(v)
                D_nodes[gene] = tuple(color)
            else :
                data = line[0:-1].split(' ; ')
                n1,n2 = data[0:2]
                D_attr = {}
                for d in data[2::] :
                    key,value = d.split(':')
                    if key == 'color' : continue # P-CEIN Network
                    try : D_attr[key] = float(value)
                    except ValueError : D_attr[key] = value
                D_edges[(n1,n2)] = D_attr

        CN_map,CE_map = [],[]

        # Add all edges with at least one query gene in their linked couples
        for (n1,n2),D_attr in D_edges.items():
            if (n1 in L_query) or (n2 in L_query) :
                G.add_edge(n1,n2)
                G[n1][n2].update(D_attr)

        # Determinate the edges' colors
        for (n1,n2) in list(G.edges):
            if 'P-CEIN' in D_attr : # Consensus network
                if G[n1][n2]['P-CEIN'] == 'NO' : CE_map.append((0.8,0.8,0.8))
                elif G[n1][n2]['P-CEIN'] < 0 : CE_map.append((1,0,0))
                elif G[n1][n2]['P-CEIN'] > 0 : CE_map.append((0,1,0))
            elif 'sign' in D_attr : # P-CEIN Network
                if D_attr['sign'] < 0 : CE_map.append((1,0,0))
                elif D_attr['sign'] > 0 : CE_map.append((0,1,0))
            else : CE_map.append((0.8,0.8,0.8))


        # Determinate the query genes' neighbors
        L_neighbors = []
        for node in L_query:
            for n in list(G.neighbors(node)) :
                if n not in L_neighbors : L_neighbors.append(n)
        
        # Determinate the nodes' colors
        for node in list(G.nodes):
            CN_map.append(D_nodes[node])
            nx.set_node_attributes(G,{node:{'color':D_nodes[node]}})
            

print(f"Graph loaded : {len(G.nodes)} nodes and {len(G.edges)} edges")
print(len(CN_map),len(CE_map),'\n')

for gene in L_query :
    print(f"{gene} : {len(G[gene])} edges\n")
    DF_features = {'Neighbors':[],'Color':[]}
    for (n1,n2),D_attr in D_edges.items():
        if (n1 == gene) or (n2 == gene) :
            if n1 == gene :
                DF_features['Neighbors'].append(n2)
                DF_features['Color'].append(G.nodes[n2]['color'])
            else :
                DF_features['Neighbors'].append(n1)
                DF_features['Color'].append(G.nodes[n2]['color'])
            for k,v in D_attr.items():
                if k not in DF_features.keys() : DF_features[k] = [v]
                else : DF_features[k].append(v)
    df = pd.DataFrame.from_dict(DF_features)
    if 'Consensus' in DF_features.keys() : df = df.sort_values(by=['Consensus','Neighbors'],ascending = [False,True], ignore_index=True)
    else : df = df.sort_values(by=['Neighbors'], ignore_index=True)
    pd.set_option('display.max_columns',None)
    print(df)
    print("-------------------------")

nx.draw_spring(G , node_color=CN_map , edge_color=CE_map , node_size = 7)
plt.show()
