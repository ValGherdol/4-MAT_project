import sys
import os
from datetime import datetime
import pandas as pd
import pickle

#=====================================================================
#=====================================================================

# List of the functions defined in this file :
#   - time_count
#   - settings
#   - globalGenePool
#   - anchorGenePools
#   - linkageDictionary

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

def settings(setting_file , Nb_datasets) :

    """
    Generates the setting dictionary.

    input1 : The setting file.
    input2 : The number of provided datasets

    output1 : The setting dictionary where :
        - The keys are the settings names.
        - The values are either strings, Integers, Floats or lists.
    output2 : The linkage dictionary, which can either be :
        - New, and as such empty (is filled by another part of the code)
        - Old, and already contains some results.
    output3 : The name of the folder where the results will be put.
    """

    D_args = {}
    with open(setting_file,'r') as file :
        for line in file :
            if line[0] in ['>','\n'] : continue
#            print(line[0:-1])
            arg = line[0:-1].split(' : ')
#            print(arg[1],type(arg))
            D_args[arg[0]] = arg[1]
    
    ### Initial data settings ------------------------------------------------------------
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

    if D_args['Result folder name'] == 'default' :
        now = datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%Ss")
        folder_name = f"Result_4-MAT_{now}/"
        os.mkdir(folder_name[0:-1])
    elif D_args['Result folder name'] == 'None' : folder_name = ""
    else :
        folder_name = D_args['Result folder name']
        if not os.path.isdir(D_args['Result folder name']) : os.mkdir(folder_name)
        folder_name += '/'

    if D_args['Linkage dictionary'] == 'default' : D_assoc = {}
    else :
        with open(D_args['Linkage dictionary'],'rb') as file : D_assoc = pickle.load(file)

    ### P-CEIN settings ------------------------------------------------------------------
    D_args['Anchor centered P-CEIN'] = bool(int(D_args['Anchor centered P-CEIN']))

    if D_args['Pearson principal threshold'] == 'default' : D_args['Pearson threshold'] = 0.5
    elif ',' in D_args['Pearson principal threshold'] :
        D_args['Pearson principal threshold'] = D_args['Pearson principal threshold'].split(',')
        for i,a in enumerate(D_args['Pearson principal threshold']): D_args['Pearson principal threshold'][i] = float(a)
    else : D_args['Pearson principal threshold'] = float(D_args['Pearson principal threshold'])

    D_args['Pearson dynamic threshold'] = bool(int(D_args['Pearson dynamic threshold']))

    if D_args['Dynamic threshold factor'] == 'default' : D_args['Dynamic threshold factor'] = 1
    else : D_args['Dynamic threshold factor'] = float(D_args['Dynamic threshold factor'])

    if D_args['Minimum neighborhood P-CEIN'] != 'default' : 
        D_args['Minimum neighborhood P-CEIN'] = D_args['Minimum neighborhood P-CEIN'].split(',')
        for i,a in enumerate(D_args['Minimum neighborhood P-CEIN']): D_args['Minimum neighborhood P-CEIN'][i] = int(a)

    D_args['Research time announcement P-CEIN'] = bool(int(D_args['Research time announcement P-CEIN']))

    D_args['New Network'] = bool(int(D_args['New Network']))

    ### KNN-RBH settings -----------------------------------------------------------------
    D_args['RBH sub-step'] = bool(int(D_args['RBH sub-step']))

    if D_args['Number of neighbors'] == 'default' : D_args['Number of neighbors'] = 2
    else : D_args['Number of neighbors'] = int(D_args['Number of neighbors'])

    if D_args['Threshold factor KNN_2'] == 'default' : D_args['Threshold factor KNN_2'] = 1
    elif ',' in D_args['Threshold factor KNN_2'] :
        D_args['Threshold factor KNN_2'] = D_args['Threshold factor KNN_2'].split(',')
        for i,a in enumerate(D_args['Threshold factor KNN_2']): D_args['Threshold factor KNN_2'][i] = float(a)
    else : D_args['Threshold factor KNN_2'] = float(D_args['Threshold factor KNN_2'])

    ### NPC settings ---------------------------------------------------------------------
    if D_args['Distance threshold factor NPC'] == 'default' : D_args['Distance threshold factor NPC'] = 1
    elif ',' in D_args['Distance threshold factor NPC'] :
        D_args['Distance threshold factor NPC'] = D_args['Distance threshold factor NPC'].split(',')
        for i,a in enumerate(D_args['Threshold factor NPC']): D_args['Distance threshold factor NPC'][i] = float(a)
    else : D_args['Distance threshold factor NPC'] = float(D_args['Distance threshold factor NPC'])

    D_args['Minimum neighborhood NPC'] = D_args['Minimum neighborhood NPC'].split(',')
    for i,a in enumerate(D_args['Minimum neighborhood NPC']):
        if a != 'Dynamic' : D_args['Minimum neighborhood NPC'][i] = int(a)

    if D_args['Minimum redundancy'] == 'default' : D_args['Minimum redundancy'] = Nb_datasets
    else : D_args['Minimum redundancy'] = int(D_args['Minimum redundancy'])

    ### ClusterPath settings -------------------------------------------------------------
    if D_args['Number of clusters'] == 'default' : D_args['Number of clusters'] = 3
    else : D_args['Number of clusters'] = int(D_args['Number of clusters'])

    if D_args['Maximum deviation'] == 'default' : D_args['Maximum deviation'] = 0
    elif ',' in D_args['Maximum deviation'] :
        D_args['Maximum deviation'] = D_args['Maximum deviation'].split(',')
        for i,a in enumerate(D_args['Maximum deviation']): D_args['Maximum deviation'][i] = int(a)
    else : D_args['Maximum deviation'] = int(D_args['Maximum deviation'])

    D_args['Anchor centered CP'] = bool(int(D_args['Anchor centered CP']))

    D_args['Research time announcement CP'] = bool(int(D_args['Research time announcement CP']))

    ### Consensus settings ---------------------------------------------------------------
    if D_args['Consensus levels list'] == 'default' : D_args['Consensus levels list'] = [1,2,3,4]
    else :
        D_args['Consensus levels list'] = D_args['Consensus levels list'].split(',')
        for i,a in enumerate(D_args['Consensus levels list']): D_args['Consensus levels list'][i] = int(a)

    return D_args , D_assoc , folder_name

#=====================================================================

def globalGenePool(Pool_file):

    """
    Loads the global gene pool.

    input1 : The global gene pool file.

    output1 : The list of genes in the global pool.
    """

    L_pool = []
    with open(Pool_file,'r') as file :
        for line in file : L_pool.append(line[0:-1])
    
    return L_pool

#=====================================================================

def anchorGenePools(anchor_files_list , globalGenePool):

    """
    Loads the anchor gene pools.

    input1 : The list of anchor gene files.
    input2 : The list of genes in the global gene pool.

    output1 : The list of sub-lists of all anchor genes from each anchor gene file.
    output2 : The list of sub-lists of anchor genes from each anchor gene file that are in the global gene pool.
    output3 : The total list of anchor genes that are in the global gene pool.
    """

    L_L_All_anchors = [] # List of all complete anchor genes lists
    L_L_anchors = [] # List of anchor genes lists whose anchors are in the global gene pool
    
    for i,anchor_file in enumerate(anchor_files_list) :
        L_L_All_anchors.append([])
        L_L_anchors.append([])
        with open(anchor_file,'r') as file :
            for line in file : L_L_All_anchors[i].append(line[0:-1])
    
    for gene in globalGenePool :
        for i,L_All_anchors in enumerate(L_L_All_anchors) :
            if gene in L_All_anchors : L_L_anchors[i].append(gene)

    L_Final_anchors = [] # List of all unique anchors
    for L_anchors in L_L_anchors : L_Final_anchors += L_anchors
    
    return L_L_All_anchors , L_L_anchors , L_Final_anchors

#=====================================================================

def linkageDictionary(link_dict , setting_dict , folder_name , globalGenePool , anchorGeneLists , gene_labels):

    """
    Initializes a new, empty linkage dictionary.
    Instanciates the name of the variable pointing to the dictionary, be it new or loaded.

    input1 : A dictionary.
        If empty, fills it with tuples of genes and labels as keys and empty sub-dictionaries as values.
    input2 : A dictionary of processed parameters.
    input3 : A name for a folder in which the results will be put.
    inpupt4 : A list of genes.
    input5 : A list of anchor genes sub-lists.
    input6 : A list of anchor labels.

    output1 : The linkage dictionary.
    output2 : The name of the linkage dictionary file.
    """

    if link_dict == {} : # If the linkage dictionary is new
        Result_dict_file = f"{folder_name}Result_0_dictionary.pickle"
        if setting_dict['Anchor Genes lists'] != [] : # If anchors are provided
            for n in globalGenePool :
                for i,L_anchors in enumerate(anchorGeneLists) :
                    if n in L_anchors :
                        link_dict[(n,gene_labels[i])] = {}
                        break
        else : # If no anchors are provided
            for n in globalGenePool : link_dict[(n,'Candidate')] = {}
    else : Result_dict_file = f"{folder_name}{setting_dict['Linkage dictionary']}"

    return link_dict , Result_dict_file
#=====================================================================
#=====================================================================
#=====================================================================
