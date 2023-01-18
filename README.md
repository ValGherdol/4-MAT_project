# 4-MAT_project

This README explains the content and proper usage of the 4-MAT package.
The 4-MAT package's goal is to subject transcriptomic data to a multi-method analysis in order to find statistically relevant links between genes of interest (anchors) who play a known role in given metabolic pathways and candidats genes who may have a yet unknown role in those pathways.
At time of creation, this package has been used in the hope to find missing genes in the Dormancy-Germination regulation process in Arabidopsis thaliana.

Package usage rules :
- This package is written in the Python language. Make sure a Python interpreter (like IDLE) is installed on the computer you're using to run the program as well as the following Python modules : time, collections, sys, copy, numpy, matplotlib.pyplot, sklearn (scikit-learn), pandas, scipy, networkx, pickle, warnings.
	NOTE : Some of these modules are installed by default with Python IDLE.
- When writing the command line to run the program, be sure to put the program file first, the arguments file in second, and then at least one dataset csv file (unless you set the argument file to only run the Consensus Step, in which case no dataset are required).
- When using multiple datasets, make sure all of them contains the same genes and the same number of genes (more detailed on the datasets' structure below).
- When customizing the settings the arguments file, make sure that all settings that are number-of-dataset dependent are correctly set according to the number of datasets you are using. Same for settings that are number-of-anchor-lists dependent.
- The program is organized in (currently) 4 Analysis Steps plus a Consensus Step. You can customize the argument file to specify which steps you want to run or not. All steps that you'll run will produce one or several result files (detailed below).

This package folder initially contains the following files (arranged by type):

- Python files :
	- Program_XXXX_YY_ZZ.py : the core code file of the package. The XYZ characters stands for the date of the lastest update.
	- Stage_functions.py : the first auxiliary code file containing most of the program's functions' definitions.
	- CVI_functions.py : the seconde auxiliary code file containing the functions' definitions for the [CVI] analysis step.
- Text files :
	- Arguments.txt : this file describs the settings of the program and how to customize them.
	- Pool_9103.txt : this file contains an ordered list of 9103 genes (anchors and candidats) to be analysed by the program.
	- Pool_16428.txt : this file contains an ordered list of 16428 genes (anchors and candidats) to be analysed by the program.
	- Pool_Dormancy.txt : this file contains the ordered list of anchor genes whose role is to promote Dormancy.
	- Pool_Germination.txt : this file contains the ordered list of anchor genes whose role is to promote Germination.
- Csv files, each containing a transcriptomic dataset of the genes to be analysed:
	- Data_X.csv (with X = 1,2,3) contains transcriptomic data for Pool_9103.
	- Data_0.csv contains the merged data from Data_X.csv.
	- Data_Ya.csv (with Y = 2,3,5,6) contains transcriptomic data for Pool_16428.
	- Data_0a.csv contains the merged data from Data_Ya.scv.
- Docx file, the user manual which explains in details how the program works.

Description of the program's steps :
- Step 1 - Co-Expression Network : This step calculates the Pearson's Correlation Coefficient (PCC) of couples of genes throughout all datasets. 
  You can either ask for all couples to be analyzed or give a list of specific genes from which the correlation to all other genes in the datasets will be caluclated.
	A threshold (in absolute value) can be used to select PCCs that are considered significant. It is by default set to 0.5 for all datasets but you can customize this value in the argument file or put different values for each dataset. You can as well explicitly put a threshold of 0 in order to keep all correlation coefficients.
	For each couple of genes analysed, a PCC is calculated in each dataset. If all PCC are above their respective threshold (in absolute value) and are of the same sign (either all positive or all negative), the couple is linked by an edge in a network and the arithmetic mean of the couple's PCCs are associated to the edge.
	When all couples have been analysed and the network is built, you can set a limite of neighbors that each gene can't go above. It is however highly unadvised to set such a limite because the filtering method is currently dependant of the order in which the genes have been added to the network. As such, two genes pools containing the exact same genes but in different orders could result in different network.
	You can also ask for a second, dynamic threshold to be calculated from all the created edges. This threshold follows the formula [Mean PCC - (Multiplication Factor * Standard Deviation of the PCCs)] and cut out all edges with a lower absolute value.
	You can finally set a minimum number of neighbors of interest (you can put different values for each list of genes of interest) to sort out candidates.
	These second and third filters are more objective than the first one.
	WARNING : this step is the single most time consuming part of the program. If correlation between all genes are to be calculated, count at least 1h for a dataset of 9000 genes (1h15 for 3 of those) and up to 5 hours for a dataset of 16500 genes. Also, this step specifically needs datasets with at least 3 value columns, since any PCC calculated on vectors of length 2 will give a result of 1 (in absolute value), and as such will not be useful.
  
- Step 2 - K-Nearest Neighbors (KNN) with Reciprocal Best Hit (RBH) : This step is composed of the following 3 substeps.
	First, for each dataset, it uses the KNN algorithm to select the nearest neighbors of each gene.
	Second, again for each dataset, it crosses all calculated neighborhoods and only keeps couples of genes that consider each other one of their respective nearest neighbors (based on the RBH concept).
	Third, it crosses the filtered neighborhoods of all datasets and only keeps couples of genes that are reciprocal neighbors in all datasets.
	The KNN algorithm can be set up two ways : either using the standard algorithm with a fix neighborood for all genes or using an altered version that dynamically select the neighborhhod size for each gene. This dynamic selection consists in calculating for each gene it's distances to the others, then calculating a threshold with the formula [Mean Distance - (Multiplication Factor * Standard Deviation of the distances)].
	The Multiplcation Factor is by default set to 1 but you can customize this value in the argument file or put different values for each dataset.
	WARNING : for liberty purposes, the RBH substep can also be passed over but it is highly unadvised to do so.
  
- Step 3 - Netowrk Properties Closeness (NPC) : This step selects candidate genes based on network properties.
	For each dataset, a network is built for each pool of genes of interest you provide as anchor points. All others genes' likelihood of being associated with the anchor points are calculated through several methods and then sorted by descending order of the number of anchor points they are associated with. For each pool of genes of interest, you can customize the minimum number of associations a candidate gene must have with the anchor points to be considered relevant.
	All network are then crossed to sort for each candidate how many associations they have with genes of interest from each pool and in how many datasets a specific association is found. You can customize the minimum number of datasets a candidate has to be found to be coonsidered relevant. 
	NOTE : Due to the networks' building being based on only one pool of interest at a time, it is possible that some candidates for a given pool come 
	       from another pool of interest. Also, this step specifically needs at least one anchor genes list to be able to run.
- Step 4 - Cluster Path : This step segregates the genes by using several 1-dimensional K-Means algorithms.
	For each dataset, a K-Means clustering is run on each of the dataset's timepoints. The clusters are sorted by ascending order of their centroids' values. Each gene is then associated to a vector indicating which cluster it belongs to in each timepoint. Once all genes have received a vector for all datasets, they are regrouped by common vectors suits. The idea is that if a given group of genes are always together in a common cluster from a timepoint to another across all datastes, then maybe they have some relevant relationship.

- End Step - Analysis Consensus : This final step of the program crosses all results from all the above steps and sorts all genes of interest based on how many candidates they have been associated with and how many methods consider those associations relevant. You can customize which levels of association you want to be saved (a level being the number of method in which an association is found relevant).


Datasets' structure : All datasets must be structured in the following way :
- The first column must be called 'ID_REF' and contain the names of all genes to be analysed. The order doesn't have to be the same in all datasets 
	but all datasets must contain the entirety of the global gene pool, no more no less.
- The next columns can be called whatever you want but must contain finite float values (no NaN or inf). The number of value columns can change from 
	one dataset to another but keep in mind that at least 3 of those value columns have to exist in all datasets on which you run the Co-Expression 
	Step (all the other steps can work with at least 1 value column) and as a general rule of thumb, the more columns the more reliable the results 
	will be.

Result files : After launching the program, the following result files will appear in the folder (according to the steps you will have set to be run).
- Result_0_dictionnary.pickle : a pickle file containing a two-level dictionnary object. The first level's keys are the genes of interest and each of them is associated to a sub-dictionnary. The sub-dictionnaries' keys are the code names of the different analysis steps and those keys' values are the lists of candidate genes associated to a given gene of interest according to the corresponding analysis step. The dictionnary is instantiated with only it's first level with empty sub-dictionnaries. Those sub-dictionnaries are then filled after each analysis steps run. Finally, this dictionnary is used in the Consensus step to sort the results.
	In the sub-dictionnaries :
		- the P-CEIN key refers to the 1st step (Co-Expression Network).
		- the KNN-RBH key refers to the 2nd step.
		- the NPC-X keys refer to the 3rd step with the X indicating the number of datasets the associated list of genes are found.
		- the ClusterPath keys refer to the 4th step.
- Result_1a_RICEP_graph.txt : a text file containing all edges of the co-expression network built in the 1st step of the program. Each line indicates either a gene name that is in the graph or the names of the two linked genes, followed by the weigh of the edge (i.e. the absolute value of the average PCC of the cople of genes), then the sign of the correlation, and finally the color of the edge (red for negative correlations, green for positive correlations).
- Result_1a_RICEP_data.csv : a csv file containing the general data of each gene in the co-expression network. For each gene, you can know if it's a candidate gene or an anchor gene (and which kind of anchor), it's number of anchor neighbors (as a list of integers, each of which corresponds to an anchor list), it's total number of edges (which equals it's total number of neighbors), it's numbers of positive and negative correlations, and in order the average value, standard deviation value, maximum value and minimum value of all Pearson's Correlation Coefficients between the current gene and all it's neighbors.
- Result_1b_RICEP_dynamic_pearson_graph.txt : an optional text file, only created if you activate the dynamic PCC threshold filter. It is organized the same way the initial graph file is.
- Result_1b_RICEP_dynamic_pearson_data.csv : a csv file containing the general data of each gene in the co-expression network after the dynamic PCC threshold filter has been used. It is organized the same way the initial data file is.
- Result_1c_RICEP_relevant_neighborhood_graph.txt : an optional text file, only created if you set the parameters with a minimum number of neighbors of interest for the candidate genes. It is organized the same way the initial graph file is.
- Result_1c_RICEP_relevant_neighborhood_data.csv : a csv file containing the general data of each gene in the co-expression network after the neighborhood filter has been used. It is organized the same way the initial data file is.
- Result_2_KNN-RBH_graph.txt : a text file containing the KNN-RBH network built during the 2nd step of the program. Each line indicates either the name of a gene and it's color in the network (cyan for Dormancy genes, orange for Germination genes and black for candidate genes), or a couple of genes that are linked according to the network.
- Result_3a_CVI_X_for_Dataset_Y_minZ.txt : a group of text files each containing the results of a loop of the 3rd step of the program. The X mark is either 'Dorm' or 'Germ' and corresponds to the pool of genes of interest used in the loop. The Y mark is the number of the dataset used in the loop ; the Y is determined by the order in which the datasets have been given at the program's launch, it is your responsability keep track of which number correspond to which dataset you provide. The Z mark is the minimum number of genes of interest the candidate genes present in the file have been associated with. The number N of Result_3a files follows an N = m*n formula where n is the number of datasets annd m is the number of pools of gene of interest (note that any value of m higher than 2 will clash with most of the rest of program's steps). Each file is structured as a tab and each line (strating from the 2nd) indicates a candidate gene, the number of neighbors of interest it has been associated to, it's total number of neighbors, the number of cliques it is found in, the number of special cliques it is found in, the size of the maximal clique it is found in, it's average distance to it's neighbors, and the list of genes of interest it has been associated to.
- Result_3t_CandidatLists_file.txt : an auxiliary file in which all Result_3a files are sorted by the X mark (i.e. all Dormancy files together and all Germination files together). This file's purpose is to be used by the program as it crosses all the Result_3a files, it is of no direct use for the user.
- Result_3b_CVI_CandidateGenes.txt : a text file structured as a tab where each line (strating from the 2nd) indicates the name of a candidate gene, it's number of Dormancy neighbors in each dataset, ditto for the Germination neighbors, it's number of unique neighbors, and finally the lists of neighbors of interest found in a specific number of dataset (i.e. 1-Neighborhood means the genes are only associated in 1 dataset, 2-Neighborhood means the association is found in 2 datasets, and so on...).
- Result_3c_CVI_graph.txt : a text file containing the CVI network built during the 3rd step of the program. Each line indicates either the name of a gene and it's color in the network (cyan for Dormancy genes, orange for Germination genes and black for candidate genes), or a couple of genes that are linked according to the network followed by the weight of the edge which correspond to the number of datasets the association has been found in.
- Result_4a_ClusterPath_list.txt : a text file indicating for each combination of cluster vectors the number of genes it regroups, the number of Dormancy and Germination genes it contains, and the list of genes.
- Result_4b_ClusterPath_graph.txt : a text file containing the Cluster Path network built during the 4th step of the program. Each line indicates either the name of a gene and it's color in the network (cyan for Dormancy genes, orange for Germination genes and black for candidate genes), or a couple of genes that are linked according to the network.
- Result_END_consensus.csv : a csv file indicating for each gene of interest it's pool of origine (Dormancy or Germination), it's number of unique genes associated at least once to it, it's number of neighbors found in all combinations of X methods (X going from 1 to the total number of methods used), and number of neighbors each method has found.
- Result_END_GenesOfInterest_ListByAnchors.txt : a text file where all genes of interest have been sorted by descending order of the number of candidate genes associated to it and in descending order of the number of methods that have found those candidates. At the end of the file is written for each number of methods the total number of associations found in all combinations of this level of recurrence.
- Result_END_GenesOfInterest_ListByCandidates.txt : a text file in a tab format where each line (starting from the 2nd) indicates a candidate gene, the gene of interest it's is associated to, the function of the associated gene of interest (Dormancy or Germination), the number of methods the association has been found relevant and which methods have found the association. The lines are sorted the same way they are in the ListByAnchors file.
- Result_END_graph.txt : a text file containing the final consensus network. Each line indicates either the name of a gene and it's color in the network (cyan for Dormancy genes, orange for Germination genes and black for candidate genes), or the data of an edge, namely the two linked genes, the consensus level and a value for each step of the program. If a given edge has not been found by a given step, this step's value is 'NO'. If it has, the value can be the Pearson's value (Co-Expression step), a 'YES' (KNN and Cluster Path steps), or a rank (CNI step).
