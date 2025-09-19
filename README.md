# 4-MAT_project

The 4-MAT program's purpose is to apply a multi-method analysis to transcriptomic data in order to find statistically relevant links between genes of interest (anchors) who play a known role in given metabolic pathways and candidate genes who may have a yet unknown role in those pathways.
At time of creation, this package has been used in the hope to find missing genes in the Dormancy-Germination regulation process in Arabidopsis thaliana.

For a detailed description of the program (rules of use, analysis methods' explanation, resulting files), please check the 4-MAT USER MANUAL.

This package initially contains the following files :

- You will find in the main folder (other than the current README) :
	- 4-MAT USER MANUAL.docx : the user manual which explains in details how the program works. A pdf version of this file is also available.
	- Block Diagram.pdf : a diagram to visualize how the different code files interact with one another.
- You will find in the Code folder :
	- 4-MAT_XXXX_YY_ZZ.py : the main code file of the program. The XYZ characters stands for the date of the lastest update.
	- Setting_functions.py : the first auxiliary code file containing the definitions of functions the main code file uses to set the parameters and load the data.
	- PCEIN_functions.py : the second auxiliary code file containing the definitions of functions the main code file uses during the first analysis method.
	- KNN_RBH_function.py : the third auxiliary code file containing the definitions of functions the main code file uses during the second analysis method.
	- NPC_functions.py : the fourth auxiliary code file containing the definitions of functions the main code file uses during the third analysis method.
	- ClusterPath_functions.py : the fifth auxiliary code file containing the definitions of functions the main code file uses during the fourth analysis method.
	- Consensus_functions.py : the sixth auxiliary code file containing the definitions of functions the main code file uses during the consensus calculation.
	- Parameters.txt : this file describs the settings of the program and how to customize them.
	- Visualize_Network.py : an auxiliary code file allowing to visualize any network file created during a run of 4-MAT.
- You will find in the Data folder :
	- GlobalPool_9103.txt : this file contains an ordered list of 9103 genes (anchors and candidats) to be analysed by the program.
	- GlobalPool_16428.txt : this file contains an ordered list of 16428 genes (anchors and candidats) to be analysed by the program.
	- AnchorPool_Dormancy.txt : this file contains an ordered list of anchor genes whose role is to promote Dormancy.
	- AnchorPool_Germination.txt : this file contains an ordered list of anchor genes whose role is to promote Germination.
	- Data_X_9103.csv (with X = 1,2,3) contains transcriptomic data for Pool_9103.
	- Data_1-2-3_9103.csv contains the merged data from Data_X.csv.
	- Data_Y_16428.csv (with Y = 2,3,5,6) contains transcriptomic data for Pool_16428.
	- Data_2-3-5-6_16428.csv contains the merged data from Data_Y.csv.
 	- NOTE : "merging" the data means that for each gene, the data vectors from the concerned datasets have been concatenated into a single vector.
- You will find in the Results folder most of the expected results from a run of the program on the Data_2-3-5-6_16428.csv file according to the current state of the Parameters.txt file. For a description of all the results files, see the User Manual.
	- Results are stored into two sub-folders titled 'Results 2023 (obsolete)' and 'Results 2025'. These sub-folders contain results from runs of two different versions of the program. (The 2023 version no longer being reproducible after a upgrade of the 'scikit-learn' python module used in Method 4 changed it's results in 2025).
	- Three result files are missing in the sub-folder, for they are too large to be uploaded here. There respective default names are "Result_1a_P-CEIN_graph.txt", "Result_2_KNN-RBH_Graph.txt" and "Result_END_graph.txt". See the User Manual for their detailed descriptions.

Quick description of 4-MATS's analysis methods :
- Method 1 - Pearson based Co-Expression Intersected Network : This method calculates the Pearson's Correlation Coefficient (PCC) of couples of genes throughout all datasets. A couple of two given genes is retained if in all provided dataset, the couple shows PCCs that are higher than a threshold in absolute value and are of the same sign (either all positives or all negatives).

- Method 2 - K-Nearest Neighbors (KNN) with Reciprocal Best Hit (RBH) : This method calculates the nearest neighborhood for each gene and retains couples whose two genes have each other in their respective neighborhood. When several datasets are provided, a given couple must be found in all of them to be retained at the end of the method.

- Method 3 - Network Properties Closeness (NPC) : This method build networks around anchor genes and selects candidates that are show good association to the anchors based on properties such as number and size of cliques. If several datasets are provided, a given candidate is retained at the end of the method only if it is retained by all datasets separatly. However, this candidate can be associated to different anchor genes from one dataset to another, so to each association is attributed a rank indicating how many datasets has retained this particular association.

- Method 4 - Cluster Path : This method segregates the genes by using several 1-dimensional K-Means algorithms. A K-Means clustering is run on each timepoint of a dataset and as the clusters are sorted by ascending order of their centroids' values, each gene can be associated to a vector indicating which cluster it belongs to in each timepoint. A couple of two given genes is retained if the two genes are associated to the same vector. If several datasets are provided, their vectors are put together to form a longer vector and the same logic as before applies to retain couples of genes.

- End - Analysis Consensus : This final step of the program crosses all results from all the above methods and attributes to each association a level indicating how many methods have found it. The higher the level, it more relevant the association ought to be.

- Network Visualization : When running the Visualize_Network.py code file, you can either :
 	- Just give a network file. The code will then draw all nodes and edges.
 	- Give a query of specific genes in addition to the network file. The code will then only draw those genes, their neighbors and the edges between them. For each gene in the query, the console will also display a dataframe of all neighbors as well as the attributes of the edges.
