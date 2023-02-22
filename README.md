# 4-MAT_project

The 4-MAT program's purpose is to apply a multi-method analysis to transcriptomic data in order to find statistically relevant links between genes of interest (anchors) who play a known role in given metabolic pathways and candidate genes who may have a yet unknown role in those pathways.
At time of creation, this package has been used in the hope to find missing genes in the Dormancy-Germination regulation process in Arabidopsis thaliana.

For a detailed description of the program (rules of use, analysis methods' explanation, resulting files), please check the 4-MAT USER MANUAL.

This package folder initially contains the following files (arranged by type):

- Python files :
	- 4-MAT_XXXX_YY_ZZ.py : the main code file of the program. The XYZ characters stands for the date of the lastest update.
	- Setting_functions.py : the first auxiliary code file containing the definitions of functions the main code file uses to set the parameters and load the data.
	- PCEIN_functions.py : the second auxiliary code file containing the definitions of functions the main code file uses during the first analysis method.
	- KNN_RBH_function.py : the third auxiliary code file containing the definitions of functions the main code file uses during the second analysis method.
	- NPC_functions.py : the fourth auxiliary code file containing the definitions of functions the main code file uses during the third analysis method.
	- ClusterPath_functions.py : the fifth auxiliary code file containing the definitions of functions the main code file uses during the fourth analysis method.
	- Consensus_functions.py : the sixth auxiliary code file containing the definitions of functions the main code file uses during the consensus calculation.
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
	NOTE : "merging" the data means that for each gene, the data vectors from the concerned datasets have been regrouped into a single vector.
- Docx file : the user manual which explains in details how the program works.

Quick description of 4-MATS's analysis methods :
- Method 1 - Pearson based Co-Expression Intersected Network : This method calculates the Pearson's Correlation Coefficient (PCC) of couples of genes throughout all datasets. A couple of two given genes is retained if in all provided dataset, the couple shows PCCs that are higher than a threshold in absolute value and are of the same sign (either all positives or all negatives).

- Method 2 - K-Nearest Neighbors (KNN) with Reciprocal Best Hit (RBH) : This method calculates the nearest neighborhood for each gene and retains couples whose two genes have each other in their respective neighborhood. When several datasets are provided, a given couple must be found in all of them to be retained at the end of the method.

-Method 3 - Network Properties Closeness (NPC) : This method build networks around anchor genes and selects candidates that are show good association to the anchors based on properties such as number and size of cliques. If several datasets are provided, a given candidate is retained at the end of the method only if it is retained by all datasets separatly. However, this candidate can be associated to different anchor genes from one dataset to another, so to each association is attributed a rank indicating how many datasets has retained this particular association.

- Method 4 - Cluster Path : This method segregates the genes by using several 1-dimensional K-Means algorithms. A K-Means clustering is run on each timepoint of a dataset and as the clusters are sorted by ascending order of their centroids' values, each gene can be associated to a vector indicating which cluster it belongs to in each timepoint. A couple of two given genes is retained if the two genes are associated to the same vector. If several datasets are provided, their vectors are put together to form a longer vector and the same logic as before applies to retain couples of genes.

- End - Analysis Consensus : This final step of the program crosses all results from all the above methods and attributes to each association a level indicating how many methods have found it. The higher the level, it more relevant the association ought to be.
