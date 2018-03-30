# Lab6 - Data Exploration


## Submission 
* Please note I have only pushed the content that I have produced in the lab, and any immediate dependencies.
Other reference files originally cloned from the master repository are not included

	* the jupyter notebook - shows all experiments that were conducted

	* the source code file (.py) is also included - note this shows on the latest version. Earlier versions (for earlier experiments) are referenced in the jupyter notebook

	* any images (charts etc.) generated during the experiments


## Analysis (Overview) 
* Following points summarise the experimental findings as detailed in the Jupyter notebook

	* Agglomerative CLustering produces identical clusters ervy time (for a give #clusters) - unlike KNN where initial seeding effects the resultant cluster

	* According to silhouette score, optimal agglomerative clustering (4 clusters) is slightly more cohesive than optimal KNN (3 clusters)

	* For cluster visualtion, note that I used correlation scores provided in lecture notes, to select the bets attributes for visualisation

	* I also tried PCA reduction (setting 3 dimensions). In this case the same clusters are produced every time (even for KNN), so no confidence interval was relevant (hence why it wasnt shown in the notebook) - these clusters are more cohesive.






