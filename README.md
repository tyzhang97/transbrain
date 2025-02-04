![Fig3_TR_pattern](https://github.com/user-attachments/assets/3c7187d1-96a7-4c6a-8151-4a50ff8412ab)# Transbrain

The code of paper 'TransBrain: A computational framework for translating brain-wide phenotypes between humans and mice'.

![Fig1](https://github.com/user-attachments/assets/7feaddc4-13cb-472d-95e1-5c3084b3d074)


# Overview

The source introduces the initial transcriptional and graph embeddings and outlines the process for transforming mouse whole-brain phenotypes into human equivalents. Python and Jupyter are used for analysis.

# Files and Analysis

The analysis process and figures can be viewed in the Jupyter Notebook. The necessary files for completing the analysis are included in the notebook's folder, while other files are organized in a separate folder.

## Files

* [TR_embeddings](https://github.com/ibpshangzheng/Transbrain/tree/main/TRembeddings/FinalModels): the region-specific transcriptional embeddings of detached model.

  * [MixData_Cortical_Train_Repeat1000](https://github.com/ibpshangzheng/Transbrain/tree/main/TRembeddings/FinalModels/MixData_Cortical_Train_Repeat1000): cortical embeddings of the fused data repeated 100 times, with each dataset subjected to 10 iterations of model training.

  * [MixData_SubCortical_Train_Repeat1000](https://github.com/ibpshangzheng/Transbrain/tree/main/TRembeddings/FinalModels/MixData_SubCortical_Train_Repeat1000): subcortical embeddings of the fused data repeated 100 times, with each dataset subjected to 10 iterations of model training.

* [Graphembeddings](https://github.com/ibpshangzheng/Transbrain/tree/main/Graphembeddings): graph embeddings of Human-Mouse generated using graph walk embedding algorithms.

* [atlas](https://github.com/ibpshangzheng/Transbrain/tree/main/atlas): human and mouse atlas, templates, and corresponding csv files.

* [sc_matrix](https://github.com/ibpshangzheng/Transbrain/tree/main/sc_matrix): human dti matrix and mouse tracer matrix used in our study.

## Analysis

* [TRembeddings_analysis](https://github.com/ibpshangzheng/Transbrain/tree/main/notebook/TRembeddings_analysis): cross-species correspondence of TR embeddings and conserved transcriptional gradients.

![Fig3_TR_pattern](https://github.com/user-attachments/assets/a5a768db-2901-4b5a-9437-4b581577ef05)


* [generate_graph_embeddings](https://github.com/ibpshangzheng/Transbrain/tree/main/generate_graph_embeddings): code to generate graph embeddings.

  * [Human_Mouse_embeddings](https://github.com/ibpshangzheng/Transbrain/blob/main/generate_graph_embeddings/Human_Mouse_Embedding.ipynb): notebook to generate Human-Mouse Graph.
 
  * [Connectome-embeddings](https://github.com/ibpshangzheng/Transbrain/tree/main/generate_graph_embeddings/Connectome-embeddings): the source node embedding code was provided by [Gideon Rosenthal et al](https://www.nature.com/articles/s41467-018-04614-w).
 

![graph_embedding](https://github.com/user-attachments/assets/06232867-7213-4357-9737-6f41c8cc5066)


* [Graphembedding_analysis](https://github.com/ibpshangzheng/Transbrain/tree/main/notebook/Graphembedding_analysis): evaluate whether graph embeddings incorporate structural connectivity information while preserving cross-species transcriptional similarity.

* [Mouse_to_Human](https://github.com/ibpshangzheng/Transbrain/tree/main/notebook/Mouse_to_Human): the process for transforming mouse whole-brain phenotypes into human equivalents.

![Fig7_HSZ_check_git](https://github.com/user-attachments/assets/b0526b91-9bca-4963-bae5-c262217ab9d8)

# Python Dependencies

Code mainly depends on the Python (>= 3.8.5) scientific stack.

```
numpy V1.22.4
pandas V1.5.3
matplotlib V3.7.3
Seaborn V0.13.2
nilearn V0.10.2
scipy V1.10.1
scikit-learn V1.3.2
```
# License
This project is covered under the MIT 2.0 License.


