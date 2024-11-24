# Transbrain

The core code of paper 'Transbrain: A translational framework for whole-brain mapping linking human and mouse'.

![Fig1_HSZ_check](https://github.com/user-attachments/assets/da7ebcf1-43ad-4ca3-a6c4-95cbbe654891)

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

![Fig3_HSZ_check_画板 1](https://github.com/user-attachments/assets/5df54559-25ac-4b68-9b77-05848c448f9c)

* [Graphembedding_analysis](https://github.com/ibpshangzheng/Transbrain/tree/main/notebook/Graphembedding_analysis): evaluate whether graph embeddings incorporate structural connectivity information while preserving cross-species transcriptional similarity.

* [Mouse_to_Human](https://github.com/ibpshangzheng/Transbrain/tree/main/notebook/Mouse_to_Human): The process for transforming mouse whole-brain phenotypes into human equivalents.

# Python Dependencies

Code mainly depends on the Python (>= 3.8.5) scientific stack.

```python
numpy V1.22.4
pandas V1.5.3
matplotlib V3.7.3
Seaborn V0.13.2
nilearn V0.10.2
scipy V1.10.1
scikit-learn 1.3.2
```



