# Transbrain

The core code of paper 'Transbrain: A translational framework for whole-brain mapping linking human and mouse'

![Fig1_HSZ_check](https://github.com/user-attachments/assets/da7ebcf1-43ad-4ca3-a6c4-95cbbe654891)

# Overview

The source introduces the initial transcriptional and graph embeddings and outlines the process for transforming mouse whole-brain phenotypes into human equivalents. Python and Jupyter is used for analysis.

# Files and Analysis

## Files

* [atlas](https://github.com/ibpshangzheng/Transbrain/tree/main/atlas): human and mouse atlas, templates, and corresponding csv files.

* [sc_matrix](https://github.com/ibpshangzheng/Transbrain/tree/main/sc_matrix): human dti matrix and mouse tracer matrix used in our study.

* [TR_embeddings](https://github.com/ibpshangzheng/Transbrain/tree/main/TRembeddings/FinalModels): the region-specific transcriptional embeddings of detached model.

** [MixData_Cortical_Train_Repeat1000](https://github.com/ibpshangzheng/Transbrain/tree/main/TRembeddings/FinalModels/MixData_Cortical_Train_Repeat1000): cortical embeddings of the fused data repeated 100 times, with each dataset subjected to 10 iterations of model training.

** [MixData_SubCortical_Train_Repeat1000](https://github.com/ibpshangzheng/Transbrain/tree/main/TRembeddings/FinalModels/MixData_SubCortical_Train_Repeat1000): subcortical embeddings of the fused data repeated 100 times, with each dataset subjected to 10 iterations of model training.

* [Graphembeddings](https://github.com/ibpshangzheng/Transbrain/tree/main/Graphembeddings): graph embeddings of Human-Mouse generated using graph walk embedding algorithms.
