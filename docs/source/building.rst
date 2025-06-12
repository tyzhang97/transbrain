.. _build_method:

Construction Pipeline
========================================

- Here we provide a step-by-step guide to follow the entire TransBrain construction pipeline.

- If you have more specific needs (such as constructing mapping matrices based on your own data or atlases), this tutorial will help you.

- At the top of each page you can find a brief introduction of the data and methods, for detailed description please refer to our `paper <https://www.biorxiv.org/content/10.1101/2025.01.27.635016v2.abstract>`_.

- Due to the large data size, the dataset is not included in the GitHub repository. If needed, you can download it from the following link in ``pipeline/datasets/README.md`` and place it in the same directory.


1. **Spatial Transcriptomic Matching**

- We integrated complementary human transcriptomic datasets, including `microarray data <https://www.nature.com/articles/nature11405>`_ and large-scale `single-nucleus RNA sequencing data <https://www.science.org/doi/abs/10.1126/science.add7046>`_ to enhance sample size and diversity.  
- A detached deep neural network model was trained on the integrated transcriptomic data to learn region-specific latent embeddings that are generalizable across species.

2. **Graph-based Random Walk**

- A heterogeneous graph was constructed to connect brain regions within and across species.   
- Latent embeddings capturing integrated transcriptomic, anatomical, and connectivity information were generated.

3. **Bidirectional Mapping**

- `Dual-regression <./methods/mapping.rst>`_ method was employed to translate brain-wide patterns.  
- Establishing a unified cross-species latent space that enables cross-species analysis and quantitative comparison.




.. toctree::
   :maxdepth: 1
   :caption: Steps


   pipeline/run_pipeline/integration.ipynb
   pipeline/run_pipeline/dnn.ipynb
   pipeline/run_pipeline/graph_embedding.ipynb
   ./methods/mapping
   ./methods/embeddings_analysis.ipynb