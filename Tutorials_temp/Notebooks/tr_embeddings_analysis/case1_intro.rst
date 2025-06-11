Introduction
--------------------

This example demonstrates how to perform cross-species similarity analysis of transcriptional embeddings.

- The aim of this experiment is to explore the cross-species homology of region-specific gene expression patterns.
- In tutorials of :ref:`build_method`, we have demonstrated how to train the detached-model.
- Here we first applied the trained models to region-averaged gene expression data from publicly available `mouse spatial transcriptomics resources <https://www.science.org/doi/full/10.1126/sciadv.abb3446>`_ and then quantitatively evaluated cross-species matching performance by assessing their `similarity ranks <https://elifesciences.org/articles/79418>`_ among all human cortical ROIs.

Data
----

- Human and Mouse transcriptional embeddings generated from trained detached models (`link <https://github.com/ibpshangzheng/transbrain/tree/main/tutorials/tr_embeddings/FinalModels>`_)

