Transcriptional Data Integration
==================================


AHBA data
--------------------------

- The `Allen Human Brain Atlas (AHBA) <https://www.nature.com/articles/nature11405>`_ provides comprehensive, high-resolution transcriptional profiles across the adult human brain. The data was downloaded from the Allen Institute’s API and preprocessed using the `abagen package <https://abagen.readthedocs.io/en/stable/index.html>`_.
- After processing, we obtained a gene expression matrix comprising 4,729 samples and 20,232 genes across all donors.

Human single-nucleus data
--------------------------

- We used the `large-scale single-nucleus dataset <https://www.science.org/doi/abs/10.1126/science.add7046>`_ for the human brain provided by the Linnarsson laboratory.
- This dataset includes single-nucleus transcriptomic data collected from multiple brain regions of four adult human donors.
- We retained the data from donors (H19.30.001 and H19.30.002) for further analysis and conducted preprocessing with `scanpy <https://scanpy.readthedocs.io/en/stable>`_.
