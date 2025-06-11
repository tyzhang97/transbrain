Graph-based Random Walk
==================================



Human DTI
--------------

- We utilized diffusion-weighted imaging (DWI) data from the `Human Connectome Project (HCP) <https://www.humanconnectome.org/study/hcp-young-adult/data-releases>`_, which have been preprocessed following HCP's standard pipeline.
- Further post-processing steps of tractography and connectome construction were carried out using `MRtrix3 <http://mrtrix.readthedocs.io>`_.
- The final set of streamlines was then parcellated into brain regions according to our human brain atlas.

Mouse Tracer
-----------------

- Tracer data maps neural connectivity by tracking the movement of injected tracers along axonal pathways in animal brains.
- The mouse connectome data we used was initially sourced from the `Allen Mouse Connectivity Atlas <https://connectivity.brain-map.org>`_.
- We averaged all voxel-level connectivity values between each pair of brain regions according to index correspondence between the voxel connectivity matrix and the voxel-scale connectome template, resulting in a 66 ROIs connectivity matrix.

Embedding space generated from Human-Mouse graph
----------------------------------------------------

- Intra-species edges are defined by anatomical connectivity, derived from mouse viral tracer data and human diffusion tensor imaging (DTI) tractography.
- Cross-species edges were established using transcriptomic latent embeddings from DNN model, constrained by coarse-scale anatomical hierarchies.
- For the pruned Human-Mouse graph, we used the `Node2vec algorithm by Sporns et al. <https://www.nature.com/articles/s41467-018-04614-w>`_ to construct graph embedding.
