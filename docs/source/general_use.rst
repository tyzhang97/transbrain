.. _general-use:

General Usage
===============

- Here is a brief demonstration on how to use TransBrain. Before getting started, please `install <./installation.rst>`_ TransBrain and run ``test_transbrain.py`` to ensure successful installation.
- Refer to this `basic usage notebook <./tests/basic_usage.ipynb>`_ for a quick introduction to TransBrain's mapping workflow.
- Please see detailed documentation in :ref:`API` and :ref:`tutorial-section`.

~~~~ 

Supported Atlases
----------------------
- We provide several example data in the ``exampledata`` directory of the our `GitHub repository <https://github.com/ibpshangzheng/transbrain/tree/main/transbrain/exampledata>`_.
- You can download and check these files to learn the required input format. **Note** that when replacing with your own data, please **strictly follow** the format and region order in the provided template file. Mapping will **fail** if the structure is incorrect.
- For detailed atlases information, please refer to our `paper <https://www.biorxiv.org/content/10.1101/2025.01.27.635016v1>`_ and `transbrain/atlas <https://github.com/ibpshangzheng/transbrain/tree/main/transbrain/atlas>`_
- Support for additional atlases will be expanded in future updates.


Human Brain Atlases
^^^^^^^^^^^^^^^^^^^^^

We currently provide the following the options. 

- `BN (Brainnetome Atlas) <https://atlas.brainnetome.org/>`_

.. note::
    The naming of regions in the Brainnetome (BN) atlas are defined based on the anatomical locations from `Brodmann atlas <https://en.wikipedia.org/wiki/Brodmann_area>`_. You can check the correspondence in the BN website or in this `table <https://github.com/ibpshangzheng/Transbrain/blob/main/transbrain/atlas/BNA_subregions.xlsx>`_ to help understand. For subcortical regions, we adopted a `hybrid approach (22 ROIs) <https://github.com/ibpshangzheng/transbrain/tree/main/transbrain/atlas>`_ that integrates the Brainnetome Atlas, the `Allen Brain Atlas <https://community.brain-map.org/t/allen-human-reference-atlas-3d-2020-new/405>`_, and `public manual delineations <https://www.sciencedirect.com/science/article/abs/pii/S1053811913001237?via%3Dihub>`_.

- `DK (Desikan-Killiany Atlas) <https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation>`_
- `AAL (Automated Anatomical Labeling) <https://www.gin.cnrs.fr/en/tools/aal/>`_


Mouse Brain Atlas
^^^^^^^^^^^^^^^^^^^^^

- `CCFv3 (Allen Mouse Common Coordinate Framework v3) <https://atlas.brain-map.org/>`_

~~~~ 

Step 1: Initialization
--------------------------
Before starting the mapping, you need to initialize TransBrain by creating an instance of the ``SpeciesTrans`` class.

.. code-block:: python

    import pandas as pd
    import transbrain as tb

    # Initialize TransBrain for BN atlas
    Transformer = tb.trans.SpeciesTrans('bn')

~~~~ 

Step 2: Prepare input data
---------------------------------
There are two ways to input data. One is that you already have ROI-level phenotype data as ``CSV`` table which follows the format and region order in the `provided template file <https://github.com/ibpshangzheng/transbrain/tree/main/transbrain/exampledata>`_. The DataFrame contains two columns: ``['Anatomical Name', 'Phenotype']``.

.. code-block:: python

    # Example mouse data
    mouse_phenotype = pd.read_csv('/transbrain/exampledata/mouse/mouse_all_example_data.csv',index_col=0)

.. code-block:: python

    # Example human data
    human_phenotype = pd.read_csv('/transbrain/exampledata/human/bn/human_bn_all_example_data.csv',index_col=0)


Or if you source data is volumetric data in ``.nii`` or ``.nii.gz`` format that has been aligned to the atlas space required by TransBrain. You can use the following function to extract ROI-level data.


Fetch atlas in TransBrain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can fetch the Human or Mouse atlas using the functions provided in the `atlas` module of TransBrain. These functions return both the atlas image and detailed region information for further mapping.

- ``atlas_type`` can be choose from ``['bn', 'dk', 'aal']``.
- ``region_type`` can be set to ``'cortex'``, ``'subcortex'``, or ``'all'`` (the entire brain). 
- returns a ``dictionary`` containing the following keys: ``[atlas, atlas_data, region_info, info_table]``

.. code-block:: python
    
    #fetch human atlas
    human_atlas = tb.atlas.fetch_human_atlas(atlas_type='bn',region_type='cortex')

.. code-block:: python

    #fetch mouse atlas
    mouse_atlas = tb.atlas.fetch_mouse_atlas(region_type='all')


Get phenotypes from image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Get phenotypes in Human atlas used in TransBrain
    phenotype_nii_path = '/transbrain/exampledata/human/human_example_phenotype_data.nii.gz'
    human_phenptype_extracted = tb.base.get_region_phenotypes(phenotype_nii_path, atlas_dict = human_atlas)

.. code-block:: python

    # Get phenotypes in Mouse atlas used in TransBrain
    phenotype_nii_path = ('/transbrain/exampledata/mouse/mouse_example_phenotype_data.nii.gz')
    mouse_phenptype_extracted = tb.base.get_region_phenotypes(phenotype_nii_path, atlas_dict = mouse_atlas)

~~~~ 

Step 3: Mapping
-------------------

Mouse to Human Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This function supports several optional parameters:

- ``region_type`` can be set to ``'cortex'``, ``'subcortex'``, or ``'all'`` (the entire brain). 
- ``normalize`` determines whether to normalize the input data before mapping. Default is ``True``, if your data has already been normalized, you can set it to ``False``.

.. code-block:: python

    # Example from mouse to human
    mouse_phenotype_in_human = Transformer.mouse_to_human(
        mouse_phenotype, 
        region_type='all', 
        normalize=True
        )

Human to Mouse Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mapping from human to mouse follows similar usage and requirements, but in the opposite direction.

.. code-block:: python

    # Example from human to mouse
    human_phenotype_in_mouse = Transformer.human_to_mouse(
        human_phenotype, 
        region_type='all', 
        normalize=True
        )

~~~~ 

Get Graph Embeddings
---------------------------
- Load the graph embedding matrix obtained from the construction progress, which serves as the foundation for training dual-regression mapping model. 
- This step is not required if you only want to use our precomputed matrices for mapping, as it has already been integrated into the function above.

.. code-block:: python

    # Get graph embeddings for BN atlas
    Transformer = tb.trans.SpeciesTrans('bn')
    Human_Mouse_embedding_bn = Transformer._load_embeddings()




