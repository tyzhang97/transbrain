

Welcome to TransBrain
====================================
|DOI| |Github| |PyPI| |PyPI_downloads|

.. |DOI| image:: https://img.shields.io/badge/DOI-10.1101%2F2025.01.27.635016-blue
   :target: https://doi.org/10.1101/2025.01.27.635016
   :alt: View DOI
.. |Github| image:: https://img.shields.io/badge/View_in_Github-6C51D4?logo=github&logoColor=white
   :target: https://github.com/ibpshangzheng/transbrain
.. |PyPI| image:: https://img.shields.io/pypi/v/transbrain
   :target: https://pypi.org/project/TransBrain/
.. |PyPI_downloads| image:: https://static.pepy.tech/badge/transbrain
   :target: https://pepy.tech/project/transbrain


TransBrain is an integrated computational framework for bidirectional translation of brain-wide phenotypes between humans and mice. Specifically, TransBrain provides a systematic approach for cross-species quantitative comparison and mechanistic investigation of both normal and pathological brain functions.

.. image:: ../../figure/transbrain_fig1.jpg
    :align: center

~~~~ 

Key Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**🧬 Spatial Transcriptomic Matching**

- We integrated complementary human transcriptomic datasets, including `microarray data <https://www.nature.com/articles/nature11405>`_ and large-scale `single-nucleus RNA sequencing data <https://www.science.org/doi/abs/10.1126/science.add7046>`_ to enhance sample size and diversity.  
- A detached deep neural network model was trained on the integrated transcriptomic data to learn region-specific latent embeddings that are generalizable across species.

**🧠 Graph-based Random Walk**  

- A heterogeneous graph was constructed to connect brain regions within and across species.  
- Intra-species edges were defined based on anatomical connectivity, using `viral tracer data <https://www.nature.com/articles/nature13186>`_ for mouse and diffusion tensor imaging (DTI) tractography for human from `Human Connectome Project (HCP) <https://www.humanconnectome.org/study/hcp-young-adult/data-releases>`_.  
- Cross-species edges were determined by transcriptomic embeddings learned from the first phase, constrained by coarse-scale anatomical hierarchies.  
- Latent embeddings capturing integrated transcriptomic, anatomical, and connectivity information were generated.

**🔄 Bidirectional Mapping**  

- `Dual-regression <./methods/mapping.rst>`_ method was employed to translate brain-wide patterns, such as imaging phenotypes.  
- Establishing a unified cross-species latent space that enables cross-species analysis and quantitative comparison.

~~~~ 

What can TransBrain do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**TransBrain** offers many promising applications. You can learn how to implement these functions in the :ref:`tutorial-section` section of this documentation or through our `GitHub repository <https://github.com/ibpshangzheng/transbrain>`_.

1. Exploring the similarity relationships at the transcriptional level.
2. Inferring the conservation of whole-brain phenotypes.
3. Transforming and annotating whole-brain functional circuits.
4. Linking specific mouse models with human diseases.


~~~~ 

Usage Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We provide **three levels** of usage options to accommodate users with different needs. See detials in :ref:`user-guide`.

General Use
^^^^^^^^^^^^^^^^
- If you want to use the embedded functions of TransBrain to map your data and have some basic knowledge of ``Python``, you can refer to our :ref:`API` and :ref:`tutorial-section` for detailed guidance.

Online Mapping
^^^^^^^^^^^^^^^^
- We also provide a code-free website  (https://transbrain.streamlit.app/), enabling users to directly upload your data for online mapping and visualization.

Advanced Use
^^^^^^^^^^^^^^^^
- If you want to understand the detailed methodology of TransBrain or adapt it to your specific needs, please refer to :ref:`build_method` section.


~~~~ 

Contents
============

.. toctree::
   :maxdepth: 2
   :caption: Setup & Usage

   installation
   user


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   building
   cases


.. toctree::
   :maxdepth: 2
   :caption: API

   api/utils
   api/external_api

 


.. toctree::
   :maxdepth: 2
   :caption: Other

   faq
   citing

~~~~ 

About
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- 🌐 [`GitHub Repository <https://github.com/ibpshangzheng/transbrain>`_]
- 📦 [`Install from PyPI <https://pypi.org/project/transbrain/>`_] 
- 📜 [`Our Paper <https://www.biorxiv.org/content/10.1101/2025.01.27.635016v1>`_]
- 📧 For questions, contact the author: Shangzheng Huang (huangshangzheng@ibp.ac.cn)