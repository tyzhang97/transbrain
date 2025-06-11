Introduction
--------------------

- While mouse models enable direct investigation of gene-phenotype relationships, translating these findings to human clinical presentations remains challenging.

- Here, we demonstrate TransBrain's utility in bridging this gap, establishing an objective approach to link human imaging phenotypes with genetic mouse models in autism.

- `Deformation-based morphology (DBM) <https://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch6.pdf>`_ method was used to quantify regional volumes based on structural images. For mouse models, we quantified mutation-specific structural alterations by computing standardized volume differences (Cohen's d) between mutant and control groups. For human patients, we calculated risk scores through a normative model (see **Methods** section of our `paper <https://www.biorxiv.org/content/10.1101/2025.01.27.635016v2.abstract>`_).

- The `normative modeling <https://pubmed.ncbi.nlm.nih.gov/35650452/>`_ approach characterizes atypical brain features by comparing each individual against a statistical model of typical brain development, to quantify region-specific brain volume deviations.

- Using TransBrain, we can compare the deviation patterns and link it with specific genetic mechanisms via `Allen Human Brain Data <https://www.nature.com/articles/nature11405>`_.


Data
--------

- Autism mouse models image: (`link <https://www.nature.com/articles/mp201498>`_)

- Human ABIDE I data: (`link <https://fcon_1000.projects.nitrc.org/indi/abide>`_)

- `Allen Human Brain Genetic Data <https://www.nature.com/articles/nature11405>`_ obtained via the `Abagen toolbox <https://abagen.readthedocs.io/en/stable/index.html>`_

- Pre-saved data: (`link <https://github.com/ibpshangzheng/Transbrain/tree/main/tutorials/notebooks/translation/autism_mutation>`_)
