# TransBrain

TransBrain is an integrated computational framework for bidirectional translation of brain-wide phenotypes between humans and mice. Specifically, TransBrain provides a systematic approach for cross-species quantitative comparison and mechanistic investigation of both normal and pathological brain functions.

![TransBrain_FIG1](https://github.com/user-attachments/assets/1d87b730-3928-480a-98fa-7a0492754ee5)

What can TransBrain do?

1. Exploring the similarity relationships at the transcriptional level.

2. Inferring the conservation of whole-brain phenotypes.

3. Transforming and annotating whole-brain functional circuits.

4. Linking specific mouse models with human diseases.

## Further Reading

If you wish to learn more about the construction details of this method, please refer to our article: [https://www.biorxiv.org/content/10.1101/2025.01.27.635016v1](https://www.biorxiv.org/content/10.1101/2025.01.27.635016v1) (in preprint).


## Installation
TransBrain is on pypi: https://pypi.org/project/TransBrain/

To install TransBrain as a package, run:

```sh
pip install TransBrain
```

You can also create a conda environment from the environment.yml file:

* First, clone this repository,
```sh
git clone https://github.com/ibpshangzheng/Transbrain.git
```

* Then, create the environment,
```sh
cd TransBrain
conda env create -f environment.yml
```

* Activate the environment,
```sh
conda activate transbrain_env
```

## Python Dependencies

The project mainly depends on Python (>= 3.8.5).

```
matplotlib==3.7.5,
matplotlib-inline==0.1.7,
nibabel==5.2.1,
nilearn==0.10.4,
numpy==1.24.4,
openpyxl==3.1.5,
pandas==2.0.3,
scikit-learn==1.3.2,
scipy==1.10.1,
seaborn==0.13.2,
six==1.17.0
```
See full list in environment.yml file. 


## Getting Started
### Usage
After installation, you can refer to the 'Test.py' file in the root directory of TransBrain to explore usage examples and other functions.

```python
import pandas as pd
import TransBrain as TB

# Initialize TransBrain
Transformer = TB.trans.SpeciesTrans()
```

```python
# example from mouse to human

mouse_phenotype = pd.read_csv('./TransBrain/ExampleData/Mouse_cortex_example_data.csv',index_col=0)
mouse_phenotype_in_human = Transformer.mouse_to_human(mouse_phenotype, region_type='cortex', normalize_input=True, restore_output=False)
```

```python
# example from human to mouse

human_phenotype = pd.read_csv('./TransBrain/ExampleData/Human_cortex_example_data.csv',index_col=0)
human_phenotype_in_mouse = Transformer.human_to_mouse(human_phenotype, region_type='cortex', normalize_input=True, restore_output=False)
```

```python
#####get graph embeddings 

Human_Mouse_embedding = Transformer._load_graph_embeddings()
```

This allows seamless integration into your existing workflows. 🚀


### Toturials
We provided [**Tutorial Cases**](#tutorial-cases) demonstrating how to apply TransBrain for cross-species translation and comparison, which includes:

* Analyzing and visualizing transcriptional similarity between humans and mice.

* Characterizing the evolutionary spectrum of resting-state fMRI network phenotypes.

* Annotating the optogenetic circuits in mice using Neurosynth.

* Linking gene mutations to imaging phenotype deviations in autism.

The analysis process and figures can be viewed in the Jupyter Notebook. The necessary files and datas for completing these analysis are included in the notebook's folder.



## License
This project is covered under the Apache 2.0 License.

## Support
For questions and comments, please file a Github issue and/or email Shangzheng Huang(huangshangzheng@ibp.ac.cn)


