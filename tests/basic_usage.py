from pathlib import Path
import pandas as pd
import numpy as np
import transbrain as tb
import os

current_file = Path(__file__).resolve()
base_dir = current_file.parent.parent


#####Example from mouse to human
#Initialize TransBrain for bn atlas
Transformer = tb.trans.SpeciesTrans('bn')

mouse_phenotype = pd.read_csv(base_dir.joinpath('transbrain/exampledata/mouse/mouse_all_example_data.csv'),index_col=0)
mouse_phenotype_in_human = Transformer.mouse_to_human(mouse_phenotype, region_type='all', normalize=True)
print(mouse_phenotype_in_human)



#####Example from human to mouse
Transformer = tb.trans.SpeciesTrans('bn')

human_phenotype = pd.read_csv(base_dir.joinpath('transbrain/exampledata/human/bn/human_bn_all_example_data.csv'),index_col=0)
human_phenotype_in_mouse = Transformer.human_to_mouse(human_phenotype, region_type='all', normalize=True)
print(human_phenotype_in_mouse)



##### Get phenotypes in Human atlas used in TransBrain
human_atlas = tb.atlas.fetch_human_atlas(atlas_type='bn',region_type='cortex')
phenotype_nii_path = base_dir.joinpath('transbrain/exampledata/human/human_example_phenotype_data.nii.gz')
human_phenptype_extracted = tb.base.get_region_phenotypes(phenotype_nii_path, atlas_dict = human_atlas)
print(human_phenptype_extracted)

#Mapping
Transformer = tb.trans.SpeciesTrans('bn')
human_phenotype_in_mouse = Transformer.human_to_mouse(human_phenptype_extracted, region_type='cortex', normalize=True)
print(human_phenotype_in_mouse)



##### Get phenotypes in Mouse atlas used in TransBrain
mouse_atlas = tb.atlas.fetch_mouse_atlas(region_type='all')
phenotype_nii_path = base_dir.joinpath('transbrain/exampledata/mouse/mouse_example_phenotype_data.nii.gz')
mouse_phenptype_extracted = tb.base.get_region_phenotypes(phenotype_nii_path, atlas_dict = mouse_atlas)
print(mouse_phenptype_extracted)

#Mapping
Transformer = tb.trans.SpeciesTrans('bn')
mouse_phenotype_in_human = Transformer.mouse_to_human(mouse_phenptype_extracted, region_type='all', normalize=True)
print(mouse_phenotype_in_human)


#####Get graphembeddings 
Transformer = tb.trans.SpeciesTrans('bn')
Human_Mouse_embedding_bn = Transformer._load_embeddings()


