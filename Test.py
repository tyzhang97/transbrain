import pandas as pd
import numpy  as np
import TransBrain as TB


#Initialize TransBrain
Transformer = TB.trans.SpeciesTrans()

#####get graphembeddings 
Human_Mouse_embedding = Transformer._load_graph_embeddings()

#####Example from mouse to human

mouse_phenotype = pd.read_csv('./TransBrain/ExampleData/Mouse_cortex_example_data.csv',index_col=0)
mouse_phenotype_in_human = Transformer.mouse_to_human(mouse_phenotype, region_type='cortex', normalize_input=True, restore_output=False)
print(mouse_phenotype_in_human)

#####Example from human to mouse

human_phenotype = pd.read_csv('./TransBrain/ExampleData/Human_cortex_example_data.csv',index_col=0)
human_phenotype_in_mouse = Transformer.human_to_mouse(human_phenotype, region_type='cortex', normalize_input=True, restore_output=False)
print(human_phenotype_in_mouse)

#####fetch atlas in TransBrain

human_atlas = TB.atlas.fetch_human_atlas(region_type='cortex')
mouse_atlas = TB.atlas.fetch_mouse_atlas(region_type='cortex')


##### get phenotypes in Human and Mouse atlas used in TransBrain

phenotype_nii_path = './TransBrain/ExampleData/Example_human_phenotype.nii.gz'

human_phenptypes = TB.base.get_region_phenotypes(phenotype_nii_path)

print(human_phenptypes)