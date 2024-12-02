import pandas as pd
import networkx as nx
import pickle
import os
import sys
import  argparse

CE_file_path='./Connectome-embeddings'
dir_name_parameter='./Graphembeddings'

sys.path.append(CE_file_path)
from connectome_embed_nature import create_embedding


def get_path_paramater():

    parser=argparse.ArgumentParser(add_help=True,description='Generate graph embeddings',formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i','--input',type=str,default='',help='input matrix path')
    parser.add_argument('-t','--type',type=str,default='homologous',help='homologous or human')
    parser.add_argument('-pm','--permutation',type=int,default=15,help='permutation number')
    parser.add_argument('-d','--dimensions',type=int,default=40,help='graph embedding vector dimensions')
    parser.add_argument('-wl','--walk_length',type=int,default=40,help='random walk node length')
    parser.add_argument('-ws','--window_size',type=int,default=5,help='the number of input adjacent node')
    parser.add_argument('-p','--preturn',type=float,default=0.01,help='return parameter')
    parser.add_argument('-q','--qinout',type=float,default=0.1,help='in out parameter')
    parser.add_argument('-n','--null',type=str,default='False',help='null model test')
    args=parser.parse_args()
    if (args.input=='') or (not os.path.exists(args.input)):
        raise Exception('Input error: no such path')
    if (args.type != 'homologous') and (args.type !='human'):
        raise Exception('Input type error')
    if (args.null !='False') and (args.null !='True'):
        raise Exception('Input error: False or True')
    return (args.input,args.type,args.permutation,args.dimensions,args.walk_length,args.window_size,args.preturn,args.qinout,args.null)

def Run_CE(CE_matrix,dir_name,input_edge_list,output_embedding,current_name,permutation,d,wl,ws,p,q):
    
    current_dti=CE_matrix
    G=nx.DiGraph(current_dti)
    print(nx.is_weakly_connected(G))
    
    word2Vecmodelsorted=create_embedding(dir_name, input_edge_list, output_embedding, current_dti, current_name,permutation_no=permutation,dimensions=d,walk_length=wl,window_size=ws,p=p,q=q)
    
    with open(output_embedding+'_graph_embeddings.pkl','wb') as f:
        pickle.dump(word2Vecmodelsorted,f)

if __name__=='__main__':

    arg=get_path_paramater()
    
    CE_input_path,CE_type,permutation,d,wl,ws,p,q,null=arg
    
    atlas_type=CE_input_path.split('/')[-1].split('.')[0].split('_')[-2]+'_'+CE_input_path.split('/')[-1].split('.')[0].split('_')[-1]
    

    if null=='True':

        times=CE_input_path.split('.')[0].split('_')[-1]
        info='Null_Human_Mouse_{}_p{}_q{}'.format(times,p,q)
        dir_name=os.path.join(dir_name_null,atlas_type)

    else:

        if CE_type=='homologous':
            info='Human_Mouse_p{}_q{}'.format(p,q)
        else:
            info='Human_p{}_q{}'.format(p,q)
        dir_name=os.path.join(dir_name_parameter,atlas_type)

    os.makedirs(dir_name,exist_ok=True)

    input_edge_list = os.path.join(dir_name,'{0}_graph_only_positive.txt'.format(info))
    output_embedding = os.path.join(dir_name,'{0}_graph'.format(info))
    current_name='{0}_graph_embeddings_test'.format(info)
    CE_matrix=pd.read_csv(CE_input_path)
    CE_matrix.set_index('Unnamed: 0',inplace=True,drop=True)
    CE_matrix=CE_matrix.values

    Run_CE(CE_matrix,dir_name, input_edge_list, output_embedding,current_name,permutation,d,wl,ws,p,q)
