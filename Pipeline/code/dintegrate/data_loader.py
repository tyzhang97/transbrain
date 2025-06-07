import pandas as pd
from typing import Tuple
from config import Config,parse_arguments

def load_data() -> Tuple[pd.DataFrame, ...]:
    """
    Load all required files
    
    Args:
        config: Pipeline configuration object
        
    Returns:
        Tuple of (align_csv,ahba_mean_ex,sample_ex,h19_data,thre_df,gene_filter)

    """
    args = parse_arguments()

    config = Config()
    
    config.update_parameters(
        pool = args.pool,
        integrate = args.integrate,
        downsample = args.downsample,
        zscore = args.zscore,
        pool_type = args.pool_type,
        ahba_weight = args.ahba_weight,
        sn_weight = args.sn_weight,
        jobs = args.jobs,
        iterations = args.iterations,
        downsample_n = args.downsample_n,
        corr_threshold = args.corr_threshold,
        screening_thre = args.screening_thre,
        max_value = args.max_value
    )

    config.update_paths(
        mean_path = args.mean_path,
        sample_path = args.sample_path,
        h19_path = args.h19_path,
        alignment_file = args.alignment_file,
        gene_file = args.gene_file,
        thre_file = args.thre_file,
        pool_s_path = args.pool_s_path,
        integrated_s_path = args.integrated_s_path 
    )
    
    align_csv = pd.read_csv(config.data_files['alignment_file'], index_col=0)   
    ahba_mean_ex = pd.read_csv(config.data_files['mean_path'], index_col=0)
    sample_ex = pd.read_csv(config.data_files['sample_path'], index_col=0)
    h19_data = pd.read_feather(config.data_files['h19_path'])
    h19_data.set_index('index',inplace=True,drop=True)
    thre_df = pd.read_csv(config.data_files['thre_file'], index_col=0)
    gene_filter = pd.read_csv(config.data_files['gene_file'])['genes'].tolist()
    
    if config.pool_type != 'all':
        is_cortex = h19_data.index.str.contains('Cerebral cortex')
        h19_data = h19_data[is_cortex] if config.pool_type == 'cortex' else h19_data[~is_cortex]

    ahba_mean_ex = ahba_mean_ex[gene_filter]
    sample_ex = sample_ex[gene_filter]
    h19_data = h19_data[gene_filter]

    h19_data['cell_id'] = range(len(h19_data))
    
    return align_csv, ahba_mean_ex, sample_ex, h19_data, thre_df, gene_filter,config