#%%
import numpy as np
import pandas as pd
import math
import re, os
import glob
import logging

from pynwb import NWBFile, TimeSeries, NWBHDF5IO

from analysis.util import foraging_eff_baiting, foraging_eff_no_baiting

script_dir = os.path.dirname(os.path.abspath(__file__))

LEFT, RIGHT = 0, 1

def nwb_to_df(nwb):
    #%%
    df_trials = nwb.trials.to_dataframe()

    # Reformat data
    choice_history = df_trials.animal_response.map({0: 0, 1: 1, 2: np.nan}).values
    reward_history = np.vstack([df_trials.rewarded_historyL, df_trials.rewarded_historyR])
    p_reward = np.vstack([df_trials.reward_probabilityL, df_trials.reward_probabilityR])

    #%%
    # -- Session-based table --
    # - Meta data -
    subject_id, session_date, nwb_suffix = re.match(r"(?P<subject_id>\d+)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<n>\d+))?\.json", 
                                        nwb.session_id).groups()
    nwb_suffix = int(nwb_suffix) if nwb_suffix is not None else 0
    session_index = pd.MultiIndex.from_tuples([(subject_id, session_date, nwb_suffix)], 
                                            names=['subject_id', 'session_date', 'nwb_suffix'])

    # Parse meta info
    # TODO: when generating nwb, put meta info in nwb.scratch and get rid of the regular expression
    extra_water, rig = re.search(r"Give extra water.*:(\d*(?:\.\d+)?)? .*tower:(.*)?", nwb.session_description).groups()
    weight_after_session = re.search(r"Weight after.*:(\d*(?:\.\d+)?)?", nwb.subject.description).groups()[0]
    
    extra_water = float(extra_water) if extra_water !='' else 0
    weight_after_session = float(weight_after_session) if weight_after_session != '' else np.nan
    weight_before_session = float(nwb.subject.weight) if nwb.subject.weight != '' else np.nan

    dict_meta = {
        'rig': rig,
        'experimenter': nwb.experimenter[0],
        'experiment_description': nwb.experiment_description,
        'protocol': nwb.protocol,
        'session_start_time': nwb.session_start_time,
        'weight_before_session': weight_before_session,
        'weight_after_session': weight_after_session,
        'water_during_session': weight_after_session - weight_before_session,
        'water_extra': extra_water
        }

    df_session = pd.DataFrame(dict_meta, 
                            index=session_index,
                            )
    # Use hierarchical index (type = {'metadata', 'session_stats'}, variable = {...}, etc.)
    df_session.columns = pd.MultiIndex.from_product([['metadata'], dict_meta.keys()],
                                                names=['type', 'variable'])

    # - Compute session-level stats -
    # TODO: Ideally, all these simple stats could be computed in the GUI and send a copy to the meta session.json file
    
    total_trials = len(df_trials)
    finished_trials = np.sum(~np.isnan(choice_history))
    reward_trials = np.sum(reward_history)

    reward_rate = reward_trials / finished_trials

    # TODO: add more stats
    # early_lick_ratio = 
    # double_dipping_ratio = 
    # block_num
    # mean_block_length
    # mean_reward_sum
    # mean_reward_contrast
    # autowater_num
    # autowater_ratio
    #
    # mean_iti
    # mean_reward_sum
    # mean_reward_contrast 
    # ...


    foraging_eff_func = foraging_eff_baiting if 'bait' in nwb.protocol.lower() else foraging_eff_no_baiting
    foraging_eff, foraging_eff_random_seed = foraging_eff_func(reward_rate, p_reward[LEFT,:], p_reward[RIGHT, :])

    # -- Add session stats here --
    dict_session_stat = {
        'total_trials': total_trials,
        'finished_trials': finished_trials,
        'finished_rate': finished_trials / total_trials,
        'ignore_rate': np.sum(np.isnan(choice_history)) / total_trials,
        'reward_trials': reward_trials,
        'reward_rate': reward_rate,
        'foraging_eff': foraging_eff,
        
        # TODO: add more stats here
    }

    # Generate df_session_stat
    df_session_stat = pd.DataFrame(dict_session_stat, 
                                index=session_index)
    df_session_stat.columns = pd.MultiIndex.from_product([['session_stats'], dict_session_stat.keys()],
                                                        names=['type', 'variable'])

    # -- Merge to df_session --
    df_session = pd.concat([df_session, df_session_stat], axis=1)

    return df_session


def process_one_nwb(nwb_file_name, result_folder):
    logging.info(f'{nwb_file_name} processing...')
    
    try:
        io = NWBHDF5IO(nwb_file_name, mode='r')
        nwb = io.read()
        df_session = nwb_to_df(nwb)
        
        pickle_file_name = result_folder + '/' + nwb_file_name.split('/')[-1].replace('.nwb', '.pkl')
        pd.to_pickle(df_session, pickle_file_name)
        
        logging.info(f'{nwb_file_name} done.')
    except:
        logging.error(f'{nwb_file_name} failed!!')
        
    return

def combine_all_dfs(result_folder):
    df_all = pd.DataFrame()
    for file_name in os.listdir(result_folder):
        if file_name.endswith('.pkl'):
            df = pd.read_pickle(result_folder + '/' + file_name)
            df_all = pd.concat([df_all, df])
            
    pd.to_pickle(df_all, result_folder + '/all_sessions.pkl')
    
    return df_all


if __name__ == '__main__':
    #%%
    data_folder = os.path.join(script_dir, '../data/foraging_nwb_bonsai')
    result_folder = os.path.join(script_dir, '../results')
    
    logging.basicConfig(#filename=f"{result_folder}/logfile.log",
                                level=logging.INFO,
                                format='%(asctime)s %(levelname)s [%(filename)s:%(funcName)s]: %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')


    # By default, process all nwb files under /data/foraging_nwb_bonsai folder
    nwb_file_names = glob.glob(f'{data_folder}/*.nwb')
        
    for nwb_file_name in nwb_file_names:
        process_one_nwb(nwb_file_name, result_folder)
        
    combine_all_dfs(result_folder)
