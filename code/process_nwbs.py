#%%
import numpy as np
import pandas as pd
import math
import re, os, sys
import glob
import logging
import s3fs
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from pathlib import Path
import json
from matplotlib import pyplot as plt

from analysis.util import foraging_eff_baiting, foraging_eff_no_baiting
from plot.foraging_matplotlib import plot_session_lightweight
from util.aws import upload_result_to_s3

script_dir = os.path.dirname(os.path.abspath(__file__))

LEFT, RIGHT = 0, 1

#%%
def nwb_to_df(nwb):
    df_trials = nwb.trials.to_dataframe()

    # Reformat data
    choice_history = df_trials.animal_response.map({0: 0, 1: 1, 2: np.nan}).values
    reward_history = np.vstack([df_trials.rewarded_historyL, df_trials.rewarded_historyR])
    p_reward = np.vstack([df_trials.reward_probabilityL, df_trials.reward_probabilityR])

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
        'user_name': nwb.experimenter[0],
        'experiment_description': nwb.experiment_description,
        'task': nwb.protocol,
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
    # TODO: Ideally, all these simple stats could be computed in the GUI, and 
    # the GUI sends a copy to the meta session.json file and to the nwb file as well.
    
    total_trials = len(df_trials)
    finished_trials = np.sum(~np.isnan(choice_history))
    reward_trials = np.sum(reward_history)

    reward_rate = reward_trials / finished_trials

    # TODO: add more stats
    # See code here: https://github.com/AllenNeuralDynamics/map-ephys/blob/7a06a5178cc621638d849457abb003151f7234ea/pipeline/foraging_analysis.py#L70C8-L70C8
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


def plot_session_choice_history(nwb):
    
    df_trials = nwb.trials.to_dataframe()
    df_trials['trial'] = df_trials.index + 1 # Add an one-based trial number column

    # Reformat data
    choice_history = df_trials.animal_response.map({0: 0, 1: 1, 2: np.nan}).values
    reward_history = np.vstack([df_trials.rewarded_historyL, df_trials.rewarded_historyR])
    p_reward = np.vstack([df_trials.reward_probabilityL, df_trials.reward_probabilityR])

    # photostim
    photostim_trials = df_trials.laser_power > 0
    photostim = [df_trials.trial[photostim_trials], df_trials.laser_power[photostim_trials], []]

    # Plot session
    fig, ax = plot_session_lightweight([np.array([choice_history]), reward_history, p_reward], photostim=photostim)
    
    return fig


def log_error_file(file_name, result_root):
    error_file_path = Path(f'{result_root}/error_files.json')

    # Check if the file exists
    if error_file_path.exists():
        # If it exists, read the current list of error files
        with open(error_file_path, 'r') as file:
            error_files = json.load(file)
    else:
        # If it doesn't exist, start with an empty list
        error_files = []

    # Append the new file name to the list
    error_files.append(file_name)

    # Write the updated list back to the JSON file
    with open(error_file_path, 'w') as file:
        json.dump(error_files, file, indent=4)
        

def process_one_nwb(nwb_file_name, result_root):
    '''
    Process one nwb file and save the results to result_folder_root/{subject}_{session_date}/
    '''
    logging.info(f'{nwb_file_name} processing...')
    
    try:
        io = NWBHDF5IO(nwb_file_name, mode='r')
        nwb = io.read()
        df_session = nwb_to_df(nwb)
        
        # Create folder if not exist
        subject_id = df_session.index[0][0]
        session_date = df_session.index[0][1]
        nwb_suffix = df_session.index[0][2]
        session_id = f'{subject_id}_{session_date}{f"_{nwb_suffix}" if nwb_suffix else ""}'
        
        result_folder = os.path.join(result_root, session_id)
        os.makedirs(result_folder, exist_ok=True)
        
        # 1. Generate df_session
        pickle_file_name = result_folder + '/' + f'{session_id}_session_stat.pkl'
        pd.to_pickle(df_session, pickle_file_name)
        logging.info(f'{nwb_file_name} 1. df_session done.')
        
        # TODO: generate more dfs like this
        
        # 2. Plot choice history
        fig = plot_session_choice_history(nwb)
        fig.savefig(result_folder + '/' + f'{session_id}_choice_history.png',
                    bbox_inches='tight')
        logging.info(f'{nwb_file_name} 2. plot choice history done.')
        plt.close(fig)        
        
        # TODO: generate more plots like this
        
    except:
        logging.error(f'{nwb_file_name} failed!!')
        log_error_file(nwb_file_name, result_root)
    return


def add_session_number(df):
    # Parse and add session number
    # TODO: figure out how to better deal with more than one nwb files per day per mouse
    # Now I assign session number to the nwb file that has the largest finished trials, if there are more than one nwb files per day per mouse,
    # and set other sessions to nan
    
    # Sort by subject_id, session_date, and finished_trials
    df.sort_values(['subject_id', 'session_date', ('session_stats', 'finished_trials')], inplace=True)

    # Define a function to assign session numbers
    def assign_session_number(group):
        group['session'] = np.nan
        unique_dates = group['session_date'].unique()
        for i, date in enumerate(unique_dates, 1):
            mask = group['session_date'] == date
            max_idx = group.loc[mask, ('session_stats', 'finished_trials')].idxmax()
            group.loc[max_idx, 'session'] = i
        return group

    # Group by subject_id and apply the function
    df = df.groupby('subject_id').apply(assign_session_number).reset_index(drop=True)
    
    return df


def combine_all_df_session(result_folder):
    df_all = pd.DataFrame()

    for root, dirs, files in os.walk(result_folder):  # Look over all subfolders
        for file_name in files:
            if 'session_stat.pkl' in file_name:
                file_path = os.path.join(root, file_name)
                df = pd.read_pickle(file_path)
                df_all = pd.concat([df_all, df])

    df_all = add_session_number(df_all.reset_index()).set_index(['subject_id', 'session_date', 'session', 'nwb_suffix'])
    pd.to_pickle(df_all, os.path.join(result_folder, 'df_sessions.pkl'))

    return df_all
 
    
#%%
if __name__ == '__main__':
    data_folder = os.path.join(script_dir, '../data/foraging_nwb_bonsai')
    result_folder = os.path.join(script_dir, '../results')
    result_folder_s3 = 's3://aind-behavior-data/foraging_nwb_bonsai_processed/'
    
    logging.basicConfig(filename=f"{result_folder}/{os.getenv(['HOSTNAME'][0])}.log",
                                level=logging.INFO,
                                format='%(asctime)s %(levelname)s [%(filename)s:%(funcName)s]: %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')


    # By default, process all nwb files under /data/foraging_nwb_bonsai folder
    nwb_file_names = glob.glob(f'{data_folder}/**/*.nwb', recursive=True)

    if_pipeline_mode = len(sys.argv) > 1 # In pipeline, add any argument to trigger pipeline mode.

    nwb_file_to_process = nwb_file_names[: len(nwb_file_names) if if_pipeline_mode else 1]
    logging.info(f'nwb files to process: {nwb_file_to_process}')

    for nwb_file_name in nwb_file_to_process:
        process_one_nwb(nwb_file_name, result_folder)
    
    # # Combine all dfs
    # combine_all_df_session(result_folder)
    
    # # Sync to s3 (using AWS CLI)
    # upload_result_to_s3(result_folder + '/', result_folder_s3) # '/' is essential here!

# %%
