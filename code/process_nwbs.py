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

script_dir = os.path.dirname(os.path.abspath(__file__))

LEFT, RIGHT = 0, 1

#%%
def nwb_to_df_session(nwb):
    df_trials = nwb.trials.to_dataframe()

    # Reformat data
    choice_history = df_trials.animal_response.map({0: 0, 1: 1, 2: np.nan}).values
    reward_history = np.vstack([df_trials.rewarded_historyL, df_trials.rewarded_historyR])
    p_reward = np.vstack([df_trials.reward_probabilityL, df_trials.reward_probabilityR])
    reward_random_number = np.vstack([df_trials.reward_random_number_left, df_trials.reward_random_number_right])

    # -- Session-based table --
    # - Meta data -
    session_start_time_from_meta = nwb.session_start_time
    session_date_from_meta = session_start_time_from_meta.strftime("%Y-%m-%d")
    subject_id_from_meta = nwb.subject.subject_id
    
    # old file name foramt before commit https://github.com/AllenNeuralDynamics/dynamic-foraging-task/commit/62d0e9e2bb9b47a8efe8ecb91da9653381a5f551
    old_re = re.match(r"(?P<subject_id>\d+)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<n>\d+))?\.json", 
                nwb.session_id)
    
    if old_re is not None:
        # If there are more than one "bonsai sessions" (the trainer clicked "Save" button in the GUI more than once) in a certain day,
        # parse nwb_suffix from the file name (0, 1, 2, ...)
        subject_id, session_date, nwb_suffix = old_re.groups()
        nwb_suffix = int(nwb_suffix) if nwb_suffix is not None else 0
    else:
        # After https://github.com/AllenNeuralDynamics/dynamic-foraging-task/commit/62d0e9e2bb9b47a8efe8ecb91da9653381a5f551, 
        # the suffix becomes the session start time. Therefore, I use HHMMSS as the nwb suffix, which still keeps the order as before.

        # Typical situation for multiple bonsai sessions per day is that the RAs pressed more than once 
        # "Save" button but only started the session once. 
        # Therefore, I should generate nwb_suffix from the bonsai file name instead of session_start_time.
        subject_id, session_date, session_json_time = re.match(r"(?P<subject_id>\d+)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<time>.*))\.json", 
                            nwb.session_id).groups()
        nwb_suffix = int(session_json_time.replace('-', ''))
        
    # Ad-hoc bug fixes for some mistyped mouse ID
    if subject_id in ("689727"):
        subject_id_from_meta = subject_id
        
    assert subject_id == subject_id_from_meta, f"Subject name from the metadata ({subject_id_from_meta}) does not match "\
                                               f"that from json name ({subject_id})!!"
    assert session_date == session_date_from_meta, f"Session date from the metadata ({session_date_from_meta}) does not match "\
                                                   f"that from json name ({session_date})!!"
    
    session_index = pd.MultiIndex.from_tuples([(subject_id, session_date, nwb_suffix)], 
                                            names=['subject_id', 'session_date', 'nwb_suffix'])

    # Parse meta info   
    meta_dict = nwb.scratch['metadata'].to_dataframe().iloc[0].to_dict()

    dict_meta = {
        'rig': meta_dict['box'],
        'user_name': nwb.experimenter[0],
        'experiment_description': nwb.experiment_description,
        'task': nwb.protocol,
        'notes': nwb.notes,
        'session_start_time': session_start_time_from_meta,
        
        **{key: value for key, value in meta_dict.items() 
           if key not in ['box' ,
                          # There are bugs in computing foraging eff online. Let's recalculate later.
                          'foraging_efficiency', 'foraging_efficiency_with_actual_random_seed']  
           },
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
    
    n_total_trials = len(df_trials)
    n_finished_trials = np.sum(~np.isnan(choice_history))
    
    # Actual foraging trials (autowater excluded)
    _non_autowater_trials = (df_trials.auto_waterL==0) & (df_trials.auto_waterR==0)
    _non_autowater_finished_trials = _non_autowater_trials & ~np.isnan(choice_history)
    n_total_trials_non_autowater = np.sum(_non_autowater_trials)
    n_finished_trials_non_autowater = np.sum(_non_autowater_finished_trials)
    
    # Reward trials only include non-autowater trials
    n_reward_trials_non_autowater = np.sum(reward_history)  # Note that reward history only include non-autowater trials
    reward_rate_non_autowater = n_reward_trials_non_autowater / n_finished_trials_non_autowater

    # Foraging efficiency (autowater and ignored trials must be excluded)
    foraging_eff_func = foraging_eff_baiting if 'bait' in nwb.protocol.lower() else foraging_eff_no_baiting
    foraging_eff, foraging_eff_random_seed = foraging_eff_func(reward_rate_non_autowater, 
                                                               p_reward[LEFT, _non_autowater_finished_trials], 
                                                               p_reward[RIGHT, _non_autowater_finished_trials], 
                                                               reward_random_number[LEFT, _non_autowater_finished_trials], 
                                                               reward_random_number[RIGHT, _non_autowater_finished_trials]
                                                               )

    # TODO: add more stats
    # See code here: https://github.com/AllenNeuralDynamics/map-ephys/blob/7a06a5178cc621638d849457abb003151f7234ea/pipeline/foraging_analysis.py#L70C8-L70C8
    # early_lick_ratio = 
    # double_dipping_ratio = 
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
    
    # Naive bias (Bari et al) (autowater excluded)
    n_left = np.sum(choice_history[_non_autowater_trials] == LEFT)
    n_right = np.sum(choice_history[_non_autowater_trials] == RIGHT)
    bias_naive = 2 * (n_right / (n_left + n_right) - 0.5)

    # -- Add session stats here --
    dict_session_stat = {
        'total_trials': n_total_trials,
        'finished_trials': n_finished_trials,
        'finished_rate': n_finished_trials / n_total_trials,
        'finished_trials_non_autowater': n_finished_trials_non_autowater,
        'finished_rate_non_autowater': n_finished_trials_non_autowater / n_total_trials_non_autowater,
        
        'ignore_rate': np.sum(np.isnan(choice_history)) / n_total_trials,
        'ignore_rate_non_autowater': np.sum(np.isnan(choice_history[_non_autowater_trials])) / n_total_trials_non_autowater,
        
        'reward_trials_non_autowater': n_reward_trials_non_autowater,
        'reward_rate_non_autowater': reward_rate_non_autowater,
        
        # Autowater is excluded by default in foraging efficiency calculation
        'bias_naive': bias_naive,
        'foraging_eff': foraging_eff,
        'foraging_eff_random_seed': foraging_eff_random_seed,
        
        # TODO: add more stats here
    }
        
    # Generate df_session_stat
    df_session_stat = pd.DataFrame(dict_session_stat, 
                                   index=session_index)
    df_session_stat.columns = pd.MultiIndex.from_product([['session_stats'], dict_session_stat.keys()],
                                                        names=['type', 'variable'])

    # -- Add automatic training --
    if 'auto_train_engaged' in df_trials.columns:       
        df_session['auto_train', 'curriculum_name'] = np.nan if df_trials.auto_train_curriculum_name.mode()[0] == 'none' else df_trials.auto_train_curriculum_name.mode()[0]
        df_session['auto_train', 'curriculum_version'] = np.nan if df_trials.auto_train_curriculum_version.mode()[0] == 'none' else df_trials.auto_train_curriculum_version.mode()[0]
        df_session['auto_train', 'curriculum_schema_version'] = np.nan if df_trials.auto_train_curriculum_schema_version.mode()[0] == 'none' else df_trials.auto_train_curriculum_schema_version.mode()[0]
        df_session['auto_train', 'current_stage_actual'] = np.nan if df_trials.auto_train_stage.mode()[0] == 'none' else df_trials.auto_train_stage.mode()[0]
        df_session['auto_train', 'if_overriden_by_trainer'] = np.nan if all(df_trials.auto_train_stage_overridden.isna()) else df_trials.auto_train_stage_overridden.mode()[0]
        
        # Add a flag to indicate whether any of the auto train settings were changed during the training
        df_session['auto_train', 'if_consistent_within_session'] = len(df_trials.groupby(
            [col for col in df_trials.columns if 'auto_train' in col]
        )) == 1
    else:
        for field in ['curriculum_name', 
                      'curriculum_version', 
                      'curriculum_schema_version', 
                      'current_stage_actual', 
                      'if_overriden_by_trainer']:
            df_session['auto_train', field] = None
                
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
        df_session = nwb_to_df_session(nwb)
        
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
        
    except Exception as e:
        logging.error(f'{nwb_file_name} failed!!', exc_info=True)
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
 
    
#%%
if __name__ == '__main__':
    data_folder = os.path.join(script_dir, '../data/foraging_nwb_bonsai')
    result_folder = os.path.join(script_dir, '../results')
    result_folder_s3 = 's3://aind-behavior-data/foraging_nwb_bonsai_processed/'
    
    logging.basicConfig(filename=f"{result_folder}/capsule.log",
                                level=logging.INFO,
                                format='%(asctime)s %(levelname)s [%(filename)s:%(funcName)s]: %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')


    # By default, process all nwb files under /data/foraging_nwb_bonsai folder
    nwb_file_names = glob.glob(f'{data_folder}/**/*.nwb', recursive=True)

    if_debug_mode = len(sys.argv) == 1 # In pipeline, add any argument to trigger pipeline mode.

    if if_debug_mode:
        to_debug = '703548_2024-02-05_08-11-00.nwb'  # During debugging, only process this file
        nwb_file_names = [f for f in nwb_file_names if to_debug in f]
    
    logging.info(f'nwb files to process: {nwb_file_names}')

    for nwb_file_name in nwb_file_names:
        process_one_nwb(nwb_file_name, result_folder)
    

# %%
