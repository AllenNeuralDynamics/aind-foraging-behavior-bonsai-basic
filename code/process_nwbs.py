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
import shutil
import multiprocessing as mp
import tqdm

from analysis.analysis_wrapper import compute_logistic_regression
from aind_dynamic_foraging_basic_analysis import plot_foraging_session, compute_foraging_efficiency
from aind_dynamic_foraging_basic_analysis.lick_analysis import plot_lick_analysis

script_dir = os.path.dirname(os.path.abspath(__file__))

LEFT, RIGHT, IGNORE = 0, 1, 2

logger = logging.getLogger(__name__)

def _get_block_starts(p_L, p_R):
    """Find the indices of block starts
    """
    block_start_ind_left = np.where(np.hstack([True, np.diff(p_L) != 0]))[0]
    block_start_ind_right = np.where(np.hstack([True, np.diff(p_R) != 0]))[0]
    block_start_ind_effective = np.sort(np.unique(np.hstack([block_start_ind_left, block_start_ind_right])))
    return block_start_ind_left, block_start_ind_right, block_start_ind_effective
    

def _lick_analysis_in_epoch(all_left_licks, all_right_licks, choice, start_time, stop_time):
    """ Analyze lick-related stats in a given epoch.
    """
    lick_stats = {}
    
    left_licks = all_left_licks[(all_left_licks > start_time) & (all_left_licks < stop_time)]
    right_licks = all_right_licks[(all_right_licks > start_time) & (all_right_licks < stop_time)]
    all_licks = np.hstack([left_licks, right_licks])
    
    lick_stats['first_lick'] = all_licks.min() if len(all_licks) > 0 else np.nan
    lick_stats['n_lick_left'] = len(left_licks)
    lick_stats['n_lick_right'] = len(right_licks)
    lick_stats['n_lick_all'] = len(all_licks)
        
    # For double-dipping
    _lick_identity = np.hstack([np.ones(len(left_licks)) * LEFT, np.ones(len(right_licks)) * RIGHT])
    _lick_identity_sorted = [x for x, _ in sorted(zip(_lick_identity, all_licks), key=lambda pairs: pairs[1])]
    # Numer of switching side (for example, LRL -> 2 switches)
    lick_stats['n_lick_switches'] = np.sum(np.diff(_lick_identity_sorted) != 0)
    # Lick consistency = licks that are consistent with the animal's first lick / all licks (for example, LRL -> 2/3)
    lick_stats['lick_consistency'] = np.sum(_lick_identity_sorted == choice) / len(_lick_identity_sorted) \
        if len(_lick_identity_sorted) > 0 else np.nan

    return lick_stats

def compute_df_trial(nwb):
    """
    return df_trial that contains the original nwb.trials and other trial stats
    """
    
    df_trial = nwb.trials.to_dataframe().copy()
    
    # --- Add some columns to df_trial for convenience ---
    df_trial['reward'] = False  # All reward, including (collected!) autowater
    df_trial.loc[(df_trial.rewarded_historyL | df_trial.rewarded_historyR
                   | df_trial.auto_waterR | df_trial.auto_waterL 
                    & (df_trial.animal_response != IGNORE))  # because there may be "ignored" autowater...
                    > 0, 'reward'] = True
    df_trial['reward_non_autowater'] = False  # Only reward from non-autowater trials (by definition, animal must not have ignored)
    df_trial.loc[(df_trial.rewarded_historyL | df_trial.rewarded_historyR), 'reward_non_autowater'] = True
    
    df_trial['non_autowater_trial'] = False
    df_trial.loc[(df_trial.auto_waterL==0) & (df_trial.auto_waterR==0), 'non_autowater_trial'] = True
    df_trial['non_autowater_finished_trial'] = df_trial['non_autowater_trial'] & (df_trial['animal_response'] != IGNORE)
    df_trial['ignored_non_autowater'] = df_trial['non_autowater_trial'] & (df_trial['animal_response'] == IGNORE)
    df_trial['ignored_autowater'] = ~df_trial['non_autowater_trial'] & (df_trial['animal_response'] == IGNORE)
    
    # --- Lick-related stats ---    
    all_left_licks = nwb.acquisition['left_lick_time'].timestamps[:]
    all_right_licks = nwb.acquisition['right_lick_time'].timestamps[:]
    
    # Define the start and stop time for each epoch
    lick_stats_epochs = {
        # Name: (start_time_name, stop_time_name)
        'gocue_stop': ['goCue_start_time', 'stop_time'],
        'delay_period': ['delay_start_time', 'goCue_start_time'],
        'iti': ['start_time', 'delay_start_time'],        
        }
    
    # Trial-by-trial counts
    for i in range(len(df_trial)):
        for epoch_name, (start_time_name, stop_time_name) in lick_stats_epochs.items():
            start_time, stop_time = df_trial.loc[i, [start_time_name, stop_time_name]]
            lick_stats = _lick_analysis_in_epoch(
                all_left_licks=all_left_licks, 
                all_right_licks=all_right_licks,
                choice=df_trial.animal_response[i],
                start_time=start_time,
                stop_time=stop_time
            )
            
            df_trial.loc[i, f'duration_{epoch_name}'] = stop_time - start_time
            df_trial.loc[i, f'n_lick_left_{epoch_name}'] = lick_stats['n_lick_left']
            df_trial.loc[i, f'n_lick_right_{epoch_name}'] = lick_stats['n_lick_right']
            df_trial.loc[i, f'n_lick_all_{epoch_name}'] = lick_stats['n_lick_all']
            df_trial.loc[i, f'n_lick_switches_{epoch_name}'] = lick_stats['n_lick_switches']
            df_trial.loc[i, f'n_lick_consistency_{epoch_name}'] = lick_stats['lick_consistency']
            
            # Special treatment for gocue to stop
            if epoch_name == 'gocue_stop':
                df_trial.loc[i, 'reaction_time'] = lick_stats['first_lick'] - df_trial.goCue_start_time[i]
                
                # Even in ignore trials, there may be licks outside the response window, 
                # but they are invalid and should be overriden by NaN
                if df_trial.animal_response[i] == IGNORE:
                    df_trial.loc[i, 'reaction_time'] = np.nan
                    df_trial.loc[i, 'n_valid_licks_left'] = 0
                    df_trial.loc[i, 'n_valid_licks_right'] = 0
                    df_trial.loc[i, 'n_valid_licks_all'] = 0
    return df_trial

def compute_df_session_meta(nwb, df_trial):
    # -- Key meta data --
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

    # -- Meta info from nwb.scratch --
    meta_dict = nwb.scratch['metadata'].to_dataframe().iloc[0].to_dict()
    
    # -- Meta data that are only available after the session --
    p_L = df_trial.reward_probabilityL.values
    p_R = df_trial.reward_probabilityR.values
    p_contrast = np.max([p_L, p_R], axis=0) / (np.min([p_L, p_R], axis=0) + 1e-6)
    p_contrast[p_contrast > 100] = 100  # Cap the contrast at 100
    
    # Parse effective block
    block_start_left, block_start_right, block_start_effective = _get_block_starts(p_L, p_R)
    if 'uncoupled' not in nwb.protocol.lower():
        if not (len(block_start_left) == len(block_start_right) 
                and all(block_start_left == block_start_right)):
            logger.warning("Blocks are not fully aligned in a Coupled task!")
    
    # -- Pack data --
    dict_meta = {
        'rig': meta_dict['box'],        
        'user_name': nwb.experimenter[0],
        'experiment_description': nwb.experiment_description,
        'task': nwb.protocol,
        'notes': nwb.notes,
        'session_start_time': session_start_time_from_meta,
        
        **{key: value for key, value in meta_dict.items() 
           if key not in ['box' ,
                          # There are bugs in computing foraging eff online. Let's recalculate in df_session_performance later.
                          'foraging_efficiency', 'foraging_efficiency_with_actual_random_seed']  
           },
        
        'weight_after_ratio': meta_dict['weight_after'] / meta_dict['base_weight'],
        
        # Block structure
        'p_reward_sum_mean': np.mean(p_L + p_R),
        'p_reward_sum_std': np.std(p_L + p_R),
        'p_reward_sum_median': np.median(p_L + p_R),
        
        'p_reward_contrast_mean': np.mean(p_contrast),
        'p_reware_contrast_median': np.median(p_contrast),
        
        'effective_block_length_mean': np.mean(np.diff(block_start_effective)),
        'effective_block_length_std': np.std(np.diff(block_start_effective)),
        'effective_block_length_median': np.median(np.diff(block_start_effective)),
        'effective_block_length_min': np.min(np.diff(block_start_effective)),
        'effective_block_length_max': np.max(np.diff(block_start_effective)),
        
        # Durations
        'duration_gocue_stop_mean': df_trial.loc[:, 'duration_gocue_stop'].mean(),
        'duration_gocue_stop_std': df_trial.loc[:, 'duration_gocue_stop'].std(),
        'duration_gocue_stop_median': df_trial.loc[:, 'duration_gocue_stop'].median(),
        'duration_gocue_stop_min': df_trial.loc[:, 'duration_gocue_stop'].min(),
        'duration_gocue_stop_max': df_trial.loc[:, 'duration_gocue_stop'].max(),

        'duration_delay_period_mean': df_trial.loc[:, 'duration_delay_period'].mean(),
        'duration_delay_period_std': df_trial.loc[:, 'duration_delay_period'].std(),
        'duration_delay_period_median': df_trial.loc[:, 'duration_delay_period'].median(),
        'duration_delay_period_min': df_trial.loc[:, 'duration_delay_period'].min(),
        'duration_delay_period_max': df_trial.loc[:, 'duration_delay_period'].max(),

        'duration_iti_mean': df_trial.loc[:, 'duration_iti'].mean(),
        'duration_iti_std': df_trial.loc[:, 'duration_iti'].std(),
        'duration_iti_median': df_trial.loc[:, 'duration_iti'].median(),
        'duration_iti_min': df_trial.loc[:, 'duration_iti'].min(),
        'duration_iti_max': df_trial.loc[:, 'duration_iti'].max(),
        
        # Reward size
        'reward_volume_left_mean': df_trial.loc[df_trial.reward, 'reward_size_left'].mean(),
        'reward_volume_right_mean': df_trial.loc[df_trial.reward, 'reward_size_right'].mean(),
        
        # Lickspouts movement range (in um)
        **{f'lickspout_movement_range_{axis}': 
            np.ptp(df_trial[f'lickspout_position_{axis}']) for axis in 'xyz'},
        **{f'lickspout_initial_pos_{axis}': 
            df_trial[f'lickspout_position_{axis}'][0] for axis in 'xyz'},
        **{f'lickspout_median_pos_{axis}': 
            np.median(df_trial[f'lickspout_position_{axis}']) for axis in 'xyz'},
    }
    
    if 'bpod' in nwb.session_description:
        # Add a flag indicating this is an old bpod session
        dict_meta['old_bpod_session'] = True

    df_meta = pd.DataFrame(dict_meta, 
                            index=session_index,
                            )
    # Use hierarchical index (type = {'metadata', 'session_stats'}, variable = {...}, etc.)
    df_meta.columns = pd.MultiIndex.from_product([['metadata'], dict_meta.keys()],
                                                names=['type', 'variable'])
    
    # -- Add automatic training --
    if 'auto_train_engaged' in df_trial.columns:       
        df_meta['auto_train', 'curriculum_name'] = np.nan if df_trial.auto_train_curriculum_name.mode()[0].lower() == 'none' else df_trial.auto_train_curriculum_name.mode()[0]
        df_meta['auto_train', 'curriculum_version'] = np.nan if df_trial.auto_train_curriculum_version.mode()[0].lower() == 'none' else df_trial.auto_train_curriculum_version.mode()[0]
        df_meta['auto_train', 'curriculum_schema_version'] = np.nan if df_trial.auto_train_curriculum_schema_version.mode()[0].lower() == 'none' else df_trial.auto_train_curriculum_schema_version.mode()[0]
        df_meta['auto_train', 'current_stage_actual'] = np.nan if df_trial.auto_train_stage.mode()[0].lower() == 'none' else df_trial.auto_train_stage.mode()[0]
        df_meta['auto_train', 'if_overriden_by_trainer'] = np.nan if all(df_trial.auto_train_stage_overridden.isna()) else df_trial.auto_train_stage_overridden.mode()[0]
        
        # Add a flag to indicate whether any of the auto train settings were changed during the training
        df_meta['auto_train', 'if_consistent_within_session'] = len(df_trial.groupby(
            [col for col in df_trial.columns if 'auto_train' in col]
        )) == 1
    else:
        for field in ['curriculum_name', 
                      'curriculum_version', 
                      'curriculum_schema_version', 
                      'current_stage_actual', 
                      'if_overriden_by_trainer']:
            df_meta['auto_train', field] = None
    
    return df_meta

def compute_df_session_performance(nwb, df_trial):
    # TODO: Ideally, all these simple stats could be computed in the GUI, and 
    # the GUI sends a copy to the meta session.json file and to the nwb file as well.
    
    n_total_trials = len(df_trial)
    n_finished_trials = (df_trial.animal_response != IGNORE).sum()
    
    # Actual foraging trials (autowater excluded)
    n_total_trials_non_autowater = df_trial.non_autowater_trial.sum()
    n_finished_trials_non_autowater = df_trial.non_autowater_finished_trial.sum()
        
    n_reward_trials_non_autowater = df_trial.reward_non_autowater.sum()
    reward_rate_non_autowater_finished = n_reward_trials_non_autowater / n_finished_trials_non_autowater if n_finished_trials_non_autowater > 0 else np.nan

    # Foraging efficiency (autowater and ignored trials must be excluded)
    foraging_eff, foraging_eff_random_seed = compute_foraging_efficiency(
        baited='without bait' not in nwb.protocol.lower(),
        choice_history=df_trial.animal_response.map({0: 0, 1: 1, 2: np.nan}).values,
        reward_history=df_trial.rewarded_historyL | df_trial.rewarded_historyR,
        p_reward=[
            df_trial.reward_probabilityL.values,
            df_trial.reward_probabilityR.values,
        ],
        random_number=[
            df_trial.reward_random_number_left.values,
            df_trial.reward_random_number_right.values,
        ],
        autowater_offered=(df_trial.auto_waterL == 1) | (df_trial.auto_waterR == 1),
    )
    
    # Override foraging_eff_random_seed if the nwb is converted from old bpod session
    # because in DataJoint, I already computed the foraging_eff_random_seed
    if 'bpod' in nwb.session_description:
        foraging_eff_random_seed = nwb.get_scratch('metadata')['foraging_efficiency_with_actual_random_seed'].values[0]

    all_lick_number = len(nwb.acquisition['left_lick_time'].timestamps) + len(nwb.acquisition['right_lick_time'].timestamps)
    
    # --- Naive bias (Bari et al) (autowater excluded) ---
    n_left = ((df_trial.animal_response == LEFT) & (df_trial.non_autowater_trial)).sum()
    n_right = ((df_trial.animal_response == RIGHT) & (df_trial.non_autowater_trial)).sum()
    bias_naive = 2 * (n_right / (n_left + n_right) - 0.5) if n_left + n_right > 0 else np.nan
    finished_rate = n_finished_trials_non_autowater / n_total_trials_non_autowater if n_total_trials_non_autowater > 0 else np.nan

    # -- Add session stats here --
    dict_performance = {
        # 1. Basic performance
        # By default, autowater are excluded in sessions stats that are related to foraging efficiency
        # Only those with "_with_autowater" suffix include autowater trials
        'total_trials_with_autowater': n_total_trials,
        'finished_trials_with_autowater': n_finished_trials,
        'finished_rate_with_autowater': n_finished_trials / n_total_trials,
        'ignore_rate_with_autowater': 1 - n_finished_trials / n_total_trials,
        'autowater_collected': (~df_trial.non_autowater_trial & (df_trial.animal_response != IGNORE)).sum(),
        'autowater_ignored': (~df_trial.non_autowater_trial & (df_trial.animal_response == IGNORE)).sum(),
        
        'total_trials': n_total_trials_non_autowater,
        'finished_trials': n_finished_trials_non_autowater,
        'ignored_trials': n_total_trials_non_autowater - n_finished_trials_non_autowater,
        'finished_rate': finished_rate,
        'ignore_rate': 1 - finished_rate,
        
        'reward_trials': n_reward_trials_non_autowater,
        'reward_rate': reward_rate_non_autowater_finished,
        'foraging_eff': foraging_eff,
        'foraging_eff_random_seed': foraging_eff_random_seed,
        
        'foraging_performance': foraging_eff * finished_rate,
        'foraging_performance_random_seed': foraging_eff_random_seed * finished_rate,
        
        'bias_naive': bias_naive,
        
        # 2. Lick timing (including autowater trials because they are orthogonal to "foraging")
        'reaction_time_median': df_trial.loc[:, 'reaction_time'].median(),
        'reaction_time_mean': df_trial.loc[:, 'reaction_time'].mean(),
        
        'early_lick_rate':  # the proportion of trials with licks during the delay period
            (df_trial.loc[:, 'n_lick_all_delay_period'] > 0).sum() / n_total_trials,
        
        'invalid_lick_ratio':   # in all trials, licks outside gocue-stop window / all licks
            (all_lick_number - df_trial.loc[:, 'n_lick_all_gocue_stop'].sum()) / all_lick_number,
            
        # 3. Lick consistency (during the response period, in finished trials only, including autowater)
        'double_dipping_rate_finished_trials':  # In finished trials, the proportion of trials with licks on both sides 
            (df_trial.loc[(df_trial.animal_response != IGNORE), 'n_lick_switches_gocue_stop'] > 0).sum() 
            / (df_trial.animal_response != IGNORE).sum(),
        'double_dipping_rate_finished_reward_trials':  # finished and reward trials (by definition, not ignored)
            (df_trial.loc[df_trial.reward, 'n_lick_switches_gocue_stop'] > 0).sum()  
            / df_trial.reward.sum(),
        'double_dipping_rate_finished_noreward_trials':   # finished but non-reward trials
            (df_trial.loc[(df_trial.animal_response != IGNORE) & (~df_trial.reward), 'n_lick_switches_gocue_stop'] > 0).sum() 
            / ((df_trial.animal_response != IGNORE) & (~df_trial.reward)).sum(),
        'lick_consistency_mean_finished_trials': 
            df_trial.loc[(df_trial.animal_response != IGNORE), 'n_lick_consistency_gocue_stop'].mean(),
        'lick_consistency_mean_finished_reward_trials': 
            df_trial.loc[df_trial.reward, 'n_lick_consistency_gocue_stop'].mean(),
        'lick_consistency_mean_finished_noreward_trials': 
            df_trial.loc[(df_trial.animal_response != IGNORE) & (~df_trial.reward), 'n_lick_consistency_gocue_stop'].mean(),
    }
        
    # Generate df_performance
    df_performance = pd.DataFrame(dict_performance, index=[0])
    df_performance.columns = pd.MultiIndex.from_product([['session_stats'], dict_performance.keys()],
                                                  names=['type', 'variable'])
    return df_performance

#%%
def nwb_to_dfs(nwb):
    # -- 1. Trial-based table --
    df_trial = compute_df_trial(nwb)

    # -- 2. Session-based tables --
    # 2.1. Metadata
    df_session_meta = compute_df_session_meta(nwb, df_trial)  # Need df_trial to add additional meta info such as duration_iti_median    

    # 2.2. Performance
    df_session_performance = compute_df_session_performance(nwb, df_trial)
    df_session_performance.index = df_session_meta.index  # Make sure the session_key is the same as df_session_meta
    # Merge to df_session
    df_session = pd.concat([df_session_meta, df_session_performance], axis=1)

    # === Set trial_key to df_trial (trial_key = session_key + trial_id) ===
    trials = df_trial.index.values + 1  # Trial number starts from 1
    multi_index = pd.MultiIndex.from_tuples(
        [(*df_session_meta.index[0], trial) for trial in trials],
        names=[*df_session_meta.index.names, 'trial']
    )
    df_trial.index = multi_index
    
    return df_session, df_trial


def plot_session_choice_history(nwb):
    
    df_trial = nwb.trials.to_dataframe()
    df_trial['trial'] = df_trial.index + 1 # Add an one-based trial number column

    # Reformat data
    choice_history = df_trial.animal_response.map({0: 0, 1: 1, 2: np.nan}).values
    reward_history = df_trial.rewarded_historyL | df_trial.rewarded_historyR
    p_reward = np.vstack([df_trial.reward_probabilityL, df_trial.reward_probabilityR])
    autowater_offered_history = (df_trial.auto_waterL == 1) | (df_trial.auto_waterR == 1)

    # photostim
    if 'laser_power' in df_trial:  # Backward compatibility #(a quick fix -- only extract trials which have photostim on ANY side)
        laser_power_cols = ['laser_power']
    else:
        laser_power_cols = ['laser_1_power', 'laser_2_power']
    
    df_trial['laser_avg_power'] = df_trial[laser_power_cols].mean(axis=1)
    photostim = {
        "trial": df_trial.trial[df_trial['laser_avg_power'] > 0], 
        "power": df_trial.laser_avg_power[df_trial['laser_avg_power'] > 0], 
        "stim_epoch": None,
    }

    # Plot session    
    fig, axes = plot_foraging_session(
        choice_history=choice_history,
        reward_history=reward_history,
        p_reward=p_reward,
        autowater_offered=autowater_offered_history,
        photostim=photostim,        
    )
    
    return fig, axes


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
    
def load_nwb_from_data_asset(nwb_file_name):
    io = NWBHDF5IO(nwb_file_name, mode='r')
    nwb = io.read()
    return nwb


def process_one_nwb(nwb_file_name, result_root):
    '''
    Process one nwb file and save the results to result_folder_root/{subject}_{session_date}/
    '''
    logger.info(f'{nwb_file_name} processing...')
    
    try:
        nwb = load_nwb_from_data_asset(nwb_file_name)
        df_session, df_trial = nwb_to_dfs(nwb)
        
        # Create folder if not exist
        subject_id = df_session.index.get_level_values('subject_id')[0]
        session_date = df_session.index.get_level_values('session_date')[0]
        nwb_suffix = df_session.index.get_level_values('nwb_suffix')[0]
        session_id = f'{subject_id}_{session_date}{f"_{nwb_suffix}" if nwb_suffix else ""}'
        
        result_folder = os.path.join(result_root, session_id)
        os.makedirs(result_folder, exist_ok=True)
        
        # --- 1. Plot choice history ---
        fig, axes = plot_session_choice_history(nwb)
        # add some session info
        axes[0].text(0, 1.05, 
                f'{session_id}, {df_session.metadata.rig.iloc[0]}, {df_session.metadata.user_name.iloc[0]}\n'
                f'Total trials {df_session.session_stats.total_trials_with_autowater.iloc[0]} = '
                f'FORAGING finished {df_session.session_stats.finished_trials.iloc[0]} '
                f'ignored {df_session.session_stats.ignored_trials.iloc[0]} + '
                f'AUTOWATER collected {df_session.session_stats.autowater_collected.iloc[0]} '
                f'ignored {df_session.session_stats.autowater_ignored.iloc[0]}\n'
                f'FORAGING finished rate {df_session.session_stats.finished_rate.iloc[0]:.2%}, '
                f'efficiency {df_session.session_stats.foraging_eff.iloc[0]:.3f} '
                f'(r.s. {df_session.session_stats.foraging_eff_random_seed[0]:.3f}), '
                f'bias naive {df_session.session_stats.bias_naive.iloc[0]:.3f}\n',
                fontsize=8, 
                transform=axes[0].transAxes
        )
        fig.savefig(result_folder + '/' + f'{session_id}_choice_history.png',
                    bbox_inches='tight')
        logger.info(f'{nwb_file_name} 1. plot choice history done.')
        plt.close(fig)
        
        # --- 2. Logistic regression ---
        df_session_logistic_regression, dict_fig = compute_logistic_regression(nwb)
        df_session_logistic_regression.index = df_session.index  # Make sure the session_key is the same as df_session
        df_session_logistic_regression.columns = pd.MultiIndex.from_product(
            [['logistic_regression'], 
             df_session_logistic_regression.columns])
        pd.to_pickle(df_session_logistic_regression, 
                     result_folder + '/' + f'{session_id}_df_session_logistic_regression.pkl')
        # Merge df_session_logistic_regression to df_session 
        # (TODO: this should happen elsewhere, see https://github.com/orgs/AllenNeuralDynamics/projects/70/views/4?pane=issue&itemId=57992307)
        df_session = pd.concat([df_session, df_session_logistic_regression], axis=1)
        
        for key, fig in dict_fig.items():
            fig.savefig(result_folder + '/' + f'{session_id}_logistic_regression_{key}.png',
                        bbox_inches='tight')
            plt.close(fig)
        logger.info(f'{nwb_file_name} 2. Logistic regression done.')
    
        # --- 3. Lick analysis ---
        try:
            fig, _ = plot_lick_analysis(nwb)
            fig.savefig(result_folder + '/' + f'{session_id}_lick_analysis.png',
                            bbox_inches='tight')
            plt.close(fig)
            logger.info(f'{nwb_file_name} 3. Lick analysis done.')
        except Exception as e:
            logger.error(f'{nwb_file_name} Lick analysis failed!')
        
        # TODO: add more analyses like this
        
        # --- Final. Save df_session and df_trial ---
        pd.to_pickle(df_session, result_folder + '/' + f'{session_id}_df_session.pkl')
        pd.to_pickle(df_trial, result_folder + '/' + f'{session_id}_df_trial.pkl')
        logger.info(f'{nwb_file_name} 4. df_session and df_trial done.')
        
    except Exception as e:
        logger.error(f'{nwb_file_name} failed!!', exc_info=True)
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
    
    # ----------------------------
    LOCAL_MANUAL_OVERRIDE = False   # If true, local batch process all nwb files in /data/foraging_nwb_bonsai with n_CPUs = 16
    # ----------------------------
    
    data_folder = os.path.join(script_dir, '../data/foraging_nwb_bonsai')
    result_folder = os.path.join(script_dir, '../results')

    # Create a file handler with the specified file path
    logger.setLevel(level=logging.INFO)
    file_handler = logging.FileHandler(f"{result_folder}/capsule.log")
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(filename)s:%(funcName)s]: %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Send logging to terminal as well
    logger.addHandler(logging.StreamHandler())

    # By default, process all nwb files under /data/foraging_nwb_bonsai folder
    # In the CO pipeline, upstream capsule will assign jobs by putting nwb files to this folder
    nwb_file_names = glob.glob(f'{data_folder}/**/*.nwb', recursive=True)

    if_debug_mode = len(sys.argv) == 1 # In pipeline, add any argument to trigger pipeline mode.

    if if_debug_mode and not LOCAL_MANUAL_OVERRIDE:
        to_debug = [
            '697929_2024-02-22_08-38-30.nwb', # coupled baiting first session example
            '713557_2024-03-01_08-50-40.nwb', # coupled well-trained example
            '703548_2024-03-01_08-51-32.nwb',   # uncoupled baiting well-trained example
            '714314_2024-04-10_14-38-52', # new nwb format starting from https://github.com/AllenNeuralDynamics/dynamic-foraging-task/pull/369
            '727456_2024-06-12_11-10-53',  # uncoupled no baiting
        ]  
        nwb_file_names = [f for f in nwb_file_names if any(dd in f for dd in to_debug)]
            
    logger.info(f'nwb files to process: {nwb_file_names}')

    # Note that in Code Ocean, mp.cpu_count() is not necessarily the number of cores available in this session.
    # For example, even in the environment setting with 4 cores, mp.cpu_count() returns 16.
    # To really speed up when manually re-do all sessions, make sure to set core = 16 or more in the environment setting.``
    try:
        n_cpus = int(sys.argv[1])  # Input from pipeline
    except:
        n_cpus = mp.cpu_count() if LOCAL_MANUAL_OVERRIDE else 1
    
    if n_cpus > 1:
        logger.info(f'Starting multiprocessing with {n_cpus} cores...')
        with mp.Pool(processes=n_cpus) as pool:
            jobs = [pool.apply_async(process_one_nwb, args=(nwb_file_name, result_folder)) for nwb_file_name in nwb_file_names]
            for job in tqdm.tqdm(jobs):
                job.get()
    else:
        logger.info('Starting single processing...')
        for nwb_file_name in tqdm.tqdm(nwb_file_names):
            process_one_nwb(nwb_file_name, result_folder)


# %%
