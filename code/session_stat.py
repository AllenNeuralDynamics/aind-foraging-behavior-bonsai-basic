#%%
import numpy as np
import pandas as pd
import math
import re

from pynwb import NWBFile, TimeSeries, NWBHDF5IO

LEFT, RIGHT = 0, 1

#%%
nwb_folder = '/root/capsule/data/foraging_nwb_bonsai'
nwb_file = '668463_2023-07-19.nwb'

io = NWBHDF5IO(nwb_folder + '/' + nwb_file, mode='r')
nwb = io.read()


#%%
df_trials = nwb.trials.to_dataframe()
df_trials['trial'] = df_trials.index + 1 # Add an one-based trial number column

# Reformat data
choice_history = df_trials.animal_response.map({0: 0, 1: 1, 2: np.nan}).values
reward_history = np.vstack([df_trials.rewarded_historyL, df_trials.rewarded_historyR])
p_reward = np.vstack([df_trials.reward_probabilityL, df_trials.reward_probabilityR])


#%%
# -- Session-based table --
# - Meta data -
subject_id, session_date = nwb.session_id.split('.')[0].split('_')[:]

# Parse meta info
extra_water, rig = re.search(r"Give extra water.*:(\d+(?:\.\d+)?) .*tower:(.*)", nwb.session_description).groups()
weight_after_session = re.search(r"Weight after.*:(\d+(?:\.\d+)?)", nwb.subject.description).groups()[0]

dict_meta = {
    'rig': rig,
    'experimenter': nwb.experimenter[0],
    'experiment_description': nwb.experiment_description,
    'protocol': nwb.protocol,
    'session_start_time': nwb.session_start_time,
    'weight_before_session': float(nwb.subject.weight),
    'weight_after_session': float(weight_after_session),
    'water_during_session': float(weight_after_session) - float(nwb.subject.weight),
    'water_extra': float(extra_water)
    }

df_session = pd.DataFrame(dict_meta, 
                          index=pd.MultiIndex.from_tuples([(subject_id, session_date)], 
                                                          names=['subject_id', 'session_date']),
                            )
df_session.columns = pd.MultiIndex.from_product([['metadata'], dict_meta.keys()],
                                               names=['type', 'variable'])

# - Compute session-level stats -
total_trials = len(df_trials)
finished_trials = np.sum(~np.isnan(choice_history))
reward_trials = np.sum(reward_history)

reward_rate = reward_trials / finished_trials

foraging_eff_func = foraging_eff_baiting if 'bait' in nwb.protocol.lower() else foraging_eff_no_baiting
foraging_eff_use_p, foraging_eff = foraging_eff_func(reward_rate, p_reward[LEFT,:], p_reward[RIGHT, :], df_trials.bait_left, df_trials.bait_right)
    
    
#%%
    
    # Reward schedule stats
if (SessionTaskProtocol & key).fetch1('session_real_foraging'):   # Real foraging
    p_contrast = np.max([p_Ls, p_Rs], axis=0) / np.min([p_Ls, p_Rs], axis=0)
    p_contrast[np.isinf(p_contrast)] = np.nan  # A arbitrary huge number
    p_contrast_mean = np.nanmean(p_contrast)
else:
    p_contrast_mean = 100
    
session_stats.update(session_foraging_eff_optimal = for_eff_optimal,
                        session_foraging_eff_optimal_random_seed = for_eff_optimal_random_seed,
                        session_mean_reward_sum = np.nanmean(p_Ls + p_Rs), 
                        session_mean_reward_contrast = p_contrast_mean)


        finished_trials='session_pure_choices_num', 
    total_trials = 'session_total_trial_num',
    foraging_eff='session_foraging_eff_optimal',
    foraging_eff_randomseed='session_foraging_eff_optimal_random_seed',
    reward_trials='session_hit_num',
    reward_rate='session_hit_num / session_total_trial_num',
    miss_rate='session_miss_num / session_total_trial_num',
    ignore_rate='session_ignore_num / session_total_trial_num',
    early_lick_ratio='session_early_lick_ratio',
    double_dipping_ratio='session_double_dipping_ratio',
    block_num='session_block_num',
    block_length='session_total_trial_num / session_block_num',
    mean_reward_sum='session_mean_reward_sum',
    mean_reward_contrast='session_mean_reward_contrast',
    autowater_num='session_autowater_num',

#%%
# --- Trial-based table ---


# photostim
photostim_trials = df_trials.laser_power > 0
photostim = [df_trials.trial[photostim_trials], df_trials.laser_power[photostim_trials], []]

# Plot session
fig, ax = plot_session_lightweight([np.array([choice_history]), reward_history, p_reward], photostim=photostim)


#%%

df_trial_behavior

df_session_stats

#%%

def foraging_eff_no_baiting(reward_rate, p_Ls, p_Rs, random_number_L=None, random_number_R=None):  # Calculate foraging efficiency (only for 2lp)
        
    # --- Optimal-aver (use optimal expectation as 100% efficiency) ---
    for_eff_optimal = reward_rate / np.nanmean(np.max([p_Ls, p_Rs], axis=0))
    
    if random_number_L is None:
        return for_eff_optimal, np.nan
        
    # --- Optimal-actual (uses the actual random numbers by simulation)
    reward_refills = np.vstack([p_Ls >= random_number_L, p_Rs >= random_number_R])
    optimal_choices = np.argmax([p_Ls, p_Rs], axis=0)  # Greedy choice, assuming the agent knows the groundtruth
    optimal_rewards = reward_refills[0][optimal_choices==0].sum() + reward_refills[1][optimal_choices==1].sum()
    for_eff_optimal_random_seed = reward_rate / (optimal_rewards / len(optimal_choices))
    
    return for_eff_optimal, for_eff_optimal_random_seed

    

def foraging_eff(reward_rate, p_Ls, p_Rs, random_number_L=None, random_number_R=None):  # Calculate foraging efficiency (only for 2lp)
        
    # --- Optimal-aver (use optimal expectation as 100% efficiency) ---
    p_stars = np.zeros_like(p_Ls)
    for i, (p_L, p_R) in enumerate(zip(p_Ls, p_Rs)):   # Sum over all ps 
        p_max = np.max([p_L, p_R])
        p_min = np.min([p_L, p_R])
        if p_min == 0 or p_max >= 1:
            p_stars[i] = p_max
        else:
            m_star = np.floor(np.log(1-p_max)/np.log(1-p_min))
            p_stars[i] = p_max + (1-(1-p_min)**(m_star + 1)-p_max**2)/(m_star+1)

    for_eff_optimal = reward_rate / np.nanmean(p_stars)
    
    if random_number_L is None:
        return for_eff_optimal, np.nan
        
    # --- Optimal-actual (uses the actual random numbers by simulation)
    block_trans = np.where(np.diff(np.hstack([np.inf, p_Ls, np.inf])))[0].tolist()
    reward_refills = [p_Ls >= random_number_L, p_Rs >= random_number_R]
    reward_optimal_random_seed = 0
    
    # Generate optimal choice pattern
    for b_start, b_end in zip(block_trans[:-1], block_trans[1:]):
        p_max = np.max([p_Ls[b_start], p_Rs[b_start]])
        p_min = np.min([p_Ls[b_start], p_Rs[b_start]])
        side_max = np.argmax([p_Ls[b_start], p_Rs[b_start]])
        
        # Get optimal choice pattern and expected optimal rate
        if p_min == 0 or p_max >= 1:
            this_choice = np.array([1] * (b_end-b_start))  # Greedy is obviously optimal
        else:
            m_star = np.floor(np.log(1-p_max)/np.log(1-p_min))
            this_choice = np.array((([1]*int(m_star)+[0]) * (1+int((b_end-b_start)/(m_star+1)))) [:b_end-b_start])
            
        # Do simulation, using optimal choice pattern and actual random numbers
        reward_refill = np.vstack([reward_refills[1 - side_max][b_start:b_end], 
                         reward_refills[side_max][b_start:b_end]]).astype(int)  # Max = 1, Min = 0
        reward_remain = [0,0]
        for t in range(b_end - b_start):
            reward_available = reward_remain | reward_refill[:, t]
            reward_optimal_random_seed += reward_available[this_choice[t]]
            reward_remain = reward_available.copy()
            reward_remain[this_choice[t]] = 0
        
        if reward_optimal_random_seed:                
            for_eff_optimal_random_seed = reward_rate / (reward_optimal_random_seed / len(p_Ls))
        else:
            for_eff_optimal_random_seed = np.nan
    
    return for_eff_optimal, for_eff_optimal_random_seed


#%%

class SessionStats(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_total_trial_num = null : int #number of trials
    session_block_num = null : int #number of blocks, including bias check
    session_hit_num = null : int #number of hits
    session_miss_num = null : int #number of misses
    session_ignore_num = null : int #number of ignores
    session_early_lick_ratio = null: decimal(5,4)  # early lick ratio
    session_autowater_num = null : int #number of trials with autowaters
    session_length = null : decimal(10, 4) #length of the session in seconds
    session_pure_choices_num = null : int # Number of pure choices (excluding auto water)
    
    session_double_dipping_ratio = null: decimal(5,4)  # Double dipping ratio
    session_double_dipping_ratio_hit = null: decimal(5,4)  # Double dipping ratio of hit trials
    session_double_dipping_ratio_miss = null: decimal(5,4)  # Double dipping ratio of miss trials
    
    session_foraging_eff_optimal = null: decimal(5,4)   # Session-wise foraging efficiency (optimal; average sense)
    session_foraging_eff_optimal_random_seed = null: decimal(5,4)   # Session-wise foraging efficiency (optimal; random seed)
    
    session_mean_reward_sum = null: decimal(4,3)  # Median of sum of reward prob
    session_mean_reward_contrast = null: float  # Median of reward prob ratio
    session_effective_block_trans_num = null: int  # Number of effective block transitions  #!!!
    """
    # Foraging sessions only
    key_source = experiment.Session & (experiment.BehaviorTrial & 'task LIKE "foraging%"')

    def make(self, key):
        # import pdb; pdb.set_trace()
        q_all_trial = experiment.SessionTrial & key
        q_block = experiment.SessionBlock & key
        q_hit = experiment.BehaviorTrial & key & 'outcome = "hit"'
        q_miss = experiment.BehaviorTrial & key & 'outcome = "miss"'
        q_auto_water = experiment.TrialNote & key & 'trial_note_type = "autowater"'
        q_actual_finished = q_hit.proj() + q_miss.proj()  - q_auto_water.proj()   # Real finished trial = 'hit' or 'miss' but not 'autowater'
        
        session_stats = {'session_total_trial_num': len(q_all_trial),
                'session_block_num': len(q_block),
                'session_hit_num': len(q_hit),
                'session_miss_num': len(q_miss),
                'session_ignore_num': len(experiment.BehaviorTrial & key & 'outcome = "ignore"'),
                'session_early_lick_ratio': len(experiment.BehaviorTrial & key & 'early_lick="early"') / (len(q_hit) + len(q_miss)) 
                                            if len(q_hit) + len(q_miss)
                                            else np.nan,
                'session_autowater_num': len(q_auto_water),
                'session_pure_choices_num': len(q_actual_finished)}
        
        if session_stats['session_total_trial_num'] > 0:
            session_stats['session_length'] = float(((experiment.SessionTrial() & key).fetch('stop_time')).max())
        else:
            session_stats['session_length'] = 0
            
        # -- Double dipping ratio --
        q_double_dipping = TrialStats & key & 'double_dipping = 1'
        session_stats.update(session_double_dipping_ratio_hit = len(q_double_dipping & q_hit) / len(q_hit)) if len(q_hit) else np.nan     
        
        # Double dipping in missed trial is detected only for sessions later than the first day of using new lickport retraction logic 
        if (experiment.Session & key & 'session_date > "2020-08-11"'):   
            session_stats.update(session_double_dipping_ratio_miss = len(q_double_dipping & q_miss) / len(q_miss)
                                 if q_miss else np.nan,
                                 session_double_dipping_ratio = len(q_double_dipping & q_actual_finished) / len(q_actual_finished)
                                 if q_actual_finished else np.nan)
            
        # -- Session-wise foraging efficiency and schedule stats (2lp only) --
        if len(experiment.BehaviorTrial & key & 'task="foraging"'):
            # Get reward rate (hit but not autowater) / (hit but not autowater + miss but not autowater)
            q_pure_hit_num = q_hit.proj() - q_auto_water.proj()
            reward_rate = len(q_pure_hit_num) / len(q_actual_finished) if q_actual_finished else np.nan
            
            q_actual_finished_reward_prob = (experiment.SessionTrial * experiment.SessionBlock.BlockTrial  # Session-block-trial
                                           * experiment.SessionBlock.WaterPortRewardProbability  # Block-trial-p_reward
                                           & q_actual_finished)  # Select finished trials
                                
            # Get reward probability (only pure finished trials)
            p_Ls = (q_actual_finished_reward_prob & 'water_port="left"').fetch(
                'reward_probability', order_by='trial').astype(float)  # Note 'order_by'!!!
            p_Rs = (q_actual_finished_reward_prob & 'water_port="right"').fetch(
                'reward_probability', order_by='trial').astype(float)
            
            # Recover actual random numbers
            random_number_Ls = np.empty(len(q_all_trial))
            random_number_Ls[:] = np.nan
            random_number_Rs = random_number_Ls.copy()
            
            rand_seed_starts = (experiment.TrialNote()  & key & 'trial_note_type="random_seed_start"').fetch('trial', 'trial_note', order_by='trial')
            
            if len(rand_seed_starts[0]):  # Random seed exists
                for start_idx, start_seed in zip(rand_seed_starts[0], rand_seed_starts[1]):  # For each pybpod session
                    # Must be exactly the same as the pybpod protocol 
                    # https://github.com/hanhou/Foraging-Pybpod/blob/5e19e1d227657ed19e27c6e1221495e9f180c323/pybpod_protocols/Foraging_baptize_by_fire_new_lickport_retraction.py#L478
                    np.random.seed(int(start_seed))
                    random_number_L_this = np.random.uniform(0.,1.,4000).tolist()
                    random_number_R_this = np.random.uniform(0.,1.,4000).tolist()
                    
                    # Fill in random numbers
                    random_number_Ls[start_idx - 1 :] = random_number_L_this[: len(random_number_Ls) - start_idx + 1]
                    random_number_Rs[start_idx - 1 :] = random_number_R_this[: len(random_number_Rs) - start_idx + 1]
                    
                # Select finished trials
                actual_finished_idx = q_actual_finished.fetch('trial', order_by='trial')-1
                random_number_Ls = random_number_Ls[actual_finished_idx]
                random_number_Rs = random_number_Rs[actual_finished_idx]
            else:  # No random seed (backward compatibility)
                print(f'No random seeds for {key}')
                random_number_Ls = None
                random_number_Rs = None
                
            # Compute foraging efficiency
            no_baiting = (SessionTaskProtocol & key).fetch('session_task_protocol') in (110, 120)
            if q_actual_finished:
                if no_baiting:
                    for_eff_optimal, for_eff_optimal_random_seed = foraging_eff_no_baiting(reward_rate, p_Ls, p_Rs, random_number_Ls, random_number_Rs)
                else:
                    for_eff_optimal, for_eff_optimal_random_seed = foraging_eff(reward_rate, p_Ls, p_Rs, random_number_Ls, random_number_Rs)
            else:
                for_eff_optimal, for_eff_optimal_random_seed = np.nan, np.nan

            # Reward schedule stats
            if (SessionTaskProtocol & key).fetch1('session_real_foraging'):   # Real foraging
                p_contrast = np.max([p_Ls, p_Rs], axis=0) / np.min([p_Ls, p_Rs], axis=0)
                p_contrast[np.isinf(p_contrast)] = np.nan  # A arbitrary huge number
                p_contrast_mean = np.nanmean(p_contrast)
            else:
                p_contrast_mean = 100
                
            session_stats.update(session_foraging_eff_optimal = for_eff_optimal,
                                 session_foraging_eff_optimal_random_seed = for_eff_optimal_random_seed,
                                 session_mean_reward_sum = np.nanmean(p_Ls + p_Rs), 
                                 session_mean_reward_contrast = p_contrast_mean)
            
        self.insert1({**key, **session_stats})