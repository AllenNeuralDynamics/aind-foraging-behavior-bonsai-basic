"""
Convert old bpod nwb to new bonsai nwb
Han Hou 04/2024
"""
from uuid import uuid4
import numpy as np
import json
import os
from datetime import datetime, timedelta, date
import re
import glob
import logging
import pandas as pd
from dateutil.tz import tzlocal

from pynwb import NWBHDF5IO, NWBFile, TimeSeries, behavior
from pynwb.file import Subject
from scipy.io import loadmat

bpod_nwb_folder = R'/root/capsule/data/s3_foraging_all_nwb/'
save_folder=R'/root/capsule/results'

logger = logging.getLogger(__name__)

TASK_MAPPER = {
    'coupled_block_baiting': 'Coupled Baiting',
    'decoupled_no_baiting': 'Uncoupled Without Baiting',
    'random_walk': 'Random Walk',
}

df_session_bpod = pd.read_pickle(os.path.join("/root/capsule/data/s3_foraging_all_nwb/df_sessions.pkl"))

def get_meta_dict_from_session_pkl(bpod_session_id):
    """ Get metainfo from df_session.pkl for bpod sessions
    I haven't exported all necessary metadata to nwb but luckily I have them in df_session.pkl
    """
    regex = r"(?P<h2o>\w+)_(?P<date>\d{8})_(\d+)_s(?P<session>\d+)"
    match = re.search(regex, bpod_session_id)
    if match:
        h2o = match['h2o']
        date = datetime.strptime(match['date'], '%Y%m%d').date()
        session = int(match['session'])
        df_this = df_session_bpod[
            (df_session_bpod['h2o'] == h2o) 
            & (df_session_bpod['session_date'] == date) 
        ].to_dict(orient='records')
            
    if not match or len(df_this) == 0:
        logger.warning(f"Cannot find meta info from df_sessions.pkl for {bpod_session_id}")
        # return a dict with all nan but with the same columns
        return None
    return df_this[0]


def _get_trial_event_time(nwb, event_name, trial_start_time, trial_stop_time):
    """ Get trialized event time from bpod nwb
    """
    if event_name not in nwb.acquisition['BehavioralEvents'].time_series:
        return np.nan
    
    all_event_times = nwb.acquisition['BehavioralEvents'][event_name].timestamps[:]
    this_trial = np.where((trial_start_time <= all_event_times) 
                          & (all_event_times <= trial_stop_time)
                          )[0]
    if this_trial.size == 0:
        return np.nan
    else:
        return all_event_times[this_trial[0]]
    

def nwb_bpod_to_bonsai(bpod_nwb, meta_dict_from_pkl, save_folder=save_folder):
    
    # Time info
    session_start_time = bpod_nwb.session_start_time.replace(tzinfo=tzlocal())
    session_run_time_in_min = meta_dict_from_pkl['session_length_in_hrs'] * 60
    session_end_time = (session_start_time + timedelta(minutes=session_run_time_in_min))\
        if not np.isnan(session_run_time_in_min) else np.nan

    # New NWB file name
    bonsai_nwb_name = (f"{bpod_nwb.subject.subject_id}_"
                       f"{session_start_time.strftime(r'%Y-%m-%d')}_"
                       f"{session_start_time.strftime(r'%H-%M-%S')}.nwb")

    # --- Create a new NWB file ---
    bonsai_nwb = NWBFile(
        session_description='From old bpod sessions',  
        identifier=str(uuid4()),  # required
        session_start_time=session_start_time,
        session_id=bonsai_nwb_name.replace("nwb", "json"),  # A fake json file
        experimenter=bpod_nwb.experimenter,  # optional
        lab=("" 
            if session_start_time > datetime(2022, 1, 1, tzinfo=tzlocal())
            else "Svoboda lab"),  # optional
        institution=("Allen Institute for Neural Dynamics" 
                     if session_start_time > datetime(2022, 1, 1, tzinfo=tzlocal())
                     else "Janelia Research Campus"),  # optional
        experiment_description=bpod_nwb.experiment_description,  # optional
        related_publications="",  # optional
        notes=bpod_nwb.session_description,  # optional
        protocol=TASK_MAPPER[meta_dict_from_pkl['task']],  # optional
    )

    # --- Subject ---
    bonsai_nwb.subject = Subject(
        subject_id=bpod_nwb.subject.subject_id,
        description=bpod_nwb.subject.description, # Animal's alias (h2o)
        species="Mus musculus",
        weight=meta_dict_from_pkl['session_weight'], # weight before the session
        date_of_birth=bpod_nwb.subject.date_of_birth, 
    )
    
    ### Add some meta data to the scratch (rather than the session description) ###
    metadata = {
        # Meta
        'box': meta_dict_from_pkl['rig'] + '_bpod', # Add _bpod suffix to distinguish from bonsai
        'session_end_time': session_end_time.strftime(r"%Y-%m-%d %H:%M:%S.%s") if isinstance(session_end_time, (date, datetime)) else np.nan,
        'session_run_time_in_min': session_run_time_in_min, 
        'has_video': 'BehavioralTimeSeries' in bpod_nwb.acquisition,
        'has_ephys': hasattr(bpod_nwb, 'units'),
        
        
        # Water (all in mL)
        'water_in_session_foraging': np.nan, # Not directly available in old bpod nwb
        'water_in_session_manual': np.nan, # Not directly available in old bpod nwb
        'water_in_session_total':  meta_dict_from_pkl['water_earned'],
        'water_after_session': meta_dict_from_pkl['water_extra'],
        'water_day_total': meta_dict_from_pkl['water_total'],

        # Weight
        'base_weight': meta_dict_from_pkl['start_weight'],
        'target_weight': np.nan,
        'target_weight_ratio': np.nan,
        'weight_after': meta_dict_from_pkl['session_weight'] + meta_dict_from_pkl['water_earned'],
        
        # Performance
        'foraging_efficiency': np.nan,
        'foraging_efficiency_with_actual_random_seed': meta_dict_from_pkl['foraging_eff'],
        
        # A copy of all fields from df_session.pkl so that we keep as much info from datajoint as possible
        **{f"bpod_backup_{key}": (value 
                                    if not isinstance(value, (datetime, date)) else
                                    value.strftime(r"%Y-%m-%d %H:%M:%S.%s")
                                  )
           for key, value in meta_dict_from_pkl.items()},
    }

    # Turn the metadata into a DataFrame in order to add it to the scratch
    df_metadata = pd.DataFrame(metadata, index=[0])

    # Are there any better places to add arbitrary meta data in nwb?
    # I don't bother creating an nwb "extension"...
    # To retrieve the metadata, use:
    # nwbfile.scratch['metadata'].to_dataframe()
    bonsai_nwb.add_scratch(df_metadata, 
                        name="metadata",
                        description="Some important session-wise meta data")


    # ------- Add trial -------
    # I'm keeping the same fields as the bonsai nwb while filling in bpod info as much as possible
    
    ## behavior events (including trial start/end time; left/right lick time; give left/right reward time) ##
    bonsai_nwb.add_trial_column(name='animal_response', description=f'The response of the animal. 0, left choice; 1, right choice; 2, no response')
    bonsai_nwb.add_trial_column(name='rewarded_historyL', description=f'The reward history of left lick port')
    bonsai_nwb.add_trial_column(name='rewarded_historyR', description=f'The reward history of right lick port')
    bonsai_nwb.add_trial_column(name='delay_start_time', description=f'The delay start time')
    bonsai_nwb.add_trial_column(name='goCue_start_time', description=f'The go cue start time')
    bonsai_nwb.add_trial_column(name='reward_outcome_time', description=f'The reward outcome time (reward/no reward/no response)')
    ## training paramters ##
    # behavior structure
    bonsai_nwb.add_trial_column(name='bait_left', description=f'Whether the current left lickport has a bait or not')
    bonsai_nwb.add_trial_column(name='bait_right', description=f'Whether the current right lickport has a bait or not')
    bonsai_nwb.add_trial_column(name='base_reward_probability_sum', description=f'The summation of left and right reward probability')
    bonsai_nwb.add_trial_column(name='reward_probabilityL', description=f'The reward probability of left lick port')
    bonsai_nwb.add_trial_column(name='reward_probabilityR', description=f'The reward probability of right lick port')
    bonsai_nwb.add_trial_column(name='reward_random_number_left', description=f'The random number used to determine the reward of left lick port')
    bonsai_nwb.add_trial_column(name='reward_random_number_right', description=f'The random number used to determine the reward of right lick port')
    bonsai_nwb.add_trial_column(name='left_valve_open_time', description=f'The left valve open time')
    bonsai_nwb.add_trial_column(name='right_valve_open_time', description=f'The right valve open time')
    # block
    bonsai_nwb.add_trial_column(name='block_beta', description=f'The beta of exponential distribution to generate the block length')
    bonsai_nwb.add_trial_column(name='block_min', description=f'The minimum length allowed for each block')
    bonsai_nwb.add_trial_column(name='block_max', description=f'The maxmum length allowed for each block')
    bonsai_nwb.add_trial_column(name='min_reward_each_block', description=f'The minimum reward allowed for each block')
    # delay duration
    bonsai_nwb.add_trial_column(name='delay_beta', description=f'The beta of exponential distribution to generate the delay duration(s)')
    bonsai_nwb.add_trial_column(name='delay_min', description=f'The minimum duration(s) allowed for each delay')
    bonsai_nwb.add_trial_column(name='delay_max', description=f'The maxmum duration(s) allowed for each delay')
    bonsai_nwb.add_trial_column(name='delay_duration', description=f'The expected time duration between delay start and go cue start')
    # ITI duration
    bonsai_nwb.add_trial_column(name='ITI_beta', description=f'The beta of exponential distribution to generate the ITI duration(s)')
    bonsai_nwb.add_trial_column(name='ITI_min', description=f'The minimum duration(s) allowed for each ITI')
    bonsai_nwb.add_trial_column(name='ITI_max', description=f'The maxmum duration(s) allowed for each ITI')
    bonsai_nwb.add_trial_column(name='ITI_duration', description=f'The expected time duration between trial start and ITI start')
    # response duration
    bonsai_nwb.add_trial_column(name='response_duration', description=f'The maximum time that the animal must make a choce in order to get a reward')
    # reward consumption duration
    bonsai_nwb.add_trial_column(name='reward_consumption_duration', description=f'The duration for the animal to consume the reward')
    # auto water
    bonsai_nwb.add_trial_column(name='auto_waterL', description=f'Autowater given at Left')
    bonsai_nwb.add_trial_column(name='auto_waterR', description=f'Autowater given at Right')
    # optogenetics
    bonsai_nwb.add_trial_column(name='laser_on_trial', description=f'Trials with laser stimulation')
    bonsai_nwb.add_trial_column(name='laser_wavelength', description=f'The wavelength of laser or LED')
    bonsai_nwb.add_trial_column(name='laser_location', description=f'The target brain areas')
    bonsai_nwb.add_trial_column(name='laser_power', description=f'The laser power(mw)')
    bonsai_nwb.add_trial_column(name='laser_duration', description=f'The laser duration')
    bonsai_nwb.add_trial_column(name='laser_condition', description=f'The laser on is conditioned on LaserCondition')
    bonsai_nwb.add_trial_column(name='laser_condition_probability', description=f'The laser on is conditioned on LaserCondition with a probability LaserConditionPro')
    bonsai_nwb.add_trial_column(name='laser_start', description=f'Laser start is aligned to an event')
    bonsai_nwb.add_trial_column(name='laser_start_offset', description=f'Laser start is aligned to an event with an offset')
    bonsai_nwb.add_trial_column(name='laser_end', description=f'Laser end is aligned to an event')
    bonsai_nwb.add_trial_column(name='laser_end_offset', description=f'Laser end is aligned to an event with an offset')
    bonsai_nwb.add_trial_column(name='laser_protocol', description=f'The laser waveform')
    bonsai_nwb.add_trial_column(name='laser_frequency', description=f'The laser waveform frequency')
    bonsai_nwb.add_trial_column(name='laser_rampingdown', description=f'The ramping down time of the laser')
    bonsai_nwb.add_trial_column(name='laser_pulse_duration', description=f'The pulse duration for Pulse protocol')
    
    # auto training parameters
    bonsai_nwb.add_trial_column(name='auto_train_engaged', description=f'Whether the auto training is engaged')
    bonsai_nwb.add_trial_column(name='auto_train_curriculum_name', description=f'The name of the auto training curriculum')
    bonsai_nwb.add_trial_column(name='auto_train_curriculum_version', description=f'The version of the auto training curriculum')
    bonsai_nwb.add_trial_column(name='auto_train_curriculum_schema_version', description=f'The schema version of the auto training curriculum')
    bonsai_nwb.add_trial_column(name='auto_train_stage', description=f'The current stage of auto training')
    bonsai_nwb.add_trial_column(name='auto_train_stage_overridden', description=f'Whether the auto training stage is overridden')
    
    # add lickspout position
    bonsai_nwb.add_trial_column(name='lickspout_position_x', description=f'x position (um) of the lickspout position (left-right)')
    bonsai_nwb.add_trial_column(name='lickspout_position_y', description=f'y position (um) of the lickspout position (forward-backward)')
    bonsai_nwb.add_trial_column(name='lickspout_position_z', description=f'z position (um) of the lickspout position (up-down)')

    # add reward size
    bonsai_nwb.add_trial_column(name='reward_size_left', description=f'Left reward size (uL)')
    bonsai_nwb.add_trial_column(name='reward_size_right', description=f'Right reward size (uL)')
    
    # also add all columns from bpod trial table for backup purpose
    bpod_backup_columns = [f.name for f in bpod_nwb.trials.columns 
                           if f.name not in ['start_time', 'stop_time']]
    for bpod_column in bpod_backup_columns:
        bonsai_nwb.add_trial_column(
            name=f'bpod_backup_{bpod_column}',
            description=bpod_nwb.trials[bpod_column].description)

    ## start adding trials ##
    df_trials = bpod_nwb.trials.to_dataframe()
    dict_trials = df_trials.to_dict(orient='records')
    has_photostim = all(df_trials.photostim_power != 'null')
    
    for d in dict_trials:
        bonsai_nwb.add_trial(
            start_time=d['start_time'], 
            stop_time=d['stop_time'],
            animal_response={'left': 0.0, 'right': 1.0, 'null': 2.0}[d['choice']], # 0: left, 1: right, 2: ignored
            rewarded_historyL=(d['choice']=='left') & (d['outcome']=='hit') & (d['auto_water']==0),
            rewarded_historyR=(d['choice']=='right') & (d['outcome']=='hit') & (d['auto_water']==0),
            reward_outcome_time=_get_trial_event_time(bpod_nwb, 'reward', d['start_time'], d['stop_time']),
            delay_start_time=_get_trial_event_time(bpod_nwb, 'delay', d['start_time'], d['stop_time']),
            goCue_start_time=_get_trial_event_time(bpod_nwb, 'go', d['start_time'], d['stop_time']),
            bait_left=np.nan,
            bait_right=np.nan,
            base_reward_probability_sum=d['left_reward_prob'] + d['right_reward_prob'],
            reward_probabilityL=d['left_reward_prob'],
            reward_probabilityR=d['right_reward_prob'],
            reward_random_number_left=np.nan,
            reward_random_number_right=np.nan,
            left_valve_open_time=np.nan,
            right_valve_open_time=np.nan,
            block_beta=np.nan,
            block_min=np.nan,
            block_max=np.nan,
            min_reward_each_block=np.nan,
            delay_beta=np.nan,
            delay_min=np.nan,
            delay_max=np.nan,
            delay_duration=np.nan,
            ITI_beta=np.nan,
            ITI_min=np.nan,
            ITI_max=np.nan,
            ITI_duration=np.nan,
            response_duration=np.nan,
            reward_consumption_duration=np.nan,
            auto_waterL=d['auto_water'] & (d['choice']=='left'),
            auto_waterR=d['auto_water'] & (d['choice']=='right'),
            
            # photostim
            laser_on_trial=d['photostim_power'] > 0 if has_photostim else np.nan,
            laser_wavelength=473 if has_photostim else np.nan,
            laser_location=meta_dict_from_pkl['photostim_location'] if has_photostim else np.nan,
            laser_power=d['photostim_power'] if has_photostim else np.nan,
            laser_duration=d['photostim_duration'] if has_photostim else np.nan,
            laser_condition=np.nan,
            laser_condition_probability=np.nan,
            laser_start=d['photostim_bpod_timer_align_to'] if has_photostim else np.nan,
            laser_start_offset=d['photostim_bpod_timer_offset'] if has_photostim else np.nan,
            laser_end=d['photostim_bpod_timer_align_to'] if has_photostim else np.nan,
            laser_end_offset=np.nan,
            laser_protocol='Sine' if has_photostim else np.nan,
            laser_frequency=40 if has_photostim else np.nan,
            laser_rampingdown=d['photostim_ramping_down'] if has_photostim else np.nan,
            laser_pulse_duration=np.nan,

            # add all auto training parameters (eventually should be in session.json)
            auto_train_engaged=False,
            auto_train_curriculum_name='none',
            auto_train_curriculum_version='none',
            auto_train_curriculum_schema_version='none',
            auto_train_stage='none',
            auto_train_stage_overridden=np.nan,
            
            # lickspout position
            lickspout_position_x=np.nan,
            lickspout_position_y=np.nan,
            lickspout_position_z=np.nan,
            
            # reward size
            reward_size_left=np.round(meta_dict_from_pkl['water_per_trial_in_uL'], 1),
            reward_size_right=np.round(meta_dict_from_pkl['water_per_trial_in_uL'], 1),
            
            # all bpod backup columns
            **{f'bpod_backup_{bpod_column}': d[bpod_column] 
               for bpod_column in bpod_backup_columns},
        )


    # ----  Other time series ----
    #left/right lick time; give left/right reward time
    df_trials_bonsai = bonsai_nwb.trials.to_dataframe()
    
    # Reward time
    B_LeftRewardDeliveryTime = df_trials_bonsai['reward_outcome_time'][
        df_trials_bonsai['rewarded_historyL']
    ].values
    if not len(B_LeftRewardDeliveryTime):
        B_LeftRewardDeliveryTime = [np.nan]
    LeftRewardDeliveryTime = TimeSeries(
        name="left_reward_delivery_time",
        unit="second",
        timestamps=B_LeftRewardDeliveryTime,
        data=np.ones(len(B_LeftRewardDeliveryTime)).tolist(),
        description='The reward delivery time of the left lick port'
    )
    bonsai_nwb.add_acquisition(LeftRewardDeliveryTime)
    
    B_RightRewardDeliveryTime = df_trials_bonsai['reward_outcome_time'][
        df_trials_bonsai['rewarded_historyR']
    ].values
    if not len(B_RightRewardDeliveryTime):
        B_RightRewardDeliveryTime = [np.nan]
    RightRewardDeliveryTime = TimeSeries(
        name="right_reward_delivery_time",
        unit="second",
        timestamps=B_RightRewardDeliveryTime,
        data=np.ones(len(B_RightRewardDeliveryTime)).tolist(),
        description='The reward delivery time of the right lick port'
    )    
    bonsai_nwb.add_acquisition(RightRewardDeliveryTime)

    # Lick time
    B_LeftLickTime = bpod_nwb.acquisition['BehavioralEvents']['left_lick'].timestamps[:]
    if not len(B_LeftLickTime):
        B_LeftLickTime = [np.nan]
    LeftLickTime = TimeSeries(
        name="left_lick_time",
        unit="second",
        timestamps=B_LeftLickTime,
        data=np.ones(len(B_LeftLickTime)).tolist(),
        description='The time of left licks'
    )
    bonsai_nwb.add_acquisition(LeftLickTime)
    
    B_RightLickTime = bpod_nwb.acquisition['BehavioralEvents']['right_lick'].timestamps[:]
    if not len(B_RightLickTime):
        B_RightLickTime = [np.nan]
    RightLickTime = TimeSeries(
        name="right_lick_time",
        unit="second",
        timestamps=B_RightLickTime,
        data=np.ones(len(B_RightLickTime)).tolist(),
        description='The time of left licks'
    )
    bonsai_nwb.add_acquisition(RightLickTime)

    # Add photometry time stamps
    B_PhotometryFallingTimeHarp = [np.nan]
    PhotometryFallingTimeHarp = TimeSeries(
        name="FIP_falling_time",
        unit="second",
        timestamps=B_PhotometryFallingTimeHarp,
        data=np.ones(len(B_PhotometryFallingTimeHarp)).tolist(),
        description='The time of photometry falling edge (from Harp)'
    )
    bonsai_nwb.add_acquisition(PhotometryFallingTimeHarp)

    B_PhotometryRisingTimeHarp = [np.nan]
    PhotometryRisingTimeHarp = TimeSeries(
        name="FIP_rising_time",
        unit="second",
        timestamps=B_PhotometryRisingTimeHarp,
        data=np.ones(len(B_PhotometryRisingTimeHarp)).tolist(),
        description='The time of photometry rising edge (from Harp)'
    )
    bonsai_nwb.add_acquisition(PhotometryRisingTimeHarp)
    
    # Add optogenetics time stamps
    if 'laserLon' in bpod_nwb.acquisition['BehavioralEvents'].time_series:
        B_OptogeneticsTimeHarp = bpod_nwb.acquisition['BehavioralEvents']['laserLon'].timestamps[:]
    else:
        B_OptogeneticsTimeHarp = [np.nan]
    if not len(B_OptogeneticsTimeHarp):
        B_OptogeneticsTimeHarp = [np.nan]
    OptogeneticsTimeHarp = TimeSeries(
        name="optogenetics_time",
        unit="second",
        timestamps=B_OptogeneticsTimeHarp,
        data=np.ones(len(B_OptogeneticsTimeHarp)).tolist(),
        description='Optogenetics time (from NI)'
    )
    bonsai_nwb.add_acquisition(OptogeneticsTimeHarp)
    
    # Add all bpod_nwb time series for backup
    bpod_backup_behavioral_event = behavior.BehavioralEvents(
        name='bpod_backup_BehavioralEvents'
    )
    for _, time_series in bpod_nwb.acquisition['BehavioralEvents'].time_series.items():
        bpod_backup_behavioral_event.create_timeseries(
            name=time_series.name,
            unit=time_series.unit,
            timestamps=time_series.timestamps,
            data=time_series.data,
            description=time_series.description,
        )
    bonsai_nwb.add_acquisition(bpod_backup_behavioral_event)
    
    # --- Add video and ephys, if exist ---
    if 'BehavioralTimeSeries' in bpod_nwb.acquisition:
        bpod_dlc = behavior.BehavioralTimeSeries(
            name='bpod_backup_BehavioralTimeSeries',
        )
        for acq, dlc in bpod_nwb.acquisition['BehavioralTimeSeries'].time_series.items():
            data_from_bpod = {f: v for f, v in dlc.fields.items() if f not in ['timestamps_unit', 'interval']}
            data_from_bpod['name'] = dlc.name
            
            if acq == 'pupil_size_polygon':
                # fix empty timestamps problem
                data_from_bpod['timestamps'] = bpod_nwb.acquisition['BehavioralTimeSeries']['Camera0_side_pupil_side_Down'].timestamps
                
            bpod_dlc.create_timeseries(**data_from_bpod)
        bonsai_nwb.add_acquisition(bpod_dlc)
            
    if hasattr(bpod_nwb, 'units'):
        pass
    
    # --- Save NWB file in bonsai_nwb format ---
    if len(bonsai_nwb.trials) > 0:
        NWBName = os.path.join(save_folder, bonsai_nwb_name)
        io = NWBHDF5IO(NWBName, mode="w")
        io.write(bonsai_nwb)
        io.close()
        logger.info(f'Successfully converted: {NWBName}')
        return 'success'
    else:
        logger.warning(f"No trials found! Skipping {fname}")
        return 'empty_trials'


def convert_one_bpod_to_bonsai_nwb(bpod_nwb_file, skip_existing=True):
    io = NWBHDF5IO(bpod_nwb_file, mode='r')
    bpod_nwb = io.read()
    
    # Time info
    session_start_time = bpod_nwb.session_start_time.replace(tzinfo=tzlocal())
    
    if skip_existing and len(
        glob.glob(fR"/data/foraging_nwb_bpod/{bpod_nwb.subject.subject_id}"
                  fR"_{session_start_time.strftime(r'%Y-%m-%d')}*.nwb")
    ):
        logger.info(f'Skipped {bpod_nwb_file}.')
        return 'already_exists'
    
    logger.info(f'Processing {bpod_nwb_file}...')
    try:
        meta_dict_from_pkl = get_meta_dict_from_session_pkl(bpod_session_id=bpod_nwb.identifier)
        if meta_dict_from_pkl is None:
            return 'missing_meta'
        return nwb_bpod_to_bonsai(bpod_nwb, meta_dict_from_pkl)
    except Exception as e:
        logger.error(f'{bpod_nwb_file}: {e}')
        return "uncaught_error"
   

if __name__ == '__main__':
    import multiprocessing as mp
    import tqdm

    # Create a file handler with the specified file path
    logger.setLevel(level=logging.INFO)
    file_handler = logging.FileHandler(f"/root/capsule/results/convert_bpod_to_bonsai.log")
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(filename)s:%(funcName)s]: %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Send logging to terminal as well
    logger.addHandler(logging.StreamHandler())
        
    # By default, process all nwb files under /data/foraging_nwb_bonsai folder that do not exist in /data/foraging_nwb_bpod
    bpod_nwb_files = glob.glob(f'{bpod_nwb_folder}/**/*.nwb', recursive=True)
    skip_existing = False # by default, skip existing files
    bpod_nwb_files = bpod_nwb_files[1200:]

    # For debugging
    bpod_nwb_files = ['/root/capsule/data/s3_foraging_all_nwb/HH08/HH08_20210812_49.nwb']
    
    if len(bpod_nwb_files) > 0:
        results = [convert_one_bpod_to_bonsai_nwb(bpod_nwb_file, skip_existing) for bpod_nwb_file in bpod_nwb_files]
    else:
        n_cpus = 16
        results = []
        
        logger.info(f'Starting multiprocessing with {n_cpus} cores...')
        with mp.Pool(processes=n_cpus) as pool:
            jobs = [pool.apply_async(convert_one_bpod_to_bonsai_nwb, args=(nwb_file_name, skip_existing)) 
                    for nwb_file_name in bpod_nwb_files]
            
            for job in tqdm.tqdm(jobs):
                results.append(job.get())
            
    logger.info(f'\nProcessed {len(results)} files: '
            f'{results.count("success")} successfully converted; '
            f'{results.count("already_exists")} already existed, '
            f'{results.count("missing_meta")} missing meta, '
            f'{results.count("empty_trials")} empty_trials, '
            f'{results.count("uncaught_error")} uncaught error\n\n')
    