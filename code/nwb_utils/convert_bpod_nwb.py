"""
Convert old bpod nwb to new bonsai nwb
Han Hou 04/2024
"""
from uuid import uuid4
import numpy as np
import json
import os
from datetime import datetime, timedelta
import re
import logging
import pandas as pd
from dateutil.tz import tzlocal

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.file import Subject
from scipy.io import loadmat

save_folder=R'/root/capsule/results'

logger = logging.getLogger(__name__)

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
        meta_dict = df_session_bpod[
            (df_session_bpod['h2o'] == h2o) 
            & (df_session_bpod['session_date'] == date) 
            & (df_session_bpod['session'] == session)    
        ].to_dict(orient='records')[0]
    
    if not match or len(meta_dict) == 0:
        logger.warning(f"Cannot find meta info from df_sessions.pkl for {bpod_session_id}")
        # return a dict with all nan but with the same columns
        return {k: np.nan for k in df_session_bpod.columns}
    return meta_dict
    
    
TASK_MAPPER = {
    'coupled_block_baiting': 'Coupled Baiting',
    'decoupled_no_baiting': 'Uncoupled Without Baiting',
    'random_walk': 'Random Walk',
}


def nwb_bpod_to_bonsai(bpod_nwb, meta_dict_from_pkl, save_folder=save_folder):
    
    # Time info
    session_start_time = bpod_nwb.session_start_time.replace(tzinfo=tzlocal())
    session_run_time_in_min = meta_dict_from_pkl['session_length_in_hrs'] * 60
    session_end_time = session_start_time + timedelta(minutes=session_run_time_in_min)

    # --- Create a new NWB file ---
    bonsai_nwb = NWBFile(
        session_description='Session end time:' + ' unknown',  
        identifier=str(uuid4()),  # required
        session_start_time=session_start_time,
        session_id=bpod_nwb.identifier,  # optional
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
        'session_end_time': session_end_time,
        'session_run_time_in_min': session_run_time_in_min, 
        
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
        **{f"from_bpod_pkl_{key}": value for key, value in meta_dict_from_pkl.items()},
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

    ## start adding trials ##

    for i in range(len(obj.B_TrialEndTime)):
        bonsai_nwb.add_trial(
            start_time=getattr(obj, f'B_TrialStartTime{Harp}')[i], 
            stop_time=getattr(obj, f'B_TrialEndTime{Harp}')[i],
            animal_response=obj.B_AnimalResponseHistory[i],
            rewarded_historyL=obj.B_RewardedHistory[0][i],
            rewarded_historyR=obj.B_RewardedHistory[1][i],
            reward_outcome_time=obj.B_RewardOutcomeTime[i],
            delay_start_time=getattr(obj, f'B_DelayStartTime{Harp}')[i],
            goCue_start_time=goCue_start_time_t,
            bait_left=obj.B_BaitHistory[0][i],
            bait_right=obj.B_BaitHistory[1][i],
            base_reward_probability_sum=float(obj.TP_BaseRewardSum[i]),
            reward_probabilityL=float(obj.B_RewardProHistory[0][i]),
            reward_probabilityR=float(obj.B_RewardProHistory[1][i]),
            reward_random_number_left=_get_field(obj, 'B_CurrentRewardProbRandomNumber', index=i, default=[np.nan] * 2)[0],
            reward_random_number_right=_get_field(obj, 'B_CurrentRewardProbRandomNumber', index=i, default=[np.nan] * 2)[1],
            left_valve_open_time=float(obj.TP_LeftValue[i]),
            right_valve_open_time=float(obj.TP_RightValue[i]),
            block_beta=float(obj.TP_BlockBeta[i]),
            block_min=float(obj.TP_BlockMin[i]),
            block_max=float(obj.TP_BlockMax[i]),
            min_reward_each_block=float(obj.TP_BlockMinReward[i]),
            delay_beta=float(obj.TP_DelayBeta[i]),
            delay_min=float(obj.TP_DelayMin[i]),
            delay_max=float(obj.TP_DelayMax[i]),
            delay_duration=obj.B_DelayHistory[i],
            ITI_beta=float(obj.TP_ITIBeta[i]),
            ITI_min=float(obj.TP_ITIMin[i]),
            ITI_max=float(obj.TP_ITIMax[i]),
            ITI_duration=obj.B_ITIHistory[i],
            response_duration=float(obj.TP_ResponseTime[i]),
            reward_consumption_duration=float(obj.TP_RewardConsumeTime[i]),
            auto_waterL=obj.B_AutoWaterTrial[0][i] if type(obj.B_AutoWaterTrial[0]) is list else obj.B_AutoWaterTrial[i],   # Back-compatible with old autowater format
            auto_waterR=obj.B_AutoWaterTrial[1][i] if type(obj.B_AutoWaterTrial[0]) is list else obj.B_AutoWaterTrial[i],
            laser_on_trial=obj.B_LaserOnTrial[i],
            laser_wavelength=LaserWavelengthC,
            laser_location=LaserLocationC,
            laser_power=LaserPowerC,
            laser_duration=LaserDurationC,
            laser_condition=LaserConditionC,
            laser_condition_probability=LaserConditionProC,
            laser_start=LaserStartC,
            laser_start_offset=LaserStartOffsetC,
            laser_end=LaserEndC,
            laser_end_offset=LaserEndOffsetC,
            laser_protocol=LaserProtocolC,
            laser_frequency=LaserFrequencyC,
            laser_rampingdown=LaserRampingDownC,
            laser_pulse_duration=LaserPulseDurC,

            # add all auto training parameters (eventually should be in session.json)
            auto_train_engaged=_get_field(obj, 'TP_auto_train_engaged', index=i),
            auto_train_curriculum_name=_get_field(obj, 'TP_auto_train_curriculum_name', index=i, default=None) or 'none',
            auto_train_curriculum_version=_get_field(obj, 'TP_auto_train_curriculum_version', index=i, default=None) or 'none',
            auto_train_curriculum_schema_version=_get_field(obj, 'TP_auto_train_curriculum_schema_version', index=i, default=None) or 'none',
            auto_train_stage=_get_field(obj, 'TP_auto_train_stage', index=i, default=None) or 'none',
            auto_train_stage_overridden=_get_field(obj, 'TP_auto_train_stage_overridden', index=i, default=None) or np.nan,
            
            # lickspout position
            lickspout_position_x=_get_field(obj, 'B_NewscalePositions', index=i, default=[np.nan] * 3)[0],
            lickspout_position_y=_get_field(obj, 'B_NewscalePositions', index=i, default=[np.nan] * 3)[1],
            lickspout_position_z=_get_field(obj, 'B_NewscalePositions', index=i, default=[np.nan] * 3)[2],
            
            # reward size
            reward_size_left=float(_get_field(obj, 'TP_LeftValue_volume', index=i)),
            reward_size_right=float(_get_field(obj, 'TP_RightValue_volume', index=i)),
        )


    #######  Other time series  #######
    #left/right lick time; give left/right reward time
    if getattr(obj, f'B_LeftRewardDeliveryTime{Harp}') == []:
        B_LeftRewardDeliveryTime = [np.nan]
    else:
        B_LeftRewardDeliveryTime = getattr(obj, f'B_LeftRewardDeliveryTime{Harp}')
    if getattr(obj, f'B_RightRewardDeliveryTime{Harp}') == []:
        B_RightRewardDeliveryTime = [np.nan]
    else:
        B_RightRewardDeliveryTime = getattr(obj, f'B_RightRewardDeliveryTime{Harp}')
    if obj.B_LeftLickTime == []:
        B_LeftLickTime = [np.nan]
    else:
        B_LeftLickTime = obj.B_LeftLickTime
    if obj.B_RightLickTime == []:
        B_RightLickTime = [np.nan]
    else:
        B_RightLickTime = obj.B_RightLickTime

    LeftRewardDeliveryTime = TimeSeries(
        name="left_reward_delivery_time",
        unit="second",
        timestamps=B_LeftRewardDeliveryTime,
        data=np.ones(len(B_LeftRewardDeliveryTime)).tolist(),
        description='The reward delivery time of the left lick port'
    )
    bonsai_nwb.add_acquisition(LeftRewardDeliveryTime)
    RightRewardDeliveryTime = TimeSeries(
        name="right_reward_delivery_time",
        unit="second",
        timestamps=B_RightRewardDeliveryTime,
        data=np.ones(len(B_RightRewardDeliveryTime)).tolist(),
        description='The reward delivery time of the right lick port'
    )
    bonsai_nwb.add_acquisition(RightRewardDeliveryTime)
    LeftLickTime = TimeSeries(
        name="left_lick_time",
        unit="second",
        timestamps=B_LeftLickTime,
        data=np.ones(len(B_LeftLickTime)).tolist(),
        description='The time of left licks'
    )
    bonsai_nwb.add_acquisition(LeftLickTime)
    RightLickTime = TimeSeries(
        name="right_lick_time",
        unit="second",
        timestamps=B_RightLickTime,
        data=np.ones(len(B_RightLickTime)).tolist(),
        description='The time of left licks'
    )
    bonsai_nwb.add_acquisition(RightLickTime)

    # Add photometry time stamps
    if not hasattr(obj, 'B_PhotometryFallingTimeHarp') or obj.B_PhotometryFallingTimeHarp == []:
        B_PhotometryFallingTimeHarp = [np.nan]
    else:
        B_PhotometryFallingTimeHarp = obj.B_PhotometryFallingTimeHarp
    PhotometryFallingTimeHarp = TimeSeries(
        name="FIP_falling_time",
        unit="second",
        timestamps=B_PhotometryFallingTimeHarp,
        data=np.ones(len(B_PhotometryFallingTimeHarp)).tolist(),
        description='The time of photometry falling edge (from Harp)'
    )
    bonsai_nwb.add_acquisition(PhotometryFallingTimeHarp)

    if not hasattr(obj, 'B_PhotometryRisingTimeHarp') or obj.B_PhotometryRisingTimeHarp == []:
        B_PhotometryRisingTimeHarp = [np.nan]
    else:
        B_PhotometryRisingTimeHarp = obj.B_PhotometryRisingTimeHarp
    PhotometryRisingTimeHarp = TimeSeries(
        name="FIP_rising_time",
        unit="second",
        timestamps=B_PhotometryRisingTimeHarp,
        data=np.ones(len(B_PhotometryRisingTimeHarp)).tolist(),
        description='The time of photometry rising edge (from Harp)'
    )
    bonsai_nwb.add_acquisition(PhotometryRisingTimeHarp)
    
    # Add optogenetics time stamps
    if not hasattr(obj, 'B_OptogeneticsTimeHarp') or obj.B_OptogeneticsTimeHarp == []:
        B_OptogeneticsTimeHarp = [np.nan]
    else:
        B_OptogeneticsTimeHarp = obj.B_OptogeneticsTimeHarp
    OptogeneticsTimeHarp = TimeSeries(
        name="optogenetics_time",
        unit="second",
        timestamps=B_OptogeneticsTimeHarp,
        data=np.ones(len(B_OptogeneticsTimeHarp)).tolist(),
        description='Optogenetics time (from Harp)'
    )
    bonsai_nwb.add_acquisition(OptogeneticsTimeHarp)

    # save NWB file
    base_filename = os.path.splitext(os.path.basename(fname))[0] + '.nwb'
    if len(bonsai_nwb.trials) > 0:
        NWBName = os.path.join(save_folder, base_filename)
        io = NWBHDF5IO(NWBName, mode="w")
        io.write(bonsai_nwb)
        io.close()
        logger.info(f'Successfully converted: {NWBName}')
        return 'success'
    else:
        logger.warning(f"No trials found! Skipping {fname}")
        return 'empty_trials'


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    
    nwb_folder = '/root/capsule/data/s3_foraging_all_nwb/'
    bpod_nwb_file = 'XY_23/XY_23_20230508_52.nwb'

    io = NWBHDF5IO(nwb_folder + bpod_nwb_file, mode='r')
    nwb = io.read()
    
    meta_dict_from_pkl = get_meta_dict_from_session_pkl(bpod_session_id=nwb.identifier)
    nwb_bpod_to_bonsai(nwb, meta_dict_from_pkl)
    