from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.figure

from aind_dynamic_foraging_models.logistic_regression import (
    fit_logistic_regression, plot_logistic_regression, MODEL_MAPPER
)


def compute_logistic_regression(nwb) -> Tuple[
    pd.DataFrame, Dict[str, matplotlib.figure.Figure]]:
    """Fit a logistic regression model to the data in the NWB file.

    Parameters
    ----------
    nwb : loaded NWB file

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, matplotlib.figure.Figure]]
        pd.DataFrame : the fitted parameters of the logistic regression model
        dict: {{logistic_regression_model}: the generated figure}
    """
    df_session_logistic_regression = pd.DataFrame()
    dict_figures = {}

    df_trial = nwb.trials.to_dataframe()
    
    # Turn to 0 and 1 coding
    # I decided to also include autowaters because it makes sense to assume that 
    # the autowater will affect animal's subsequent choices
    choice_history = df_trial['animal_response'].values
    choice_history[choice_history == 2] = np.nan
    reward_history = (df_trial.rewarded_historyL | df_trial.rewarded_historyR | 
                      df_trial.auto_waterR | df_trial.auto_waterL 
                      & (df_trial.animal_response != 2)
                      ).astype(int).values
        
    for model_name in MODEL_MAPPER.keys():
        # Do fitting
        dict_logistic_result = fit_logistic_regression(choice_history, reward_history,
                                logistic_model=model_name,
                                n_trial_back=15,
                                selected_trial_idx=None,
                                solver='liblinear', 
                                penalty='l2',
                                Cs=10,
                                cv=10,
                                n_jobs_cross_validation=1,
                                n_bootstrap_iters=1000, 
                                n_bootstrap_samplesize=None,)
        ax = plot_logistic_regression(dict_logistic_result)
        
        # Pack data
        df_beta_exp_fit = dict_logistic_result['df_beta_exp_fit']
        for logistic_var in df_beta_exp_fit.index:
            for exp_fit_var in set(df_beta_exp_fit.columns.get_level_values(0)):
                df_session_logistic_regression.loc[
                    0, f'logistic_{model_name}_{logistic_var}_{exp_fit_var}'
                ] = df_beta_exp_fit.loc[logistic_var, (exp_fit_var, 'fitted')]
                
        # Add bias
        df_session_logistic_regression.loc[
                    0, f'logistic_{model_name}_bias'
                ] = dict_logistic_result['df_beta'].loc['bias']['cross_validation'].values[0]

        dict_figures[model_name] = ax.figure
    
    return df_session_logistic_regression, dict_figures