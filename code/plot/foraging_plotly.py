import numpy as np
import plotly.graph_objs as go
import plotly.express as px


def moving_average(a, n=3) :
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_session_from_nwb(nwb, **kargs):
    
    df_trials = nwb.trials.to_dataframe()

    choice_history = df_trials.animal_response.map({0: 0, 1: 1, 2: np.nan}).values
    reward_history = np.vstack([df_trials.rewarded_historyL, df_trials.rewarded_historyR])
    p_reward = np.vstack([df_trials.reward_probabilityL, df_trials.reward_probabilityR])
    
    plot_session_lightweight_plotly(choice_history, reward_history, p_reward, smooth_factor=5, 
                                bait_history=np.vstack([df_trials.bait_left, df_trials.bait_right]))
    
def plot_session_lightweight(choice_history, 
                             reward_history, 
                             p_reward, 
                             smooth_factor=5, 
                             trial_range=None,
                             bait_history=None):

    # == Fetch data ==
    n_trials = len(choice_history)

    p_reward_fraction = p_reward[1, :] / (np.sum(p_reward, axis=0))

    ignored_trials = np.isnan(choice_history)
    rewarded_trials = np.any(reward_history, axis=0)
    unrewarded_trials = np.logical_not(np.logical_or(rewarded_trials, ignored_trials))

    # == Choice trace ==
    fig = go.Figure()

    # Rewarded trials
    xx = np.nonzero(rewarded_trials)[0] + 1
    yy = 0.5 + (choice_history[rewarded_trials] - 0.5) * 1.4
    fig.add_trace(go.Scattergl(x=xx, y=yy, mode='markers', marker=dict(color='black', size=15, symbol='line-ns', line=dict(width=2, color='black')), name='rewarded'))

    # Unrewarded trials
    xx = np.nonzero(unrewarded_trials)[0] + 1
    yy = 0.5 + (choice_history[unrewarded_trials] - 0.5) * 1.4
    fig.add_trace(go.Scattergl(x=xx, y=yy, mode='markers', marker=dict(color='lightgray', size=7, symbol='line-ns', line=dict(width=2, color='gray')), name='unrewarded'))

    # Ignored trials
    xx = np.nonzero(ignored_trials)[0] + 1
    yy = [1.1] * sum(ignored_trials)
    fig.add_trace(go.Scattergl(x=xx, y=yy, mode='markers', marker=dict(color='red', size=7, symbol='x'), name='ignored'))

    # Base probability
    xx = np.arange(0, n_trials) + 1
    yy = p_reward_fraction
    fig.add_trace(go.Scattergl(x=xx, y=yy, mode='lines', line=dict(color='darkgoldenrod', width=1.5), name='base rew. prob.'))

    # Smoothed choice history
    y = moving_average(choice_history, smooth_factor) / moving_average(~np.isnan(choice_history), smooth_factor)
    x = np.arange(0, len(y)) + int(smooth_factor / 2) + 1
    fig.add_trace(go.Scattergl(x=x, y=y, mode='lines', line=dict(color='black', width=2), name='choice (smooth = %g)' % smooth_factor))
    
    # finished ratio
    if np.sum(np.isnan(choice_history)):
        x = np.arange(0, len(y)) + int(smooth_factor / 2) + 1
        y = moving_average(~np.isnan(choice_history), smooth_factor)
        fig.add_trace(go.Scattergl(x=x, y=y,
                                mode='lines', 
                                line=dict(color='magenta', width=1),
                                name='finished (smooth = %g)' % smooth_factor))
    
    # trial range    
    if trial_range is not None:
        fig.add_vrect(x0=trial_range[0], x1=trial_range[1], fillcolor='grey', opacity=0.2, line_width=0)
        
    # bait history
    if bait_history is not None:
        for side in (0, 1):
            xx = np.nonzero(bait_history[side, :])[0] + 1
            yy = [-0.05 if side == 0 else 1.05] * len(xx)
            fig.add_trace(go.Scattergl(x=xx, y=yy, mode='markers', 
                                       marker=dict(color='green', size=4, symbol='circle'), 
                                       name='reward assigned',
                                       showlegend=side==0,
                                       legendgroup='reward assigned'))
     
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", xanchor="left", y=1.2, x=0),
        yaxis=dict(tickvals=[0, 1], ticktext=['Left', 'Right'], range=[-0.4, 1.4], showgrid=False, zeroline=False),
        xaxis=dict(title='Trial', showgrid=False, zeroline=False),
        margin=dict(l=60, r=30, t=30, b=40),
        hovermode='closest',
        height=400,
        width=1200
    )
    
    return fig