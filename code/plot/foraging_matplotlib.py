import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def moving_average(a, n=3) :
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
def plot_session_lightweight(data,   # choice_history, reward_history, p_reward
                             fitted_data=None, 
                             photostim=None,    # trial, power, s_type
                             valid_range=None,
                             smooth_factor=5, 
                             base_color='y', 
                             ax=None, 
                             vertical=False):
    # sns.reset_orig()
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 3) if not vertical else (3, 12), dpi=200)
        plt.subplots_adjust(left=0.1, right=0.8, bottom=0.05, top=0.8)

    if not vertical:
        gs = ax._subplotspec.subgridspec(2, 1, height_ratios=[1, 0.2], hspace=0.1)
        ax_1 = ax.get_figure().add_subplot(gs[0, 0])
        ax_2 = ax.get_figure().add_subplot(gs[1, 0])
        ax_1.get_shared_x_axes().join(ax_1, ax_2)
    else:
        gs = ax._subplotspec.subgridspec(1, 2, width_ratios=[0.2, 1], wspace=0.1)
        ax_1 = ax.get_figure().add_subplot(gs[0, 1])
        ax_2 = ax.get_figure().add_subplot(gs[0, 0])
        ax_1.get_shared_y_axes().join(ax_1, ax_2)
        
    with sns.plotting_context("notebook", font_scale=1):

        choice_history, reward_history, p_reward = data

        # == Fetch data ==
        n_trials = np.shape(choice_history)[1]

        p_reward_fraction = p_reward[1, :] / (np.sum(p_reward, axis=0))

        ignored_trials = np.isnan(choice_history[0])
        rewarded_trials = np.any(reward_history, axis=0)
        unrewarded_trials = np.logical_not(np.logical_or(rewarded_trials, ignored_trials))

        # == Choice trace ==
        # Rewarded trials
        xx = np.nonzero(rewarded_trials)[0] + 1
        yy = 0.5 + (choice_history[0, rewarded_trials] - 0.5) * 1.4
        ax_1.plot(*(xx, yy) if not vertical else [*(yy, xx)], 
                '|' if not vertical else '_', color='black', markersize=10, markeredgewidth=2)

        # Unrewarded trials
        xx = np.nonzero(unrewarded_trials)[0] + 1
        yy = 0.5 + (choice_history[0, unrewarded_trials] - 0.5) * 1.4
        ax_1.plot(*(xx, yy) if not vertical else [*(yy, xx)],
                '|' if not vertical else '_', color='gray', markersize=6, markeredgewidth=1)

        # Ignored trials
        xx = np.nonzero(ignored_trials)[0] + 1
        yy = [1.1] * sum(ignored_trials)
        ax_1.plot(*(xx, yy) if not vertical else [*(yy, xx)],
                'x', color='red', markersize=2, markeredgewidth=0.5, label='ignored')

        # Base probability
        xx = np.arange(0, n_trials) + 1
        yy = p_reward_fraction
        ax_1.plot(*(xx, yy) if not vertical else [*(yy, xx)],
                color=base_color, label='base rew. prob.', lw=1.5)

        # Smoothed choice history
        y = moving_average(choice_history, smooth_factor) / moving_average(~np.isnan(choice_history), smooth_factor)
        x = np.arange(0, len(y)) + int(smooth_factor / 2) + 1
        ax_1.plot(*(x, y) if not vertical else [*(y, x)],
                linewidth=1.5, color='black', label='choice (smooth = %g)' % smooth_factor)
        
        # finished ratio
        if np.sum(np.isnan(choice_history)):
            x = np.arange(0, len(y)) + int(smooth_factor / 2) + 1
            y = moving_average(~np.isnan(choice_history), smooth_factor)
            ax_1.plot(*(x, y) if not vertical else [*(y, x)],
                    linewidth=0.8, color='m', alpha=1,
                    label='finished (smooth = %g)' % smooth_factor)
             
        # add valid ranage
        if valid_range is not None:
            add_range = ax_1.axhline if vertical else ax_1.axvline
            add_range(valid_range[0], color='m', ls='--', lw=1, label='motivation good')
            add_range(valid_range[1], color='m', ls='--', lw=1)
                
        # For each session, if any fitted_data
        if fitted_data is not None:
            ax_1.plot(np.arange(0, n_trials), fitted_data[1, :], linewidth=1.5, label='model')
        
        # == photo stim ==
        if photostim is not None:
            plot_spec_photostim = { 'after iti start': 'cyan',  
                                    'before go cue': 'cyan',
                                    'after go cue': 'green',
                                    'whole trial': 'blue'}
            
            trial, power, s_type = photostim
            x = trial
            y = np.ones_like(trial) + 0.4
            scatter = ax_1.scatter(
                                *(x, y) if not vertical else [*(y, x)],
                                s=power.astype(float)*2,
                                edgecolors=[plot_spec_photostim[t] for t in s_type]
                                  if any(s_type) else 'darkcyan',
                                marker='v' if not vertical else '<',
                                facecolors='none',
                                linewidth=0.5,
                                label='photostim')

        # p_reward    
        xx = np.arange(0, n_trials) + 1
        ll = p_reward[0, :]
        rr = p_reward[1, :]
        ax_2.plot(*(xx, rr) if not vertical else [*(rr, xx)],
                color='b', label='p_right', lw=1)
        ax_2.plot(*(xx, ll) if not vertical else [*(ll, xx)],
                color='r', label='p_left', lw=1)
        ax_2.legend(fontsize=5, ncol=1, loc='upper left', bbox_to_anchor=(0, 1))
        
        if not vertical:
            ax_1.set_yticks([0, 1])
            ax_1.set_yticklabels(['Left', 'Right'])
            ax_1.legend(fontsize=6, loc='upper left', bbox_to_anchor=(0.6, 1.3), ncol=2)
            ax_1.set_xticks([])

            sns.despine(trim=True, bottom=True, ax=ax_1)
            sns.despine(trim=True, ax=ax_2)
        else:
            ax_1.set_xticks([0, 1])
            ax_1.set_xticklabels(['Left', 'Right'])
            ax_1.invert_yaxis()
            ax_1.legend(fontsize=6, loc='upper left', bbox_to_anchor=(0, 1.05), ncol=2)
            ax_1.set_yticks([])

            sns.despine(trim=True, left=True, ax=ax_1)
            sns.despine(trim=True, ax=ax_2)


        # ax_2.set(ylim=(0, 1))
    
    # ax.set_xlim(0,300)

    # fig.tight_layout()
    ax_2.set(xlabel='Trial number')
    ax.remove()

    return ax_1.get_figure(), [ax_1, ax_2]