
import numpy as np

"""
NOTES on foraging efficiency

Definition of foraging efficiency (i.e., a single value measurement to quantify "performance") is a complex and unsolved topic especially in the context of reward baiting. I have a presentation for this (https://alleninstitute.sharepoint.com/:p:/s/NeuralDynamics/EejfBIEvFA5DjmfOZV8atWgBx7q68GsKnavkVrfghL9y8g?e=OnR5r4). Simply speaking, foraging eff = actual reward rate of the mouse / reward rate of an ideal observer in the same session. The question is how to define the ideal observer. For the coupled-block-with-baiting task (Jeremiah's 2019 Neuron paper), I assume the ideal observer knows the underlying reward probability and the baiting dynamics, and do the optimal choice pattern ("fix-and-sample" in this case, see references on p.24 of my slides). For the non-baiting task (Cooper Grossman), I assume the ideal observer knows the underlying probability and makes the greedy choice. To account for the randomness of each actual session, I simulated the ideal observers using the actual random seed I used during the experiment.

This might not be the best way because the ideal observers I assumed is kind of cheating in the sense that they already know the underlying probability, but it sets an upper bound for all realistic agents, at least in an average sense. For a single session, however, there are chances where the efficiency can be larger than 1 because of the randomness of the task (sometimes the mice are really lucky that they get more reward than performing "optimally")
"""

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



def foraging_eff_baiting(reward_rate, p_Ls, p_Rs, random_number_L=None, random_number_R=None):  # Calculate foraging efficiency (only for 2lp)
        
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
    block_start_ind_left = np.where(np.diff(np.hstack([np.inf, p_Ls, np.inf])))[0].tolist()
    block_start_ind_right = np.where(np.diff(np.hstack([np.inf, p_Rs, np.inf])))[0].tolist()
    block_start_ind_effective = np.sort(np.unique(np.hstack([block_start_ind_left, block_start_ind_right])))
        
    reward_refills = [p_Ls >= random_number_L, p_Rs >= random_number_R]
    reward_optimal_random_seed = 0
    for_eff_optimal_random_seed = np.nan
    
    # Generate optimal choice pattern
    for b_start, b_end in zip(block_start_ind_effective[:-1], block_start_ind_effective[1:]):
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
    
    return for_eff_optimal, for_eff_optimal_random_seed
