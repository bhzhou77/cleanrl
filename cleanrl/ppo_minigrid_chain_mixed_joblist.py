# !/usr/bin/env python
import numpy as np

env_id_list = list(np.random.choice(['MiniGrid-Unlock-v0', 'MiniGrid-UnlockPickup-v0', 'MiniGrid-BlockedUnlockPickup-v0'], 99))
env_id_list.insert(0, np.str_('MiniGrid-Unlock-v0'))
timestep_dict = {'MiniGrid-Unlock-v0':1000000, 
                 'MiniGrid-UnlockPickup-v0':100000, 
                 'MiniGrid-BlockedUnlockPickup-v0':100000}

# Save info to a joblist file
log_file = 'ppo_minigrid_chain_mixed_joblist.sh'
with open(log_file, 'w') as f:
    f.truncate()
    f.write('#!/bin/bash\n')
    f.write(f'python ppo_minigrid_chain.py --parent_folder=blocked_unlock_pickup_chain_mixed_2 --env_id={env_id_list[0].item()} --order=1 --total_timesteps={timestep_dict[env_id_list[0].item()]}\n')
    for order in range(2, 101):
        f.write(f'python ppo_minigrid_chain.py --parent_folder=blocked_unlock_pickup_chain_mixed_2 --env_id={env_id_list[order-1].item()} --env_id_pre={env_id_list[order-2].item()} --order={order} --total_timesteps={timestep_dict[env_id_list[order-1].item()]}\n')