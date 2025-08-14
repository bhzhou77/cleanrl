#!/bin/bash
python ppo_minigrid_chain.py --parent_folder=blocked_unlock_pickup_chain --env_id=MiniGrid-Unlock-v0 --order=1
python ppo_minigrid_chain.py --parent_folder=blocked_unlock_pickup_chain --env_id=MiniGrid-UnlockPickup-v0 --env_id_pre=MiniGrid-Unlock-v0 --order=2
python ppo_minigrid_chain.py --parent_folder=blocked_unlock_pickup_chain --env_id=MiniGrid-BlockedUnlockPickup-v0 --env_id_pre=MiniGrid-UnlockPickup-v0 --order=3
