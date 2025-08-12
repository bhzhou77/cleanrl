import sys
sys.path.append('../../')
sys.path.append('../../Gymnasium/')

from Gymnasium import gymnasium as gym
import mujoco as mj
import mujoco.viewer as mjcv
import time
import torch

import ppo_continuous_action_spot as ppospot

model = mj.MjModel.from_xml_path('../../Gymnasium/gymnasium/envs/mujoco/assets/spot_scene_v0.xml')
data = mj.MjData(model)

cam = mj.MjvCamera()
opt = mj.MjvOption()
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)

joint_name_list = ['fl_hx', 'fl_hy', 'fl_kn', 'fr_hx', 'fr_hy', 'fr_kn', 'hl_hx', 'hl_hy', 'hl_kn', 'hr_hx', 'hr_hy', 'hr_kn']
actuator_name_list = joint_name_list.copy()

# env setup
env_id = 'Spot-v0'
capture_video = False
run_name = f""
gamma = 0.99
num_envs = 1
envs = gym.vector.SyncVectorEnv(
        [ppospot.make_env(env_id, i, capture_video, run_name, gamma) for i in range(num_envs)]
    )
agent = ppospot.Agent(envs)

folder_name = 'Spot-v0__ppo_continuous_action_spot__1__1754958041'
model_path = f'runs_sync/{folder_name}/ppo_continuous_action_spot.cleanrl_model'
agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def get_next_obs(data):
  next_obs_qpos = torch.tensor(data.qpos[1:].copy().reshape(1, -1)).float()
  next_obs_qvel = torch.tensor(data.qvel[:].copy().reshape(1, -1)).float()
  next_obs = torch.cat([next_obs_qpos, next_obs_qvel], dim=1)
  return next_obs

next_obs = get_next_obs(data)
with mjcv.launch_passive(model, data) as viewer:
  start = time.time()
  counter = 0
  while viewer.is_running() and time.time() - start < 1000:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    if counter % 10 == 0:
      action, _, _, _ = agent.get_action_and_value(next_obs)
      data.ctrl = action.detach().numpy()
    mj.mj_step(model, data)
    next_obs = get_next_obs(data)
    viewer.cam.lookat[:] = data.body('body').xpos
    
    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

    counter = counter + 1

