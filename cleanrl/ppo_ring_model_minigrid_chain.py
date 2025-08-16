# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import sys
sys.path.append('../../cognition_ring_attractor/src')

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import minigrid
from minigrid.wrappers import ImgObsWrapper

import ring_attractor as ra

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    parent_folder: str = "ppo_minigrid_chain"

    # Algorithm specific arguments
    order: str = "1"
    env_id: str = "MiniGrid-BlockedUnlockPickup-v0"
    """the id of the environment"""
    env_id_pre: str = "MiniGrid-BlockedUnlockPickup-v0"
    """the id of the previous environment in the chain"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-3
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    features_dim: int = 128
    """the feature dimension after cnn"""
    nring: int = 4
    """the number of rings"""
    nupdate: int = 5
    """the number of updates for each cycle of the rnn"""
    nneuron: int = 256
    """the number of nneurons in the rnn"""
    taus: tuple = (0.04, 1.0, 1.0, 0.04)

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = ImgObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_nobias(layer, std=np.sqrt(2)):
    torch.nn.init.orthogonal_(layer.weight, std)
    return layer


# adapted from minigrid documentations
class MinigridFeaturesExtractor(nn.Module):
    def __init__(self, x, features_dim=128, normalized_image=False):
        super().__init__()
        x = x.permute(0, 3, 1, 2)
        n_input_channels = x.size()[1]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(x).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.linear(self.cnn(x))


class Agent(nn.Module):
    def __init__(self, envs, features_dim, nring, nneuron, taus, nupdate):
        super().__init__()
        self.nupdate = nupdate
        self.ra_network_critic = ra.RingAttractorNetwork(nring, nneuron, taus)
        self.ra_network_actor = ra.RingAttractorNetwork(nring, nneuron, taus)

        self.project_in_critic = layer_init_nobias(nn.Linear(features_dim, nneuron, bias=False))
        self.project_in_actor = layer_init_nobias(nn.Linear(features_dim, nneuron, bias=False))
        self.project_out_critic = layer_init_nobias(nn.Linear(nneuron, 1, bias=False), std=0.01)
        self.project_out_actor = layer_init_nobias(nn.Linear(nneuron, envs.single_action_space.n, bias=False), std=0.01)

        # the last three dimensions of x_test need to be compatible with the original data
        self.register_buffer('x_test', torch.rand(1, 7, 7, 3))
        self.cnn_input = MinigridFeaturesExtractor(self.x_test, features_dim)

    def get_value(self, rs_current, x):
        xp = self.cnn_input.forward(x)
        xp = self.project_in_critic(xp)
        rs_current, rs_delta7 = self.ra_network_critic.update_firing_rate_full_cycle(rs_current, self.nupdate, xp)
        value = self.project_out_critic(rs_delta7)

        return rs_current, value

    def get_action(self, rs_current, x, action=None, reward=1.0):
        xp = self.cnn_input.forward(x)
        xp = self.project_in_actor(xp)
        rs_current, rs_delta7 = self.ra_network_actor.update_firing_rate_full_cycle(rs_current, self.nupdate, xp)
        logits = self.project_out_actor(rs_delta7)
        if reward < 0.1:
            logits = torch.where(torch.rand(1) < 0.7, logits, torch.ones(logits.size()))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return rs_current, action, probs.log_prob(action), probs.entropy()

# class Agent(nn.Module):
#     def __init__(self, envs, features_dim, nring, nneuron, nupdate):
#         super().__init__()
#         self.nupdate = nupdate
#         self.ra_network_critic = nn.Sequential(
#             layer_init(nn.Linear(features_dim, 256)),
#             nn.Tanh(),
#             layer_init(nn.Linear(256, 256)),
#             nn.Tanh(),
#             layer_init(nn.Linear(256, 1), std=1.0),
#         )
#         self.ra_network_actor = nn.Sequential(
#             layer_init(nn.Linear(features_dim, 256)),
#             nn.Tanh(),
#             layer_init(nn.Linear(256, 256)),
#             nn.Tanh(),
#             layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01),
#         )

#         # the last three dimensions of x_test need to be compatible with the original data
#         self.register_buffer('x_test', torch.rand(1, 7, 7, 3))
#         self.cnn_input = MinigridFeaturesExtractor(self.x_test, features_dim)

#     def get_value(self, rs_current, x):
#         xp = self.cnn_input.forward(x)
#         return rs_current, self.ra_network_critic(xp)

#     def get_action(self, rs_current, x, action=None, reward=1.0):
#         xp = self.cnn_input.forward(x)
#         logits = self.ra_network_actor(xp)
#         if reward < 0.1:
#             logits = torch.where(torch.rand(1) < 0.7, logits, torch.ones(logits.size()))
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()

#         return rs_current, action, probs.log_prob(action), probs.entropy()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.parent_folder}/{args.env_id}__{args.exp_name}__{args.seed}__order{args.order}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    print(f"Training in {args.env_id}.")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    agent = Agent(envs, args.features_dim, args.nring, args.nneuron, args.taus, args.nupdate).to(device)
    if int(args.order) > 1:
        run_name_pre = f"{args.parent_folder}/{args.env_id_pre}__{args.exp_name}__{args.seed}__order{int(args.order)-1}"
        model_path_pre = f"runs/{run_name_pre}/{args.exp_name}.cleanrl_model"
        agent.load_state_dict(torch.load(model_path_pre))

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Save the initial model
    model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model_init"
    torch.save(agent.state_dict(), model_path)
    print(f"initial model saved to {model_path}")

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Get the initial firing rate
    ra_network_init = ra.RingAttractorNetwork(args.nring, args.nneuron, args.taus)
    rs_initial = ra_network_init.rs_initial
    for param in ra_network_init.parameters():
        param.requires_grad = False

    # rs_initial = torch.rand(1, 256)
    # rs_initial = rs_initial / torch.norm(rs_initial, dim=1, keepdim=True)

    for iteration in range(1, args.num_iterations + 1):
        rs_current_critic = rs_initial.clone()
        rs_current_actor = rs_initial.clone()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                rs_current_actor, action, logprob, _ = agent.get_action(rs_initial, next_obs, reward=rewards[step])
                rs_current_critic, value = agent.get_value(rs_initial, next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            rs_current_critic, next_value = agent.get_value(rs_initial, next_obs)
            next_value = next_value.reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        rs_current_critic = rs_initial.repeat(args.minibatch_size, 1)
        rs_current_actor = rs_initial.repeat(args.minibatch_size, 1)
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                rs_current_actor, _, newlogprob, entropy = agent.get_action(rs_initial, b_obs[mb_inds], b_actions.long()[mb_inds])
                rs_current_critic, newvalue = agent.get_value(rs_initial, b_obs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if iteration % 10 == 0:
            # print("SPS:", int(global_step / (time.time() - start_time)))
            print("iteration", iteration, "value loss", v_loss.item(), "policy loss", pg_loss.item(), "entropy loss", entropy_loss.item())

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
