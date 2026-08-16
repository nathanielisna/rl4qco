from source.model.Policy import *
from source.model.QuantCircOptEnv import *
from source.model.Logger import OptimizationLogger, OptimizationLoggingCallback

## Choose the reinforment learning algorithm to import and use
# from sb3_contrib import TRPO
from stable_baselines3 import A2C
from stable_baselines3 import PPO # For loading value head pretrained with PPO

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

## Set the seed
# BASE_SEED = 
NUM_ENV = 16

## Path to a training checkpoint to extract the pretrained value function from
PPO_CHECKPOINT_PATH = ""

def make_env(rank, base_seed=BASE_SEED):
    def _init():
        return Monitor(
            QuantCircOptEnv(
                qubit_dim=12,
                depth_dim=350,
                num_gates=150,
                seed=base_seed + rank,
            ),
            info_keywords=("opt_episode",),
        )
    return _init


def transfer_critic_weights(a2c_model, ppo_checkpoint_path):
    """
    Load PPO checkpoint and transfer only the critic weights
    (shared_cnn + value_head) into the A2C model.
    Policy head is left randomly initialized to isolate
    the critic's contribution.
    """
    ppo_model = PPO.load(ppo_checkpoint_path, device=a2c_model.device)

    ppo_state = ppo_model.policy.state_dict()
    a2c_state = a2c_model.policy.state_dict()

    transferred_keys = []
    skipped_keys = []

    for key in a2c_state:
        # only transfer shared_cnn and value_head, leave policy_head random
        if key.startswith("shared_cnn.") or key.startswith("value_head."):
            if key in ppo_state and ppo_state[key].shape == a2c_state[key].shape:
                a2c_state[key] = ppo_state[key].clone()
                transferred_keys.append(key)
            else:
                skipped_keys.append(key)

    a2c_model.policy.load_state_dict(a2c_state)

    return a2c_model


if __name__ == "__main__":
    tmp_path = "./tmp/sb3_log/"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    csv_logger = OptimizationLogger("./tmp/opt.csv")

    env = SubprocVecEnv(
        [make_env(i, BASE_SEED) for i in range(NUM_ENV)]
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./logs/",
        name_prefix="A2C_pretrained_critic_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    opt_callback = OptimizationLoggingCallback(csv_logger)
    callback = CallbackList([checkpoint_callback, opt_callback])

    model = A2C(
        FCNActorCriticPolicy,
        env,
        verbose=1,
        device="cuda",
        seed=BASE_SEED,
        n_steps=60,
    )

    # transfer PPO critic weights into A2C
    model = transfer_critic_weights(model, PPO_CHECKPOINT_PATH)

    model.set_logger(new_logger)
    model.learn(
        total_timesteps=1_000_000,
        callback=callback,
        progress_bar=True,
        log_interval=1,
    )