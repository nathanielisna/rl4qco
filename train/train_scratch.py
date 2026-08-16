from source.model.Policy import *
from source.model.QuantCircOptEnv import *
from source.model.Logger import OptimizationLogger, OptimizationLoggingCallback

# # Choose the reinforment learning algorithm to import and use
# from sb3_contrib import TRPO
# from stable_baselines3 import A2C
from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

# # Set the seed
BASE_SEED = 42

# # Set the number of parallel enviroments
NUM_ENV = 16


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
        name_prefix="PPO_model",
    )
    opt_callback = OptimizationLoggingCallback(csv_logger)

    callback = CallbackList([checkpoint_callback, opt_callback])

    '''
    Choose the reinforment learning algorithm.
    The following is an example with PPO. Refer to the stable-baselines3 documentation
    for A2C and TRPO, or to adjust hyperparmeters:
        https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
        https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
        https://sb3-contrib.readthedocs.io/en/master/modules/trpo.html
    '''
    model = PPO(
        FCNActorCriticPolicy,
        env,
        verbose=1,
        device="cuda",
        seed=BASE_SEED,
    )
    model.set_logger(new_logger)
    model.learn(
        total_timesteps=2_500_000,
        callback=callback, progress_bar=True
    )
