from typing import Dict
import yaml, collections
import argparse
from rl_zoo3.exp_manager import ExperimentManager
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_data = []
        self.rollout_end_count = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"]
        for i in info:
            if "episode" in i:
                self.episode_data.append(
                    {
                        "reward": i["episode"]["r"],
                        "length": i["episode"]["l"],
                        "time": self.num_timesteps,  # or time.time()
                    }
                )
                print("Last episode data", self.episode_data[-1])
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_end_count += 1


def manual_training_reproduction(
    model_name: str, env_name: str, exp_manager_args: Dict
):
    def odict_ctor(loader, node):
        data = loader.construct_sequence(node, deep=True)
        inner = collections.OrderedDict(data[0])
        return {env_name: inner}

    yaml.SafeLoader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:collections.OrderedDict",
        odict_ctor,
    )

    exp_manager = ExperimentManager(
        argparse.Namespace(), model_name, env_name, "/tmp/", **exp_manager_args
    )

    logger = EpisodeLogger()
    results = exp_manager.setup_experiment()
    exp_manager.callbacks = [logger]
    if results is not None:
        model, saved_hyperparams = results
        exp_manager.learn(model)

if __name__ == "__main__":
    config_file = "/home/gabor/projects/r2l/r2l-verification/rl-trained-agents/a2c/Acrobot-v1_1/Acrobot-v1/config.yml"
    manager_args = {
        "env_kwargs": None,
        "n_eval_episodes": 10,
        "eval_freq": 10000,
        "hyperparams": None,
        "log_interval": -1,
        "n_evaluations": 20,
        "n_jobs": 1,
        "n_startup_trials": 10,
        "n_timesteps": -1,
        "n_trials": 10,
        "optimize_hyperparameters": False,
        "pruner": "median",
        "sampler": "tpe",
        "save_freq": -1,
        "save_replay_buffer": False,
        "seed": 951484142,
        "storage": None,
        "study_name": None,
        "tensorboard_log": "",
        "trained_agent": "",
        "truncate_last_trajectory": True,
        "uuid_str": "5627250e-c0e6-428c-b5ce-ea9204617cda",
        "vec_env_type": "dummy",
        "verbose": 1,
        "config": config_file,
    }
    manual_training_reproduction("a2c", "Acrobot-v1", manager_args)
