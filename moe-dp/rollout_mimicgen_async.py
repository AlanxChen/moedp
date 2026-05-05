import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
import copy
import random
import wandb
import numpy as np
import threading
from hydra.core.hydra_config import HydraConfig
from moe_dp.policy.base_policy import BasePolicy
import datetime
OmegaConf.register_new_resolver("eval", eval, replace=True)

APP_ROOT = pathlib.Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent

max_steps = {
    "Mug_Cleanup_T0": 700,
    "Coffee_Preparation_T0": 1200,
    "Hammer_Cleanup_T0": 500,
    "Kitchen_T0": 1200,
    "Kitchen_Cleanup_T0": 900,
    "Table_Cleanup_T0": 1000,
}
OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)


def _resolve_repo_path(path_value):
    path = pathlib.Path(path_value)
    if path.is_absolute():
        return path

    candidates = [
        APP_ROOT / path,
        REPO_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _resolve_cfg_paths(cfg: OmegaConf):
    if "task" in cfg and "env_runner" in cfg.task and "dataset_path" in cfg.task.env_runner:
        cfg.task.env_runner.dataset_path = str(_resolve_repo_path(cfg.task.env_runner.dataset_path))


def _get_rollout_cfg(cfg: OmegaConf):
    rollout_cfg = cfg.get("rollout")
    if rollout_cfg is None:
        rollout_cfg = OmegaConf.create()
    return rollout_cfg


def _resolve_checkpoint_path(checkpoint_path):
    if checkpoint_path is None:
        return None
    return _resolve_repo_path(checkpoint_path)

class RolloutWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed

        # configure model
        self.model: BasePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: BasePolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except Exception:
                # Some modules cannot be deep-copied, so recreate them from config.
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        
    def env_runner(self, policy, cfg, output_dir=None):
        class_path = cfg["task"]["env_runner"]["_target_"]
        params = {key: value for key, value in cfg["task"]["env_runner"].items() if key != '_target_'}

        module_name, class_name = class_path.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        EnvRunner = getattr(module, class_name)

        if output_dir is None:
            raise ValueError("output_dir must be specified")

        env_runner = EnvRunner(**params, output_dir=output_dir,exp_name=cfg.exp_name)

        runner_log = env_runner.run(policy)

        return runner_log
        
    def run(self):
        cfg = copy.deepcopy(self.cfg)
        _resolve_cfg_paths(cfg)
        rollout_cfg = _get_rollout_cfg(cfg)

        checkpoint_path = _resolve_checkpoint_path(rollout_cfg.get("checkpoint_path"))
        checkpoint_tag = rollout_cfg.get("checkpoint_tag", "latest")
        load_include_keys = ("global_step", "epoch")

        if checkpoint_path is not None:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint file was not found: {checkpoint_path}")
            print(f"Loading checkpoint {checkpoint_path}")
            self.load_checkpoint(path=checkpoint_path, include_keys=load_include_keys)
        elif cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path(tag=checkpoint_tag)
            if lastest_ckpt_path.is_file():
                print(f"Loading checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path, include_keys=load_include_keys)
            else:
                raise FileNotFoundError(
                    f"Checkpoint tag '{checkpoint_tag}' was not found under {self.output_dir}"
                )
        else:
            raise ValueError(
                "Rollout requires a checkpoint. Set +rollout.checkpoint_path=... "
                "or enable training.resume with a valid checkpoint in the run directory."
            )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
      
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = rollout_cfg.get("output_subdir", "rollout")
        output_dir = pathlib.Path(self.output_dir).joinpath(output_subdir, str(cfg.exp_name), current_time)
        output_dir.mkdir(parents=True, exist_ok=True)

        runner_log = self.env_runner(policy, cfg, output_dir=str(output_dir))

        log_file_path = output_dir.joinpath("runner_log.txt")

        filtered_runner_log = {k: v for k, v in runner_log.items() if not isinstance(v, wandb.sdk.data_types.video.Video)}

        with log_file_path.open("w") as log_file:
            for key, value in filtered_runner_log.items():
                log_file.write(f"{key}: {value}\n")
        print(f"Runner log saved to {log_file_path}")

        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        USE_SEED_0 = False
        if tag=='latest':
            if USE_SEED_0:
                output_dir_parts = self.output_dir.split('_')
                output_dir_parts[-1] = 'seed0'
                self.output_dir_seed = '_'.join(output_dir_parts)
                return pathlib.Path(self.output_dir_seed).joinpath('checkpoints', f'{tag}.ckpt')
            else:
                return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            
            

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()
        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                if key in ['model','ema_model']:
                    new_state_dict = {}
                    for param_key, param_value in value.items():
                        if param_key.startswith("module."):
                            param_key = param_key[7:]
                        new_state_dict[param_key] = param_value
                    self.__dict__[key].load_state_dict(new_state_dict, **kwargs)
                else:
                    self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'moe_dp', 'config'))
)
def main(cfg):
    workspace = RolloutWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
