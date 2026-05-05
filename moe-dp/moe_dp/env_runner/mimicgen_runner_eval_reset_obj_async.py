import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from moe_dp.gym_util.async_vector_env import AsyncVectorEnv
from moe_dp.gym_util.sync_vector_env import SyncVectorEnv
from moe_dp.gym_util.multistep_wrapper import MultiStepWrapper
from moe_dp.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from moe_dp.model.common.rotation_transformer import RotationTransformer

from moe_dp.policy.base_policy import BasePolicy
from moe_dp.common.pytorch_util import dict_apply
from moe_dp.env_runner.base_runner import BaseRunner
from moe_dp.env.mimicgen.mimicgen_wrapper import MimicgenEnv
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import mimicgen
from datetime import datetime
import time 

def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env


class MimicgenRunner(BaseRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            device="cuda:0",
            exp_name=None,
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    MimicgenEnv(
                        task_name=robomimic_env.name,
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    MimicgenEnv(
                        env=robomimic_env,
                        task_name=robomimic_env.name,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        self.task_name=env_meta['env_name']
        TASK_RESET_KEY_BY_ENV_NAME = {
            "Coffee_Preparation_T0": "CoffeePreparation",
            "Kitchen_Cleanup_T0": "KitchenCleanupEnv",
            "KitchenCleanupEnv": "KitchenCleanupEnv",
            "Hammer_Cleanup_T0": "HammerCleanup",
            "Table_Cleanup_T0": "TableCleanup",
            "TableCleanup_D0": "TableCleanup",
            "Kitchen_T0": "Kitchen",
            "Mug_Cleanup_T0": "MugCleanup",
            "MugCleanup_D0": "MugCleanup",
        }
        ALL_TASK_RESET_DICT = {
            "HammerCleanup": {
                "reset_objs": [
                    {
                        "reset_obj_name": "hammer_joint0",
                        "reset_trigger_step": 2,
                        "reset_delay_steps": 4 
                    }
                ]
            },
            'MugCleanup':{
                "reset_objs": [ {
                        "reset_obj_name": "cleanup_object_joint0",
                        "reset_trigger_step": 2,
                        "reset_delay_steps": 4  
                    }]
            },
            "Kitchen": {
                "reset_objs": [
                    {
                        "reset_obj_name": "PotObject_joint0",
                        "reset_trigger_step": 2,
                        "reset_delay_steps": 3 
                    },
                    {
                        "reset_obj_name": "cube_bread_joint0",
                        "reset_trigger_step": 4,
                        "reset_delay_steps": 2  
                    },

                ]},
            "CoffeePreparation":{
                "reset_objs": [
                    {
                        "reset_obj_name": "mug_joint0",
                        "reset_trigger_step": 1,
                        "reset_delay_steps": 2  
                    },
                ]},
            "KitchenCleanupEnv":{
                "reset_objs":[
                    {
                        "reset_obj_name": "cube_bread_joint0",
                        "reset_trigger_step": 1,
                        "reset_delay_steps": 4 
                    },
                ]
            },
            "TableCleanup":{
                "reset_objs": [
                    {
                        "reset_obj_name": "hammer_joint0",
                        "reset_trigger_step": 1,
                        "reset_delay_steps": 4 
                    }
                  ]
            }

        }
        task_reset_key = TASK_RESET_KEY_BY_ENV_NAME.get(
            self.task_name,
            self.task_name.split('_')[0]
        )
        task_reset_dict = ALL_TASK_RESET_DICT.get(task_reset_key)
        if task_reset_dict is None:
            raise KeyError(f"Unsupported MimicGen reset task: {self.task_name}")
        reset_objs_list = task_reset_dict.get('reset_objs')
        self.reset_objs_list = reset_objs_list

        # norm test without any reset
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render,output_dir=output_dir):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = pathlib.Path(output_dir).joinpath(
                        'test', f"seed_{seed}_{current_time}.mp4")
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env,MimicgenEnv)
                env.env.env.init_state = None
                env.seed(seed)

                env.env.env.reset_obj_name = None
                env.env.env.reset_obj_position = None
                env.env.env.reset_trigger_step = None
                env.env.env.reset_delay_steps = None
                env.env.env.step_counter_in_trigger = 0

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test_reset (include reset objects)
        for obj in reset_objs_list:
            reset_obj_name = obj['reset_obj_name']
            reset_trigger_step = obj['reset_trigger_step']
            reset_delay_steps = obj['reset_delay_steps']

            for i in range(n_test):
                seed = test_start_seed + i
                enable_render = i < n_test_vis

                def init_fn(env, 
                            seed=seed, 
                            enable_render=enable_render,
                            output_dir=output_dir,
                            reset_obj_name=reset_obj_name,
                            reset_trigger_step=reset_trigger_step,
                            reset_delay_steps=reset_delay_steps):
                    # setup rendering
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = pathlib.Path(output_dir).joinpath(
                            'test_reset', f"{reset_obj_name}_seed_{seed}_{current_time}.mp4")
                        filename.parent.mkdir(parents=True, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to seed reset
                    assert isinstance(env.env.env, MimicgenEnv)
                    env.env.env.init_state = None
                    env.seed(seed)

                    env.env.env.reset_obj_name = reset_obj_name
                    env.env.env.reset_obj_position = None
                    env.env.env.reset_trigger_step = reset_trigger_step
                    env.env.env.reset_delay_steps = reset_delay_steps
                    env.env.env.step_counter_in_trigger = 0

                env_seeds.append(seed)
                env_prefixs.append(f'test_reset/{reset_obj_name}/')
                env_init_fn_dills.append(dill.dumps(init_fn))


        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None

            #get the reset obj poseition
            env.get_reset_obj_position()

            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                #reset object to eval the policy
                env.reset_obj(obs_dict['task_step'][:, 0, 0].int().tolist())
                
                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("NaN or Inf action detected during rollout")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None and i < 2:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
