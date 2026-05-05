import gym
import numpy as np
import random

from gym import spaces
from robomimic.envs.env_robosuite import EnvRobosuite
from typing import Optional
import collections
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import mimicgen.env_interfaces.robosuite 


MIMICGEN_INTERFACE_BY_ENV_NAME = {
    "Coffee_Preparation_T0": "MG_CoffeePreparation",
    "CoffeePreparation_D1": "MG_CoffeePreparation",
    "Kitchen_Cleanup_T0": "MG_KitchenCleanup",
    "KitchenCleanupEnv": "MG_KitchenCleanup",
    "Hammer_Cleanup_T0": "MG_HammerCleanup",
    "Table_Cleanup_T0": "MG_TableCleanup",
    "TableCleanup_D0": "MG_TableCleanup",
    "Kitchen_T0": "MG_Kitchen",
    "Mug_Cleanup_T0": "MG_MugCleanup",
    "MugCleanup_D0": "MG_MugCleanup",
}


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
    env.seed(0)
    return env

class MimicgenEnv(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        shape_meta: dict,
        task_name: str,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        #reset related
        reset_obj_name=None,          # Object name to reset
        reset_obj_position=None,      # Target reset position [x, y, z]

        reset_trigger_step=None,      # Task step to trigger reset
        reset_delay_steps=None,       # Delay (in steps) before reset after trigger
        step_counter_in_trigger=0     # Step count in trigger step
        ):
        # simulation environment
        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state

        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        self.task_name = task_name

        self.image_size=480

        # MimicGen env inference is used to derive task_step for each task.
        mimicgen_env_inference_name = MIMICGEN_INTERFACE_BY_ENV_NAME.get(
            task_name,
            f"MG_{task_name.split('_')[0]}"
        )
        #Reference https://github.com/NVlabs/mimicgen/blob/4c7b46e6a912c49cf9072e4c0f873e1aadd42b24/mimicgen/scripts/generate_dataset.py#L261-L265
        self.mimicgen_env_inference = getattr(mimicgen.env_interfaces.robosuite, mimicgen_env_inference_name)(env.base_env)

        # observation and action space
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            elif key.endswith('task_step'):
                min_value, max_value = -1, 5
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

        #reset related
        self.reset_obj_name = reset_obj_name
        self.reset_obj_position = reset_obj_position
        self.reset_trigger_step = reset_trigger_step
        self.reset_delay_steps = reset_delay_steps
        self.step_counter_in_trigger = step_counter_in_trigger

    def get_task_step(self):
        task_step = self.mimicgen_env_inference.get_task_step()
        return np.array([task_step], dtype=int) 


    def get_observation(self, raw_obs=None):
        # raw_obs is the robosuite observation dictionary returned by robomimic.
        if raw_obs is None:
            raw_obs = self.env.get_observation()   
        self.render_cache = raw_obs[self.render_obs_key]

        task_step=self.get_task_step()
        raw_obs['task_step']=task_step
        
        if "render_image" in self.observation_space.keys():
            raw_obs['render_image'] = self.render()
        if "render_hand_image" in self.observation_space.keys():
            raw_obs['render_hand_image'] = self.render_hand()

        obs = dict()
        for key in self.observation_space.keys(): 
            obs[key] = raw_obs[key]    

        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        random.seed(seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # check if init_state is a dict
            if isinstance(self.init_state, dict):
                raw_obs = self.env.reset_to(self.init_state)
            else:
                # Always reset to the same state to stay compatible with gym.
                raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            np.random.seed(seed=seed)
            random.seed(seed)
            raw_obs = self.env.reset()
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        self.cur_step = 0
        if  hasattr(self.mimicgen_env_inference, "current_task_step"):
            self.mimicgen_env_inference.current_task_step = 0
        obs = self.get_observation(raw_obs)
        return obs
    
    def reset_to(self, state):
        raw_obs = self.env.reset_to(state)
        self.cur_step = 0
        obs = self.get_observation(raw_obs)
        return obs
     
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self.cur_step += 1
        obs = self.get_observation(raw_obs)

        return obs, reward, done, info
    
    def get_rgb(self):
        img = self.env.render(mode='rgb_array',width=self.image_size, height=self.image_size, camera_name="agentview")
        return img
    
    def render(self, mode='rgb_array'):
        img = self.get_rgb()
        return img
    
    def render_hand(self, mode='rgb_array'):
        img = self.env.render(mode='rgb_array',width=self.image_size, height=self.image_size, camera_name="robot0_eye_in_hand")
        return img

    def reset_special_object(self, object_name, position):
        self.env.env.sim.data.set_joint_qpos(object_name, position)
        self.env.env.sim.forward()

    def get_reset_obj_position(self):
        if self.reset_obj_name is not None:
            self.reset_obj_position = self.env.env.sim.data.get_joint_qpos(self.reset_obj_name).copy()
    
    def reset_obj(self,task_step:int):
        assert isinstance(task_step, int), "task_step should be an integer"
        if task_step == self.reset_trigger_step:
            self.step_counter_in_trigger += 1
            if self.step_counter_in_trigger == self.reset_delay_steps:
                self.reset_special_object(self.reset_obj_name, self.reset_obj_position)
