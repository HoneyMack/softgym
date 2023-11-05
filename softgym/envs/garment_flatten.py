from collections import defaultdict
import numpy as np
import random
import pyflex
from softgym.envs.cloth_env import ClothEnv
from copy import deepcopy
from softgym.utils.misc import vectorized_range, vectorized_meshgrid
from softgym.utils.pyflex_utils import center_object
from scipy.spatial.transform import Rotation as R
from typing import Dict, Any
import cv2
import pickle
import os
import gym
import gym.spaces as spaces

#TODO:プログラムの整理・コメントの追加
class GarmentFlattenEnv(ClothEnv):
    def __init__(self, cached_states_path='tshirt_flatten_init_states.pkl', cloth_type='tank_male', **kwargs):
        """
        Initialize the GarmentFlattenEnv environment.

        Parameters
        ----------
        cached_states_path : str
            Path to the cached states.
        cloth_type : str
            Type of the cloth (e.g., 'tank_male').
        kwargs : dict
            keyword arguments for ClothEnv.
        """
        ##### Need to set before super().__init__()
        self.cloth_type = cloth_type
        self.prev_covered_area = None  # Should not be used until initialized, used for computing reward
        #####
        
        super().__init__(**kwargs)
        
        # action spaceの上書き
        self.action_space = spaces.Dict({
            #pick_posはcloth_envのpickerのlow,highを参考に合わせている(action_spaceのlow,highではない)
            "pick_pos":spaces.Box(low=np.array([-0.2,0.0,-0.2]),high=np.array([0.2,0.05,0.2]),dtype=np.float32),
            "place_pos":spaces.Box(low=np.array([-0.2,0.01,-0.2]),high=np.array([0.2,0.05,0.2]),dtype=np.float32),
        })
        
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        

    def generate_env_variation(self, num_variations=1, vary_cloth_size=True,max_wait_step=500,stable_vel_threshold=0.15):
        """
        Generate initial states for the environment.
        Note: This will also change the current states!

        Parameters
        ----------
        num_variations : int, optional
            Number of generated variations, by default 1
        vary_cloth_size : bool, optional
            Make cloth size random or not, by default True 
        max_wait_step : int, optional
            Maximum number of steps waiting for the cloth to stabilize, by default 100
        stable_vel_threshold : float, optional
            Cloth stable when all particles' vel are smaller than this, by default 0.25

        Returns
        -------
        Tuple[List[Dict[str,Any]], List[np.ndarray]]
            Tuple of generated configs and states.
        """        
        
        #TODO: 布のサイズをrandomizeする場合の処理を追加
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        

        for var_idx in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            self.set_scene(config)
            max_flatten_area = self._set_to_flat()# 布が完全に広がったときの領域も保持
            config["flatten_area"] = max_flatten_area
            
            # 布をカメラ内に移動
            self.move_to_pos([0, 0.05, 0])

            # 布が安定するまで待つ
            self._wait_for_stable(max_wait_step,stable_vel_threshold)

            self.action_tool.reset([0., -1., 0.])
            # 服の1点を持ち上げて落とす
            pos = self.get_positions()
            num_particle = pos.shape[0]
            pick_idx = random.randint(0, num_particle - 1)
            
            inv_mass_original = pos[pick_idx,3]
            pos[pick_idx,3] = 0  # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            self.set_positions(pos)
            
            # 持ち上げる
            pickup_t = 20 # Number of steps to pick up the cloth
            pickup_height = 0.1 # Height to pick up the cloth
            for _ in range(pickup_t):
                curr_pos = self.get_positions()
                curr_vel = self.get_velocities()
                curr_pos[pick_idx,1] += pickup_height / pickup_t
                curr_vel[pick_idx,:3] = [0, 0, 0] # Set the velocity of the pickup point to zero
                self.set_positions(curr_pos)
                self.set_velocities(curr_vel)
                self.step_simulation()

            # 落とす
            ## 持ち上げていた点の質量を元に戻す
            curr_pos = self.get_positions()
            curr_pos[pick_idx,3] = inv_mass_original
            self.set_positions(curr_pos)
            ## 安定するまで待つ
            self._wait_for_stable(max_wait_step,stable_vel_threshold) 
            
            # 布を中央に
            center_object()
            
            # pickerの位置を初期化
            if self.action_mode == 'sphere' or self.action_mode.startswith('picker'):
                picker_pos = curr_pos[pick_idx,:3] + [0., 0.2, 0.]
                self.action_tool.reset(picker_pos)
                
            generated_configs.append(deepcopy(config))
            generated_states.append(deepcopy(self.get_state()))

            print('config {}: camera params {}, flatten area: {}'.format(var_idx, config['camera_params'], generated_configs[-1]['flatten_area']))


        return generated_configs, generated_states
            
    def reset(self, to_flat=True):
        """
        Reset the environment.

        Parameters
        ----------
        to_flat : bool, optional
            Whether to reset the cloth to flat state or not, by default True

        Returns
        -------
        np.ndarray
            Observation of the environment.
        """
        obs = super().reset()
        
        if to_flat:
            self._set_to_flat()
            
        return obs

    def _reset(self):
        """ Right now only use one initial state"""
        self.prev_covered_area = self._get_current_covered_area(self.get_positions())
        if hasattr(self, 'action_tool'):
            curr_pos = self.get_positions()
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.2, cy])
        self.step_simulation()
        self.init_covered_area = None
        info = self._get_info()
        self.init_covered_area = info['performance']
        return self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('key_point'):
            # TODO ad action_repeat
            print('Need to add action repeat')
            raise NotImplementedError
        else:
            self.action_tool.step(action)
            if self.action_mode in ['sawyer', 'franka']:
                pyflex.step(self.action_tool.next_action)
            else:
                pyflex.step()
        return
    
    def step(self, action):
        pick_pos = action["pick_pos"]
        place_pos = action["place_pos"]
        
        self._perform_picking_action(pick_pos,place_pos)
        self._wait_for_stable()
        
        null_action = np.zeros(4)
        obs,reward,done,info = super().step(null_action)
        return obs, reward, done, info

    def normalize_action(self,action:Dict[str,np.ndarray])->Dict[str,np.ndarray]:
        #環境に合わせてactionの値を[-1,1]に正規化
        pick_pos = action["pick_pos"] /[0.2,0.05,0.2]
        place_pos = action["place_pos"]/[0.2,0.05,0.2]
        return {"pick_pos":pick_pos,"place_pos":place_pos}
    
    def denormalize_action(self,action:Dict[str,np.ndarray])->Dict[str,np.ndarray]:
        #[-1,1](zに関しては[0,1])で正規化されているactionの値を環境に合わせて変換
        pick_pos = action["pick_pos"] * [0.2,0.05,0.2] 
        place_pos = action["place_pos"] * [0.2,0.05,0.2] 
        return {"pick_pos":pick_pos,"place_pos":place_pos}
    
    def _perform_picking_action(self, pick_pos: np.ndarray, place_pos:np.ndarray,move_time_steps:int = 40):
        total_movement = place_pos - pick_pos
        one_step_movement = total_movement / move_time_steps

        action = np.zeros((move_time_steps, 4))
        action[:, 3] = 1 #pick
        #action[:, [0, 2]] = one_step_movement
        action[:,0:3] = one_step_movement.reshape(1,-1)
        
        # move picker to the target position
        shape_states = pyflex.get_shape_states().reshape((-1, 14))
        shape_states[0, :3] = pick_pos
        shape_states[0, 7:10] = pick_pos

        pyflex.set_shape_states(shape_states)
        
        self.step_simulation()

        # pick and move
        for a in action:
            _, _, _, _ = super().step(a)
        # release
        a = np.zeros(4)
        _, _, _, _ = super().step(a)
        
    def _wait_for_stable(self,max_wait_steps:int = 100,stable_vel_threshold:float = 0.25):
        """
        Wait for the cloth to stabilize.

        Parameters
        ----------
        max_wait_steps : int, optional
            Maximum number of steps waiting for the cloth to stabilize, by default 100
        stable_vel_threshold : float, optional
            Cloth stable when all particles' vel are smaller than this, by default 0.25
        """
        # wait for stable
        for _ in range(max_wait_steps):
            self.step_simulation()
            curr_vel = self.get_velocities()
            if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
                break
        else:
            print('Warning: cloth not stable')
        
    def sample_pick_pos(self)->np.ndarray:
        #布の点をランダムに選択して確実に持てる位置を選ぶ
        particle_pos= self.get_positions()[:,:3]
        pick_idx = np.random.randint(0, len(particle_pos))
        picking_pos = particle_pos[pick_idx]
        #ちょっと上めを持つ
        picking_pos[1] += 0.01
        return picking_pos

    def _get_current_covered_area(self, pos: np.ndarray):
        """
        Calculate the covered area by taking max x,y cood and min x,y coord, create a discritized grid between the points
        :param pos: Current positions of the particle states
        """
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = pos[:, [0, 2]]

        offset = pos2d - init
        slotted_x_low = np.maximum(np.round((offset[:, 0] - self.cloth_particle_radius) / span[0]).astype(int), 0)
        slotted_x_high = np.minimum(np.round((offset[:, 0] + self.cloth_particle_radius) / span[0]).astype(int), 100)
        slotted_y_low = np.maximum(np.round((offset[:, 1] - self.cloth_particle_radius) / span[1]).astype(int), 0)
        slotted_y_high = np.minimum(np.round((offset[:, 1] + self.cloth_particle_radius) / span[1]).astype(int), 100)
        # Method 1
        grid = np.zeros(10000)  # Discretization
        listx = vectorized_range(slotted_x_low, slotted_x_high)
        listy = vectorized_range(slotted_y_low, slotted_y_high)
        listxx, listyy = vectorized_meshgrid(listx, listy)
        idx = listxx * 100 + listyy
        idx = np.clip(idx.flatten(), 0, 9999)
        grid[idx] = 1

        return np.sum(grid) * span[0] * span[1]

        # Method 2
        # grid_copy = np.zeros([100, 100])
        # for x_low, x_high, y_low, y_high in zip(slotted_x_low, slotted_x_high, slotted_y_low, slotted_y_high):
        #     grid_copy[x_low:x_high, y_low:y_high] = 1
        # assert np.allclose(grid_copy, grid)
        # return np.sum(grid_copy) * span[0] * span[1]

    def _get_center_point(self, pos):
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        particle_pos = self.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        return curr_covered_area

    def _get_info(self):
        # Duplicate of the compute reward function!
        particle_pos = self.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        init_covered_area = curr_covered_area if self.init_covered_area is None else self.init_covered_area
        max_covered_area = self.get_current_config()['flatten_area']
        info = {
            'performance': curr_covered_area,
            'normalized_performance': (curr_covered_area - init_covered_area) / (max_covered_area - init_covered_area),
            'normalized_performance_2': (curr_covered_area) / (max_covered_area),
        }
        if 'qpg' in self.action_mode:
            info['total_steps'] = self.action_tool.total_steps
        return info

    def get_picked_particle(self):
        pps = np.ones(shape=self.action_tool.num_picker, dtype=np.int32) * -1  # -1 means no particles picked
        for i, pp in enumerate(self.action_tool.picked_particles):
            if pp is not None:
                pps[i] = pp
        return pps

    def get_picked_particle_new_position(self):
        intermediate_picked_particle_new_pos = self.action_tool.intermediate_picked_particle_pos
        if len(intermediate_picked_particle_new_pos) > 0:
            return np.vstack(intermediate_picked_particle_new_pos)
        else:
            return []

    def set_scene(self, config, state=None):
        # 描画モードの設定
        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3

        #衣服の種類の設定
        if self.cloth_type == 'tshirt':
            cloth_type = 0
        elif self.cloth_type == 'shorts':
            cloth_type = 1
        elif self.cloth_type == 'tshirt-small':
            cloth_type = 2
        elif self.cloth_type == 'tank_male':
            cloth_type = 10
        else:
            raise NotImplementedError(f"cloth_type {self.cloth_type} is not implemented")
        
        # シーンのパラメータの設定
        camera_params = config['camera_params'][config['camera_name']]
        scene_params = np.concatenate(
            [config['pos'][:], [config['scale'], config['rot']], config['vel'][:], [config['stiff'], config['mass'], config['radius']],
             camera_params['pos'][:], camera_params['angle'][:], [camera_params['width'], camera_params['height']], [render_mode], [cloth_type]])
        
        # シーンの設定: バージョンごとに設定が異なるので注意
        env_idx = 3 #服広げで使用する環境？
        if self.version == 2:
            robot_params = []
            self.params = (scene_params, robot_params)
            pyflex.set_scene(env_idx, scene_params, 0, robot_params)
        elif self.version == 1:
            pyflex.set_scene(env_idx, scene_params, 0)


        # 衣服の種類に応じて回転
        if cloth_type == 10:
            self.rotate_particles([0, 0, 90])
            self.move_to_pos([0, 0.04, 0])
        else:
            self.rotate_particles([0, 0, -90])
            self.move_to_pos([0, 0.05, 0])
        
        self._wait_for_stable()
        
        self.default_pos = self.get_positions()

        if state is not None:
            self.set_state(state)

        self.current_config = deepcopy(config)

    def get_default_config(self):
        """
        Get the default config of the environment.

        Returns
        -------
        Dict[str, Any]
            Default config of the environment.
        """
        #各服のスケール
        cloth_scales = defaultdict(lambda: -1)
        cloth_scales["tank_male"] = 0.004
        
        # 布に関するconfig
        config_cloth = {
            'pos': [0.01, 0.15, 0.01],
            'scale': cloth_scales[self.cloth_type],
            'rot': 0.0,
            'vel': [0., 0., 0.],
            'stiff': 2.0*1e-1,
            'mass': 0.5 / (40 * 40)*((1e-1)**2),
            'radius': self.cloth_particle_radius,  # / 1.8,
            'cloth_type': 0
        }
        # カメラに関するconfig
        config_camera = {
            'camera_name': 'default_camera',
            'camera_params': {
                'default_camera':{
                    'pos': np.array([-0.0, 0.82, 0.82]),
                    'angle': np.array([0, -45 / 180. * np.pi, 0.]),
                    'width': self.camera_width,
                    'height': self.camera_height
                },
                'top_down_camera_full': {
                    'pos': np.array([0, 0.55, 0]),
                    'angle': np.array([0, -90 / 180 * np.pi, 0]),
                    'width': self.camera_width,
                    'height': self.camera_height
                },
            },
        }
        # その他のconfig
        config_other ={
            'drop_height': 0.2,
        }
        
        # configの統合
        config = {}
        config.update(config_cloth)
        config.update(config_camera)
        config.update(config_other)
        
        return config

    def rotate_particles(self, angle):
        r = R.from_euler('zyx', angle, degrees=True)
        pos = self.get_positions()
        center = np.mean(pos, axis=0)
        pos -= center
        rot_pos_xyz = pos.copy()[:, :3]
        rot_pos_xyz = r.apply(rot_pos_xyz)
        rot_pos_xyzm = np.column_stack([rot_pos_xyz, pos[:, 3]])
        rot_pos_xyzm += center
        self.set_positions(rot_pos_xyzm)

    def _set_to_flat(self, pos:np.ndarray=None):
        """
        服を完全に広げる関数

        Parameters
        ----------
        pos : np.ndarray, optional
            位置を表す配列(N,4)
            N: 粒子数
            4: 座標(x,y,z)と各粒子の質量, by default None

        Returns
        -------
        float
            布が完全に広がったときの面積
        """

        if pos is None:
            pos = self.default_pos
        
        self.set_positions(pos)
        self.step_simulation()

        return self._get_current_covered_area(pos)

    def move_to_pos(self, new_pos):
        """
        重心をnew_posに移動する関数

        Parameters
        ----------
        new_pos : np.ndarray
            移動先の重心座標
        """
        pos = self.get_positions()
        center = np.mean(pos, axis=0)
        pos[:, :3] -= center[:3]
        pos[:, :3] += np.asarray(new_pos)
        self.set_positions(pos)



if __name__ == '__main__':
    from softgym.registered_env import env_arg_dict
    from softgym.registered_env import SOFTGYM_ENVS
    import imageio
    
    def get_softgym_env_args(env_name = 'GarmentFlatten',cloth_type='tshirt-small') -> dict:
        args:Dict = env_arg_dict[env_name].copy()
        args.update({
            'env_name': env_name,
            'render_mode': 'cloth',
            'observation_mode': "cloth_rgb",#'cam_rgb',
            'render': True,
            'camera_height': 64,
            'camera_width': 64,
            'camera_name': 'top_down_camera_full',
            'horizon': 10000,
            'headless': True,
            'action_repeat': 1,
            'picker_radius': 0.0001,
            'picker_threshold': 0.015,#0.00625,
            'num_variations': 1000,
            'cached_states_path': f"{env_name}_{cloth_type}_init_states.pkl",
            'use_cached_states':  True,
            'save_cached_states': False,
            'cloth_type': cloth_type,
        })
        
        return args

    env_name = 'GarmentFlatten'
    cloth_type = 'tank_male'
    # cloth_type = 'tshirt-small'
    softgym_env_args = get_softgym_env_args(env_name,cloth_type)
    env:GarmentFlattenEnv = SOFTGYM_ENVS[env_name](**softgym_env_args)
    
    obs = env.reset()
    history = [obs]
    
    for i in range(1,5):
        action = env.action_space.sample()
        action['pick_pos'] = env.sample_pick_pos()
        obs, reward, done, _ = env.step(action)
        history.append(obs) #observation_modeに応じた描画

    env.close()
    
    
    # 画像を保存
    if not os.path.exists("images"):
        os.mkdir("images")
    for i, obs in enumerate(history):
        imageio.imwrite(f"images/{i}.png", obs)
    
    # import copy
    # import cv2


    # def prepare_policy(env):
    #     print("preparing policy! ", flush=True)

    #     # move one of the picker to be under ground
    #     shape_states = pyflex.get_shape_states().reshape(-1, 14)
    #     shape_states[1, :3] = -1
    #     shape_states[1, 7:10] = -1

    #     # move another picker to be above the cloth
    #     pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
    #     pp = np.random.randint(len(pos))
    #     shape_states[0, :3] = pos[pp] + [0., 0.06, 0.]
    #     shape_states[0, 7:10] = pos[pp] + [0., 0.06, 0.]
    #     pyflex.set_shape_states(shape_states.flatten())


    # env_name = 'TshirtFlattenCFM'
    # env_args = copy.deepcopy(env_arg_dict[env_name])
    # env_args['render_mode'] = 'cloth'
    # env_args['observation_mode'] = 'cam_rgb'
    # env_args['render'] = True
    # env_args['camera_height'] = 720 #128
    # env_args['camera_width'] = 720#128
    # env_args['camera_name'] ='default_camera'
    # env_args['headless'] = True
    # env_args['action_repeat'] = 1
    # env_args['picker_radius'] = 0.01
    # env_args['picker_threshold'] = 0.00625
    # env_args['cached_states_path'] = 'tshirt_flatten_cfm_init_states_small_2021_05_28_01_16.pkl'
    # env_args['num_variations'] = 1
    # env_args['use_cached_states'] = False
    # env_args['save_cached_states'] = False
    # env_args['cloth_type'] = 'tshirt-small'
    # # pkl_path = './softgym/cached_initial_states/shorts_flatten.pkl'

    # env = SOFTGYM_ENVS[env_name](**env_args)
    # print("before reset")
    # env.reset()
    # print("after reset")
    # env._set_to_flat()
    # print("after reset")
    # # env.move_to_pos([0, 0.1, 0])
    # # pyflex.step()
    # # i = 0
    # # import pickle

    # # while (1):
    # #     pyflex.step(render=True)
    # #     if i % 500 == 0:
    # #         print('saving pkl to ' + pkl_path)
    # #         pos = pyflex.get_positions()
    # #         with open(pkl_path, 'wb') as f:
    # #             pickle.dump(pos, f)
    # #     i += 1
    # #     print(i)

    # obs = env._get_obs()
    # cv2.imwrite('./small_tshirt.png', obs)
    # # cv2.imshow('obs', obs)
    # # cv2.waitKey()

    # prepare_policy(env)

    # particle_positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
    # n_particles = particle_positions.shape[0]
    # # p_idx = np.random.randint(0, n_particles)
    # # p_idx = 100
    # pos = particle_positions
    # ok = False
    # while not ok:
    #     pp = np.random.randint(len(pos))
    #     if np.any(np.logical_and(np.logical_and(np.abs(pos[:, 0] - pos[pp][0]) < 0.00625, np.abs(pos[:, 2] - pos[pp][2]) < 0.00625),
    #                              pos[:, 1] > pos[pp][1])):
    #         ok = False
    #     else:
    #         ok = True
    # picker_pos = particle_positions[pp] + [0, 0.01, 0]

    # timestep = 50
    # movement = np.random.uniform(0, 1, size=(3)) * 0.4 / timestep
    # movement = np.array([0.2, 0.2, 0.2]) / timestep
    # action = np.zeros((timestep, 8))
    # action[:, 3] = 1
    # action[:, :3] = movement

    # shape_states = pyflex.get_shape_states().reshape((-1, 14))
    # shape_states[1, :3] = -1
    # shape_states[1, 7:10] = -1

    # shape_states[0, :3] = picker_pos
    # shape_states[0, 7:10] = picker_pos

    # pyflex.set_shape_states(shape_states)
    # pyflex.step()

    # obs_list = []

    # for a in action:
    #     obs, _, _, _ = env.step(a)
    #     obs_list.append(obs)
    #     # cv2.imshow("move obs", obs)
    #     # cv2.waitKey()

    # for t in range(30):
    #     a = np.zeros(8)
    #     obs, _, _, _ = env.step(a)
    #     obs_list.append(obs)
    #     # cv2.imshow("move obs", obs)
    #     # cv2.waitKey()

    # from softgym.utils.visualization import save_numpy_as_gif

    # save_numpy_as_gif(np.array(obs_list), '{}.gif'.format(
    #     env_args['cloth_type']
    # ))
    # print("before reset")
    # env.reset()
    # print("after reset")
    # pyflex.set_shape_states(shape_states)
    # pyflex.step()

    # obs_list = []

    # for a in action:
    #     obs, _, _, _ = env.step(a)
    #     obs_list.append(obs)
    #     # cv2.imshow("move obs", obs)
    #     # cv2.waitKey()

    # for t in range(30):
    #     a = np.zeros(8)
    #     obs, _, _, _ = env.step(a)
    #     obs_list.append(obs)
    #     # cv2.imshow("move obs", obs)
    #     # cv2.waitKey()
    # save_numpy_as_gif(np.array(obs_list), '{}_2.gif'.format(
    #     env_args['cloth_type']
    # ))
