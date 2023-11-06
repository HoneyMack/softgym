import pyflex
import numpy as np
from typing import Tuple
from softgym.envs.flex_env import FlexEnv


def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def intrinsic_from_fov(height:int, width:int, fov=45)->np.ndarray:
	"""
	カメラをピンホールカメラとしたときの内部パラメータを計算する関数

	Parameters
	----------
	height : int
		カメラの縦の画素数[px]
	width : int
		カメラの横の画素数[px]
	fov : int, optional
		水平方向の視野角, by default 45

	Returns
	-------
	np.ndarray
		K: 	内部パラメータ行列[4, 4]
			(カメラ座標を画像座標に変換する行列)
	"""
	# 視野角・焦点距離fを計算
	hfov = fov / 360. * 2. * np.pi # 水平方向の視野角[rad]
	fx = (width / 2.) / np.tan(hfov / 2.)

	## 垂直方向の視野角vfovをアスペクト比とhfovから計算
	aspect = height / width
	vfov = 2. * np.arctan(np.tan(hfov / 2) * aspect)# 垂直方向の視野角[rad]
	fy = (height / 2.) / np.tan(vfov / 2.)
 
	# 画像の中心座標
	px, py = (width / 2, height / 2)

	return np.array([[fx, 0, px, 0.],
						[0, fy, py, 0.],
						[0, 0, 1., 0.],
						[0., 0., 0., 1.]])

def get_rotation_matrix(angle, axis):
	axis = axis / np.linalg.norm(axis)
	s = np.sin(angle)
	c = np.cos(angle)

	m = np.zeros((4, 4))

	m[0][0] = axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * c
	# m[0][1] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
	m[0][1] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
	# m[0][2] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
	m[0][2] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
	m[0][3] = 0.0

	# m[1][0] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
	m[1][0] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
	m[1][1] = axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * c
	# m[1][2] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
	m[1][2] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
	m[1][3] = 0.0

	# m[2][0] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
	m[2][0] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
	# m[2][1] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
	m[2][1] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
	m[2][2] = axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * c
	m[2][3] = 0.0

	m[3][0] = 0.0
	m[3][1] = 0.0
	m[3][2] = 0.0
	m[3][3] = 1.0

	return m



def get_world_coords(rgb:np.ndarray, depth:np.ndarray, env:FlexEnv):
	"""
	rgb画像とdepth画像からworld座標を計算する関数

	Parameters
	----------
	rgb : np.ndarray
		rgb画像
	depth : np.ndarray
		depth画像
	env : FlexEnv
		環境

	Returns
	-------
	np.ndarray
		各画像のピクセルに対応するworld座標[height, width, [x,y,z,1]]
	"""
	height, width, _ = rgb.shape
	K = intrinsic_from_fov(height, width, 90) # the fov is 90 degrees

	cam_coords = back_project_pixels(depth, K)

	rotation_matrix, translation_matrix = get_world2camera_transform(env)

	world_coords = convert_cam_to_world(cam_coords, rotation_matrix, translation_matrix)

	return world_coords

def back_project_pixels(depth:np.ndarray, K:np.ndarray)->np.ndarray:
    """画像座標系->カメラ座標系へ変換"""
    height, width = depth.shape
    cam_coords = np.zeros((height, width, 4))
    u0, v0, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]

    for v in range(height):
        for u in range(width):
            x = (u - u0) * depth[v, u] / fx
            y = (v - v0) * depth[v, u] / fy
            z = depth[v, u]
            cam_coords[v][u][:3] = (x, y, z)
            cam_coords[v][u][3] = 1.0 # homogenous coordinate

    return cam_coords


def get_world2camera_transform(env:FlexEnv)->Tuple[np.ndarray,np.ndarray]:
    """
    点のワールド座標->カメラ座標への変換行列を取得
    
	Parameters
	----------
	env : FlexEnv
 		環境
	
	Returns
	-------
	Tuple[np.ndarray,np.ndarray]
		回転行列[4,4], 並進行列[4,4]
    """
    camera_name = env.camera_name
    cam_pos = env.camera_params[camera_name]['pos']
    cam_angle = env.camera_params[camera_name]['angle']

	# 回転行列
    matrix1 = get_rotation_matrix(-cam_angle[0], [0, 1, 0])
    matrix2 = get_rotation_matrix(-cam_angle[1] - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1
	# 並進行列
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = -np.array(cam_pos)

    return rotation_matrix, translation_matrix


def convert_cam_to_world(cam_coords:np.ndarray, rotation_matrix:np.ndarray, translation_matrix:np.ndarray):
    height, width = cam_coords.shape[:2]
    cam_coords = cam_coords.reshape((-1, 4)).transpose() # to [4, height x width]
    world_coords:np.ndarray = np.linalg.solve(rotation_matrix @ translation_matrix, cam_coords)
    world_coords = world_coords.transpose().reshape((height, width, 4))# to [height, width, 4]

    return world_coords

# def get_world_coords(rgb, depth, env):
# 	height, width, _ = rgb.shape
# 	K = intrinsic_from_fov(height, width, 45) # the fov is 90 degrees

# 	# Apply back-projection: K_inv @ pixels * depth
# 	cam_coords = np.ones((height, width, 4))
# 	u0 = K[0, 2]
# 	v0 = K[1, 2]
# 	fx = K[0, 0]
# 	fy = K[1, 1]
# 	# Loop through each pixel in the image
# 	for v in range(height):
# 		for u in range(width):
# 			# Apply equation in fig 3
# 			x = (u - u0) * depth[v, u] / fx
# 			y = (v - v0) * depth[v, u] / fy
# 			z = depth[v, u]
# 			cam_coords[v][u][:3] = (x, y, z)

# 	particle_pos = pyflex.get_positions().reshape((-1, 4))
# 	print('cloth pixels: ', np.count_nonzero(depth))
# 	print("cloth particle num: ", pyflex.get_n_particles())

# 	# debug: print camera coordinates
# 	# print(cam_coords.shape)
# 	# cnt = 0
# 	# for v in range(height):
# 	#     for u in range(width):
# 	#         if depth[v][u] > 0:
# 	#             print("v: {} u: {} cnt: {} cam_coord: {} approximate particle pos: {}".format(
# 	#                     v, u, cnt, cam_coords[v][u], particle_pos[cnt]))
# 	#             rgb = rgbd[:, :, :3].copy()
# 	#             rgb[v][u][0] = 255
# 	#             rgb[v][u][1] = 0
# 	#             rgb[v][u][2] = 0
# 	#             cv2.imshow('rgb', rgb[:, :, ::-1])
# 	#             cv2.waitKey()
# 	#             cnt += 1

# 	# from cam coord to world coord
# 	cam_x, cam_y, cam_z = env.camera_params['default_camera']['pos'][0], env.camera_params['default_camera']['pos'][1], env.camera_params['default_camera']['pos'][2]
# 	cam_x_angle, cam_y_angle, cam_z_angle = env.camera_params['default_camera']['angle'][0], env.camera_params['default_camera']['angle'][1], env.camera_params['default_camera']['angle'][2]

# 	# get rotation matrix: from world to camera
# 	matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0]) 
# 	# matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [np.cos(cam_x_angle), 0, np.sin(cam_x_angle)])
# 	matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
# 	rotation_matrix = matrix2 @ matrix1
	
# 	# get translation matrix: from world to camera
# 	translation_matrix = np.zeros((4, 4))
# 	translation_matrix[0][0] = 1
# 	translation_matrix[1][1] = 1
# 	translation_matrix[2][2] = 1
# 	translation_matrix[3][3] = 1
# 	translation_matrix[0][3] = - cam_x
# 	translation_matrix[1][3] = - cam_y
# 	translation_matrix[2][3] = - cam_z

# 	# debug: from world to camera
# 	cloth_x, cloth_y = env.current_config['ClothSize'][0], env.current_config['ClothSize'][1]
# 	# cnt = 0
# 	# for u in range(height):
# 	#     for v in range(width):
# 	#         if depth[u][v] > 0:
# 	#             world_coord = np.ones(4)
# 	#             world_coord[:3] = particle_pos[cnt][:3]
# 	#             convert_cam_coord =  rotation_matrix @ translation_matrix @ world_coord
# 	#             # convert_cam_coord =  translation_matrix  @ matrix2 @ matrix1 @ world_coord
# 	#             print("u {} v {} \n world coord {} \n convert camera coord {} \n real camera coord {}".format(
# 	#                 u, v, world_coord, convert_cam_coord, cam_coords[u][v]
# 	#             ))
# 	#             cnt += 1
# 	#             input('wait...')


# 	# convert the camera coordinate back to the world coordinate using the rotation and translation matrix
# 	cam_coords = cam_coords.reshape((-1, 4)).transpose() # 4 x (height x width)
# 	world_coords = np.linalg.inv(rotation_matrix @ translation_matrix) @ cam_coords # 4 x (height x width)
# 	world_coords = world_coords.transpose().reshape((height, width, 4))

# 	# roughly check the final world coordinate with the actual coordinate
# 	# firstu = 0
# 	# firstv = 0
# 	# for u in range(height):
# 	#     for v in range(width):
# 	#         if depth[u][v]:
# 	#             if u > firstu: # move to a new line
# 	#                 firstu = u
# 	#                 firstv = v

# 	#             cnt = (u - firstu) * cloth_x + (v - firstv)  
# 	#             print("u {} v {} cnt{}\nworld_coord\t{}\nparticle coord\t{}\nerror\t{}".format(
# 	#                 u, v, cnt, world_coords[u][v], particle_pos[cnt], np.linalg.norm( world_coords[u][v] - particle_pos[cnt])))
# 	#             rgb = rgbd[:, :, :3].copy()
# 	#             rgb[u][v][0] = 255
# 	#             rgb[u][v][1] = 0
# 	#             rgb[u][v][2] = 0
# 	#             cv2.imshow('rgb', rgb[:, :, ::-1])
# 	#             cv2.waitKey()
# 	# exit()
# 	return world_coords

def get_observable_particle_index(world_coords, particle_pos, rgb, depth):
	height, width, _ = rgb.shape
	# perform the matching of pixel particle to real particle
	observable_particle_idxes = []
	particle_pos = particle_pos[:, :3]
	for u in range(height):
		for v in range(width):
			if depth[u][v] > 0:
				estimated_world_coord = world_coords[u][v][:3]
				distance = np.linalg.norm(estimated_world_coord - particle_pos, axis=1)
				estimated_particle_idx = np.argmin(distance)
				# print("u {} v {} estimated particle idx {}".format(u, v, estimated_particle_idx))
				observable_particle_idxes.append(estimated_particle_idx)
				# rgb = rgbd[:, :, :3].copy()
				# rgb[u][v][0] = 255
				# rgb[u][v][1] = 0
				# rgb[u][v][2] = 0
				# cv2.imshow('chosen_idx', rgb[:, :, ::-1])
				# cv2.waitKey()
	# exit()
	return observable_particle_idxes