import gymnasium as gym
import numpy as np
from transforms3d.quaternions import quat2mat

import mani_skill2_real2sim.envs


def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    assert H.shape == (4, 4), H.shape
    assert pts.ndim == 2 and pts.shape[1] == 3, pts.shape
    return pts @ H[:3, :3].T + H[:3, 3]


def invert_transform(H: np.ndarray):
    assert H.shape[-2:] == (4, 4), H.shape
    H_inv = H.copy()
    R_T = np.swapaxes(H[..., :3, :3], -1, -2)
    H_inv[..., :3, :3] = R_T
    H_inv[..., :3, 3:] = -R_T @ H[..., :3, 3:]
    return H_inv


def get_pointcloud_in_camera(obs, camera_name):
    camera_param = obs["camera_param"][camera_name]
    depth = obs["image"][camera_name]["depth"]
    intrinsic_cv = camera_param["intrinsic_cv"]
    extrinsic_cv = camera_param["extrinsic_cv"]
    # cam2world_gl = camera_param['cam2world_gl']
    h, w = depth.shape[:2]
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    zz = depth.flatten()
    xx = (xx - intrinsic_cv[0, 2]) / intrinsic_cv[0, 0] * zz
    yy = (yy - intrinsic_cv[1, 2]) / intrinsic_cv[1, 1] * zz
    ones = np.ones_like(zz)
    points = np.stack([xx, yy, zz, ones], axis=0)  # [3, N]
    points_world = invert_transform(extrinsic_cv) @ points
    # Convert to base frame
    base_pose = obs["agent"]["base_pose"]  # [p, q]
    base_T = np.eye(4)
    base_T[:3, :3] = quat2mat(base_pose[3:])
    base_T[:3, 3] = base_pose[:3]
    points_base = invert_transform(base_T) @ points_world
    points_rgb = obs["image"][camera_name]["rgb"].reshape(-1, 3)
    # import matplotlib.pyplot as plt
    # plt.imshow(obs['image'][camera_name]['rgb'])
    # plt.show()
    return points[:3].T, points_base[:3].T, points_rgb


def main():
    env_id = "GraspSingleOpenedCokeCanInScene-v0"
    # env_id = "PutSpoonOnTableClothInScene-v0"
    # control_mode = "arm_pd_joint_pos_gripper_pd_joint_pos"
    control_mode = "arm_pd_ee_pose_gripper_pd_joint_pos"
    env = gym.make(
        env_id, obs_mode="rgbd", control_mode=control_mode, prepackaged_config=True
    )
    # Get the control freq which can be used for time parameterization in motion planning
    print(env.control_freq, env.sim_freq)
    # Get joint names for mapping if necessary
    joint_names = [joint.name for joint in env.agent.controller.joints]
    print(joint_names)

    # print(env.observation_space)
    # Dict('agent': Dict('qpos': Box(-inf, inf, (11,), float32), 'qvel': Box(-inf, inf, (11,), float32), 'controller': Dict('gripper': Dict('target_qpos': Box(-inf, inf, (2,), float32))), 'base_pose': Box(-inf, inf, (7,), float32)), 'extra': Dict('tcp_pose': Box(-inf, inf, (7,), float32)), 'camera_param': Dict('base_camera': Dict('extrinsic_cv': Box(-inf, inf, (4, 4), float32), 'cam2world_gl': Box(-inf, inf, (4, 4), float32), 'intrinsic_cv': Box(-inf, inf, (3, 3), float32)), 'overhead_camera': Dict('extrinsic_cv': Box(-inf, inf, (4, 4), float32), 'cam2world_gl': Box(-inf, inf, (4, 4), float32), 'intrinsic_cv': Box(-inf, inf, (3, 3), float32))), 'image': Dict('base_camera': Dict('rgb': Box(0, 255, (128, 128, 3), uint8), 'depth': Box(0.0, inf, (128, 128, 1), float32), 'Segmentation': Box(0, 4294967295, (128, 128, 4), uint32)), 'overhead_camera': Dict('rgb': Box(0, 255, (512, 640, 3), uint8), 'depth': Box(0.0, inf, (512, 640, 1), float32), 'Segmentation': Box(0, 4294967295, (512, 640, 4), uint32))))

    obs, reset_info = env.reset()

    # Example: how to get point clouds

    if "GoogleRobot" in env.agent.__class__.__name__:
        camera_name = "overhead_camera"
    elif "WidowX" in env.agent.__class__.__name__:
        camera_name = "3rd_view_camera"
    else:
        raise NotImplementedError(env.agent.__class__.__name__)
        
    points_camera, points_base, points_rgb = get_pointcloud_in_camera(
        obs, camera_name
    )
    # Uncomment to Visualize pointclouds (pyglet<2 is needed)
    # import trimesh
    # trimesh.points.PointCloud(points_base, points_rgb).show()
    # trimesh.points.PointCloud(points_camera, points_rgb).show()
    # NOTE: Since we green-screen the background, the depth is not matching the RGB image.

    # You can also get scene point cloud for motion planning
    scene_points = env.gen_scene_pcd(seed=0)  # The default number of points is 1e5. The object surface might not be sampled.
    # import trimesh
    # trimesh.points.PointCloud(scene_points).show()
    # # It may include the whole large scene, so you may want to filter out the points outside the table.
    # base_pose = obs["agent"]["base_pose"]
    # base_T = np.eye(4)
    # base_T[:3, :3] = quat2mat(base_pose[3:])
    # base_T[:3, 3] = base_pose[:3]
    # scene_points_base = transform_points(invert_transform(base_T), scene_points)
    # mask = (scene_points_base[:, 0] > -0.5) & (scene_points_base[:, 0] < 1) & (scene_points_base[:, 1] > -0.5) & (scene_points_base[:, 1] < 0.5) & (scene_points_base[:, 2] > -0.1) & (scene_points_base[:, 2] < 2)
    # scene_points_near_robot = scene_points_base[mask]
    # trimesh.points.PointCloud(scene_points_near_robot).show()

    # Random take action
    # env.render_human()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, truncated, done, info = env.step(action)
        env.render_human()
        # import matplotlib.pyplot as plt
        # plt.imshow(obs['image']['overhead_camera']['rgb'])
        # plt.show()

    # import pdb
    # pdb.set_trace()


if __name__ == "__main__":
    main()
