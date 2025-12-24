import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
from PIL import Image
import math
from transform_utils import *
import mplib
from utils import get_config

class SapienDataGenerator:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'config.yaml')
        self.urdf_path = get_config(config_path=config_path)['urdf_path']
        self.srdf_path = get_config(config_path=config_path)['srdf_path']
        engine = sapien.Engine()   # 创建引擎，负责管理物理仿真和渲染
        self.renderer = sapien.SapienRenderer()   # 创建渲染器
        engine.set_renderer(self.renderer)   # 将渲染器与引擎关联
        
        self.scene = engine.create_scene()   # 创建场景
        self.scene.set_timestep(1 / 100.0)   # 一秒进行100次仿真更新

        loader = self.scene.create_urdf_loader()   # 创建URDF加载器
        loader.fix_root_link = True     # 在创建机器人时固定root link
        
        asset = loader.load_kinematic(self.urdf_path)   # 加载URDF文件
        assert asset, 'URDF not loaded.'
        self.robot = asset  # 将加载的机器人资产赋值给self.robot
        
        self.add_light()
        self.add_camera()
        self.camera_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sensor')
        self.add_box()
        
        print(f"URDF 文件存在: {os.path.exists(self.urdf_path)}")
        print(f"SRDF 文件存在: {os.path.exists(self.srdf_path)}")
        
        # 初始化 mplib 的 Planner
        self.planner = mplib.Planner(
            urdf=self.urdf_path,
            srdf=self.srdf_path,
            move_group='panda_hand',
        )
        # 获取机器人的关节列表
        self.active_joints = self.robot.get_joints()
        
    def create_box(self, scene: sapien.Scene, pose: sapien.Pose, half_size, color=None, name='') -> sapien.Actor:
        half_size = np.array(half_size)
        builder: sapien.ActorBuilder = scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, color=color)
        box: sapien.Actor = builder.build(name=name)
        box.set_pose(pose)
        return box
    
    def add_box(self):
        self.scene.add_ground(altitude=0)
        self.box1 = self.create_box(
            self.scene,
            sapien.Pose(p=[0.3, -0.3, 0.02]),
            half_size=[0.02, 0.02, 0.02],
            color=[1., 0., 0.],
            name='box1',
        ) 
        self.box2 = self.create_box(
            self.scene,
            sapien.Pose(p=[0.3, 0.3, 0.04]),
            half_size=[0.04, 0.04, 0.04],
            color=[0., 1., 0.],
            name='box2',
        ) 
        
    def add_light(self):
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
        
    def add_camera(self):
        near, far = 0.1, 100
        width, height = 640, 480
        self.camera = self.scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(35),
            near=near,
            far=far,
        )
        self.camera.set_pose(sapien.Pose(p=[0.7, 0, 2], q=euler2quat(np.array([0, -np.pi/2, 0]))))
        
    def get_camera_data(self):
        self.scene.step()
        self.scene.update_render()
        self.camera.take_picture()
        
        rgba = self.camera.get_float_texture('Color')
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        rgb_img = rgba_img[:, :, :3]
        rgb_pil = Image.fromarray(rgb_img)
        rgb_pil.save(os.path.join(self.camera_data_path, 'color.png'))  
        
        position = self.camera.get_float_texture('Position')
        points_opengl = position[..., :3][position[..., 3] < 1]
        points_color = rgba[position[..., 3] < 1][..., :3]
        model_matrix = self.camera.get_model_matrix()
        points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
        points_world_reshaped = points_world.reshape(480, 640, 3)
        np.save(os.path.join(self.camera_data_path, 'points_world.npy'), points_world_reshaped)
        
        seg_labels = self.camera.get_uint32_texture("Segmentation")
        actor_masks = seg_labels[..., 1]
        np.save(os.path.join(self.camera_data_path, 'actor_mask.npy'), actor_masks)

        depth = -position[..., 2]
        depth_image = (depth * 1000.0).astype(np.uint16)
        depth_pil = Image.fromarray(depth_image)
        depth_pil.save(os.path.join(self.camera_data_path, 'depth.png'))
    
    def view_window(self):
        viewer = Viewer(self.renderer)
        viewer.set_scene(self.scene)
         # 设置相机的位置（稍微提高相机高度）
        viewer.set_camera_xyz(x=1.2, y=0, z=1)  # 提高相机的位置，确保能看到物块和机械臂
        viewer.set_camera_rpy(r=0, p=-np.pi / 6, y=np.pi)  # 调整俯视角度
        # viewer.set_camera_xyz(x=0.7, y=0, z=0.7)
        # viewer.set_camera_rpy(r=0, p=-np.pi/6, y=-np.pi/2)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
        while not viewer.closed:
            self.scene.step()
            self.scene.update_render()
            viewer.render()


if __name__ == "__main__":
    sapien_data_generator = SapienDataGenerator()
    # print(sapien_data_generator.active_joints)
    robot = sapien_data_generator.robot
    # robot.set_qpos(np.concatenate([[0,0,0,0,0,0,0],[0.03, 0.02]]))
    # robot.set_qpos(np.array([0.    ,  0.    ,  0.    , -0.0698,  0.    ,  0.    ,  0.    ,
    #     0.03  ,  0.02]))
    # ee_link = robot.get_links()[-1]
    # ee_pose = ee_link.get_pose()
    # print(ee_pose)
    
    # ee_link = robot.get_links()[9]
    # ee_pose = ee_link.get_pose()
    # print(ee_pose)
    
    links = robot.get_links()
    print(len(links))
    count = 0
    for link in links:
        if link.get_name() == "panda_hand":
            ee_link = link
            break
        count += 1
    ee_pose = ee_link.get_pose()
    print(ee_pose)
    print(count)
    
    print(robot.get_qpos())

    
    sapien_data_generator.view_window()
