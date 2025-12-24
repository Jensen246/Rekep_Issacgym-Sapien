# OpenGL (Open Graphics Library)
#   跨平台的图形编程接口(API)，用于渲染2D和3D矢量图形
#   在SAPIEN中，OpenGL用于渲染3D场景
# Blender (3D modeling and animation software)
#   3D建模和动画软件，广泛用于游戏开发、电影制作和3D打印
#   在SAPIEN中，Blender用于渲染3D场景
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
from PIL import Image
import os
import math
from transform_utils import euler2quat

class SapienDataGenerator:
    def __init__(self):
        engine = sapien.Engine()   # 创建引擎，负责管理物理仿真和渲染
        self.renderer = sapien.SapienRenderer()   # 创建渲染器
        engine.set_renderer(self.renderer)   # 将渲染器与引擎关联
        
        self.scene = engine.create_scene()   # 创建场景
        self.scene.set_timestep(1 / 100.0)   # 一秒进行100次仿真更新

        loader = self.scene.create_urdf_loader()   # 创建URDF加载器
        loader.fix_root_link = True     # 在创建机器人时固定root link
        
        urdf_path = "data/panda/panda.urdf"
        asset = loader.load_kinematic(urdf_path)   # 加载URDF文件
        assert asset, 'URDF not loaded.'
        
        self.add_light()
        self.add_camera()
        self.camara_data_path = './data/sensor'
        self.add_box()
    
    def create_box(
        self,
        scene: sapien.Scene,
        pose: sapien.Pose,
        half_size,
        color=None,
        name=''
    ) -> sapien.Actor:
        """Create a box.

        Args:
            scene: sapien.Scene to create a box.
            pose: 6D pose of the box.
            half_size: [3], half size along x, y, z axes.
            color: [3] or [4], rgb or rgba
            name: name of the actor.

        Returns:
            sapien.Actor
        """
        half_size = np.array(half_size)
        builder: sapien.ActorBuilder = scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)  # Add collision shape
        builder.add_box_visual(half_size=half_size, color=color)  # Add visual shape
        box: sapien.Actor = builder.build(name=name)
        # Or you can set_name after building the actor
        # box.set_name(name)
        box.set_pose(pose)
        return box
    
    def add_box(self):
        self.scene.add_ground(altitude=0)  # The ground is in fact a special actor.
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
        self.scene.set_ambient_light([0.5, 0.5, 0.5])    # 设置环境光，参数为RGB值，即中等强度的白色环境光
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)   # 添加方向光，参数为方向向量和颜色，阴影为True
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True) # 添加点光源
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True) 
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)  
        
    def add_camera(self):
        near, far = 0.1, 100    # 相机的近平面和远平面，即相机能看到最近的距离和最远的距离
        width, height = 640, 480    # 相机的宽度（像素）和高度（像素），乘积即分辨率
        self.camera = self.scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(35),    # 垂直视场角Field of View, 除了fovy还有fovx
            near=near,
            far=far,
        )
        self.camera.set_pose(sapien.Pose(p=[0.7, 0, 2], q=euler2quat(np.array([0, -np.pi/2, 0]))))    # 设置相机的位置
        
        # print('Camara 1 Intrinsic matrix\n', self.camera.get_intrinsic_matrix())
        
    def get_camera_data(self):
        # 获取相机数据
        self.scene.step()
        self.scene.update_render()
        self.camera.take_picture()
        
        # 获取RGB图像
        rgba = self.camera.get_float_texture('Color')   # [H,W,4]
        # 也可以用 rgba = camera.get_color_rgba() 
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        rgb_img = rgba_img[:, :, :3]  # 只保留前三个通道(RGB)
        rgb_pil = Image.fromarray(rgb_img)
        rgb_pil.save(os.path.join(self.camara_data_path, 'color.png'))  
        
        # 获取点云
        # 每个像素都是相机坐标系下的 (x, y, z, render_depth)
        position = self.camera.get_float_texture('Position')  # [H, W, 4]
        # OpenGL/Blender: y up and -z forward
        points_opengl = position[..., :3][position[..., 3] < 1]
        points_color = rgba[position[..., 3] < 1][..., :3]
        # Model matrix 是OpenGL相机坐标系到SAPIEN世界坐标系的变换矩阵
        # scene.update_render() 后必须调用 camera.get_model_matrix()
        model_matrix = self.camera.get_model_matrix()
        points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
        # 将一维点云数据重塑为图像尺寸 640x480x3
        points_world_reshaped = points_world.reshape(480, 640, 3)
        # 将点云数据保存为npy文件
        np.save(os.path.join(self.camara_data_path, 'points_world.npy'), points_world_reshaped)
        
        # 获取物体掩码（mask）
        # 获取分割ID图像
        seg_labels = self.camera.get_uint32_texture("Segmentation")  # [H, W]
        # 只提取actor-level的分割ID（索引1）并保持uint32格式
        actor_masks = seg_labels[..., 1]  # 保持uint32格式
        # 保存actor-level掩码为npy文件
        np.save(os.path.join(self.camara_data_path, 'actor_mask.npy'), actor_masks)

        # 获取深度图像        
        depth = -position[..., 2]
        depth_image = (depth * 1000.0).astype(np.uint16)
        depth_pil = Image.fromarray(depth_image)
        depth_pil.save(os.path.join(self.camara_data_path, 'depth.png'))
    
    def view_window(self):
        viewer = Viewer(self.renderer)
        viewer.set_scene(self.scene)
        viewer.set_camera_xyz(x=0, y=0, z=3)
        viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 2), y=0)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
        while not viewer.closed:
            sapien_data_generator.scene.step()
            sapien_data_generator.scene.update_render()
            viewer.render()

if __name__ == "__main__":
    sapien_data_generator = SapienDataGenerator()
    
    # 查看可视化窗口
    sapien_data_generator.view_window()
    sapien_data_generator.get_camera_data()
