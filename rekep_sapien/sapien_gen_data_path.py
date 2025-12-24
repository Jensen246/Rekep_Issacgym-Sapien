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
import mplib
from utils import get_config

class SapienDataGenerator:
    def __init__(self):
        self.urdf_path = get_config(config_path='./configs/config.yaml')['urdf_path']
        self.srdf_path = get_config(config_path='./configs/config.yaml')['srdf_path']
        engine = sapien.Engine()   # 创建引擎，负责管理物理仿真和渲染
        self.renderer = sapien.SapienRenderer()   # 创建渲染器
        engine.set_renderer(self.renderer)   # 将渲染器与引擎关联
        
        # 创建场景配置
        scene_config = sapien.SceneConfig()
        self.scene = engine.create_scene(scene_config)   # 创建场景
        self.scene.set_timestep(1 / 1000.0)   # 一秒进行100次仿真更新
        
        # 设置默认物理材质，增加摩擦力
        self.scene.default_physical_material = self.scene.create_physical_material(5.0, 5.0, 0.0)

        loader = self.scene.create_urdf_loader()   # 创建URDF加载器
        loader.fix_root_link = True     # 在创建机器人时固定root link
        
        # 加载机器人 - 使用运动学控制，这是适合机械臂的方式
        asset = loader.load_kinematic(self.urdf_path)   # 加载URDF文件为运动学控制
        assert asset, 'URDF not loaded.'
        self.robot = asset  # 将加载的机器人资产赋值给self.robot
        
        # 设置夹爪关节驱动属性
        self.finger_joints = []
        for joint in self.robot.get_joints():
            if joint.get_name() in ['panda_finger_joint1', 'panda_finger_joint2']:
                self.finger_joints.append(joint)
        
        self.add_light()
        self.add_camera()
        self.camera_data_path = './data/sensor'
        self.add_box()
        
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
        
        # 创建高摩擦力物理材质 - 增加摩擦系数
        material = scene.create_physical_material(5.0, 5.0, 0.0)
        # material = scene.create_physical_material(1.0, 1.0, 0.0)
        
        builder.add_box_collision(half_size=half_size, material=material)
        builder.add_box_visual(half_size=half_size, color=color)
        
        # 使用build_kinematic创建运动学物体，这样更容易被夹取
        if name == 'box1' or name == 'box2':
            box: sapien.Actor = builder.build(name=name)  # 动态物体可以被抓取
        else:
            box: sapien.Actor = builder.build_kinematic(name=name)  # 静态物体不会移动
            
        box.set_pose(pose)
        return box
    
    def add_box(self):
        self.scene.add_ground(altitude=0)
        
        # 创建一个固定的基座
        base_material = self.scene.create_physical_material(5.0, 5.0, 0.0)
        base_builder = self.scene.create_actor_builder()
        base_builder.add_box_collision(half_size=[0.5, 0.5, 0.01], material=base_material)
        base_builder.add_box_visual(half_size=[0.5, 0.5, 0.01], color=[0.8, 0.8, 0.8])
        base = base_builder.build_kinematic(name="base")
        base.set_pose(sapien.Pose([0.4, 0, 0]))
        
        # 创建两个可移动的盒子
        self.box1 = self.create_box(
            self.scene,
            sapien.Pose(p=[0.3, -0.3, 0.02]),
            half_size=[0.01, 0.01, 0.01],
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
        viewer.set_camera_xyz(x=0.9, y=0, z=0.4)  # 提高相机的位置，确保能看到物块和机械臂
        viewer.set_camera_rpy(r=0, p=-np.pi / 10, y=np.pi)  # 调整俯视角度
        # viewer.set_camera_xyz(x=0.7, y=0, z=0.7)
        # viewer.set_camera_rpy(r=0, p=-np.pi/6, y=-np.pi/2)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
        while not viewer.closed:
            self.scene.step()
            self.scene.update_render()
            viewer.render()
    
    def follow_path(self, result, viewer=None, is_grasp_stage=False, is_release_stage=False):
        """执行规划的路径，可选择使用viewer进行可视化
        
        Args:
            result: 规划结果，包含position字段，即关节角度序列
            viewer: 可选的Viewer对象，用于可视化执行过程
        """
        n_step = result["position"].shape[0]
        
        # 决定整个路径过程中使用的夹爪状态
        current_qpos = self.robot.get_qpos()
        gripper_state = current_qpos[-2:].copy()  # 默认保持当前夹爪状态
        
        # 根据阶段设置夹爪状态
        if is_grasp_stage:
            # 在抓取阶段，夹爪应始终保持打开直到路径结束
            gripper_state = np.array([1.0, 1.0])
        # 在释放阶段和普通移动阶段，保持当前夹爪状态（通常是闭合的）
        
        # 执行路径
        for i in range(n_step):
            qpos = result["position"][i]
            # 设置机械臂关节位置（不包括夹爪）
            arm_qpos = qpos
            current_qpos = self.robot.get_qpos()
            new_qpos = current_qpos.copy()
            new_qpos[:len(arm_qpos)] = arm_qpos
            
            # 设置夹爪状态，在整个路径执行过程中保持不变
            new_qpos[-2:] = gripper_state
                
            self.robot.set_qpos(new_qpos)
            
            self.scene.step()
            self.scene.update_render()
            if viewer is not None:
                viewer.render()
        
        # # 只在阶段结束时改变夹爪状态
        # if is_grasp_stage:
        #     # 抓取阶段结束时闭合夹爪
        #     self.set_gripper(0, viewer, steps=100)  # 0表示完全闭合
        
        # if is_release_stage:
        #     # 释放阶段结束时打开夹爪
        #     self.set_gripper(1, viewer, steps=50)  # 1表示完全打开
    
    def plan_and_execute_path(self, end_effector_path, visualize=True, viewer=None,
                              is_grasp_stage=False, is_release_stage=False):
        """根据末端执行器路径规划并执行机器人运动
        
        Args:
            end_effector_path: 末端执行器路径点的列表，每个点是一个[x,y,z,qx,qy,qz,qw]数组或相应的位姿表示，而在mplib中，位姿表示为[x,y,z,qw,qx,qy,qz]，需要转换
            visualize: 是否可视化执行过程
        
        Returns:
            bool: 执行是否成功
        """
        # 将位姿转换为mplib需要的格式，即[x,y,z,qx,qy,qz,qw]变为[x,y,z,qw,qx,qy,qz]
        end_effector_path = [np.concatenate([pose[:3], [pose[6]], pose[3:6]]) for pose in end_effector_path]
        success = True
        # 获取当前关节位置，使用robot的get_qpos方法
        current_qpos = self.robot.get_qpos()
        
        # 检查planner需要的关节数量
        # 通常mplib中如果move_group是panda_hand，会需要9个关节(7个机械臂关节+2个手爪关节)
        expected_joint_num = 9  # 通常Panda机器人有7个关节加上2个手爪关节
        
        # 确保current_qpos的长度正确
        if len(current_qpos) != expected_joint_num:
            # 如果长度不一致，调整current_qpos的长度
            # 如果当前值少于期望值，补充0；如果多于期望值，截断
            if len(current_qpos) < expected_joint_num:
                current_qpos = np.concatenate([current_qpos, np.zeros(expected_joint_num - len(current_qpos))])
            else:
                current_qpos = current_qpos[:expected_joint_num]
        
        for target_pose in end_effector_path:
            # 如果输入是位置和四元数数组，转换为sapien.Pose对象
            if isinstance(target_pose, (list, np.ndarray)) and len(target_pose) >= 7:
                pose = sapien.Pose(p=target_pose[:3], q=target_pose[3:7])
            elif isinstance(target_pose, sapien.Pose):
                pose = target_pose
            else:
                print(f"不支持的位姿格式: {type(target_pose)}")
                success = False
                continue
            
            # 规划到目标位姿的路径
            result = self.planner.plan_pose(
                goal_pose=pose,
                current_qpos=current_qpos,
                time_step=1 / 100.0
            )
            
            if result['status'] != "Success":
                print(f"规划失败: {result['status']}")
                success = False
                continue
            
            # 执行规划的路径
            self.follow_path(result, viewer, is_grasp_stage, is_release_stage)
            
            
            # 更新当前关节位置为最后一个规划点
            if len(result["position"]) > 0:
                # 确保更新后的关节位置长度也是正确的
                last_pos = result["position"][-1]
                if len(last_pos) != expected_joint_num:
                    if len(last_pos) < expected_joint_num:
                        last_pos = np.concatenate([last_pos, np.zeros(expected_joint_num - len(last_pos))])
                    else:
                        last_pos = last_pos[:expected_joint_num]
                current_qpos = last_pos
        
        if is_grasp_stage:
            self.set_gripper(0, viewer, steps=100)  # 0表示完全闭合
        if is_release_stage:
            self.set_gripper(1, viewer, steps=50)  # 1表示完全打开
            
        # if visualize:
        #     while not viewer.closed:
        #         self.scene.step()
        #         self.scene.update_render()
        #         viewer.render()
        
        return success

    def plan_and_execute_path_from_stage_result_list(self, stage_result_list):
        """根据stage_result_list中的关节位置和末端执行器路径规划并执行机器人运动"""
        viewer = None

        # 创建Viewer进行可视化
        viewer = Viewer(self.renderer)
        viewer.set_scene(self.scene)
        viewer.set_camera_xyz(x=0.9, y=0, z=0.4)  # 位置更近更低
        viewer.set_camera_rpy(r=0, p=-np.pi/10, y=np.pi)  # 调整俯角以更好地看到场景
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
        for stage_result in stage_result_list:
            self.plan_and_execute_path(stage_result['path'], visualize=True, viewer=viewer,
                                        is_grasp_stage=stage_result['is_grasp_stage'],
                                        is_release_stage=stage_result['is_release_stage'])
        while not viewer.closed:
            self.scene.step()
            self.scene.update_render()
            viewer.render()
            
    def set_gripper(self, position, viewer=None, steps=100):
        """控制夹爪开合
        
        Args:
            position: 夹爪位置，0表示闭合，1表示打开
            viewer: 可选的Viewer对象，用于可视化执行过程
            steps: 执行步骤数，越大动作越平滑
        """
        # 获取当前机器人关节位置
        current_qpos = self.robot.get_qpos()
        
        # 分步执行夹爪开合，使动作更平滑
        for i in range(steps):
            # 更新夹爪位置 - 线性插值，让动作更平滑
            current_pos = current_qpos[-2]
            target_pos = position
            # 逐渐移动到目标位置
            t = (i + 1) / steps
            gripper_pos = current_pos * (1 - t) + target_pos * t
            
            # 更新最后两个关节位置（夹爪关节）
            new_qpos = current_qpos.copy()
            new_qpos[-2:] = [gripper_pos, gripper_pos]
            self.robot.set_qpos(new_qpos)
            
            # 执行仿真步
            self.scene.step()
            self.scene.update_render()
            if viewer is not None:
                viewer.render()
        
        # 如果是闭合操作，多执行几次确保抓紧
        if position == 0:  # 闭合状态
            for _ in range(3):  # 多次尝试闭合
                # 执行更多步骤以确保抓紧
                for _ in range(20):
                    new_qpos = self.robot.get_qpos()
                    new_qpos[-2:] = [0, 0]  # 完全闭合
                    self.robot.set_qpos(new_qpos)
                    self.scene.step()
                    self.scene.update_render()
                    if viewer is not None:
                        viewer.render()

if __name__ == "__main__":
    sapien_data_generator = SapienDataGenerator()
    sapien_data_generator.view_window()
    sapien_data_generator.get_camera_data()
    
 
