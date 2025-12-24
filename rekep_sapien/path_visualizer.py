import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
import time
import os
from mplib.planner import Planner
from mplib.pymp import Pose

class PathVisualizer:
    def __init__(self):
        # 初始化SAPIEN引擎和渲染器
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        
        # 创建场景
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(1 / 1000.0)
        
        # 添加灯光
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        
        # 加载机器人
        loader = self.scene.create_urdf_loader()
        urdf_path = "data/panda/panda.urdf"
        self.robot = loader.load(urdf_path)
        self.robot.name = "panda"
        
        # 获取机器人关节
        self.joints = self.robot.get_joints()
        self.active_joints = [joint for joint in self.joints if joint.get_dof() > 0]
        
        # 初始化planner
        self.planner = Planner(
            urdf=urdf_path,
            move_group="panda_hand",
            verbose=False
        )
        
        # 添加场景物体
        self.scene.add_ground(altitude=0)
        
        # 创建红色方块
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.02], color=[1., 0., 0.])
        self.box1 = builder.build(name='box1')
        self.box1.set_pose(sapien.Pose(p=[0.3, -0.3, 0.02]))
        
        # 创建绿色方块
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.04, 0.04, 0.04])
        builder.add_box_visual(half_size=[0.04, 0.04, 0.04], color=[0., 1., 0.])
        self.box2 = builder.build(name='box2')
        self.box2.set_pose(sapien.Pose(p=[0.3, 0.3, 0.04]))
        
        # 创建查看器
        self.viewer = Viewer(self.renderer)
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_xyz(x=1.0, y=0.0, z=1.0)
        self.viewer.set_camera_rpy(r=0, p=-0.7, y=3.14)
        
        # 初始化路径变量
        self.path = None
        self.planned_path = None  # 存储规划好的路径
        self.current_step = 0
        self.loop = True
        
        print("PathVisualizer初始化完成")
    
    def set_path(self, path):
        """设置要跟踪的路径"""
        self.path = path
        self.current_step = 0
        self.planned_path = None
        print(f"已设置路径，共{len(path)}个路径点")
    
    def plan_full_path(self):
        """规划完整的路径"""
        if self.path is None or len(self.path) == 0:
            print("没有设置路径，无法规划")
            return False
        
        print("开始规划路径...")
        
        # 获取当前机器人状态
        current_qpos = self.robot.get_qpos()
        
        # 获取planner中move_group的关节索引
        move_group_indices = self.planner.move_group_joint_indices
        print(f"Planner的move_group关节索引: {move_group_indices}")
        
        # 完整的关节数量
        full_qpos_len = len(current_qpos)
        print(f"机器人完整关节数量: {full_qpos_len}")
        
        # 用于规划的关节位置 - 根据move_group_joint_indices选择
        planning_qpos = np.zeros(len(move_group_indices))
        for i, joint_idx in enumerate(move_group_indices):
            if joint_idx < full_qpos_len:
                planning_qpos[i] = current_qpos[joint_idx]
        
        print(f"用于规划的关节位置维度: {planning_qpos.shape}")
        
        # 存储所有子路径
        all_subpaths = []
        
        # 为每个路径点规划轨迹
        for i, target_pose in enumerate(self.path):
            # 创建mplib的Pose对象
            pos = target_pose[:3]
            quat = target_pose[3:7] if len(target_pose) >= 7 else np.array([0, 0, 0, 1])
            pose_obj = Pose(p=pos, q=quat)
            
            # 使用RRT规划
            result = self.planner.plan_pose(
                pose_obj, 
                planning_qpos,          # 使用与move_group对应的关节位置
                time_step=0.02,         # 时间步长
                planning_time=1.0,      # 规划时间限制
                simplify=True,          # 简化路径
                wrt_world=True          # 相对于世界坐标系
            )
            
            # 检查规划结果
            if result["status"] == "Success":
                print(f"路径点 {i+1}/{len(self.path)}: 规划成功")
                
                # 扩展规划结果至完整关节数
                if "position" in result:
                    # 创建新的position和velocity数组
                    planned_pos_len = result["position"].shape[0]  # 路径点数量
                    planned_joints = result["position"].shape[1]   # 规划的关节数量
                    
                    # 创建完整维度的数组
                    full_positions = np.zeros((planned_pos_len, full_qpos_len))
                    full_velocities = np.zeros((planned_pos_len, full_qpos_len))
                    
                    # 填充规划数据到对应的关节位置
                    for j in range(planned_pos_len):
                        for k, joint_idx in enumerate(move_group_indices):
                            if k < planned_joints and joint_idx < full_qpos_len:
                                full_positions[j, joint_idx] = result["position"][j, k]
                                full_velocities[j, joint_idx] = result["velocity"][j, k]
                    
                    # 保存原始数据，用于下次规划
                    result["original_position"] = result["position"].copy()
                    
                    # 替换为完整维度的数据
                    result["position"] = full_positions
                    result["velocity"] = full_velocities
                
                all_subpaths.append(result)
                
                # 更新当前关节位置为轨迹终点
                if "original_position" in result:
                    planning_qpos = result["original_position"][-1]
                else:
                    # 确保维度匹配
                    if result["position"].shape[1] == len(planning_qpos):
                        planning_qpos = result["position"][-1]
                    else:
                        # 从完整关节位置中提取对应于move_group的关节位置
                        for k, joint_idx in enumerate(move_group_indices):
                            if joint_idx < full_qpos_len and k < len(planning_qpos):
                                planning_qpos[k] = result["position"][-1][joint_idx]
            else:
                print(f"路径点 {i+1}/{len(self.path)}: 规划失败: {result['status']}")
                if i == 0:  # 第一个点失败直接返回
                    return False
                break  # 否则使用已规划的路径
        
        if not all_subpaths:
            print("路径规划失败")
            return False
        
        # 存储规划好的路径
        self.planned_path = all_subpaths
        print(f"路径规划完成，共{len(all_subpaths)}段轨迹")
        return True
    
    def reset_robot(self):
        """重置机器人到初始姿态"""
        reset_pos = [0.0, -0.3, 0.0, -2.0, 0.0, 2.0, 0.0, 0.0, 0.0]
        for i, joint in enumerate(self.active_joints):
            if i < len(reset_pos):
                joint.set_drive_target(reset_pos[i])
                joint.set_drive_property(stiffness=1000, damping=100)
        
        # 执行几步仿真使机器人回到初始位置
        for _ in range(100):
            self.scene.step()
        
        # 重置路径执行进度
        self.current_step = 0
        self.planned_path = None
    
    def execute_step(self):
        """执行一步路径跟踪"""
        # 如果没有规划好的路径，先规划
        if self.planned_path is None:
            if not self.plan_full_path():
                return False
        
        # 检查是否执行完所有子路径
        if self.current_step >= len(self.planned_path):
            if self.loop:
                print("路径完成，重新开始")
                self.reset_robot()
                if not self.plan_full_path():
                    return False
                return True
            else:
                print("路径完成")
                return False
        
        # 获取当前子路径
        current_subpath = self.planned_path[self.current_step]
        
        # 获取子路径中的所有路点
        positions = current_subpath["position"]
        velocities = current_subpath["velocity"]
        
        # 执行子路径中的每个路点
        for i in range(positions.shape[0]):
            # 设置关节目标位置和速度
            for j, joint in enumerate(self.active_joints):
                if j < positions.shape[1]:  # 只设置包含在路径中的关节
                    joint.set_drive_target(positions[i][j])
                    joint.set_drive_velocity_target(velocities[i][j])
                    joint.set_drive_property(stiffness=1000, damping=100)
            
            # 执行仿真步骤
            self.scene.step()
            
            # 每4步更新一次渲染
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()
        
        # 最后一次渲染确保显示最终状态
        self.scene.update_render()
        self.viewer.render()
        
        # 当前子路径执行完毕，移动到下一个
        self.current_step += 1
        return True
    
    def run_visualization(self, path=None):
        """运行可视化"""
        if path is not None:
            self.set_path(path)
        
        if self.path is None:
            print("请先设置路径")
            return
        
        print("开始可视化路径，按ESC退出...")
        
        # 重置机器人
        self.reset_robot()
        
        # 运行循环
        while not self.viewer.closed:
            # 执行一步动作
            success = self.execute_step()
            
            # 如果路径已完成且不循环，则退出
            if not success and not self.loop:
                break
        
        print("可视化结束")

def visualize_path(path, loop=True):
    """从外部调用的主函数"""
    visualizer = PathVisualizer()
    visualizer.loop = loop
    visualizer.run_visualization(path)

if __name__ == "__main__":
    # 测试代码：创建一个简单的圆形路径
    t = np.linspace(0, 2*np.pi, 10)  # 减少路径点数量
    positions = np.zeros((len(t), 3))
    positions[:, 0] = 0.5 + 0.2 * np.cos(t)  # x坐标
    positions[:, 1] = 0.2 * np.sin(t)        # y坐标
    positions[:, 2] = 0.5                    # z坐标固定
    
    # 始终朝下的方向
    orientations = np.zeros((len(t), 4))
    orientations[:, 0] = 0  # qx
    orientations[:, 1] = 1  # qy
    orientations[:, 2] = 0  # qz
    orientations[:, 3] = 0  # qw
    
    # 组合位置和方向
    path = np.concatenate([positions, orientations], axis=1)
    
    # 运行可视化
    visualize_path(path) 