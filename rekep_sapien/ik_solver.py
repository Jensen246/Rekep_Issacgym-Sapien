"""
使用mplib实现的IK求解器，替代原Lula IK solver
"""
from mplib import Planner
import numpy as np
from mplib.pymp import Pose
import transform_utils as T
import os
class IKSolver:
    def __init__(self, reset_joint_pos, world2robot_homo):
        self.urdf_path = "./data/panda/panda.urdf"
        self.srdf_path = "./data/panda/panda.srdf"
        self.planner = Planner(
            urdf=self.urdf_path,
            srdf=self.srdf_path if os.path.exists(self.srdf_path) else None,
            move_group="panda_hand",  # 使用机器人手掌作为末端执行器
            verbose=False,
        )
        # 定义IK掩码：0表示需要IK求解，1表示固定（不参与求解）
        # Panda机器人有7个旋转关节+2个夹爪关节，设置前7个为求解对象
        self.mask = [0, 0, 0, 0, 0, 0, 0, 1, 1]  # 前7个关节求解，夹爪不求解
        self.world2robot_homo = world2robot_homo
        self.reset_joint_pos = reset_joint_pos
    
    def solve(self, target_pose, start_joint_pos):
        """
            求解IK
        Args:
            target_pose: 目标位姿
            start_joint_pos: 初始关节位置, 7维数组 np.ndarray
            可选参数：n_init_qpos, threshold, return_closest, verbose
        Returns:
            ik_result: IK求解结果: 
                status: 求解状态,str
                position_error: 位置误差,float
                qgoal: 关节角度解,np.ndarray
        """
        if start_joint_pos is None:
            start_joint_pos = self.reset_joint_pos
            
        # 将7维关节角扩展为9维（7个旋转关节+2个夹爪关节）
        # 初始化完整关节空间，保持夹爪位置不变（通常是0.0或某个预设值）
        full_joint_state = np.zeros(9)
        # 前7个关节设置为输入的关节位置
        full_joint_state[:7] = start_joint_pos
        # 夹爪位置可以设置为预设值，例如0.0表示闭合
        # full_joint_state[7:9] = [0.0, 0.0]  # 可根据实际需求调整夹爪状态
        
        if target_pose.shape == (6,):
            # 将欧拉角转换为四元数表示
            quat = T.euler2quat(target_pose[3:])
            # 创建目标位姿，使用位置和四元数
        else:
            quat = target_pose[3:]
        quat = np.concatenate([[quat[3]], quat[0:3]])
        target_pose = Pose(p=target_pose[:3], q=quat)
        # 使用完整的关节状态进行IK求解
        ik_result = self.planner.FullIK(target_pose, full_joint_state, mask=self.mask)
        
        if not ik_result.success:
            return ik_result
            
        # 只返回7个旋转关节的值
        ik_result.cspace_position = ik_result.cspace_position[:7]
        return ik_result
    