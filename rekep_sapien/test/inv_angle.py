import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from transform_utils import *
import numpy as np

def inverse_rotation(quaternion, rotated_vector):
    """
    计算经过四元数旋转后得到的向量的原始向量
    
    Args:
        quaternion (np.array): 表示旋转的四元数，格式为(x,y,z,w)
        rotated_vector (np.array): 旋转后的向量
        
    Returns:
        np.array: 原始向量
    """
    # 将四元数转换为旋转矩阵
    rotation_matrix = quat2mat(quaternion)
    
    # 旋转矩阵的逆等于其转置
    inverse_rotation_matrix = rotation_matrix.T
    
    # 使用逆旋转矩阵将旋转后的向量变换回原始向量
    original_vector = inverse_rotation_matrix @ rotated_vector
    
    return original_vector

# 测试示例
if __name__ == "__main__":
    # 示例四元数 (x,y,z,w)
    quat = np.array([0.259, 0.0, 0.0, 0.966])
    
    # 原始向量
    original_vec = np.array([1.0, 0.0, 0.0])
    
    # 使用四元数旋转向量
    rotation_matrix = quat2mat(quat)
    rotated_vec = rotation_matrix @ original_vec
    
    print("原始向量:", original_vec)
    print("旋转后的向量:", rotated_vec)
    
    # 使用我们的函数计算原始向量
    calculated_original = inverse_rotation(quat, rotated_vec)
    print("计算得到的原始向量:", calculated_original)
    
    # 验证结果
    print("误差:", np.linalg.norm(original_vec - calculated_original))
