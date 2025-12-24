import numpy as np

def quat2mat(quaternion):
    """
    将四元数转换为旋转矩阵
    
    Args:
        quaternion (np.array): 表示旋转的四元数，格式为(x,y,z,w)
        
    Returns:
        np.array: 3x3旋转矩阵
    """
    x, y, z, w = quaternion
    
    # 计算旋转矩阵的元素
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    
    yy = y * y
    yz = y * z
    yw = y * w
    
    zz = z * z
    zw = z * w
    
    # 构建旋转矩阵
    rotation_matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
        [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
        [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]
    ])
    
    return rotation_matrix

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