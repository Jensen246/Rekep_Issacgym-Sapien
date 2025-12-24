"""
简单地获取Panda机器人末端执行器的坐标范围
通过随机采样关节角度来确定末端执行器的活动范围
"""
import sapien.core as sapien
import numpy as np
import time

# 创建SAPIEN引擎和场景
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene()

# 加载Panda机器人URDF
loader = scene.create_urdf_loader()
loader.fix_root_link = True  # 固定基座
robot = loader.load_kinematic("data/panda/panda.urdf")
assert robot, "无法加载Panda机器人URDF文件"

# 获取所有活动关节和末端执行器
active_joints = []
joint_limits = []

print("机器人关节信息:")
for joint in robot.get_joints():
    if joint.get_dof() > 0:  # 只考虑有自由度的关节
        active_joints.append(joint)
        # 获取关节限制
        try:
            lower = joint.get_limits()[0]
            upper = joint.get_limits()[1]
            print(f"关节名称: {joint.get_name()}, 限制: [{lower}, {upper}]")
        except:
            # 从URDF文件中的limits标签读取关节限制
            # 根据panda.urdf文件手动设置对应的关节限制
            if "panda_joint1" in joint.get_name():
                lower, upper = -2.8973, 2.8973
            elif "panda_joint2" in joint.get_name():
                lower, upper = -1.7628, 1.7628
            elif "panda_joint3" in joint.get_name():
                lower, upper = -2.8973, 2.8973
            elif "panda_joint4" in joint.get_name():
                lower, upper = -3.0718, -0.0698
            elif "panda_joint5" in joint.get_name():
                lower, upper = -2.8973, 2.8973
            elif "panda_joint6" in joint.get_name():
                lower, upper = -0.0175, 3.7525
            elif "panda_joint7" in joint.get_name():
                lower, upper = -2.8973, 2.8973
            elif "panda_finger_joint1" in joint.get_name() or "panda_finger_joint2" in joint.get_name():
                lower, upper = 0.0, 0.04
            else:
                lower, upper = -3.14, 3.14
            print(f"关节名称: {joint.get_name()}, 使用URDF限制: [{lower}, {upper}]")
        joint_limits.append((lower, upper))

# 获取末端执行器链接
links = robot.get_links()
end_effector = None

# 查找名称包含"hand"或"gripper"或"panda_link8"的链接作为末端执行器
for link in links:
    name = link.get_name()
    if "hand" in name or "gripper" in name or "panda_link8" in name:
        end_effector = link
        print(f"找到末端执行器: {name}")
        break

# 如果找不到特定命名的末端执行器，则使用最后一个链接
if end_effector is None:
    end_effector = links[-1]
    print(f"使用最后一个链接作为末端执行器: {end_effector.get_name()}")

# 随机采样获取末端执行器位置范围
num_samples = 5000
ee_positions = []

print(f"开始采样末端执行器位置 (共{num_samples}个样本)...")
start_time = time.time()

# 正确设置关节角度的方法
def set_robot_qpos(robot, qpos):
    """设置机器人的关节角度"""
    robot.set_qpos(qpos)
    scene.step()  # 更新运动学

# 测试所有关节角度为0时的位置(机器人默认姿态)
print("\n测试默认姿态 - 设置所有关节为0...")
qpos_size = robot.get_qpos().shape[0]
zero_qpos = np.zeros(qpos_size)
set_robot_qpos(robot, zero_qpos)
ee_pose = end_effector.get_pose()
print(f"末端执行器在默认姿态:")
print(f"- 位置 (xyz): {ee_pose.p}")
print(f"- 旋转 (四元数 wxyz): {ee_pose.q}")
print(f"- 完整6D姿态 (xyz + wxyz): [{ee_pose.p[0]}, {ee_pose.p[1]}, {ee_pose.p[2]}, {ee_pose.q[0]}, {ee_pose.q[1]}, {ee_pose.q[2]}, {ee_pose.q[3]}]")

# 获取当前机器人的关节角度
print(f"关节数量: {qpos_size}")

for i in range(num_samples):
    # 生成随机关节角度
    random_qpos = np.zeros(qpos_size)
    
    # 为每个关节设置随机角度
    joint_idx = 0
    for j, joint in enumerate(active_joints):
        lower, upper = joint_limits[j]
        dof = joint.get_dof()
        
        for k in range(dof):
            if joint_idx < qpos_size:
                random_qpos[joint_idx] = np.random.uniform(lower, upper)
                joint_idx += 1
    
    # 设置关节角度并更新运动学
    set_robot_qpos(robot, random_qpos)
    
    # 获取末端执行器位置
    ee_pose = end_effector.get_pose()
    ee_positions.append(ee_pose.p)
    
    # 显示进度和当前位置
    if (i+1) % (num_samples // 10) == 0:
        print(f"已完成 {i+1}/{num_samples} 个样本 ({(i+1)/num_samples*100:.1f}%)")
        print(f"样例位置: {ee_pose.p}")

# 计算边界
ee_positions = np.array(ee_positions)
bounds_min = np.min(ee_positions, axis=0)
bounds_max = np.max(ee_positions, axis=0)

elapsed_time = time.time() - start_time
print(f"\n采样完成，用时 {elapsed_time:.2f} 秒")
print("\n==== 末端执行器活动范围 ====")
print(f"最小坐标: {bounds_min}")
print(f"最大坐标: {bounds_max}")
print(f"尺寸: {bounds_max - bounds_min}")

# 将结果保存到文件
np.savetxt("robot_bounds.txt", np.vstack((bounds_min, bounds_max)), fmt="%.6f")
print("\n结果已保存到 robot_bounds.txt") 