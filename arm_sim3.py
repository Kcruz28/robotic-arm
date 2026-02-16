import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, String

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import time

# --- ROS 2 VISUALIZER NODE ---
class ArmVisualizer(Node):
    def __init__(self):
        super().__init__('arm_sim_visualizer')
        self.marker_pub = self.create_publisher(MarkerArray, '/viz_markers', 10)
        self.dist_pub = self.create_publisher(Float32, '/arm_distance', 10)
        self.status_pub = self.create_publisher(String, '/arm_status', 10)

    def publish_env_state(self, env, dist, status_text="RUNNING"):
        markers = MarkerArray()
        timestamp = self.get_clock().now().to_msg()
        
        # 1. TRASH MARKER (RED CUBE) - ID 100
        trash_pos, _ = p.getBasePositionAndOrientation(env.trash_id)
        m_trash = Marker()
        m_trash.header.frame_id = "map"
        m_trash.header.stamp = timestamp
        m_trash.ns = "trash_target" # Unique Namespace
        m_trash.id = 100            # Unique ID
        m_trash.type = Marker.CUBE
        m_trash.action = Marker.ADD
        m_trash.pose.position.x = trash_pos[0]
        m_trash.pose.position.y = trash_pos[1]
        m_trash.pose.position.z = trash_pos[2]
        m_trash.scale.x = 0.05; m_trash.scale.y = 0.05; m_trash.scale.z = 0.05
        m_trash.color.r = 1.0; m_trash.color.a = 1.0 # Red, Solid
        markers.markers.append(m_trash)

        # 2. ARM JOINTS (BLUE SPHERES) - ID 200
        m_joints = Marker()
        m_joints.header.frame_id = "map"
        m_joints.header.stamp = timestamp
        m_joints.ns = "arm_skeleton"
        m_joints.id = 200
        m_joints.type = Marker.SPHERE_LIST
        m_joints.action = Marker.ADD
        m_joints.scale.x = 0.03; m_joints.scale.y = 0.03; m_joints.scale.z = 0.03
        m_joints.color.b = 1.0; m_joints.color.a = 1.0 # Blue
        
        # 3. ARM BONES (WHITE LINES) - ID 300
        m_bones = Marker()
        m_bones.header.frame_id = "map"
        m_bones.header.stamp = timestamp
        m_bones.ns = "arm_bones"
        m_bones.id = 300
        m_bones.type = Marker.LINE_STRIP
        m_bones.action = Marker.ADD
        m_bones.scale.x = 0.01 # Line width
        m_bones.color.r = 1.0; m_bones.color.g = 1.0; m_bones.color.b = 1.0; m_bones.color.a = 1.0

        # Collect points
        p_base = Point(); p_base.x=0.0; p_base.y=0.0; p_base.z=0.0
        m_joints.points.append(p_base)
        m_bones.points.append(p_base)
        
        for i in range(p.getNumJoints(env.arm_id)):
            link_pos = p.getLinkState(env.arm_id, i)[0]
            pt = Point()
            pt.x = link_pos[0]; pt.y = link_pos[1]; pt.z = link_pos[2]
            m_joints.points.append(pt)
            m_bones.points.append(pt)

        markers.markers.append(m_joints)
        markers.markers.append(m_bones)

        # Publish
        self.marker_pub.publish(markers)
        self.dist_pub.publish(Float32(data=dist))
        self.status_pub.publish(String(data=status_text))


class RobotArmEnv(gym.Env):
    def __init__(self, render_mode=False):
        super(RobotArmEnv, self).__init__()
        # FORCE DIRECT MODE TO FIX ROS LAG
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        self.arm_id = p.loadURDF("/Users/laflame/ros2_ws/src/tech_team/so_arm_description/urdf/so101_new_calib.urdf", useFixedBase=True)
        
        rand_x = np.random.uniform(0.25, 0.35) 
        rand_y = np.random.uniform(-0.1, 0.1)
        self.trash_id = p.loadURDF("cube_small.urdf", basePosition=[rand_x, rand_y, 0.02], globalScaling=1.2)
        p.changeVisualShape(self.trash_id, -1, rgbaColor=[1, 0, 0, 1])
        
        self.step_counter = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_counter += 1
        current_joints = [p.getJointState(self.arm_id, i)[0] for i in range(6)]
        new_joints = np.array(current_joints) + (action * 0.1)
        for i in range(6):
            p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=new_joints[i])
        p.stepSimulation()
        
        gripper_pos = p.getLinkState(self.arm_id, 5)[0] 
        trash_pos, _ = p.getBasePositionAndOrientation(self.trash_id)
        dist = np.linalg.norm(np.array(gripper_pos) - np.array(trash_pos))
        
        reward = -dist
        if dist < 0.05: reward += 100.0
        
        truncated = False
        if self.step_counter > 500: truncated = True
        return self._get_obs(), reward, False, truncated, {}

    def _get_obs(self):
        joints = [p.getJointState(self.arm_id, i)[0] for i in range(6)]
        trash_pos, _ = p.getBasePositionAndOrientation(self.trash_id)
        gripper_pos = p.getLinkState(self.arm_id, 5)[0]
        dist = np.linalg.norm(np.array(gripper_pos) - np.array(trash_pos))
        return np.array(joints + list(trash_pos) + list(gripper_pos) + [dist], dtype=np.float32)

# --- MAIN ---
if __name__ == "__main__":
    rclpy.init()
    ros_node = ArmVisualizer()
    
    model_path = "so101_fast_model"
    # Ensure model exists (or you can insert training code here)
    if not os.path.exists(model_path + ".zip"):
        print("Model not found! Run the training script first.")
        exit()
        
    env = RobotArmEnv(render_mode=False) 
    model = PPO.load(model_path)
    obs, _ = env.reset()
    
    # FIND JAW
    jaw_index = -1
    for i in range(p.getNumJoints(env.arm_id)):
        info = p.getJointInfo(env.arm_id, i)
        if "Jaw" in info[1].decode('utf-8') or "gripper" in info[1].decode('utf-8'):
            jaw_index = i

    print("✅ SIMULATION RUNNING")
    print("👉 Foxglove: Connect to Rosbridge/Foxglove Bridge")
    print("👉 3D Panel: Set Frame to 'map', Enable '/viz_markers'")

    try:
        while rclpy.ok():
            action, _ = model.predict(obs)
            obs, _, done, truncated, _ = env.step(action)
            
            # CALC DIST
            gripper_pos = p.getLinkState(env.arm_id, 5)[0]
            trash_pos, _ = p.getBasePositionAndOrientation(env.trash_id)
            dist = np.linalg.norm(np.array(gripper_pos) - np.array(trash_pos))
            
            # PUBLISH & SPIN
            ros_node.publish_env_state(env, dist, "RUNNING")
            rclpy.spin_once(ros_node, timeout_sec=0)

            # TRAP LOGIC
            if dist < 0.05:
                if jaw_index >= 0:
                    p.setJointMotorControl2(env.arm_id, jaw_index, p.POSITION_CONTROL, targetPosition=0.0, force=50)
                
                # Show the grab
                for _ in range(30):
                    p.stepSimulation()
                    ros_node.publish_env_state(env, dist, "GRABBING")
                    rclpy.spin_once(ros_node, timeout_sec=0)
                    time.sleep(0.01)
                done = True

            if done or truncated:
                obs, _ = env.reset()
            
            time.sleep(1/30) # 30 FPS
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        rclpy.shutdown()