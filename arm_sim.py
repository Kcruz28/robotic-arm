import os
import glob
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import pybullet as p
import time
import math

# Force FastRTPS for Mac stability
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

class ArmSim(Node):
    def __init__(self):
        super().__init__('arm_sim')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # Start PyBullet Headless

        p.connect(p.DIRECT)
        # p.connect(p.GUI)  ## works
        
        # 1. THE FIX: Set search path to the parent directory
        p.setAdditionalSearchPath("/Users/laflame/ros2_ws/src/tech_team/")
        
        # 2. Update the glob search to use the newly renamed folder
        urdf_search_path = "/Users/laflame/ros2_ws/src/tech_team/so_arm_description/urdf/*.urdf"
        urdf_files = glob.glob(urdf_search_path)
        
        if not urdf_files:
            self.get_logger().error(f"Could not find URDF at {urdf_search_path}. Check your path!")
            raise FileNotFoundError("URDF missing.")
            
        # 3. Load the full arm
        try:
            self.arm_id = p.loadURDF(urdf_files[0], useFixedBase=True)
        except p.error as e:
            self.get_logger().error(f"Failed to load URDF: {e}")
            raise

        # --- DYNAMICALLY GET JOINTS ---
        self.active_joints = []
        self.joint_names = []
        
        # PyBullet counts fixed joints, so we filter only for moving ones
        for i in range(p.getNumJoints(self.arm_id)):
            info = p.getJointInfo(self.arm_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            # Type 0 is REVOLUTE, Type 4 is CONTINUOUS.
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_CONTINUOUS:
                self.active_joints.append(i)
                self.joint_names.append(joint_name)
                
        self.get_logger().info(f"Found active joints: {self.joint_names}")
        
        self.timer = self.create_timer(0.05, self.publish_joints) # 20Hz
        self.start_time = time.time()
        self.get_logger().info("Simulation started. Open Foxglove!")

    def publish_joints(self):
        t = time.time() - self.start_time
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = []
        
        # Generate movement for however many joints the SO-101 has
        for i, joint_id in enumerate(self.active_joints):
            # Give each joint a slightly different frequency so it "dances"
            angle = math.sin(t * 0.5 + i) * 0.5 
            
            # Apply to PyBullet
            p.resetJointState(self.arm_id, joint_id, angle)
            # Add to ROS 2 message
            msg.position.append(float(angle))

        # Publish to ROS 2
        self.publisher.publish(msg)

def main():
    rclpy.init()
    node = ArmSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    p.disconnect()
    rclpy.shutdown()

if __name__ == '__main__':
    main()