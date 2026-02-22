import os
# --- MAC M1/M2 FIX ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import time

# --- 1. THE PROGRESS BAR (The Visualizer) ---
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = tqdm(total=total_timesteps, desc="Training Neural Network")

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()

# --- 2. THE ROBOT ENVIRONMENT (The Game) ---
class RobotArmEnv(gym.Env):
    def __init__(self, render_mode=False):
        super(RobotArmEnv, self).__init__()
        self.render_mode = render_mode
        
        # HEADLESS for Training (Fast), GUI for Watching (Slow)
        if self.render_mode:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # ACTION: 6 Motor Velocities (-1 to +1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        
        # OBSERVATION: 20 Data Points (Joints + Trash Pos + Gripper Pos + Gripper Ori + Rel Pos + Dist)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        try:
            p.resetSimulation(self.client)
        except p.error:
            # Reconnect if window was closed
            if self.render_mode:
                self.client = p.connect(p.GUI)
            else:
                self.client = p.connect(p.DIRECT)
            p.resetSimulation(self.client)
            
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        
        # Load Arm (CHECK YOUR PATH!)
        urdf_path = os.path.join(os.path.dirname(__file__), "so_arm_description", "urdf", "so101_new_calib.urdf")
        self.arm_id = p.loadURDF(urdf_path, useFixedBase=True)
        
        # Random Trash
        # Make the range smaller so the trash is easier to reach
        rand_x = np.random.uniform(0.28, 0.32) 
        rand_y = np.random.uniform(-0.05, 0.05)
        # Load block AND THEN change its properties
        try:
            # SHRUNK THE BLOCK: So the small metal jaws can easily fit completely around it!
            self.trash_id = p.loadURDF("cube_small.urdf", basePosition=[rand_x, rand_y, 0.02], globalScaling=0.7, physicsClientId=self.client)
            p.changeVisualShape(self.trash_id, -1, rgbaColor=[1, 0, 0, 1], physicsClientId=self.client)
            
            # MAKE GRABBING PHYSICALLY EASIER: High friction so the block doesn't slip like ice!
            p.changeDynamics(self.trash_id, -1, lateralFriction=2.0, physicsClientId=self.client)
            p.changeDynamics(self.arm_id, 5, lateralFriction=2.0, physicsClientId=self.client)
        except p.error:
            # Fallback if there's a weird Pybullet visual shape error during fast resets
            pass

        self.step_counter = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_counter += 1
        
        # Move Motors
        current_joints = [p.getJointState(self.arm_id, i)[0] for i in range(6)]
        # 0.05 is the "Learning Rate" for physics. 
        # Change 0.05 to 0.1 so it moves with more confidence
        new_joints = np.array(current_joints) + (action * 0.1)
        
        for i in range(6):
            # ADDED FORCE: The arm needs high torque (50.0) so it has the strength to grip and lift!
            p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=new_joints[i], force=50.0, physicsClientId=self.client)
        
        p.stepSimulation(physicsClientId=self.client)
        if self.render_mode: time.sleep(1./240.) 
        
        # Safe Check: Did the block get deleted by the physics engine?
        try:
            trash_pos, _ = p.getBasePositionAndOrientation(self.trash_id)
        except p.error:
            # The trash fell out of the world! End game with penalty.
            return self._get_obs(fallback_pos=[0,0,-10]), -50.0, True, False, {}
            
        # Fail Condition: Knocked block off the table
        if trash_pos[2] < -0.1:
            return self._get_obs(fallback_pos=trash_pos), -50.0, True, False, {}

        # Safe Check: Did PyBullet freeze trying to find the link?
        try:
            link_state = p.getLinkState(self.arm_id, 5, computeForwardKinematics=1, physicsClientId=self.client)
            gripper_pos = link_state[0] 
        except p.error:
            return self._get_obs(fallback_pos=trash_pos), 0.0, True, False, {}

        dist = np.linalg.norm(np.array(gripper_pos) - np.array(trash_pos))
        
        # 1. Approach Reward (encourage getting close)
        # Aim for the TOP of the block (trash_z + small offset) so it hovers correctly
        target_pos = np.array(trash_pos)
        target_pos[2] += 0.01 
        
        real_dist = np.linalg.norm(np.array(gripper_pos) - target_pos)
        reward = -real_dist * 5.0 
        
        # 1.5 The "Crane" Approach (Hover above, then drop)
        xy_dist = np.linalg.norm(np.array(gripper_pos[:2]) - np.array(trash_pos[:2]))
        
        # Prevent "sweeping": If it's too far horizontally (> 0.04m), FORCE it to stay high!
        if xy_dist > 0.04:
            if gripper_pos[2] < 0.10: 
                reward -= 10.0 
        else:
            # Once hovering, reward being at the ideal grasp height (approx 0.04m)
            # This prevents it from trying to drive the gripper into the table.
            grasp_height_reward = max(0, 0.1 - abs(gripper_pos[2] - 0.04))
            reward += grasp_height_reward * 10.0
               
        # 1.8 REALISTIC GRASP REWARD
        # Reward the 'Sandwich': Contact on BOTH stationary gripper (link 4) AND moving jaw (link 5)
        contact_stationary = p.getContactPoints(bodyA=self.arm_id, bodyB=self.trash_id, linkIndexA=4, physicsClientId=self.client)
        contact_moving = p.getContactPoints(bodyA=self.arm_id, bodyB=self.trash_id, linkIndexA=5, physicsClientId=self.client)
        
        if len(contact_stationary) > 0 and len(contact_moving) > 0:
            reward += 50.0 # JACKPOT: The block is secured between jaws!
        elif len(contact_stationary) > 0 or len(contact_moving) > 0:
            reward += 10.0 # Just touching it
            
        # 1.9 ANTI-CRUSH PENALTY
        # If the shoulder, elbow, or wrist (links 0-3) hits the block, punish it.
        # This teaches the robot to ONLY use its 'fingers' to interact with the block.
        for link_idx in range(4): 
            bad_contact = p.getContactPoints(bodyA=self.arm_id, bodyB=self.trash_id, linkIndexA=link_idx, physicsClientId=self.client)
            if len(bad_contact) > 0:
                reward -= 10.0 

        # 2. Smoothness Penalty
        motor_effort = np.sum(np.square(action))
        reward -= motor_effort * 0.001 

        # 3. Lift Reward (The "Jackpot" path)
        trash_z = trash_pos[2]
        if trash_z > 0.03: # Block starts at ~0.02
            # Very high reward for every centimeter of lift
            reward += (trash_z - 0.02) * 1000 
            
        terminated = False
        # 3. Ultimate Goal: Lift the block more than 10cm!
        if trash_z > 0.1:
            reward += 500.0 # JACKPOT FOR LIFTING!
            terminated = True 
            if self.render_mode: print("TARGET GRABBED AND LIFTED!")

        truncated = False
        if self.step_counter > 800: # Increased from 500 to give it time to grab and lift
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self, fallback_pos=None):
        try:
            joints = [p.getJointState(self.arm_id, i, physicsClientId=self.client)[0] for i in range(6)]
        except p.error:
            joints = [0.0] * 6
        
        if fallback_pos is not None:
            trash_pos = fallback_pos
        else:
            try:
                trash_pos, _ = p.getBasePositionAndOrientation(self.trash_id, physicsClientId=self.client)
            except p.error:
                trash_pos = [0, 0, -1]
                
        try:
            link_state = p.getLinkState(self.arm_id, 5, computeForwardKinematics=1, physicsClientId=self.client)
            gripper_pos = link_state[0]
            gripper_ori = link_state[1] # Orientation quaternion (x,y,z,w)
        except p.error:
            gripper_pos = [0, 0, 0]
            gripper_ori = [0, 0, 0, 1]
        
        rel_pos = np.array(trash_pos) - np.array(gripper_pos)
        dist = np.linalg.norm(rel_pos)
        
        # 6 + 3 + 3 + 4 + 3 + 1 = 20 points of data
        return np.array(
            joints + 
            list(trash_pos) + 
            list(gripper_pos) + 
            list(gripper_ori) + 
            list(rel_pos) + 
            [dist], 
            dtype=np.float32
        )
    
    def close(self):
        p.disconnect()

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    # Use absolute path so it finds the file no matter what folder you run it from!
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "so101_grab_crane_v4") 
    
    # 400k is great for learning to reach! Let's bump it up slightly more to let it master the grab!
    training_steps = 600000 

    # --- PART A: TRAINING ---
    # If the model doesn't exist, we train it.
    if not os.path.exists(model_path + ".zip"):
        print(f"STARTING FAST TRAINING ({training_steps} steps)...")
        
        env = RobotArmEnv(render_mode=False)
        # `device="cpu"` is CRITICAL here! It stops the GPU/CUDA warning and makes it much faster for smaller models.
        # BIGGER BRAIN: We added `policy_kwargs=dict(net_arch=[256, 256])` so it has way more neurons to solve this hard physics puzzle!
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./ppo_tensorboard/", device="cpu", policy_kwargs=dict(net_arch=[256, 256])) 
        
        # Run Training with Progress Bar
        callback = ProgressBarCallback(training_steps)
        model.learn(total_timesteps=training_steps, callback=callback)
        
        model.save(model_path)
        print("\nTRAINING DONE! Saving Brain...")
        env.close()
    else:
        print("MODEL FOUND! Skipping training.")

    # --- PART B: WATCH IT WORK ---
    print("LAUNCHING GUI...")
    env = RobotArmEnv(render_mode=True) # Turn GUI ON
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        
        if done:
            print("Success! Resetting...")
            time.sleep(1.0)
            obs, _ = env.reset()
        
        if truncated:
            obs, _ = env.reset()