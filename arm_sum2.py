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
import torch

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
        
        # OBSERVATION: 13 Data Points (Joints + Trash Pos + Gripper Pos + Dist)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        # Load Arm (CHECK YOUR PATH!)
        urdf_path = os.path.join(os.path.dirname(__file__), "so_arm_description", "urdf", "so101_new_calib.urdf")
        self.arm_id = p.loadURDF(urdf_path, useFixedBase=True)
        
        # Random Trash
        # Make the range smaller so the trash is easier to reach
        rand_x = np.random.uniform(0.25, 0.35) 
        rand_y = np.random.uniform(-0.1, 0.1)
        self.trash_id = p.loadURDF("cube_small.urdf", basePosition=[rand_x, rand_y, 0.02], globalScaling=1.2)
        p.changeVisualShape(self.trash_id, -1, rgbaColor=[1, 0, 0, 1])

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
            p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL, targetPosition=new_joints[i])
        
        p.stepSimulation()
        if self.render_mode: time.sleep(1./240.) 
        
        # Reward Function
        gripper_pos = p.getLinkState(self.arm_id, 5)[0] 
        trash_pos, _ = p.getBasePositionAndOrientation(self.trash_id)
        dist = np.linalg.norm(np.array(gripper_pos) - np.array(trash_pos))
        
        reward = -dist # Closer is better
        
        terminated = False
        if dist < 0.05:
            reward += 100.0 # JACKPOT
            terminated = True 
            if self.render_mode: print("TARGET REACHED!")

        truncated = False
        if self.step_counter > 500: # Shorter episodes for faster iterations
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        joints = [p.getJointState(self.arm_id, i)[0] for i in range(6)]
        trash_pos, _ = p.getBasePositionAndOrientation(self.trash_id)
        gripper_pos = p.getLinkState(self.arm_id, 5)[0]
        dist = np.linalg.norm(np.array(gripper_pos) - np.array(trash_pos))
        return np.array(joints + list(trash_pos) + list(gripper_pos) + [dist], dtype=np.float32)
    
    def close(self):
        p.disconnect()

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    model_path = "so101_fast_model"
    training_steps = 1000000  # reduced for speed

    print(torch.version.cuda)   
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # --- GPU / Device Setup ---
    import torch
    if torch.backends.mps.is_available():
        device = "mps"  # Apple M1/M2
    elif torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # --- PART A: TRAINING ---
    if not os.path.exists(model_path + ".zip"):
        print(f"STARTING FAST TRAINING ({training_steps} steps)...")
        
        env = RobotArmEnv(render_mode=False)
        model = PPO("MlpPolicy", env, verbose=0, device=device)  # Pass device
        
        callback = ProgressBarCallback(training_steps)
        model.learn(total_timesteps=training_steps, callback=callback)
        
        model.save(model_path)
        print("\nTRAINING DONE! Saving Brain...")
        env.close()
    else:
        print("MODEL FOUND! Skipping training.")

    # --- PART B: WATCH IT WORK ---
    print("LAUNCHING GUI...")
    env = RobotArmEnv(render_mode=True)
    model = PPO.load(model_path, device=device)  # Pass device
    
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
