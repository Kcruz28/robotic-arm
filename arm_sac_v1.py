import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Safe on all platforms

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import time

# --- GPU DETECTION ---
# Priority: CUDA (A40 / RTX 5090) > Apple MPS (M1/M2/M3/M4) > CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"[GPU] CUDA detected: {torch.cuda.get_device_name(0)}")
    print(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"
    print("[GPU] Apple MPS detected (M1/M2/M3/M4). Using Metal GPU acceleration.")
    print("      Note: if you see any MPS tensor errors, set DEVICE = 'cpu' manually.")
else:
    DEVICE = "cpu"
    print("[CPU] No GPU detected. Training on CPU.")

# --- 1. PROGRESS BAR CALLBACK ---
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = tqdm(total=total_timesteps, desc="Training SAC")

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


# --- 2. ROBOT ARM ENVIRONMENT ---
class RobotArmEnv(gym.Env):
    def __init__(self, render_mode=False):
        super(RobotArmEnv, self).__init__()
        self.render_mode = render_mode

        if self.render_mode:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # ACTION: 6 joint deltas in [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # OBSERVATION: 20 values — joints(6) + block_pos(3) + gripper_pos(3) + gripper_ori(4) + rel_pos(3) + dist(1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        try:
            p.resetSimulation(self.client)
        except p.error:
            if self.render_mode:
                self.client = p.connect(p.GUI)
            else:
                self.client = p.connect(p.DIRECT)
            p.resetSimulation(self.client)

        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        urdf_path = os.path.join(os.path.dirname(__file__), "so_arm_description", "urdf", "so101_new_calib.urdf")
        self.arm_id = p.loadURDF(urdf_path, useFixedBase=True)

        rand_x = np.random.uniform(0.28, 0.32)
        rand_y = np.random.uniform(-0.05, 0.05)

        try:
            self.trash_id = p.loadURDF("cube_small.urdf", basePosition=[rand_x, rand_y, 0.02],
                                       globalScaling=0.7, physicsClientId=self.client)
            p.changeVisualShape(self.trash_id, -1, rgbaColor=[1, 0, 0, 1], physicsClientId=self.client)
            # High friction so the block doesn't slip once gripped
            p.changeDynamics(self.trash_id, -1, lateralFriction=2.0, physicsClientId=self.client)
            p.changeDynamics(self.arm_id, 5, lateralFriction=2.0, physicsClientId=self.client)
        except p.error:
            pass

        # Start with jaws open (joint 5)
        try:
            p.resetJointState(self.arm_id, 5, targetValue=0.5)
        except p.error:
            pass

        self.step_counter = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_counter += 1

        # FIX: physicsClientId added — was missing in original, caused multi-env crashes
        current_joints = [p.getJointState(self.arm_id, i, physicsClientId=self.client)[0] for i in range(6)]
        new_joints = np.array(current_joints) + (action * 0.1)

        for i in range(6):
            p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL,
                                    targetPosition=new_joints[i], force=50.0,
                                    physicsClientId=self.client)

        p.stepSimulation(physicsClientId=self.client)
        if self.render_mode:
            time.sleep(1. / 240.)

        # --- Safety checks ---
        try:
            trash_pos, _ = p.getBasePositionAndOrientation(self.trash_id, physicsClientId=self.client)
        except p.error:
            return self._get_obs(fallback_pos=[0, 0, -10]), -50.0, True, False, {}

        if trash_pos[2] < -0.1:
            return self._get_obs(fallback_pos=trash_pos), -50.0, True, False, {}

        try:
            link_state = p.getLinkState(self.arm_id, 5, computeForwardKinematics=1, physicsClientId=self.client)
            gripper_pos = link_state[0]
        except p.error:
            return self._get_obs(fallback_pos=trash_pos), 0.0, True, False, {}

        dist = np.linalg.norm(np.array(gripper_pos) - np.array(trash_pos))

        # --- Reward 1: Approach ---
        # FIX: target 6cm ABOVE block center (was 1cm = inside the block, caused push-down behaviour)
        target_pos = np.array(trash_pos)
        target_pos[2] += 0.06
        real_dist = np.linalg.norm(np.array(gripper_pos) - target_pos)
        reward = -real_dist * 5.0

        # --- Reward 2: Crane approach — stay high until directly over block ---
        xy_dist = np.linalg.norm(np.array(gripper_pos[:2]) - np.array(trash_pos[:2]))
        if xy_dist > 0.06:
            # Too far laterally → penalise being low
            if gripper_pos[2] < 0.10:
                reward -= 10.0
        else:
            # Hovering above block → reward being at ideal descend height (~0.06m)
            # FIX: was 0.04m (= block top) which drove gripper into block
            grasp_height_reward = max(0, 0.1 - abs(gripper_pos[2] - 0.06))
            reward += grasp_height_reward * 10.0

        # --- Reward 3: Grasp (both jaw faces must contact block) ---
        contact_stationary = p.getContactPoints(bodyA=self.arm_id, bodyB=self.trash_id,
                                                linkIndexA=4, physicsClientId=self.client)
        contact_moving = p.getContactPoints(bodyA=self.arm_id, bodyB=self.trash_id,
                                            linkIndexA=5, physicsClientId=self.client)
        gripped = len(contact_stationary) > 0 and len(contact_moving) > 0

        if gripped:
            reward += 50.0  # Jackpot: block sandwiched between jaws

        # --- Reward 4: Keep jaws open while approaching ---
        gripper_joint = current_joints[5]
        if dist > 0.05:
            reward += gripper_joint * 5.0

        # --- Penalty 1: Hammering down without a grip ---
        # FIX: threshold raised 0.03→0.05m (above block top), penalty -5→-15
        if gripper_pos[2] < 0.05 and not gripped:
            reward -= 15.0

        # --- Penalty 2: Arm body hitting block (only fingers should touch) ---
        for link_idx in range(4):
            bad_contact = p.getContactPoints(bodyA=self.arm_id, bodyB=self.trash_id,
                                             linkIndexA=link_idx, physicsClientId=self.client)
            if len(bad_contact) > 0:
                reward -= 10.0

        # --- Penalty 3: Jittery/high-effort movement ---
        reward -= np.sum(np.square(action)) * 0.001

        # --- Reward 5: Lift ---
        trash_z = trash_pos[2]
        # FIX: lifted threshold 0.03→0.05m to prevent physics jitter triggering false reward at reset
        if trash_z > 0.05:
            reward += (trash_z - 0.02) * 1000

        terminated = False
        if trash_z > 0.1:
            reward += 500.0
            terminated = True
            if self.render_mode:
                print("TARGET GRABBED AND LIFTED!")

        truncated = self.step_counter > 800

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
            gripper_ori = link_state[1]
        except p.error:
            gripper_pos = [0, 0, 0]
            gripper_ori = [0, 0, 0, 1]

        rel_pos = np.array(trash_pos) - np.array(gripper_pos)
        dist = np.linalg.norm(rel_pos)

        # 6 + 3 + 3 + 4 + 3 + 1 = 20
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
        # FIX: self.client passed so only this env's physics instance is disconnected
        try:
            p.disconnect(self.client)
        except p.error:
            pass


# --- 3. MAIN ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_model_path = os.path.join(script_dir, "so101_SAC_v0")
    new_model_path  = os.path.join(script_dir, "so101_SAC_v1")

    # SAC is ~3x more sample-efficient than PPO for continuous robotic control.
    # 2M steps on an A40/5090 beats 4M PPO steps on CPU in both wall-clock time
    # and final policy quality.

    # ------------------------------------------------------------------ #
    # Training Steps                                                     #
    # ------------------------------------------------------------------ #
    training_steps = 10_000

    # ------------------------------------------------------------------ #
    #  SAC HYPERPARAMETERS                                               #
    # ------------------------------------------------------------------ #
    sac_kwargs = dict(
        policy="MlpPolicy",
        verbose=0,
        device=DEVICE,
        tensorboard_log="./sac_tensorboard/",
        # Off-policy replay buffer — SAC reuses every collected transition
        # multiple times, which is why it's so much more efficient than PPO.
        buffer_size=500_000,
        # Collect 10k random steps before training so the replay buffer
        # is populated with diverse transitions before gradient updates begin.
        learning_starts=10_000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        # batch_size=512 saturates A40/5090 CUDA cores efficiently
        batch_size=512,
        train_freq=1,
        gradient_steps=-1,  # match gradient steps to env steps collected
        # gSDE (generalized State Dependent Exploration):
        # Learns a state-conditional noise distribution per joint instead of
        # applying the same fixed Gaussian noise to every action.
        # Significantly improves arm manipulation performance over vanilla SAC.
        use_sde=True,
        sde_sample_freq=64,
        # [512, 512, 256]: deeper tapering network for 20-dim obs → 6-dim action
        policy_kwargs=dict(net_arch=[512, 512, 256]),
    )

    # ------------------------------------------------------------------ #
    #  PART A: TRAINING                                                    #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*55}")
    print(f"  RL Algorithm : SAC (Soft Actor-Critic) + gSDE")
    print(f"  Device       : {DEVICE.upper()}")
    print(f"  Steps        : {training_steps:,}")
    print(f"  Network      : [512, 512, 256]")
    print(f"  Batch size   : {sac_kwargs['batch_size']}")
    print(f"{'='*55}\n")

    env = RobotArmEnv(render_mode=False)

    if os.path.exists(base_model_path + ".zip"):
        print(f"FOUND EXISTING SAC MODEL: '{base_model_path}.zip'")
        print(f"Continuing training for {training_steps:,} more steps...")
        model = SAC.load(base_model_path, env=env, device=DEVICE)
    else:
        print(f"No existing SAC model found. Starting fresh ({training_steps:,} steps)...")
        model = SAC(env=env, **sac_kwargs)

    callback = ProgressBarCallback(training_steps)
    model.learn(
        total_timesteps=training_steps,
        callback=callback,
        reset_num_timesteps=False,  # keeps TensorBoard x-axis continuous across runs
        log_interval=1,
    )

    model.save(new_model_path)
    print(f"\nTRAINING DONE! Model saved to '{new_model_path}.zip'")
    env.close()

    # ------------------------------------------------------------------ #
    #  PART B: WATCH IT WORK (GUI)                                         #
    # ------------------------------------------------------------------ #
    print("\nLAUNCHING GUI to watch the trained model...")
    env = RobotArmEnv(render_mode=True)

    if os.path.exists(new_model_path + ".zip"):
        print(f"Loading: {new_model_path}")
        model = SAC.load(new_model_path, env=env, device=DEVICE)
    elif os.path.exists(base_model_path + ".zip"):
        print(f"Falling back to base model: {base_model_path}")
        model = SAC.load(base_model_path, env=env, device=DEVICE)
    else:
        print("ERROR: No SAC model file found. Run training first.")
        env.close()
        exit(1)

    print("Watching... close the PyBullet window to exit.")
    obs, _ = env.reset()
    episode = 0
    while True:
        # deterministic=True: policy mean — no exploration noise during eval
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)

        # FIX: combined into one block — was two separate if blocks causing double-reset
        if done or truncated:
            episode += 1
            if done:
                print(f"[Episode {episode}] SUCCESS! Block lifted! Resetting...")
                time.sleep(1.5)
            else:
                print(f"[Episode {episode}] Time limit. Resetting...")
            obs, _ = env.reset()
