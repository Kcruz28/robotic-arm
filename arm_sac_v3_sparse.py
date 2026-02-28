import os
import argparse
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Safe on all platforms

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from tqdm import tqdm
import time

# --- 1. PROGRESS BAR CALLBACK ---
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self._total = total_timesteps
        self.pbar = None
        self._last_steps = 0

    def _on_training_start(self):
        self.pbar = tqdm(total=self._total, desc="Training SAC")
        self._last_steps = self.model.num_timesteps

    def _on_step(self) -> bool:
        increment = self.model.num_timesteps - self._last_steps
        self.pbar.update(increment)
        self._last_steps = self.model.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
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

        # OBSERVATION: 22 values — joints(6) + block_pos(3) + grasp_center(3) + finger_axis(3) + rel_pos(3) + grasp_dist(1) + gripper_width(1) + jaw4_dist(1) + jaw5_dist(1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)

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
            # High friction symmetrically applied to stop the block from slipping like a watermelon seed
            p.changeDynamics(self.trash_id, -1, lateralFriction=2.0, physicsClientId=self.client)
            p.changeDynamics(self.arm_id, 4, lateralFriction=2.0, physicsClientId=self.client) # Jaw 1
            p.changeDynamics(self.arm_id, 5, lateralFriction=2.0, physicsClientId=self.client) # Jaw 2
        except p.error:
            pass

        # Start with jaws open (joint 5)
        try:
            p.resetJointState(self.arm_id, 5, targetValue=0.5)
        except p.error:
            pass

        self.step_counter = 0
        return self._get_obs(), {}

    def _compute_grasp_geometry(self):
        """
        Compute the actual grasp center point, finger axis, and gripper width
        using the URDF link frame (not COM) + the known finger extension direction.

        Returns:
            grasp_center: np.array(3) — world position where the fingers actually close
            finger_axis:  np.array(3) — unit vector in world coords pointing along the fingers
            rot:          np.array(3,3) — gripper rotation matrix (columns = local axes in world)
            jaw_4_pos:    np.array(3) — gripper COM world position (for obs compatibility)
            jaw_5_pos:    np.array(3) — jaw COM world position
            gripper_width: float — distance between the two jaw COMs (proxy for opening width)
        """
        link_4_state = p.getLinkState(self.arm_id, 4, computeForwardKinematics=1, physicsClientId=self.client)
        link_5_state = p.getLinkState(self.arm_id, 5, computeForwardKinematics=1, physicsClientId=self.client)

        # URDF link frame position (NOT the COM, which is offset into the body)
        gripper_frame_pos = np.array(link_4_state[4])   # worldLinkFramePosition
        gripper_frame_ori = link_4_state[5]              # worldLinkFrameOrientation

        rot = np.array(p.getMatrixFromQuaternion(gripper_frame_ori)).reshape(3, 3)

        # The fingers extend along the gripper's LOCAL -Z axis.
        # (Jaw joint origin is at z=-0.023 in gripper frame; jaw COM is ~5cm below gripper frame.)
        finger_axis = -rot[:, 2]   # gripper -Z in world

        # Project 4.5 cm along finger axis from the gripper frame to reach the
        # point where the two fingers actually close around an object.
        grasp_center = gripper_frame_pos + finger_axis * 0.045

        # COM positions (still useful for per-jaw distance in obs)
        jaw_4_pos = np.array(link_4_state[0])
        jaw_5_pos = np.array(link_5_state[0])
        gripper_width = float(np.linalg.norm(jaw_4_pos - jaw_5_pos))

        return grasp_center, finger_axis, rot, jaw_4_pos, jaw_5_pos, gripper_width

    def step(self, action):
        self.step_counter += 1

        current_joints = [p.getJointState(self.arm_id, i, physicsClientId=self.client)[0] for i in range(6)]
        new_joints = np.array(current_joints) + (action * 0.1)

        for i in range(6):
            p.setJointMotorControl2(self.arm_id, i, p.POSITION_CONTROL,
                                    targetPosition=new_joints[i], force=50.0,
                                    physicsClientId=self.client)

        # 4 physics sub-steps per action for more stable contacts
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client)
        if self.render_mode:
            time.sleep(1. / 60.)

        # --- Get block position ---
        try:
            trash_pos, _ = p.getBasePositionAndOrientation(self.trash_id, physicsClientId=self.client)
        except p.error:
            return self._get_obs(fallback_pos=[0, 0, -10]), -50.0, True, False, {}

        if trash_pos[2] < -0.1:
            return self._get_obs(fallback_pos=trash_pos), -50.0, True, False, {}

        block_pos = np.array(trash_pos)

        # --- Compute accurate grasp geometry ---
        try:
            grasp_center, finger_axis, rot, jaw_4_pos, jaw_5_pos, gripper_width = self._compute_grasp_geometry()
        except p.error:
            grasp_center = np.array([0, 0, 0])
            finger_axis = np.array([0, 0, -1])
            rot = np.eye(3)
            jaw_4_pos = np.array([0, 0, 0])
            jaw_5_pos = np.array([0, 0, 0])
            gripper_width = 0.0

        gripper_joint = current_joints[5]  # jaw opening (0 = closed, positive = open)

        # ===================== REWARD =====================
        reward = 0.0

        # --- 1. ORIENTATION: gripper fingers must point DOWNWARD ---
        # rot[:, 2] is the gripper's local Z in world.  When that Z points UP
        # (world +Z), the fingers (-Z) point DOWN.  orientation_score → +1 ideal.
        orientation_score = rot[2, 2]
        reward += orientation_score * 3.0

        # --- 2. XY ALIGNMENT: get directly above the block first ---
        xy_dist = np.linalg.norm(grasp_center[:2] - block_pos[:2])
        reward -= xy_dist * 15.0

        # --- 3. Z APPROACH: descend only when horizontally aligned ---
        target_grasp_z = block_pos[2]          # grasp center should reach block center height
        z_error = abs(grasp_center[2] - target_grasp_z)

        if xy_dist < 0.04:
            # Close in XY → reward for correct height
            reward -= z_error * 15.0
            reward += 2.0                       # bonus for being well-aligned
        else:
            # Not yet aligned in XY → stay at safe approach height
            safe_z = block_pos[2] + 0.08
            if grasp_center[2] < safe_z:
                reward -= (safe_z - grasp_center[2]) * 10.0

        # Overall 3D distance signal (softer, to avoid dominating the phases)
        full_dist = np.linalg.norm(grasp_center - block_pos)
        reward -= full_dist * 5.0

        # --- 4. JAW TIMING: open during approach, close when near ---
        close_threshold = 0.04
        if full_dist > close_threshold:
            # Far away ⇒ keep jaws OPEN
            target_jaw = 0.5
            jaw_error = abs(gripper_joint - target_jaw)
            reward -= jaw_error * 1.0
        else:
            # Near the block ⇒ close the jaws
            reward -= gripper_joint * 3.0       # penalty proportional to openness
            if gripper_joint < 0.15:
                reward += 2.0                   # bonus for actually closing

        # --- 5. CONTACT REWARDS ---
        contact_gripper = p.getContactPoints(bodyA=self.arm_id, bodyB=self.trash_id,
                                             linkIndexA=4, physicsClientId=self.client) or ()
        contact_jaw     = p.getContactPoints(bodyA=self.arm_id, bodyB=self.trash_id,
                                             linkIndexA=5, physicsClientId=self.client) or ()
        gripped = len(contact_gripper) > 0 and len(contact_jaw) > 0
        single_contact = len(contact_gripper) > 0 or len(contact_jaw) > 0

        if single_contact:
            reward += 3.0
        if gripped:
            reward += 10.0

        # --- 6. BAD BODY CONTACT: only fingers should touch ---
        for link_idx in range(4):
            bad = p.getContactPoints(bodyA=self.arm_id, bodyB=self.trash_id,
                                     linkIndexA=link_idx, physicsClientId=self.client) or ()
            if len(bad) > 0:
                reward -= 10.0

        # --- 7. SAFETY ---
        if grasp_center[2] < -0.01:
            reward -= 10.0
        if trash_pos[2] < 0.005:
            reward -= 15.0

        # --- 8. EFFORT ---
        reward -= np.sum(np.square(action)) * 0.003

        # --- 9. SMALL TIME PENALTY (encourages efficiency) ---
        reward -= 0.01

        # --- 10. LIFT REWARD  ---
        trash_z = trash_pos[2]
        if trash_z > 0.025 and gripped:
            reward += (trash_z - 0.025) * 3000.0

        terminated = False
        if trash_z > 0.05 and gripped:
            reward += 500.0
            terminated = True
            if self.render_mode:
                print("TARGET GRABBED AND LIFTED!")

        truncated = self.step_counter > 1000

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
            grasp_center, finger_axis, _, jaw_4_pos, jaw_5_pos, gripper_width = self._compute_grasp_geometry()
        except p.error:
            grasp_center = np.array([0, 0, 0])
            finger_axis = np.array([0, 0, -1])
            jaw_4_pos = np.array([0, 0, 0])
            jaw_5_pos = np.array([0, 0, 0])
            gripper_width = 0.0

        rel_pos = np.array(trash_pos) - grasp_center
        grasp_dist = np.linalg.norm(rel_pos)

        jaw4_dist = np.linalg.norm(jaw_4_pos - np.array(trash_pos))
        jaw5_dist = np.linalg.norm(jaw_5_pos - np.array(trash_pos))

        # 6 + 3 + 3 + 3 + 3 + 1 + 1 + 1 + 1 = 22
        return np.array(
            joints +
            list(trash_pos) +
            list(grasp_center) +
            list(finger_axis) +
            list(rel_pos) +
            [grasp_dist] +
            [gripper_width] +
            [jaw4_dist] +
            [jaw5_dist],
            dtype=np.float32
        )

    def close(self):
        # FIX: self.client passed so only this env's physics instance is disconnected
        try:
            p.disconnect(self.client)
        except p.error:
            pass


def make_env():
    """
    Factory for SubprocVecEnv. Must be a module-level function so each
    subprocess can pickle and call it independently.
    """
    def _init():
        return RobotArmEnv(render_mode=False)
    return _init


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- GPU DETECTION ---
    # Priority: CUDA (A40 / RTX 5090) > Apple MPS (M1/M2/M3/M4) > CPU
    print("Checking for available GPUs...")
    if torch.cuda.is_available():
        DEVICE = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ CUDA GPU DETECTED: {gpu_name}")
        print(f"  VRAM: {vram_gb:.1f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = "mps"
        print("✓ Apple MPS GPU DETECTED (M1/M2/M3/M4)")
        print("  Using Metal GPU acceleration.")
        print("  Note: if you see MPS tensor errors, set DEVICE='cpu' manually.")
    else:
        DEVICE = "cpu"
        print("✗ No GPU detected. Training on CPU.")
        print("\n  To enable CUDA (RTX 5090):")
        print("  1. Check NVIDIA drivers:  nvidia-smi")
        print("  2. If nvidia-smi fails, install NVIDIA drivers")
        print("  3. Reinstall PyTorch with CUDA:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        print("  4. Verify torch.cuda.is_available() = True in Python")
        print()

    # ------------------------------------------------------------------ #
    #  CLI — how to use:                                                   #
    #                                                                      #
    #  Fresh training from scratch (single env):                          #
    #    python arm_sac_v1.py train --save so101_SAC_v1                   #
    #                                                                      #
    #  Fast training on RTX 5090 with 8 parallel envs:                   #
    #    python arm_sac_v1.py train --save so101_SAC_v1 --envs 8          #
    #                                                                      #
    #  Continue training an existing model:                                #
    #    python arm_sac_v1.py train --model so101_SAC_v1 --save so101_SAC_v2 --envs 8
    #                                                                      #
    #  Override number of steps:                                           #
    #    python arm_sac_v1.py train --save so101_SAC_v1 --steps 5000000   #
    #                                                                      #
    #  Watch a saved model in the GUI:                                     #
    #    python arm_sac_v1.py watch --model so101_SAC_v1                  #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(
        description="SO-101 Arm — SAC Training & Evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- train subcommand ---
    train_parser = subparsers.add_parser("train", help="Train a SAC model.")
    train_parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL_NAME",
        help=(
            "Name of an existing model to CONTINUE training from.\n"
            "Omit this flag to start a FRESH training run.\n"
            "Example: --model so101_SAC_v1"
        ),
    )
    train_parser.add_argument(
        "--save",
        type=str,
        required=True,
        metavar="SAVE_NAME",
        help=(
            "Name to save the trained model as (no .zip needed).\n"
            "Example: --save so101_SAC_v2"
        ),
    )
    train_parser.add_argument(
        "--steps",
        type=int,
        default=2_000_000,
        metavar="N",
        help="Number of training timesteps (default: 2,000,000).",
    )
    train_parser.add_argument(
        "--envs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel PyBullet environments (default: 1).\n"
            "More envs = faster data collection = GPU stays fed.\n"
            "Recommended for RTX 5090: Match your CPU logical processors (e.g., --envs 45).\n"
            "Each env runs in its own CPU process (SubprocVecEnv).\n"
            "Example: --envs 45"
        ),
    )

    # --- watch subcommand ---
    watch_parser = subparsers.add_parser("watch", help="Watch a saved model in the GUI.")
    watch_parser.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="MODEL_NAME",
        help=(
            "Name of the model to watch (no .zip needed).\n"
            "Example: --model so101_SAC_v1"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  SAC HYPERPARAMETERS (used only when training from scratch)         #
    #  batch_size auto-scales with --envs so the GPU stays saturated.    #
    # ------------------------------------------------------------------ #
    n_envs = getattr(args, "envs", 1)  # watch mode has no --envs flag
    # For RTX 5090: Use a massive batch size to saturate the GPU in fewer passes.
    # We will scale it up directly to 4096 and use fewer gradient_steps.
    scaled_batch = min(512 * max(n_envs, 1), 4096)
    # learning_starts also scales: wait for ~10k steps worth of real data
    # regardless of how many envs are collecting simultaneously.
    scaled_learning_starts = max(10_000, 1_000 * n_envs)
    sac_kwargs = dict(
        policy="MlpPolicy",
        verbose=0,
        device=DEVICE,
        tensorboard_log="./sac_tensorboard/",
        buffer_size=1_000_000,          # doubled — parallel envs fill the buffer fast
        learning_starts=scaled_learning_starts,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=scaled_batch,        # auto-scaled to keep GPU fed
        train_freq=1,
        gradient_steps=2,               # Reduced from -1 (n_envs updates). Big batches, fewer loops!
        use_sde=True,
        sde_sample_freq=64,
        policy_kwargs=dict(net_arch=[512, 512, 256]),
    )

    # ================================================================== #
    #  TRAIN MODE                                                          #
    # ================================================================== #
    if args.mode == "train":
        save_path = os.path.join(script_dir, args.save)
        training_steps = args.steps
        n_envs = args.envs

        # ---- Build the environment (parallel or single) ----
        if n_envs > 1:
            # SubprocVecEnv: each env runs in its own OS process.
            # This bypasses Python's GIL and gives true CPU parallelism.
            # Each process gets its own PyBullet physics client (p.DIRECT).
            # "fork" is fastest on Linux; Windows requires "spawn".
            import platform
            start_method = "fork" if platform.system() != "Windows" else "spawn"
            env = SubprocVecEnv([make_env() for _ in range(n_envs)],
                                start_method=start_method)
        else:
            env = DummyVecEnv([make_env()])             # single env, no overhead

        print(f"\n{'='*55}")
        print(f"  Mode         : TRAIN")
        print(f"  RL Algorithm : SAC + gSDE")
        print(f"  Device       : {DEVICE.upper()} ")
        print(f"  Steps        : {training_steps:,}")
        print(f"  Parallel envs: {n_envs}")
        print(f"  Batch size   : {scaled_batch}")
        print(f"  Network      : [512, 512, 256]")
        print(f"{'='*55}\n")

        if args.model is not None:
            # ------ CONTINUE training from existing model ------
            load_path = os.path.join(script_dir, args.model)
            if not os.path.exists(load_path + ".zip"):
                print(f"\nERROR: Model '{load_path}.zip' not found.")
                print("       Check available models with: ls *.zip")
                env.close()
                exit(1)
            print(f"  Source model : {args.model}.zip  (continuing)")
            print(f"  Save to      : {args.save}.zip")
            print(f"{'='*55}\n")
            custom_objects = {
                "learning_rate": 3e-4,
                "lr_schedule": lambda _: 3e-4,
                "clip_range": lambda _: 0.2,
            }
            model = SAC.load(load_path, env=env, device=DEVICE, custom_objects=custom_objects)
            # Re-apply scaled batch size in case it differs from saved model
            model.batch_size = scaled_batch
            model.gradient_steps = 2
            fresh = False
        else:
            # ------ FRESH training from scratch ------
            print(f"  Source model : (none — fresh start)")
            print(f"  Save to      : {args.save}.zip")
            print(f"{'='*55}\n")
            model = SAC(env=env, **sac_kwargs)
            fresh = True

        callback = ProgressBarCallback(training_steps)
        model.learn(
            total_timesteps=training_steps,
            callback=callback,
            reset_num_timesteps=fresh,  # reset step counter only on fresh runs
            log_interval=1,
        )

        model.save(save_path)
        print(f"\nTRAINING DONE! Model saved to '{save_path}.zip'")
        env.close()

    # ================================================================== #
    #  WATCH MODE                                                          #
    # ================================================================== #
    elif args.mode == "watch":
        load_path = os.path.join(script_dir, args.model)
        if not os.path.exists(load_path + ".zip"):
            print(f"ERROR: Model '{load_path}.zip' not found.")
            print("       Check available models with: ls *.zip")
            exit(1)

        print(f"\n{'='*55}")
        print(f"  Mode   : WATCH")
        print(f"  Model  : {args.model}.zip")
        print(f"  Device : {DEVICE.upper()}")
        print(f"{'='*55}\n")

        env = RobotArmEnv(render_mode=True)
        custom_objects = {
            "learning_rate": 3e-4,
            "lr_schedule": lambda _: 3e-4,
            "clip_range": lambda _: 0.2,
        }
        model = SAC.load(load_path, env=env, device=DEVICE, custom_objects=custom_objects)

        print("Watching... close the PyBullet window to exit.")
        obs, _ = env.reset()
        episode = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)

            if done or truncated:
                episode += 1
                if done:
                    print(f"[Episode {episode}] SUCCESS! Block lifted! Resetting...")
                    time.sleep(1.5)
                else:
                    print(f"[Episode {episode}] Time limit. Resetting...")
                obs, _ = env.reset()
