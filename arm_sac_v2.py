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

        # OBSERVATION: 22 values — joints(6) + block_pos(3) + tcp_pos(3) + tcp_ori(4) + rel_pos(3) + tcp_dist(1) + jaw4_dist(1) + jaw5_dist(1)
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
            link_4_state = p.getLinkState(self.arm_id, 4, computeForwardKinematics=1, physicsClientId=self.client)
            link_5_state = p.getLinkState(self.arm_id, 5, computeForwardKinematics=1, physicsClientId=self.client)
            jaw_4_pos = np.array(link_4_state[0])
            jaw_5_pos = np.array(link_5_state[0])
            # True Tool Center Point between jaws
            tcp_pos = (jaw_4_pos + jaw_5_pos) / 2.0
            
            # The orientation of the overall gripper can simply take from the base
            tcp_ori = link_5_state[1]
        except p.error:
            # Fallback
            tcp_pos = np.array([0, 0, 0])
            jaw_4_pos = np.array([0, 0, 0])
            jaw_5_pos = np.array([0, 0, 0])
            
        dist = np.linalg.norm(tcp_pos - np.array(trash_pos))
        
        # --- Timer Penalty ---
        # Sense of urgency: constant drain every timestep to force fast grabbing
        reward = -0.5

        # --- Reward 1: Approach ---
        # Target the center of the block directly to enclose it
        target_pos = np.array(trash_pos)
        real_dist = np.linalg.norm(tcp_pos - target_pos)
        reward += -real_dist * 5.0

        # --- Reward 2: Crane approach & Centering ---
        xy_dist = np.linalg.norm(tcp_pos[:2] - np.array(trash_pos[:2]))
        if xy_dist > 0.06:
            # Too far laterally → penalise being low to prevent sweeping
            if tcp_pos[2] < 0.10:
                reward -= 10.0
        else:
            # Hovering above block → explicitly reward staying perfectly centered
            xy_centering_reward = max(0, 0.06 - xy_dist)
            reward += xy_centering_reward * 50.0

        # --- Reward 3: Grasp (both jaw faces must contact block) ---
        contact_stationary = p.getContactPoints(bodyA=self.arm_id, bodyB=self.trash_id,
                                                linkIndexA=4, physicsClientId=self.client) or ()
        contact_moving = p.getContactPoints(bodyA=self.arm_id, bodyB=self.trash_id,
                                            linkIndexA=5, physicsClientId=self.client) or ()
        gripped = len(contact_stationary) > 0 and len(contact_moving) > 0

        if gripped:
            reward += 50.0  # Jackpot: block sandwiched between jaws

        # --- Reward 4: Jaw Actuation ---
        gripper_joint = current_joints[5]
        if dist > 0.05:
            reward += gripper_joint * 5.0  # Encourage open jaws far away
        elif dist < 0.04:
            reward -= gripper_joint * 5.0  # Encourage closing jaws when super close

        # --- Penalty 1: Squishing the block & Smashing the floor ---
        if trash_pos[2] < 0.015:
            # Pushing the block into the floor
            reward -= 15.0
            
        if tcp_pos[2] < 0.01:
            # Crashing the true center into the floor
            reward -= 15.0

        # --- Penalty 2: Arm body hitting block (only fingers should touch) ---
        for link_idx in range(4):
            bad_contact = p.getContactPoints(bodyA=self.arm_id, bodyB=self.trash_id,
                                             linkIndexA=link_idx, physicsClientId=self.client) or ()
            if len(bad_contact) > 0:
                reward -= 10.0

        # --- Penalty 3: Jittery/high-effort movement ---
        reward -= np.sum(np.square(action)) * 0.001

        # --- Reward 5: Lift ---
        trash_z = trash_pos[2]
        if trash_z > 0.05 and gripped:
            reward += (trash_z - 0.02) * 1000

        terminated = False
        if trash_z > 0.1 and gripped:
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
            link_4_state = p.getLinkState(self.arm_id, 4, computeForwardKinematics=1, physicsClientId=self.client)
            link_5_state = p.getLinkState(self.arm_id, 5, computeForwardKinematics=1, physicsClientId=self.client)
            jaw_4_pos = np.array(link_4_state[0])
            jaw_5_pos = np.array(link_5_state[0])
            
            tcp_pos = (jaw_4_pos + jaw_5_pos) / 2.0
            tcp_ori = link_5_state[1]
        except p.error:
            tcp_pos = np.array([0, 0, 0])
            tcp_ori = [0, 0, 0, 1]
            jaw_4_pos = np.array([0, 0, 0])
            jaw_5_pos = np.array([0, 0, 0])

        rel_pos = np.array(trash_pos) - tcp_pos
        tcp_dist = np.linalg.norm(rel_pos)
        
        jaw4_dist = np.linalg.norm(jaw_4_pos - np.array(trash_pos))
        jaw5_dist = np.linalg.norm(jaw_5_pos - np.array(trash_pos))

        # 6 + 3 + 3 + 4 + 3 + 1 + 1 + 1 = 22
        return np.array(
            joints +
            list(trash_pos) +
            list(tcp_pos) +
            list(tcp_ori) +
            list(rel_pos) +
            [tcp_dist] +
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
            "Recommended for RTX 5090: --envs 8\n"
            "Each env runs in its own CPU process (SubprocVecEnv).\n"
            "Example: --envs 8"
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
        print(f"  Device       : {DEVICE.upper()}  ◄ THIS IS WHERE YOUR MODEL TRAINS")
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
