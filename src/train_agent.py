import os
import sys
import shutil
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from gymnasium.wrappers import RescaleAction

from drl_environment import AirfoilOptEnv

# --- 1. SCRIPT ARGUMENTS FOR RESTART/RESUME LOGIC ---
parser = argparse.ArgumentParser(description="Train a PPO agent for airfoil optimization.")
parser.add_argument('--restart', action='store_true', help='Flag to force a new training run from scratch.')
args = parser.parse_args()

# --- 2. UPDATED CONFIGURATION ---
log_dir = "ppo_airfoil_logs/"
model_dir = "ppo_airfoil_models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

INTERRUPTED_MODEL_PATH = f"{model_dir}/ppo_airfoil_model_interrupted.zip"
RUN_NAME = "PPO_run_v2_constrained" # Give a new name for the new run

# More frequent checkpoints and a longer total training time
CHECKPOINT_FREQ = 1000  # Was 10000
EVAL_FREQ = 5000
TIMESTEPS = 150000      # Was 50000

# --- SETUP THE ENVIRONMENTS ---
train_env = AirfoilOptEnv()
eval_env = AirfoilOptEnv()

# Rescale actions to [-1, 1], which is standard for PPO
train_env = RescaleAction(train_env, min_action=-1, max_action=1)
eval_env = RescaleAction(eval_env, min_action=-1, max_action=1)

check_env(train_env)

# --- SETUP THE CALLBACKS ---
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ, save_path=model_dir, name_prefix="rl_model"
)
# EvalCallback automatically saves the best performing model as 'best_model.zip'
eval_callback = EvalCallback(
    eval_env, best_model_save_path=model_dir, log_path=log_dir,
    eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=1
)

# --- 3. SETUP THE DRL AGENT (WITH RESTART/RESUME LOGIC) ---
model = None
# If --restart is used, clean up old files to ensure a fresh start
if args.restart:
    print("\n--- '--restart' flag detected. Starting a fresh training run. ---")
    if os.path.exists(INTERRUPTED_MODEL_PATH):
        os.remove(INTERRUPTED_MODEL_PATH)
        print(f"Removed old interrupted model: {INTERRUPTED_MODEL_PATH}")
    # Also clean the specific log directory for this run to avoid messy TensorBoard graphs
    run_log_dir = os.path.join(log_dir, RUN_NAME + "_1") # SB3 appends "_1"
    if os.path.exists(run_log_dir):
        shutil.rmtree(run_log_dir)
        print(f"Removed old logs for this run: {run_log_dir}")

# Default behavior: Resume if an interrupted model exists and --restart is NOT used
if not args.restart and os.path.exists(INTERRUPTED_MODEL_PATH):
    print(f"\n--- Resuming training from {INTERRUPTED_MODEL_PATH} ---")
    model = PPO.load(INTERRUPTED_MODEL_PATH, env=train_env, tensorboard_log=log_dir)
else:
    # This block runs if we are starting fresh (either no interrupted model or --restart was used)
    print("\n--- Initializing a new PPO agent ---")
    # 4. UPDATED HYPERPARAMETERS AND NETWORK ARCHITECTURE
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]) # Larger network for policy (pi) and value (vf)
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=4096,     # Increased for more stable updates
        gamma=0.995,      # Increased to encourage long-term rewards
        tensorboard_log=log_dir,
        device="cuda"
    )

# --- TRAIN THE AGENT (WITH INTERRUPTION HANDLING) ---
print(f"\n--- Starting Training for {TIMESTEPS} Timesteps ---")
print(f"TensorBoard Log Name: {RUN_NAME}")
print("To view logs, run: tensorboard --logdir={}".format(log_dir))
print("Press Ctrl+C to interrupt and save the current model.")

try:
    # Use reset_num_timesteps=False to correctly continue timestep count when resuming
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=False, 
        tb_log_name=RUN_NAME
    )
    model.save(f"{model_dir}/ppo_airfoil_model_completed_{TIMESTEPS}.zip")
    print("\n--- Training Completed Successfully ---")

except KeyboardInterrupt:
    print("\n--- Training interrupted by user ---")
    model.save(INTERRUPTED_MODEL_PATH)
    print(f"Resumable model saved to {INTERRUPTED_MODEL_PATH}")
    print("Run the script again without '--restart' to resume training.")
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
finally:
    train_env.close()
    eval_env.close()