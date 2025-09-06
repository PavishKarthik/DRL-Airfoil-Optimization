import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from gymnasium.wrappers import RescaleAction

from drl_environment import AirfoilOptEnv

# --- CONFIGURATION ---
log_dir = "ppo_airfoil_logs/"
model_dir = "ppo_airfoil_models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Define the path for the resumable model
INTERRUPTED_MODEL_PATH = f"{model_dir}/ppo_airfoil_model_interrupted.zip"

CHECKPOINT_FREQ = 10000
EVAL_FREQ = 5000

# --- SETUP THE ENVIRONMENTS ---
train_env = AirfoilOptEnv()
eval_env = AirfoilOptEnv()

train_env = RescaleAction(train_env, min_action=-1, max_action=1)
eval_env = RescaleAction(eval_env, min_action=-1, max_action=1)

check_env(train_env)

# --- SETUP THE CALLBACKS ---
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ, save_path=model_dir, name_prefix="rl_model"
)
eval_callback = EvalCallback(
    eval_env, best_model_save_path=model_dir, log_path=log_dir,
    eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=1
)

# --- SETUP THE DRL AGENT (LOAD OR CREATE) ---
# This is the "resume" logic. Check if an interrupted model exists.
if os.path.exists(INTERRUPTED_MODEL_PATH):
    print(f"\n--- Resuming training from {INTERRUPTED_MODEL_PATH} ---")
    model = PPO.load(INTERRUPTED_MODEL_PATH, env=train_env, tensorboard_log=log_dir)
else:
    print("\n--- Starting a new training session ---")
    policy_kwargs = dict(log_std_init=-2, ortho_init=False)
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir
    )

# --- TRAIN THE AGENT (WITH INTERRUPTION HANDLING) ---
TIMESTEPS = 50000
run_name = "PPO_run_final"

print(f"\n--- Starting Training for {TIMESTEPS} Timesteps ---")
print("Press Ctrl+C to interrupt and save the current model.")

try:
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=False, # This is crucial for resuming correctly
        tb_log_name=run_name
    )
    # If training completes without interruption, save a final version
    model.save(f"{model_dir}/ppo_airfoil_model_completed_{TIMESTEPS}.zip")
    print("\n--- Training Completed Successfully ---")

except KeyboardInterrupt:
    # This block runs if you press Ctrl+C
    print("\n--- Training interrupted by user ---")
    model.save(INTERRUPTED_MODEL_PATH)
    print(f"Model saved to {INTERRUPTED_MODEL_PATH}")
    print("You can run the script again to resume training.")
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

finally:
    # --- CLEANUP ---
    train_env.close()
    eval_env.close()