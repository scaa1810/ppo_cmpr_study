import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import argparse
from config import VARIANTS, PPO_DEFAULTS

class LunarLanderWrapper(gym.Wrapper):
    """Custom wrapper: obs norm + reward shaping"""
    def __init__(self, env, normalize_obs=False, custom_reward=False):
        super().__init__(env)
        self.normalize_obs = normalize_obs
        self.custom_reward = custom_reward
        self.shaping_scale = 0.01
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.custom_reward:
            x, y = obs[0], obs[1]
            distance = np.sqrt(x**2 + y**2)
            shaping_reward = -self.shaping_scale * distance
            reward += shaping_reward
        
        obs = self._process_obs(obs)
        return obs, reward, terminated, truncated, info
    
    def _process_obs(self, obs):
        if not self.normalize_obs:
            return obs
        return (obs - obs.mean()) / (obs.std() + 1e-8)

def make_env(seed=0, normalize_obs=False, custom_reward=False):
    """Returns a FUNCTION that creates the environment"""
    def _init():
        env = gym.make("LunarLanderContinuous-v2")
        env = Monitor(env)
        return LunarLanderWrapper(env, normalize_obs, custom_reward)
    return _init

def train_variant(variant_name, seeds=[0,1,2]):
    config = VARIANTS[variant_name].copy()
    config.update(PPO_DEFAULTS)
    
    print(f"\n🚀 Training {variant_name.upper()} (seeds: {seeds})")
    
    for seed in seeds:
        print(f"\n--- SEED {seed} ---")
        
        # Create environment factory
        env_fn = make_env(
            seed=seed,
            normalize_obs=config["normalize_obs"],
            custom_reward=config["custom_reward"]
        )
        
        # VecEnv
                # VecEnv
        env = DummyVecEnv([env_fn])
        
        if config["normalize_obs"]:
            env = VecNormalize(env, norm_reward=False, training=True)
        
        # Optional action noise for exploration
        action_noise = None
        if config.get("param_noise", False):
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.1 * np.ones(n_actions)
            )
        
                # PPO with optional entropy annealing
        ent_coef = 0.01 if config.get("entropy_anneal", False) else 0.0
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            device='cpu',
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            learning_rate=config["learning_rate"],
            clip_range=config["clip_range"],
            ent_coef=ent_coef,  # ← Start at 0.01
            tensorboard_log="./logs/",
        )


        
        # Train
        model.learn(total_timesteps=config["total_timesteps"])
        
        # Save
        save_path = f"results/{variant_name}_seed{seed}"
        model.save(save_path)
        if config["normalize_obs"]:
            env.save(f"{save_path}_vec_normalize.pkl")
        
        print(f"✅ SAVED: {save_path}")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="baseline", 
                       choices=list(VARIANTS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    args = parser.parse_args()
    
    train_variant(args.variant, args.seeds)
