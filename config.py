VARIANTS = {
    "baseline": {
        "policy": "MlpPolicy",
        "total_timesteps": 1_000_000,
        "normalize_obs": False,
        "custom_reward": False,
        "param_noise": False,
        "entropy_anneal": False,
    },
    "obs_norm": {
        "policy": "MlpPolicy",
        "total_timesteps": 1_000_000,
        "normalize_obs": True,
        "custom_reward": False,
        "param_noise": False,
        "entropy_anneal": False,
    },
    "obs_norm_shaping": {
        "policy": "MlpPolicy", 
        "total_timesteps": 1_000_000,
        "normalize_obs": True,
        "custom_reward": True,
        "param_noise": False,
        "entropy_anneal": False,
    },
        "obs_norm_paramnoise": {
        "policy": "MlpPolicy",
        "total_timesteps": 1_000_000,
        "normalize_obs": True,
        "custom_reward": False,
        "param_noise": True,
        "entropy_anneal": False,
    },
        "obs_norm_entropy_anneal": {
        "policy": "MlpPolicy",
        "total_timesteps": 1_000_000,
        "normalize_obs": True,
        "custom_reward": False,
        "param_noise": False,
        "entropy_anneal": True,
    },
    "obs_norm_shaping_noise": {
        "policy": "MlpPolicy",
        "total_timesteps": 1_000_000,
        "normalize_obs": True,
        "custom_reward": True,
        "param_noise": True,
        "entropy_anneal": False,
    },
    "full_combo": {
        "policy": "MlpPolicy",
        "total_timesteps": 1_000_000,
        "normalize_obs": True,
        "custom_reward": True,
        "param_noise": True,
        "entropy_anneal": True,
    }
}


PPO_DEFAULTS = {
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "clip_range": 0.2,
}
