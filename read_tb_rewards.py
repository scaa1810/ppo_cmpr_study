import os
from tensorboard.backend.event_processing import event_accumulator

LOG_DIR = "logs"

def get_final_reward(event_path, tag="rollout/ep_rew_mean"):
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return None
    scalars = ea.Scalars(tag)
    if not scalars:
        return None
    return scalars[-1].value  # last logged reward

for run_dir in sorted(os.listdir(LOG_DIR)):
    full_dir = os.path.join(LOG_DIR, run_dir)
    if not os.path.isdir(full_dir):
        continue
    # find event file
    event_files = [f for f in os.listdir(full_dir) if "tfevents" in f]
    if not event_files:
        continue
    event_path = os.path.join(full_dir, event_files[0])
    final_rew = get_final_reward(event_path)
    print(run_dir, "→ final ep_rew_mean:", final_rew)
