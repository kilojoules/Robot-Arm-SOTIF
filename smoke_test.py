"""End-to-end smoke test: Octo policy in SimplerEnv with dust."""

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.policies.octo.octo_model import OctoInference
import numpy as np

print("Creating env + policy...")
env = simpler_env.make("google_robot_pick_coke_can")
policy = OctoInference(model_type="octo-base", policy_setup="google_robot")

obs, _ = env.reset()
instruction = env.get_language_instruction()
policy.reset(instruction)
print("Instruction:", instruction)

image = get_image_from_maniskill2_obs_dict(env, obs)
print("Image shape:", image.shape)

# Apply dust model
from adversarial_dust.config import DustGridConfig
from adversarial_dust.dust_model import AdversarialDustModel

dust_config = DustGridConfig()
dust_model = AdversarialDustModel(dust_config, image.shape, budget_level=0.2)
rng = np.random.default_rng(42)
dust_params = dust_model.get_random_params(rng)
dirty_image = dust_model.apply(image, dust_params)
alpha = dust_model.grid_to_alpha_mask(dust_model.project_to_budget(dust_model.params_to_grid(dust_params)))
coverage = dust_model.compute_coverage(alpha)
print("Dust coverage:", coverage)
print("Dirty image shape:", dirty_image.shape, "dtype:", dirty_image.dtype)

# Run a few steps with dirty images
for i in range(5):
    image = get_image_from_maniskill2_obs_dict(env, obs)
    dirty_image = dust_model.apply(image, dust_params)
    raw_action, action = policy.step(dirty_image, instruction)
    action_array = np.concatenate([
        action["world_vector"],
        action["rot_axangle"],
        action["gripper"].flatten(),
    ])
    obs, reward, done, truncated, info = env.step(action_array)
    success = info.get("success", False)
    print(f"Step {i+1}: reward={reward:.3f}, done={done}, success={success}")
    if done or truncated:
        break

env.close()
print("End-to-end smoke test PASSED!")
