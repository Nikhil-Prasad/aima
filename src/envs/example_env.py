"""Example Gymnasium environment for RL experiments."""

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from crown_common import get_logger

logger = get_logger(__name__)


class ExampleEnv(gym.Env):
    """
    Simple example environment for demonstration.
    
    The agent must learn to output a target value.
    """
    
    def __init__(self, target: float = 0.5, max_steps: int = 100):
        super().__init__()
        
        self.target = target
        self.max_steps = max_steps
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
        
        self.current_step = 0
        self.current_value = 0.0
        
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_value = self.np_random.uniform(-1.0, 1.0)
        
        obs = self._get_obs()
        info = self._get_info()
        
        logger.debug("Environment reset", target=self.target, initial_value=self.current_value)
        
        return obs, info
    
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step in the environment."""
        # Update value based on action
        self.current_value += action[0] * 0.1
        self.current_value = np.clip(self.current_value, -2.0, 2.0)
        
        # Calculate reward (negative distance to target)
        distance = abs(self.current_value - self.target)
        reward = -distance
        
        # Check termination
        self.current_step += 1
        terminated = distance < 0.01  # Success condition
        truncated = self.current_step >= self.max_steps
        
        obs = self._get_obs()
        info = self._get_info()
        
        if terminated:
            logger.info("Episode succeeded", steps=self.current_step, final_value=self.current_value)
        elif truncated:
            logger.info("Episode truncated", steps=self.current_step, final_value=self.current_value)
        
        return obs, float(reward), terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return np.array([self.current_value, self.target - self.current_value], dtype=np.float32)
    
    def _get_info(self) -> dict[str, Any]:
        """Get current info dict."""
        return {
            "current_value": self.current_value,
            "target": self.target,
            "distance": abs(self.current_value - self.target),
            "step": self.current_step,
        }
    
    def render(self) -> None:
        """Render the environment (optional)."""
        print(f"Step {self.current_step}: Value={self.current_value:.3f}, Target={self.target:.3f}")


# Register the environment
gym.register(
    id="CrownExample-v0",
    entry_point="src.envs.example_env:ExampleEnv",
    max_episode_steps=100,
)