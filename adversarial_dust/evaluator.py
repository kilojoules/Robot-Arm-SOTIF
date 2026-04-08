"""Abstract base class for policy evaluation under occlusion."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PolicyEvaluator(ABC):
    """Abstract base for evaluating a manipulation policy under occlusion.

    Concrete implementations:
      - ``SimplerEnvEvaluator``: SimplerEnv / ManiSkill2 with CPU post-process.
      - ``IsaacSimEvaluator``: Isaac Sim single-env with shader occlusion.
      - ``IsaacBatchEvaluator``: Isaac Sim GPU-parallel batch evaluation.
    """

    @abstractmethod
    def run_episode(
        self, dust_params: Optional[np.ndarray], record: bool = False
    ) -> Tuple[bool, List[np.ndarray], List[np.ndarray]]:
        """Run a single episode.

        Args:
            dust_params: Occlusion parameter vector, or None for clean.
            record: If True, collect clean and dirty frames each step.

        Returns:
            (success, clean_frames, dirty_frames)
        """
        ...

    def evaluate(
        self, dust_params: Optional[np.ndarray], n_episodes: int
    ) -> float:
        """Run *n_episodes*, return mean success rate.

        Default implementation calls ``run_episode`` sequentially.
        """
        successes = []
        for ep in range(n_episodes):
            success, _, _ = self.run_episode(dust_params, record=False)
            successes.append(float(success))
            logger.debug(f"Episode {ep + 1}/{n_episodes}: success={success}")

        mean_sr = float(np.mean(successes))
        logger.info(
            f"Evaluation: {n_episodes} episodes, success_rate={mean_sr:.3f}"
        )
        return mean_sr

    def batch_evaluate(
        self,
        candidates: list,
        n_episodes: int,
    ) -> list:
        """Evaluate multiple candidate parameter vectors.

        Default implementation calls ``evaluate`` sequentially.
        GPU-parallel evaluators override this for simultaneous evaluation.

        Args:
            candidates: List of parameter arrays (one per CMA-ES candidate).
            n_episodes: Episodes per candidate.

        Returns:
            List of success rates (one per candidate).
        """
        return [
            self.evaluate(np.asarray(c), n_episodes) for c in candidates
        ]
