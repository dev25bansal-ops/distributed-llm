"""Cache warmer for pre-populating KV caches."""

import json
from pathlib import Path
from typing import List

from loguru import logger


class CacheWarmer:
    """Pre-populates KV caches by running prompts through the pipeline."""

    def warm(self, prompts: List[str], coordinator) -> int:
        """Run prompts through the pipeline to populate caches.

        Args:
            prompts: List of prompt strings to warm.
            coordinator: Coordinator instance to use for generation.

        Returns:
            Number of successfully warmed prompts.
        """
        warmed = 0
        for prompt in prompts:
            try:
                coordinator.generate(prompt, max_new_tokens=1)
                warmed += 1
            except Exception as e:
                logger.warning(f"Cache warm failed for prompt: {e}")
        logger.info(f"Cache warmer: warmed {warmed}/{len(prompts)} prompts")
        return warmed

    def warm_from_file(self, file_path: str, coordinator) -> int:
        """Load prompts from a JSON file and warm caches.

        Args:
            file_path: Path to JSON file (list of strings or {"prompts": [...]}).
            coordinator: Coordinator instance to use for generation.

        Returns:
            Number of successfully warmed prompts.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        data = json.loads(path.read_text())
        prompts = data if isinstance(data, list) else data.get("prompts", [])
        return self.warm(prompts, coordinator)
