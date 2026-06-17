"""Thread-safe pool of identical model replicas for parallel inference."""

from __future__ import annotations

import queue
import threading
from typing import Any

from deploy.inference import generate_from_loaded, load_model_and_processor


class ModelPool:
    def __init__(self, model_path: str, num_replicas: int, **load_kwargs: Any) -> None:
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")

        self.model_path = model_path
        self.replicas: list[tuple[Any, Any]] = []
        self._free: queue.Queue[int] = queue.Queue()

        for i in range(num_replicas):
            print(f"Loading model replica {i + 1}/{num_replicas} from {model_path}...")
            self.replicas.append(load_model_and_processor(model_path, **load_kwargs))
            self._free.put(i)

        self._stats_lock = threading.Lock()
        self._in_flight = 0
        self._completed = 0

    @property
    def num_replicas(self) -> int:
        return len(self.replicas)

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {
                "num_replicas": self.num_replicas,
                "in_flight": self._in_flight,
                "completed": self._completed,
            }

    def generate(self, image, prompt: str, **kwargs: Any) -> str:
        replica_idx = self._free.get()
        with self._stats_lock:
            self._in_flight += 1
        try:
            model, processor = self.replicas[replica_idx]
            return generate_from_loaded(model, processor, image, prompt, **kwargs)
        finally:
            with self._stats_lock:
                self._in_flight -= 1
                self._completed += 1
            self._free.put(replica_idx)
