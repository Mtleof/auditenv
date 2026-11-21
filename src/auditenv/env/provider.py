"""Episode providers: produce (public files, grader) pairs. The provider - not the environment -
owns the truth; it is the evaluator's side of the boundary."""
from __future__ import annotations

import json
import os

from ..generator import generate
from ..generator.episode import PUBLIC_FILES
from ..reward import Grader
from .bundle import PublicBundle


class GeneratorProvider:
    """Generates episodes on demand from (seed, config). Keeps the evaluator log."""

    def __init__(self, transform=None):
        self.log: list[dict] = []
        self._transform = transform

    def provide(self, seed: int, config: dict | None):
        ep = generate(seed, config)
        if self._transform is not None:
            ep = self._transform(ep)
        files = ep.public_files()
        bundle = PublicBundle(files)
        return bundle, Grader(ep.truth, bundle.known_ids(), self.log)


class EpisodeProvider:
    """Serves pre-built Episode objects (e.g. transformed copies for paired checks)."""

    def __init__(self, episodes):
        self.log: list[dict] = []
        self._eps = list(episodes)

    def provide(self, seed: int, config: dict | None):
        ep = self._eps[seed]
        bundle = PublicBundle(ep.public_files())
        return bundle, Grader(ep.truth, bundle.known_ids(), self.log)


class DirectoryProvider:
    """Serves episodes from disk: public files from episode_dirs[i] and, evaluator-side only, truth from
    grader_dirs[i]. reset(seed=i) selects the i-th episode. Works on files edited by third parties
    (reordered rows, renamed ids): nothing is re-derived from a generator seed."""

    def __init__(self, episode_dirs, grader_dirs):
        if len(episode_dirs) != len(grader_dirs):
            raise ValueError("episode_dirs and grader_dirs must align")
        self.log: list[dict] = []
        self._pairs = list(zip(episode_dirs, grader_dirs))

    @classmethod
    def from_split(cls, root: str, split: str) -> "DirectoryProvider":
        pub = os.path.join(root, "episodes", split)
        ids = sorted(d for d in os.listdir(pub) if os.path.isdir(os.path.join(pub, d)))
        return cls([os.path.join(pub, i) for i in ids], [os.path.join(root, "grader", split, i) for i in ids])

    def __len__(self) -> int:
        return len(self._pairs)

    def load(self, index: int):
        pub, prv = self._pairs[index]
        files = {}
        for name in PUBLIC_FILES:
            with open(os.path.join(pub, name), encoding="utf-8") as fh:
                files[name] = fh.read()
        with open(os.path.join(prv, "truth.json"), encoding="utf-8") as fh:
            truth = json.load(fh)
        bundle = PublicBundle(files)
        if truth.get("episode_id") != bundle.episode_id:
            raise ValueError("truth.json does not belong to this episode")
        return files, truth, bundle

    def provide(self, seed: int, config: dict | None):
        _, truth, bundle = self.load(seed)
        return bundle, Grader(truth, bundle.known_ids(), self.log)
