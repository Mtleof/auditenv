"""Episode providers: produce (public files, grader) pairs. The provider - not the environment -
owns the truth; it is the evaluator's side of the boundary."""
from __future__ import annotations

from ..generator import generate
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
