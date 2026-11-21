# Config versions

| config_version | generator | env | changes |
|---|---|---|---|
| 1.0.0 | 1.0.0 | 1.0.0 | Initial release: 4 error types, 2 interaction scenarios, 9 distractor kinds, `row_order`. |

* `default-<version>.json` — `DEFAULT_CONFIG`; any user config is deep-merged onto it.
* `suite-<version>.json` — the 20 train/dev configs with `config_hash` (sha256 of the canonical resolved
  config, first 16 hex chars; it is part of the RNG key). Seeds are evaluator-side and live in
  `grader/<split>/index.json`, not here.
* Bump `CONFIG_VERSION` whenever a default or the meaning of a key changes; the test
  `tests/test_artifacts.py` fails if these files drift from the code.
