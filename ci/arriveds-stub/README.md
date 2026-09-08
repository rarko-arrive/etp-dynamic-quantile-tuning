# arriveds CI stub

GitHub Actions uses this minimal package instead of the private
[`arrive-ds`](https://github.com/rarko-arrive/arrive-ds) git dependency.

`ci/patch_arriveds_source.py` swaps the `tool.uv.sources` entry before
`uv sync`. Local development keeps the real SSH git source in `pyproject.toml`.

The stub implements only what unit tests and smoke need:

- `SnowflakeConfig.from_env()`
- `create_table()` (no-op row count)
- `query_sf()` (raises if called)

Snowflake integration tests are mocked; production and local runs use real
`arriveds` from arrive-ds.
