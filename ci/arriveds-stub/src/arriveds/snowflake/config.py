from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SnowflakeConfig:
    database: str | None = None
    schema: str | None = None

    @classmethod
    def from_env(cls) -> SnowflakeConfig:
        return cls(
            database=os.environ.get("SNOWFLAKE_DATABASE"),
            schema=os.environ.get("SNOWFLAKE_SCHEMA"),
        )
