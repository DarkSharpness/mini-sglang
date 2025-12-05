from __future__ import annotations

import os


class BaseEnv:
    def init(self, name: str) -> None:
        raise NotImplementedError


class EnvBool(BaseEnv):
    def __init__(self, default_value: bool):
        self.value = default_value
        super().__init__()

    def init(self, name: str) -> None:
        env_value = os.getenv(name)
        if env_value is not None:
            self.value = env_value.lower() in ("1", "true", "yes", "on")

    def __bool__(self):
        return self.value


MINISGL_ENV_PREFIX = "MINISGL_"


class EnvClassSingleton:
    BENCHMARK_SKIP_TOKENIZE_CHECK = EnvBool(False)

    def __new__(cls):
        # single instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            attr_value.init(f"{MINISGL_ENV_PREFIX}{attr_name}")


ENV = EnvClassSingleton()
