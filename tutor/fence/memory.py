"""Shared implementation; controller-specific prompts/contracts remain separate."""
from tutor.core import memory as _shared
globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})
