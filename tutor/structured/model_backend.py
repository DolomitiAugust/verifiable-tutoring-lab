"""Shared implementation; controller-specific prompts/contracts remain separate."""
from tutor.core import model_backend as _shared
globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})
