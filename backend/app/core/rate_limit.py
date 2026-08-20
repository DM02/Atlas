from slowapi import Limiter
from slowapi.util import get_remote_address

# Single-process, in-memory rate limiting — fine until Redis lands in Phase 4,
# at which point this should move to a distributed limiter (multiple backend
# replicas would otherwise each get their own independent limit).
limiter = Limiter(key_func=get_remote_address)
