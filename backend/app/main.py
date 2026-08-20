from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.health import router as health_router
from app.api.v1.router import api_router
from app.core.config import get_settings, warn_if_insecure_defaults
from app.core.logging import configure_logging
from app.core.middleware import RequestIDMiddleware
from app.core.rate_limit import limiter

configure_logging()
settings = get_settings()
warn_if_insecure_defaults(settings)

app = FastAPI(title=settings.app_name)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Runs innermost-to-outermost relative to add order for *response* processing,
# but request processing is outermost-to-innermost — added last so its
# dispatch() wraps every other middleware, and every request/response gets a
# request_id bound and echoed back regardless of which layer handles it.
app.add_middleware(RequestIDMiddleware)

app.include_router(health_router)
app.include_router(api_router, prefix="/api/v1")
