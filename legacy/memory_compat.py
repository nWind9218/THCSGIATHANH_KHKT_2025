"""Single import point for legacy code that used the old `memory` package."""

from legacy.memory_redis import *  # noqa: F401,F403
from legacy.memory_redis import publish_admin_alert  # noqa: F401
from legacy.memory_extras import *  # noqa: F401,F403
from memory import log_emergency, send_emergency_email  # noqa: F401
