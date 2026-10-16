"""Emergency logging and email notification for the Facebook/v2 runtime."""

from .supabase_client import log_emergency, send_emergency_email

__all__ = ["log_emergency", "send_emergency_email"]
