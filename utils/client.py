from langsmith import Client
import os

_client = None

def get_client_instance() -> Client:
    global _client
    if _client is None:
        _client = Client()
    return _client
