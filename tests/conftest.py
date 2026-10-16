import pytest

import mimi_v2.console_log as console_log


@pytest.fixture(autouse=True)
def isolated_console_log(monkeypatch):
    """Tests never reach the real Redis through the process-wide console log singletons."""
    bus = console_log.EventBus()
    monkeypatch.setattr(console_log, "_bus", bus)
    monkeypatch.setattr(console_log, "_log", console_log.MemoryConsoleLog(bus=bus))
