"""
Pytest configuration.
Disables the broken langsmith plugin from the global Anaconda environment.
"""
collect_ignore_glob = []


def pytest_configure(config):
    """Disable incompatible plugins."""
    try:
        config.pluginmanager.set_blocked("langsmith")
    except Exception:
        pass
