import socket
import sys

import pytest


def _clear_remote_cache():
    mod = sys.modules.get("photosweep.remote")
    if mod is not None:
        mod._host_is_local.cache_clear()


@pytest.fixture(autouse=True)
def classifier_off(monkeypatch):
    """No image classifier unless a test installs one: inside the Docker image
    the model file exists, and every test that does not expect the classifier
    to sort would otherwise get it. Set in the environment (for test files
    that re-import photosweep) and on any config already imported."""
    monkeypatch.setenv("CLASSIFIER", "none")
    monkeypatch.setenv("CLASSIFIER_HEADS", "")  # never a heads.npz left in some data dir
    for name in ("photosweep.config",):
        mod = sys.modules.get(name)
        if mod is not None:
            monkeypatch.setattr(mod, "CLASSIFIER", "none")
            monkeypatch.setattr(mod, "CLASSIFIER_HEADS", "")
    clf = sys.modules.get("photosweep.classifier")
    if clf is not None:
        clf.reset()
    yield
    clf = sys.modules.get("photosweep.classifier")
    if clf is not None:
        clf.reset()


@pytest.fixture(autouse=True)
def model_on_the_lan(monkeypatch):
    """Every test's model server is on the local network unless the test says
    otherwise: names resolve to a private address, never through real DNS.
    Patched at the socket, not on photosweep.remote, because several test
    files delete and re-import every photosweep module. Tests about a cloud
    model patch remote._resolve themselves."""
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.5", port or 0))]
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    _clear_remote_cache()
    yield
    _clear_remote_cache()
