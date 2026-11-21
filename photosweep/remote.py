"""Is the model server outside this network, and has the owner agreed to that?

A model on a cloud service sees every photo the scanner labels. That is a
legitimate choice for someone with no GPU, and the wrong default for everyone
else, so it is never made by accident: a model URL that leaves the local
network is refused until PHOTOS_LEAVE_THIS_SERVER=yes is set, and the page says
where the photos are going for as long as it is.

"Local" means every address the host resolves to is loopback, private
(RFC 1918 / ULA), link-local, or the 100.64.0.0/10 range overlay VPNs such as
Tailscale use. A name that does not resolve is treated as remote: failing
closed costs a log line, failing open costs the photos.
"""
import functools
import ipaddress
import logging
import socket
from urllib.parse import urlsplit

from . import config

log = logging.getLogger("photosweep.remote")

_CGNAT = ipaddress.ip_network("100.64.0.0/10")


def _local_address(ip):
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_loopback or addr.is_private or addr.is_link_local or addr in _CGNAT


def _resolve(host):
    """Every address a name resolves to. A seam for tests."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return {info[4][0].split("%")[0] for info in infos}


@functools.lru_cache(maxsize=8)
def _host_is_local(host):
    if not host:
        return False
    try:
        return _local_address(host)
    except ValueError:
        pass  # a name, not an address literal
    try:
        addrs = _resolve(host)
    except OSError as exc:
        log.warning("model host %s does not resolve (%s); treating it as outside this network", host, exc)
        return False
    return bool(addrs) and all(_local_address(a) for a in addrs)


def host(url=None):
    return urlsplit(url or config.ROUTER_URL).hostname or ""


def is_remote(url=None):
    return not _host_is_local(host(url))


def allowed(url=None):
    """True when the model at `url` (default: the configured one) may be
    asked: it is on this network, or the owner said yes."""
    return config.PHOTOS_LEAVE_THIS_SERVER or not is_remote(url)


def status():
    """What the page shows about where the photos go."""
    remote = is_remote()
    return {"remote": remote, "host": host(), "allowed": (not remote) or config.PHOTOS_LEAVE_THIS_SERVER}
