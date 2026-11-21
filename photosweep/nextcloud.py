"""Deletes go through Nextcloud, never through the filesystem.

A WebDAV DELETE moves the file to the user's trash bin (retention `auto, 60`
kept up to 60 days by default), updates Nextcloud's file cache, and is
reversible from the Deleted files view. Removing it from the mount instead would
be permanent and leave Nextcloud's index pointing at nothing -- and the mount
is read-only precisely so that cannot happen by accident.

The guest resolves the Nextcloud name to its LAN address through /etc/hosts, so
the TLS certificate still matches and nothing leaves the LAN.
"""
from urllib.parse import quote

import httpx


class NotConfigured(Exception):
    pass


def dav_url(base, user, prefix, rel):
    """The WebDAV URL of one file. Each path segment is percent-encoded on its
    own, so '#', '?', '%' and spaces in names survive, and a '..' or empty
    segment is refused outright rather than normalised."""
    parts = [p for p in prefix.split("/") if p] + rel.split("/")
    for p in parts:
        if p in ("", ".", ".."):
            raise ValueError(f"refusing path segment {p!r} in {rel!r}")
    encoded = "/".join(quote(p, safe="") for p in parts)
    return f"{base.rstrip('/')}/remote.php/dav/files/{quote(user, safe='')}/{encoded}"


def client(user, password):
    if not password:
        raise NotConfigured("no Nextcloud app password saved yet")
    return httpx.Client(auth=(user, password), timeout=httpx.Timeout(30.0),
                        headers={"User-Agent": "photosweep", "OCS-APIRequest": "true"})


def check(base, user, prefix, password):
    """True when the credentials can see the folder. Used before saving a new
    app password, so a typo is caught at the form rather than at the first delete."""
    url = f"{base.rstrip('/')}/remote.php/dav/files/{quote(user, safe='')}/{quote(prefix, safe='/')}"
    with client(user, password) as c:
        resp = c.request("PROPFIND", url, headers={"Depth": "0"})
    return resp.status_code == 207


def delete(c, base, user, prefix, rel):
    """'deleted', 'gone' (already not there) or 'error:<status>'."""
    resp = c.delete(dav_url(base, user, prefix, rel))
    if resp.status_code in (200, 204):
        return "deleted"
    if resp.status_code == 404:
        return "gone"
    return f"error:{resp.status_code}"
