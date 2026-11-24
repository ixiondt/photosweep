"""Too dark, too bright: a filter on Blurry, from each picture's grey levels.

imaging.exposure() measures the stored thumbnail (average grey, the share
crushed to black, the share blown to white); the scanner stores the three in
luma, luma_dark and luma_bright for every new picture, and backfill() fills
them in for pictures from before the columns. No model.

The cut-offs (config.EXPOSE_*) were set on one library's Blurry tab on
2026-11-24, by counts only. Dark: an average under 45 or 60% crushed caught
234 of 2,660, under 60 alone 436; the defaults (50, or half crushed) sit
between. Bright: a quarter blown caught 18, an average over 200 just 2, so
over-exposure is rare among blurred photos and the tab is mostly Dark.
The owner had deleted 97% of the Blurry photos they had looked at, and 96-100%
of the dark ones, so the filter is for working through a kind at a time, not a
verdict.
"""
import logging

from PIL import Image

from . import config, db, imaging

log = logging.getLogger("photosweep.exposure")

# NULL-proof, like the Blurry test in reads.py: an unmeasured picture is
# neither dark nor bright, so it stays under "Well exposed" rather than
# vanishing from every choice. Too bright excludes too dark: a night shot with
# a lamp in it is both by the numbers, and dark is what it looks like.
_DARK = "(coalesce(luma, 128) < ? OR coalesce(luma_dark, 0) >= ?)"
_BRIGHT = "(coalesce(luma, 128) > ? OR coalesce(luma_bright, 0) >= ?)"


def filters():
    """{key: (SQL condition, its arguments)} for every choice."""
    dark = [config.EXPOSE_DARK_MEAN, config.EXPOSE_DARK_SHARE]
    bright = [config.EXPOSE_BRIGHT_MEAN, config.EXPOSE_BRIGHT_SHARE]
    return {
        "all": ("1=1", []),
        "dark": (_DARK, dark),
        "bright": (f"(NOT {_DARK} AND {_BRIGHT})", dark + bright),
        "normal": (f"(NOT {_DARK} AND NOT {_BRIGHT})", dark + bright),
    }


def count_sql():
    """(columns, arguments) counting how many of a selection each choice leaves:
    total, dark, bright, in that order; normal is the rest."""
    f = filters()
    return (f"count(*), coalesce(sum({f['dark'][0]}), 0), coalesce(sum({f['bright'][0]}), 0)",
            f["dark"][1] + f["bright"][1])


def columns(img):
    """The three columns for one stored thumbnail, all None when unmeasurable."""
    e = imaging.exposure(img)
    return dict(zip(("luma", "luma_dark", "luma_bright"), e if e else (None, None, None)))


def backfill(conn, thumb_path, stopping, batch=200):
    """Measure fingerprinted pictures from before the columns, from their
    stored thumbnails (a millisecond or two each). A missing or unreadable
    thumbnail stays NULL and counts as well exposed. Returns the number
    measured."""
    todo = [r["id"] for r in conn.execute(
        "SELECT id FROM files WHERE status IN ('pending','done') AND dhash IS NOT NULL AND luma IS NULL "
        "ORDER BY id")]
    done, pending = 0, []

    def flush():
        with db.immediate(conn):
            conn.executemany("UPDATE files SET luma=?, luma_dark=?, luma_bright=? WHERE id=? AND luma IS NULL",
                             pending)
        pending.clear()

    for file_id in todo:
        if stopping():
            break
        try:
            with Image.open(thumb_path(file_id)) as im:
                cols = columns(im)
        except (OSError, ValueError) as exc:
            log.warning("exposure backfill skipped id=%d err=%s", file_id, type(exc).__name__)
            continue
        if cols["luma"] is None:
            continue
        pending.append((cols["luma"], cols["luma_dark"], cols["luma_bright"], file_id))
        done += 1
        if len(pending) >= batch:
            flush()
    if pending:
        flush()
    return done
