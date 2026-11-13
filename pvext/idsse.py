"""The IDSSE / DFL open dataset: the 21 published files, pinned by URL, MD5 and size.

Bassek et al. (2025), "An integrated dataset of spatiotemporal and event data in elite soccer",
Scientific Data 12:195, doi:10.1038/s41597-025-04505-y. Data owner: DFL Deutsche Fussball Liga,
collected by Sportec Solutions (TRACAB). Dataset doi:10.6084/m9.figshare.28196177, version 1,
licensed CC BY 4.0. See docs/EXTERNAL_DATA.md for attribution and the changes we make.

The table below was generated from the figshare files API
(https://api.figshare.com/v2/articles/28196177/files?page_size=50, retrieved 2026-11-04), where
supplied and computed MD5 agree for every file. The match ids are the ones in the file names and in
kloppy; the Hugging Face README's J03WPF and J03WQF do not exist. Teams are home-away, from the
paper's Table 1: five of the seven matches share Fortuna Duesseldorf as the home side.

Nothing here downloads anything on import. fetch_one() is the only function that touches the network,
and the data goes to a directory outside the repository.
"""

import hashlib
import os
import shutil
import tempfile
import urllib.request

# Params
# Default: External/idsse next to the repository checkout; override with IDSSE_DIR.
DEFAULT_ROOT = os.environ.get("IDSSE_DIR") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "External", "idsse"))
CHUNK = 1 << 20
REPO = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATTRIBUTION = (
    "IDSSE open data: DFL Deutsche Fußball Liga, collected by Sportec Solutions (TRACAB), "
    "published by Bassek, M. et al. (2025), 'An integrated dataset of spatiotemporal and event "
    "data in elite soccer', Scientific Data 12:195, doi:10.1038/s41597-025-04505-y; dataset "
    "doi:10.6084/m9.figshare.28196177 (v1). Licensed CC BY 4.0, "
    "https://creativecommons.org/licenses/by/4.0/. Changes: converted from DFL XML to arrays and "
    "to the competition's ATD text format, and synthetically degraded (ball dropout, tracklets, "
    "noise) for benchmarking. No endorsement by the DFL, Sportec Solutions or the authors is "
    "implied. Provided as-is, without warranties."
)

MATCHES = {
    "J03WMX": {
        "competition": "DFL-COM-000001", "home": "1. FC Köln", "away": "FC Bayern München",
        "files": {
            "meta": {"name": "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml",
                "url": "https://ndownloader.figshare.com/files/51643475", "md5": "bc5d5b21dbcf36aa30f4cd0428309f6b", "size": 11930},
            "events": {"name": "DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml",
                "url": "https://ndownloader.figshare.com/files/51643493", "md5": "2d88e180a1f63c7e4157feb980fc7255", "size": 776578},
            "positions": {"name": "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml",
                "url": "https://ndownloader.figshare.com/files/51643514", "md5": "e4273a2adf5dbd9d4ec42def2f01905c", "size": 418524833},
        },
    },
    "J03WN1": {
        "competition": "DFL-COM-000001", "home": "VfL Bochum 1848", "away": "Bayer 04 Leverkusen",
        "files": {
            "meta": {"name": "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml",
                "url": "https://ndownloader.figshare.com/files/51643472", "md5": "37b0f85588c0acfb995fe633c7ac2e0d", "size": 11722},
            "events": {"name": "DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WN1.xml",
                "url": "https://ndownloader.figshare.com/files/51643496", "md5": "83e8ed7ea8e8ebbbdc61f61ac3534008", "size": 612114},
            "positions": {"name": "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml",
                "url": "https://ndownloader.figshare.com/files/51643517", "md5": "bacb2f609c1d1b0bfc906f6b9d4109f2", "size": 391722803},
        },
    },
    "J03WOH": {
        "competition": "DFL-COM-000002", "home": "Fortuna Düsseldorf", "away": "SSV Jahn Regensburg",
        "files": {
            "meta": {"name": "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml",
                "url": "https://ndownloader.figshare.com/files/51643478", "md5": "b9710781d400dd812f62b0a6e3d7bce8", "size": 11745},
            "events": {"name": "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml",
                "url": "https://ndownloader.figshare.com/files/51643499", "md5": "ab66e51512f39d06f49f27dd3b4c2f40", "size": 635747},
            "positions": {"name": "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml",
                "url": "https://ndownloader.figshare.com/files/51643520", "md5": "3efff25a86a61768e7d253b4bba137a3", "size": 348945715},
        },
    },
    "J03WOY": {
        "competition": "DFL-COM-000002", "home": "Fortuna Düsseldorf", "away": "F.C. Hansa Rostock",
        "files": {
            "meta": {"name": "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml",
                "url": "https://ndownloader.figshare.com/files/51643481", "md5": "7b021e577fca917a63111d55cc29ae5b", "size": 11952},
            "events": {"name": "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOY.xml",
                "url": "https://ndownloader.figshare.com/files/51643502", "md5": "64b1be01d6ff6848168efebd56f344b9", "size": 683018},
            "positions": {"name": "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml",
                "url": "https://ndownloader.figshare.com/files/51643523", "md5": "4e1063e0a0ae499920cf68ac1aa3ab4e", "size": 362567859},
        },
    },
    "J03WPY": {
        "competition": "DFL-COM-000002", "home": "Fortuna Düsseldorf", "away": "1. FC Nürnberg",
        "files": {
            "meta": {"name": "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml",
                "url": "https://ndownloader.figshare.com/files/51643487", "md5": "65a4568e4f374af94d3c1015cbc7c4f5", "size": 11922},
            "events": {"name": "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WPY.xml",
                "url": "https://ndownloader.figshare.com/files/51643505", "md5": "f68482441d38723d69f244f896482803", "size": 698898},
            "positions": {"name": "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml",
                "url": "https://ndownloader.figshare.com/files/51643526", "md5": "e56ff1336a334f8da51110fbe7cab764", "size": 371688499},
        },
    },
    "J03WQQ": {
        "competition": "DFL-COM-000002", "home": "Fortuna Düsseldorf", "away": "FC St. Pauli",
        "files": {
            "meta": {"name": "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml",
                "url": "https://ndownloader.figshare.com/files/51643484", "md5": "5979209af316a914ce57347d0633df6c", "size": 11932},
            "events": {"name": "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WQQ.xml",
                "url": "https://ndownloader.figshare.com/files/51643508", "md5": "b7e9d82af4034539d802319be6c57776", "size": 732186},
            "positions": {"name": "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml",
                "url": "https://ndownloader.figshare.com/files/51643529", "md5": "9105bad987191d3c3c362c1fce9a4fcd", "size": 356219423},
        },
    },
    "J03WR9": {
        "competition": "DFL-COM-000002", "home": "Fortuna Düsseldorf", "away": "1. FC Kaiserslautern",
        "files": {
            "meta": {"name": "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml",
                "url": "https://ndownloader.figshare.com/files/51643490", "md5": "5e2d580d3d8ed97e32959de90d04992b", "size": 11823},
            "events": {"name": "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WR9.xml",
                "url": "https://ndownloader.figshare.com/files/51643511", "md5": "c13bc5efa5eca4a022cc678210b87d5f", "size": 681050},
            "positions": {"name": "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml",
                "url": "https://ndownloader.figshare.com/files/51643532", "md5": "e060bb7d7a5a3bfbd15bbf9bc631024d", "size": 372934140},
        },
    },
}


def local_paths(match_id, root=None):
    """{part: path} for one match under root (default IDSSE_DIR)."""
    root = root or DEFAULT_ROOT
    return {part: os.path.join(root, f["name"]) for part, f in MATCHES[match_id]["files"].items()}


def check_root(root):
    """The real, absolute download directory, or ValueError if it is inside the repository.

    Compared on resolved paths (symlinks, '..', relative paths) and case-insensitively, because
    the default macOS filesystem treats /Users/X and /users/x as the same directory.
    """
    real = os.path.realpath(os.path.abspath(root))
    repo, candidate = os.path.normcase(REPO).casefold(), os.path.normcase(real).casefold()
    if candidate == repo or os.path.commonpath([repo, candidate]) == repo:
        raise ValueError("refusing to store IDSSE data inside the repository: {} -> {}".format(
            root, real))
    return real


def _check_name(name):
    if not name or name in (".", "..") or os.path.basename(name) != name or os.sep in name \
            or (os.altsep and os.altsep in name):
        raise ValueError("unsafe file name in manifest entry: {!r}".format(name))


def verify(path, md5, size):
    """True if path exists with exactly this size and MD5."""
    if not os.path.isfile(path) or os.path.getsize(path) != size:
        return False
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest() == md5


def fetch_one(entry, root):
    """Download one manifest entry into root, verified; return its path.

    Skips the download when a verified copy is already there. The bytes go to a temporary file
    that tempfile.mkstemp creates exclusively (O_CREAT | O_EXCL, mode 0600) inside the verified
    root, so no pre-existing path or symlink is ever opened for writing. Only after the size and
    MD5 check is it moved into place with os.replace, which replaces the directory entry and
    never writes through a link. A target that is itself a symlink is refused.
    """
    _check_name(entry["name"])
    root = check_root(root)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, entry["name"])
    if os.path.islink(path):
        raise ValueError("refusing to use {}: it is a symlink".format(path))
    if verify(path, entry["md5"], entry["size"]):
        return path
    fd, part = tempfile.mkstemp(dir=root, prefix=entry["name"] + ".", suffix=".download")
    try:
        with os.fdopen(fd, "wb") as handle, urllib.request.urlopen(entry["url"]) as response:
            shutil.copyfileobj(response, handle, CHUNK)
        if not verify(part, entry["md5"], entry["size"]):
            raise ValueError("{}: size or md5 does not match the manifest".format(entry["name"]))
        os.replace(part, path)
    finally:
        if os.path.lexists(part):
            os.remove(part)
    return path
