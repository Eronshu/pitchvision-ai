import hashlib
import os
import subprocess
import sys

import pytest

from pvext import idsse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FETCH = os.path.join(ROOT, "scripts", "fetch_idsse.py")
IDS = {"J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"}


def test_manifest_ids():
    assert set(idsse.MATCHES) == IDS
    assert "J03WPF" not in idsse.MATCHES and "J03WQF" not in idsse.MATCHES


@pytest.mark.parametrize("match_id, competition", [
    ("J03WMX", "DFL-COM-000001"), ("J03WN1", "DFL-COM-000001"),
    ("J03WOH", "DFL-COM-000002"), ("J03WOY", "DFL-COM-000002"),
    ("J03WPY", "DFL-COM-000002"), ("J03WQQ", "DFL-COM-000002"),
    ("J03WR9", "DFL-COM-000002"),
])
def test_manifest_competitions_and_names(match_id, competition):
    entry = idsse.MATCHES[match_id]
    assert entry["competition"] == competition
    stems = {"meta": "DFL_02_01_matchinformation", "events": "DFL_03_02_events_raw",
             "positions": "DFL_04_03_positions_raw_observed"}
    for part, stem in stems.items():
        f = entry["files"][part]
        assert f["name"] == "{}_{}_DFL-MAT-{}.xml".format(stem, competition, match_id)
        assert f["url"].startswith("https://ndownloader.figshare.com/files/")
        assert len(f["md5"]) == 32 and f["size"] > 0


def test_manifest_j03wpy_pinned():
    files = idsse.MATCHES["J03WPY"]["files"]
    assert files["positions"]["md5"] == "e56ff1336a334f8da51110fbe7cab764"
    assert files["positions"]["size"] == 371688499
    assert files["meta"]["md5"] == "65a4568e4f374af94d3c1015cbc7c4f5"


def test_teams_recorded():
    assert idsse.MATCHES["J03WPY"]["home"] == "Fortuna Düsseldorf"
    homes = [m["home"] for m in idsse.MATCHES.values()]
    assert homes.count("Fortuna Düsseldorf") == 5


def _file(tmp_path, data=b"hello idsse"):
    path = tmp_path / "f.xml"
    path.write_bytes(data)
    return path, hashlib.md5(data).hexdigest(), len(data)


@pytest.mark.parametrize("good", [True, False])
def test_verify(tmp_path, good):
    path, md5, size = _file(tmp_path)
    assert idsse.verify(str(path), md5 if good else "0" * 32, size) is good


def test_verify_wrong_size(tmp_path):
    path, md5, size = _file(tmp_path)
    assert idsse.verify(str(path), md5, size + 1) is False


def test_fetch_one_rejects_bad_md5(tmp_path):
    src, _, size = _file(tmp_path)
    entry = {"name": "out.xml", "url": src.as_uri(), "md5": "0" * 32, "size": size}
    root = tmp_path / "root"
    with pytest.raises(ValueError, match="md5"):
        idsse.fetch_one(entry, str(root))
    assert not root.exists() or os.listdir(root) == []    # no file, no temporary left behind


def test_fetch_one_downloads_and_verifies(tmp_path):
    src, md5, size = _file(tmp_path)
    entry = {"name": "out.xml", "url": src.as_uri(), "md5": md5, "size": size}
    path = idsse.fetch_one(entry, str(tmp_path / "root"))
    assert idsse.verify(path, md5, size)


def test_fetch_one_skips_verified_without_opening_url(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    path, md5, size = _file(root)
    entry = {"name": "f.xml", "url": "https://invalid.invalid/never", "md5": md5, "size": size}
    assert idsse.fetch_one(entry, str(root)) == str(path)


def test_fetch_cli_unknown_match_fails():
    result = subprocess.run([sys.executable, FETCH, "--match", "J03WPF", "--root", "/nonexistent"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0 and "J03WPF" in result.stderr


def test_fetch_cli_refuses_repo_root():
    result = subprocess.run([sys.executable, FETCH, "--match", "J03WPY",
                             "--root", os.path.join(ROOT, "data", "idsse")],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0 and "repository" in result.stderr


# -- review fix 1: download-path safety, enforced in the library, not only the CLI -------------

@pytest.mark.parametrize("make_root", [
    lambda tmp: ROOT,                                          # the repository root itself
    lambda tmp: os.path.join(ROOT, "data", "idsse"),          # inside it
    lambda tmp: os.path.join(ROOT, "data", "..", "pvext"),    # inside it, via ..
    lambda tmp: ROOT.upper(),                                  # case-only difference (APFS)
    lambda tmp: os.path.relpath(os.path.join(ROOT, "x"), os.getcwd()),   # relative
])
def test_check_root_refuses_repository(tmp_path, make_root):
    with pytest.raises(ValueError, match="repository"):
        idsse.check_root(make_root(tmp_path))


def test_check_root_refuses_symlink_into_repository(tmp_path):
    link = tmp_path / "looks_outside"
    link.symlink_to(os.path.join(ROOT, "data"))
    with pytest.raises(ValueError, match="repository"):
        idsse.check_root(str(link / "idsse"))


def test_check_root_accepts_outside(tmp_path):
    assert idsse.check_root(str(tmp_path / "idsse")) == os.path.realpath(str(tmp_path / "idsse"))


def test_fetch_one_enforces_root(tmp_path, monkeypatch):
    """Against a stand-in repository, so even a broken implementation cannot write into the real
    one (an earlier version of this test did, before the guard existed)."""
    fake_repo = tmp_path / "repo"
    (fake_repo / "data").mkdir(parents=True)
    monkeypatch.setattr(idsse, "REPO", str(fake_repo))
    src, md5, size = _file(tmp_path)
    entry = {"name": "out.xml", "url": src.as_uri(), "md5": md5, "size": size}
    with pytest.raises(ValueError, match="repository"):
        idsse.fetch_one(entry, str(fake_repo / "data"))
    assert not (fake_repo / "data" / "out.xml").exists()


@pytest.mark.parametrize("name", ["../escape.xml", "sub/dir.xml", "/abs.xml", "", ".", ".."])
def test_fetch_one_rejects_unsafe_names(tmp_path, name):
    src, md5, size = _file(tmp_path)
    with pytest.raises(ValueError, match="file name"):
        idsse.fetch_one({"name": name, "url": src.as_uri(), "md5": md5, "size": size},
                        str(tmp_path / "root"))


def test_manifest_names_are_plain():
    for match in idsse.MATCHES.values():
        for f in match["files"].values():
            assert os.path.basename(f["name"]) == f["name"] and f["name"].endswith(".xml")


# -- CC BY 4.0: one attribution text, carrying every element the licence requires --------------

@pytest.mark.parametrize("element", [
    "DFL Deutsche Fußball Liga",                      # creator / data owner
    "Sportec Solutions",                              # collector
    "Bassek",                                         # authors of the dataset paper
    "10.6084/m9.figshare.28196177",                   # source (dataset DOI)
    "10.1038/s41597-025-04505-y",                     # paper DOI
    "CC BY 4.0",                                      # licence name
    "https://creativecommons.org/licenses/by/4.0/",   # licence link
    "Changes:",                                       # modifications must be indicated
    "No endorsement",                                 # no implied endorsement
])
def test_attribution_has_required_elements(element):
    assert element in idsse.ATTRIBUTION


def test_fetch_help_shows_licence():
    result = subprocess.run([sys.executable, FETCH, "--help"], capture_output=True, text=True,
                            cwd=ROOT)
    assert "CC BY 4.0" in result.stdout and "EXTERNAL_DATA.md" in result.stdout


# -- review 2, fix 2: no write ever follows a pre-existing symlink -----------------------------

@pytest.fixture
def planted(tmp_path, monkeypatch):
    """A stand-in repository holding a precious file, and an external download directory in which
    an attacker (or an accident) left a <name>.part symlink pointing at that file."""
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    monkeypatch.setattr(idsse, "REPO", str(fake_repo))
    precious = fake_repo / "precious.txt"
    precious.write_bytes(b"ORIGINAL")
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "out.xml.part").symlink_to(precious)
    return ext, precious


@pytest.mark.parametrize("good_md5", [True, False])
def test_fetch_one_does_not_follow_part_symlink(tmp_path, planted, good_md5):
    ext, precious = planted
    src, md5, size = _file(tmp_path, data=b"downloaded bytes")
    entry = {"name": "out.xml", "url": src.as_uri(), "md5": md5 if good_md5 else "0" * 32,
             "size": size}
    if good_md5:
        path = idsse.fetch_one(entry, str(ext))
        with open(path, "rb") as handle:
            assert handle.read() == b"downloaded bytes"
        assert not os.path.islink(path)
    else:
        with pytest.raises(ValueError, match="md5"):
            idsse.fetch_one(entry, str(ext))
        assert not (ext / "out.xml").exists()
    assert precious.read_bytes() == b"ORIGINAL"          # never written through the link
    assert os.path.islink(ext / "out.xml.part")          # the planted link is left untouched
    leftovers = [p for p in os.listdir(ext) if p not in ("out.xml", "out.xml.part")]
    assert leftovers == []                               # our own temporary file is cleaned up


def test_fetch_one_refuses_symlinked_target(tmp_path, planted):
    ext, precious = planted
    (ext / "out.xml").symlink_to(precious)
    src, md5, size = _file(tmp_path, data=b"downloaded bytes")
    with pytest.raises(ValueError, match="symlink"):
        idsse.fetch_one({"name": "out.xml", "url": src.as_uri(), "md5": md5, "size": size},
                        str(ext))
    assert precious.read_bytes() == b"ORIGINAL"


def test_fetch_one_temp_file_is_private_and_inside_root(tmp_path, monkeypatch):
    seen = []
    real_mkstemp = idsse.tempfile.mkstemp

    def spy(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        seen.append((name, os.stat(name).st_mode & 0o777))
        return fd, name
    monkeypatch.setattr(idsse.tempfile, "mkstemp", spy)
    src, md5, size = _file(tmp_path)
    root = tmp_path / "ext"
    idsse.fetch_one({"name": "out.xml", "url": src.as_uri(), "md5": md5, "size": size}, str(root))
    (name, mode), = seen
    assert os.path.dirname(name) == os.path.realpath(str(root)) and mode == 0o600
