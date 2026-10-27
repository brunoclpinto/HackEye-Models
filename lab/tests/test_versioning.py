import json

from lab.store import Store
from lab.versioning import check_version


def _store(env):
    return Store(env.master, env.repo)


def _seed(s):
    (s.lab_dir / "thumbs" / "x.jpg").write_bytes(b"t")
    (s.lab_dir / "orig").mkdir(exist_ok=True)
    (s.lab_dir / "orig" / "o.png").write_bytes(b"o")
    s.db.execute("INSERT INTO jobs(id,kind,status) VALUES('j1','export','done')")
    s.db.execute("INSERT INTO frames_state(stem,status) VALUES('A_000000','ok')")
    s.db.execute("INSERT INTO edits(stem,at,tool,changed_px,delta) VALUES('A_000000','t','x',1,'')")
    s.db.execute("INSERT INTO classpix(stem,counts,updated_at) VALUES('A_000000','{}','t')")
    s.db.commit()


def test_first_run_writes_without_wipe(env):
    s = _store(env)
    _seed(s)
    assert check_version(s, "v1")["action"] == "first"
    assert (s.lab_dir / ".lab-version").read_text().strip() == "v1"
    assert (s.lab_dir / "thumbs" / "x.jpg").exists()
    assert s.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
    s.db.close()


def test_same_version_no_wipe(env):
    s = _store(env)
    check_version(s, "v1")
    _seed(s)
    assert check_version(s, "v1")["action"] == "same"
    assert (s.lab_dir / "thumbs" / "x.jpg").exists()
    assert s.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
    assert s.db.execute("SELECT count(*) FROM versions").fetchone()[0] == 0
    s.db.close()


def test_new_version_wipes_only_scratch(env):
    s = _store(env)
    check_version(s, "v1")
    _seed(s)
    masks = sorted(p.name for p in (env.master).rglob("*.png"))
    assert check_version(s, "v2")["action"] == "wiped"
    assert not (s.lab_dir / "thumbs" / "x.jpg").exists()
    assert s.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    assert (s.lab_dir / "orig" / "o.png").exists()
    assert sorted(p.name for p in (env.master).rglob("*.png")) == masks
    for t in ("frames_state", "edits", "classpix"):
        assert s.db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] == 1
    v = s.db.execute("SELECT from_version,to_version FROM versions").fetchall()
    assert [tuple(r) for r in v] == [("v1", "v2")]
    assert (s.lab_dir / ".lab-version").read_text().strip() == "v2"
    s.db.close()


def test_running_job_marked_failed(env):
    s = _store(env)
    s.db.execute("INSERT INTO jobs(id,kind,status) VALUES('r','export','running')")
    s.db.commit()
    check_version(s, "v1")
    r = s.db.execute("SELECT status,result FROM jobs WHERE id='r'").fetchone()
    assert r["status"] == "failed" and "interrupted by restart" in json.loads(r["result"])["error"]
    s.db.close()
