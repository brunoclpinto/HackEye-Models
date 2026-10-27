"""Version-scoped Lab scratch state.

`lab/VERSION` is the Lab's version; `<dataset>/lab/.lab-version` records which version last
owned the regenerable state. On a change only thumbnails and the job history are wiped. Masks,
`lab/orig/`, frames_state, edits, classpix, tax_log, taxonomy.json and the yaml are the user's
curation and are never touched here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .store import Store, now

MARKER = ".lab-version"


def check_version(store_or_root, version: str) -> dict:
    """Run once at startup. Returns {"action": "first"|"same"|"wiped", ...}."""
    store = store_or_root if isinstance(store_or_root, Store) else Store(Path(store_or_root), Path("."))
    marker = store.lab_dir / MARKER
    with store.lock:
        store.db.execute(
            "UPDATE jobs SET status='failed', finished=?, result=? WHERE status='running'",
            (now(), json.dumps({"error": "interrupted by restart"})))
        store.db.commit()
        old = marker.read_text().strip() if marker.exists() else None
        if old == version:
            return {"action": "same"}
        if old is None:
            marker.write_text(version + "\n")
            return {"action": "first"}
        wiped = []
        thumbs = store.lab_dir / "thumbs"
        if thumbs.exists():
            shutil.rmtree(thumbs)
            wiped.append("thumbs")
        thumbs.mkdir(exist_ok=True)
        n = store.db.execute("DELETE FROM jobs").rowcount
        wiped.append(f"jobs({n})")
        store.db.execute("INSERT INTO versions(at, from_version, to_version, wiped) VALUES(?,?,?,?)",
                         (now(), old, version, ",".join(wiped)))
        store.db.commit()
        marker.write_text(version + "\n")
        return {"action": "wiped", "from": old, "to": version, "wiped": wiped}
