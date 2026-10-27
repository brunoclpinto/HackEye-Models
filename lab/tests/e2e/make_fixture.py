"""Build the E2E fixture: 20 real frames of the master copied into /runs/lab-e2e/HackEye-fixture.

Runs inside a lab container (stdlib only):
  docker compose run --rm --no-deps --entrypoint python lab /app/lab/tests/e2e/make_fixture.py
Result: /runs/lab-e2e/HackEye-fixture/{images,masks}/<split>/, frames.csv, meta/{classpix,labels}_<split>.csv, taxonomy.json,
HackEye-fixture.yaml, dataset_manifest.json; plus the empty exports root /runs/lab-e2e/datasets/.
Picks 3 frames with >= 2000 px of "nature snow" (id 62; 2 train + 1 val) and 17 others spread over both splits (14 train, 3 val).
"""
import csv
import os
import re
import shutil
import sys

MASTER = os.environ.get("E2E_MASTER", "/datasets/HackEye26.1-124")
OUT = os.environ.get("E2E_FIXTURE", "/runs/lab-e2e/HackEye-fixture")
SNOW = 62
N_SNOW = {"train": 2, "val": 1}
N_OTHER = {"train": 14, "val": 3}


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def snow_px(counts):
    for kv in counts.split(";"):
        k, _, v = kv.partition(":")
        if k == str(SNOW):
            return int(v)
    return 0


def spread(items, n):
    if n >= len(items):
        return list(items)
    return [items[int(i * len(items) / n)] for i in range(n)]


def main():
    frames = read_csv(f"{MASTER}/frames.csv")
    by_split = {"train": [], "val": []}
    for r in frames:
        by_split[r["split"]].append(r)
    classpix = {}
    for split in by_split:
        for r in read_csv(f"{MASTER}/meta/classpix_{split}.csv"):
            classpix[r["stem"]] = r["counts"]
    chosen = []
    for split in by_split:
        snowy = [r for r in by_split[split] if snow_px(classpix.get(r["stem"], "")) >= 2000]
        if len(snowy) < N_SNOW[split]:
            sys.exit(f"only {len(snowy)} {split} frames with snow, need {N_SNOW[split]}")
        pick = spread(snowy, N_SNOW[split])
        ids = {r["stem"] for r in pick}
        rest = [r for r in by_split[split] if r["stem"] not in ids and snow_px(classpix.get(r["stem"], "")) < 2000]
        chosen += pick + spread(rest, N_OTHER[split])
    assert len(chosen) == 20, len(chosen)

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for split in by_split:
        os.makedirs(f"{OUT}/images/{split}")
        os.makedirs(f"{OUT}/masks/{split}")
    os.makedirs(f"{OUT}/meta")
    os.makedirs(os.path.dirname(OUT) + "/datasets", exist_ok=True)
    for r in chosen:
        s, sp = r["stem"], r["split"]
        shutil.copyfile(f"{MASTER}/images/{sp}/{s}.jpg", f"{OUT}/images/{sp}/{s}.jpg")
        shutil.copyfile(f"{MASTER}/masks/{sp}/{s}.png", f"{OUT}/masks/{sp}/{s}.png")

    def write(path, header, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)

    write(f"{OUT}/frames.csv", list(frames[0].keys()), chosen)
    stems = {r["stem"] for r in chosen}
    for split in by_split:
        for name in ("classpix", "labels"):
            src = read_csv(f"{MASTER}/meta/{name}_{split}.csv")
            rows = [r for r in src if r["stem"] in stems]
            write(f"{OUT}/meta/{name}_{split}.csv", list(src[0].keys()), rows)
    shutil.copyfile(f"{MASTER}/taxonomy.json", f"{OUT}/taxonomy.json")
    if os.path.exists(f"{MASTER}/dataset_manifest.json"):
        shutil.copyfile(f"{MASTER}/dataset_manifest.json", f"{OUT}/dataset_manifest.json")
    yaml = open(f"{MASTER}/HackEye26.1-124.yaml").read()
    yaml = re.sub(r"(?m)^path:.*$", f"path: {OUT}", yaml)
    with open(f"{OUT}/HackEye-fixture.yaml", "w") as f:
        f.write(yaml)
    print(f"fixture: {len(chosen)} frames -> {OUT}")


if __name__ == "__main__":
    main()
