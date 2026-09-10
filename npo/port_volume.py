"""
Copy a Modal Volume between workspaces, cloud-to-cloud.

    modal run npo/port_volume.py --dry-run     # inventory + byte count
    modal run npo/port_volume.py               # do the copy

Volumes are workspace-scoped, so switching Modal profiles strands everything
the old account held -- here, six 8B checkpoints and the NPO reference cache.

The two obvious routes are both bad: `modal volume get` + `put` drags ~100 GB
down to a laptop and back up, and staging on the HF Hub needs a write-scoped
token and doubles the transfer. Instead this runs a container in the NEW
workspace that opens a SECOND authenticated client against the OLD workspace
(Client.from_credentials + Volume.hydrate) and streams file-by-file straight
into the mounted destination volume. Compute bills to the new account, no
bytes touch local disk, no HF token involved.

Resumable: a file already present at the destination with a matching size is
skipped, so a timeout or crash just means running it again.
"""

from __future__ import annotations

import modal

SRC_VOLUME = "npo-work"
DST_VOLUME = "npo-work"
HOURS = 60 * 60

image = modal.Image.debian_slim(python_version="3.11").pip_install("modal==1.2.1")
app = modal.App("npo-port", image=image)
dst_vol = modal.Volume.from_name(DST_VOLUME, create_if_missing=True)


@app.function(
    volumes={"/work": dst_vol},
    secrets=[modal.Secret.from_name("modal-src-creds")],
    timeout=6 * HOURS,
    cpu=4.0,
    memory=8192,
)
def port(prefix: str = "", dry_run: bool = False, workers: int = 6):
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    import modal

    src_client = modal.Client.from_credentials(
        os.environ["MODAL_SRC_TOKEN_ID"], os.environ["MODAL_SRC_TOKEN_SECRET"])
    src = modal.Volume.from_name(SRC_VOLUME)
    src.hydrate(src_client)

    try:
        from modal.volume import FileEntryType
        is_file = lambda e: e.type == FileEntryType.FILE  # noqa: E731
    except Exception:                                     # pragma: no cover
        is_file = lambda e: getattr(e, "size", 0) > 0     # noqa: E731

    files = [e for e in src.iterdir(prefix or "/", recursive=True) if is_file(e)]
    total = sum(e.size for e in files)
    print(f"source {SRC_VOLUME}: {len(files)} files, {total / 2**30:.1f} GiB", flush=True)

    todo = []
    for e in files:
        dst = os.path.join("/work", e.path.lstrip("/"))
        if os.path.exists(dst) and os.path.getsize(dst) == e.size:
            continue
        todo.append((e, dst))
    skipped = len(files) - len(todo)
    pend = sum(e.size for e, _ in todo)
    print(f"already present: {skipped} | to copy: {len(todo)} "
          f"({pend / 2**30:.1f} GiB)", flush=True)

    if dry_run:
        by_dir = {}
        for e in files:
            by_dir.setdefault(e.path.split("/")[1] if "/" in e.path else ".", [0, 0])
            by_dir[e.path.split("/")[1] if "/" in e.path else "."][0] += 1
            by_dir[e.path.split("/")[1] if "/" in e.path else "."][1] += e.size
        for d, (n, b) in sorted(by_dir.items(), key=lambda kv: -kv[1][1]):
            print(f"  {b / 2**30:8.2f} GiB  {n:>4} files  {d}")
        return {"files": len(files), "bytes": total, "to_copy": len(todo)}

    done = {"n": 0, "b": 0}
    t0 = time.time()

    def copy(job):
        e, dst = job
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        with open(tmp, "wb") as fh:
            for chunk in src.read_file(e.path):
                fh.write(chunk)
        os.replace(tmp, dst)  # atomic, so a crash never leaves a short file
        done["n"] += 1
        done["b"] += e.size
        el = time.time() - t0
        print(f"[{done['n']}/{len(todo)}] {done['b'] / 2**30:6.1f} GiB "
              f"@ {done['b'] / 2**20 / max(el, 1):.0f} MiB/s  {e.path}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(copy, todo))

    dst_vol.commit()
    el = time.time() - t0
    print(f"\ncopied {done['n']} files, {done['b'] / 2**30:.1f} GiB in {el / 60:.1f} min",
          flush=True)
    return {"copied": done["n"], "bytes": done["b"], "skipped": skipped, "seconds": el}


@app.local_entrypoint()
def main(prefix: str = "", dry_run: bool = False, workers: int = 6):
    import json
    print(json.dumps(port.remote(prefix=prefix, dry_run=dry_run, workers=workers), indent=2))
