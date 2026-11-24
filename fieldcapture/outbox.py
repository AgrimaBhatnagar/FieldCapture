"""SQLite offline outbox with deduplication, backoff, and latency tracking."""
import json, sqlite3, time


class Outbox:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS q(id TEXT PRIMARY KEY, body TEXT, "
            "state TEXT DEFAULT 'queued', tries INT DEFAULT 0, next_at REAL DEFAULT 0, "
            "enqueued_at REAL DEFAULT 0, synced_at REAL)"
        )
        self.db.commit()

    def close(self):
        if getattr(self, "db", None) is not None:
            self.db.close()
            self.db = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def put(self, cid, body, now=None):
        ts = time.time() if now is None else now
        self.db.execute(
            "INSERT OR IGNORE INTO q(id,body,enqueued_at) VALUES(?,?,?)",
            (cid, json.dumps(body), ts),
        )
        self.db.commit()

    def pending(self):
        return self.db.execute("SELECT COUNT(*) FROM q WHERE state='queued'").fetchone()[0]

    def flush(self, send, now=None, base=2.0, cap=300.0):
        now = time.time() if now is None else now
        sent = 0
        rows = self.db.execute(
            "SELECT id,body,tries FROM q WHERE state='queued' AND next_at<=? ORDER BY rowid", (now,)
        ).fetchall()
        for cid, body, tries in rows:
            ok = False
            try:
                ok = bool(send(json.loads(body)))
            except Exception:
                ok = False
            if ok:
                self.db.execute(
                    "UPDATE q SET state='synced', synced_at=? WHERE id=?",
                    (now, cid),
                )
                sent += 1
            else:
                self.db.execute(
                    "UPDATE q SET tries=?, next_at=? WHERE id=?",
                    (tries + 1, now + min(cap, base * 2 ** tries), cid),
                )
        self.db.commit()
        return sent

    def avg_sync_latency_sec(self):
        rows = self.db.execute(
            "SELECT synced_at - enqueued_at FROM q WHERE state='synced' AND synced_at IS NOT NULL"
        ).fetchall()
        if not rows:
            return None
        return round(sum(r[0] for r in rows) / len(rows), 3)
