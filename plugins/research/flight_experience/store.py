"""Versioned local experience storage; no Agent/Runtime dependencies.

SQL owns identity, immutable evidence/version records and publication state.
Models never issue SQL. Writers use short transactions; search uses a read-only
snapshot. Experimental sources cannot be promoted to production by publication.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import unicodedata
import uuid

from .evidence import read_run


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def digest(value):
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def tokens(text):
    """NFKC lowercase Latin words and overlapping Han bigrams (single Han kept).

    Deliberately lexical: no unverified synonym expansion or embeddings. All
    tokens are generated here, so caller text cannot inject FTS query syntax.
    """
    result = []
    for part in re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", unicodedata.normalize("NFKC", text).lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            result.extend(part[i:i+2] for i in range(max(1, len(part)-1)))
        else:
            result.append(part)
    return list(dict.fromkeys(result))


FIELDS = ("applicability", "observed_evidence", "recommendation", "limitations")
TAG_FIELDS = {"decision", "action", "observation", "anomaly", "platform", "control", "sensor"}


def normalize_body(body):
    """Only the experience contract crosses from model output into storage."""
    if not isinstance(body, dict) or any(not isinstance(body.get(k), str) or not body[k].strip() for k in FIELDS):
        raise ValueError("Experience requires all four nonempty content fields")
    tags = body.get("tags", {})
    if not isinstance(tags, dict) or set(tags) - TAG_FIELDS:
        raise ValueError("Unknown tag dimension")
    if any(not isinstance(v, list) or any(not isinstance(x, str) or not x.strip() for x in v) for v in tags.values()):
        raise ValueError("Tags must be nonempty string lists")
    result = {k: body[k].strip() for k in FIELDS}
    result.update(title=str(body.get("title") or body["recommendation"][:100]), tags={k:v for k,v in tags.items() if v})
    # Old immutable versions are not backfilled with invented decision questions.
    if 'decision_question' in body:
        if not isinstance(body['decision_question'], str) or not body['decision_question'].strip():
            raise ValueError('Decision question must be nonempty')
        result['decision_question'] = body['decision_question'].strip()
    result["description"] = (f"适用需求：{result['applicability']}／解决问题：{result['recommendation']}"
        f"／内容范围：{result['observed_evidence']}／能力边界：{result['limitations']}")
    return result


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT OR IGNORE INTO meta VALUES ('schema_version','1');
CREATE TABLE IF NOT EXISTS sources (
 id TEXT PRIMARY KEY, root TEXT NOT NULL, scope TEXT NOT NULL CHECK(scope IN ('experiment','production')),
 instruction_id TEXT, instruction TEXT NOT NULL, evidence_json TEXT NOT NULL,
 availability TEXT NOT NULL DEFAULT 'available', issue TEXT);
CREATE TABLE IF NOT EXISTS imports (
 id TEXT PRIMARY KEY, path TEXT NOT NULL, source_id TEXT NOT NULL REFERENCES sources(id), result_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiences (
 id TEXT PRIMARY KEY, published_version INTEGER, state TEXT NOT NULL DEFAULT 'candidate');
CREATE TABLE IF NOT EXISTS versions (
 experience_id TEXT NOT NULL REFERENCES experiences(id), version INTEGER NOT NULL,
 body_json TEXT NOT NULL, created REAL NOT NULL,
 PRIMARY KEY(experience_id,version));
CREATE TRIGGER IF NOT EXISTS immutable_versions BEFORE UPDATE ON versions BEGIN SELECT RAISE(ABORT,'immutable version'); END;
CREATE TRIGGER IF NOT EXISTS keep_versions BEFORE DELETE ON versions BEGIN SELECT RAISE(ABORT,'retain version history'); END;
CREATE TABLE IF NOT EXISTS evidence_links (
 experience_id TEXT NOT NULL, version INTEGER NOT NULL, source_id TEXT NOT NULL REFERENCES sources(id),
 refs_json TEXT NOT NULL, observations_json TEXT NOT NULL, relation TEXT NOT NULL,
 FOREIGN KEY(experience_id,version) REFERENCES versions(experience_id,version));
CREATE INDEX IF NOT EXISTS evidence_by_version ON evidence_links(experience_id,version);
CREATE TABLE IF NOT EXISTS tags (
 experience_id TEXT NOT NULL, version INTEGER NOT NULL, dimension TEXT NOT NULL, value TEXT NOT NULL,
 FOREIGN KEY(experience_id,version) REFERENCES versions(experience_id,version));
CREATE INDEX IF NOT EXISTS tags_lookup ON tags(dimension,value);
CREATE TABLE IF NOT EXISTS reviews (
 id INTEGER PRIMARY KEY, experience_id TEXT, version INTEGER, decision TEXT NOT NULL,
 reason TEXT NOT NULL, actor TEXT NOT NULL, created REAL NOT NULL, detail_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS relations (
 origin TEXT NOT NULL, destination TEXT NOT NULL, kind TEXT NOT NULL, proposal_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, dedup TEXT UNIQUE NOT NULL, payload_json TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'pending', owner TEXT, lease_until REAL, attempts INTEGER NOT NULL DEFAULT 0,
 created REAL NOT NULL, error TEXT);
CREATE INDEX IF NOT EXISTS queued_jobs ON jobs(state,created,id);
CREATE TABLE IF NOT EXISTS attempts (
 id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), started REAL NOT NULL,
 ended REAL, state TEXT NOT NULL, error TEXT, audit_path TEXT);
CREATE TABLE IF NOT EXISTS proposals (
 id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), base_json TEXT NOT NULL,
 change_json TEXT NOT NULL, review_json TEXT NOT NULL, state TEXT NOT NULL,
 created REAL NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(experience_id UNINDEXED, version UNINDEXED, text);
"""


class Store:
    """Each operation opens its own connection, safe across worker heartbeat threads."""
    def __init__(self, path):
        self.path = Path(path).resolve()

    @contextmanager
    def connect(self, *, write=False):
        connection = sqlite3.connect(str(self.path) if write else self.path.as_uri()+"?mode=ro", uri=not write, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        if not write:
            connection.execute("PRAGMA query_only=ON")
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='meta'").fetchone():
                version = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
                if version and version[0] not in ('1','2'):
                    raise ValueError('Unsupported database schema')
            db.execute("PRAGMA journal_mode=WAL")
            try:
                db.executescript(SCHEMA)
            except sqlite3.OperationalError as exc:
                raise RuntimeError("SQLite schema initialization requires FTS5") from exc
            self._migrate_v2(db)

    @staticmethod
    def _migrate_v2(db):
        """Transactional additive migration; never rewrites old immutable bodies."""
        db.execute('BEGIN IMMEDIATE')
        try:
            db.execute('CREATE TABLE IF NOT EXISTS maintenance_groups (fingerprint TEXT PRIMARY KEY, proposal_id TEXT NOT NULL)')
            db.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
            db.commit()
        except BaseException:
            db.rollback()
            raise

    def import_experiment(self, directory, *, scope="experiment", instruction_id=None):
        """Import artifacts, not their claimed acceptance. Every candidate stays unpublished.

        Historical experiments explicitly marked experiment_only cannot be used as
        production material; supported production data needs a separate approved
        collection workflow rather than a scope flag on a benchmark artifact.
        """
        directory = Path(directory).resolve()
        raw = (directory / "experiment.json").read_bytes()
        result = json.loads(raw)
        if scope != "experiment":
            raise ValueError("Offline experiment imports are restricted to experiment scope")
        saved = json.loads((directory / "evidence_index.json").read_text(encoding="utf-8"))
        root = Path(result["source"]).resolve()
        if self.path.is_relative_to(root) or self.path.is_relative_to(directory):
            raise ValueError("Library database must be outside original run and experiment directories")
        try:
            bundle = read_run(root, instruction_id=instruction_id, instruction=saved["instruction"])
            available = bundle.index() == saved
        except (OSError, ValueError, KeyError):
            available = False
        source_id = digest({"root": str(root), "index": saved, "instruction_id": instruction_id})
        import_id = (digest({k:result.get(k) for k in ('protocol_version','source','candidates','reviews','accepted','calls')})
            if result.get('protocol_version')=='v2' else hashlib.sha256(raw).hexdigest())
        ids = []
        with self.connect(write=True) as db:
            if db.execute("SELECT 1 FROM imports WHERE id=?", (import_id,)).fetchone():
                return {"import_id": import_id, "duplicate": True, "experience_ids": []}
            db.execute("INSERT OR IGNORE INTO sources VALUES (?,?,?,?,?,?,?,?)", (source_id, str(root), scope,
                instruction_id, saved["instruction"], dumps(saved), "available" if available else "unavailable",
                None if available else "source_changed_since_experiment"))
            db.execute("INSERT INTO imports VALUES (?,?,?,?)", (import_id, str(directory), source_id,
                dumps({k: result.get(k) for k in ("status", "reviews", "accepted", "error_type")})))
            for offset, candidate in enumerate(result.get("candidates", [])):
                eid = digest({"import": import_id, "offset": offset})[:24]
                body = normalize_body(candidate)
                db.execute("INSERT INTO experiences(id) VALUES (?)", (eid,))
                links = [{"source_id": source_id, "evidence_refs": candidate.get("evidence_refs", []),
                    "observation_ids": candidate.get("observation_ids", []), "relation": "support"}]
                self._version(db, eid, body, links)
                self._review(db, eid, 1, "candidate", "Imported; historical review retained separately, not publication", "import", {})
                ids.append(eid)
        return {"import_id": import_id, "duplicate": False, "experience_ids": ids, "source_available": available}

    def _version(self, db, eid, body, links):
        version = db.execute("SELECT COALESCE(MAX(version),0)+1 FROM versions WHERE experience_id=?", (eid,)).fetchone()[0]
        db.execute("INSERT INTO versions VALUES (?,?,?,?)", (eid, version, dumps(normalize_body(body)), time.time()))
        for link in links:
            db.execute("INSERT INTO evidence_links VALUES (?,?,?,?,?,?)", (eid, version, link["source_id"],
                dumps(link["evidence_refs"]), dumps(link["observation_ids"]), link.get("relation", "support")))
        for dimension, values in body.get("tags", {}).items():
            db.executemany("INSERT INTO tags VALUES (?,?,?,?)", [(eid, version, dimension, value) for value in set(values)])
        index_text = " ".join(str(body.get(k, "")) for k in (*FIELDS, "title", "decision_question")) + " " + dumps(body.get("tags", {}))
        db.execute("INSERT INTO search_index VALUES (?,?,?)", (eid, version, " ".join(tokens(index_text))))
        return version

    @staticmethod
    def _review(db, eid, version, decision, reason, actor, detail):
        db.execute("INSERT INTO reviews(experience_id,version,decision,reason,actor,created,detail_json) VALUES (?,?,?,?,?,?,?)",
            (eid, version, decision, reason, actor, time.time(), dumps(detail)))

    def get(self, eid, version=None):
        with self.connect() as db:
            return self._get(db, eid, version)

    @staticmethod
    def _get(db, eid, version=None):
        row = db.execute("SELECT v.*,e.state,e.published_version FROM versions v JOIN experiences e ON e.id=v.experience_id "
            "WHERE v.experience_id=? AND (? IS NULL OR v.version=?) ORDER BY v.version DESC LIMIT 1", (eid, version, version)).fetchone()
        if not row:
            raise ValueError("Unknown experience/version")
        result = {"id": eid, "version": row["version"], "state": row["state"], "published_version": row["published_version"],
            **json.loads(row["body_json"])}
        result["evidence"] = [{"source_id": x["source_id"], "evidence_refs": json.loads(x["refs_json"]),
            "observation_ids": json.loads(x["observations_json"]), "relation": x["relation"]} for x in db.execute(
            "SELECT * FROM evidence_links WHERE experience_id=? AND version=?", (eid, row["version"]))]
        return result

    def list_ids(self):
        with self.connect() as db:
            return [r[0] for r in db.execute("SELECT id FROM experiences ORDER BY id")]

    def source(self, sid):
        with self.connect() as db:
            row = db.execute("SELECT * FROM sources WHERE id=?", (sid,)).fetchone()
        if not row:
            raise ValueError("Unknown source")
        return dict(row)

    def check_source(self, sid):
        """Reload through the original allowlist; never substitute another log or RGB."""
        source = self.source(sid)
        try:
            bundle = read_run(Path(source["root"]), instruction_id=source["instruction_id"], instruction=source["instruction"])
            if bundle.index() != json.loads(source["evidence_json"]):
                raise ValueError("Source changed")
        except (OSError, ValueError, KeyError):
            with self.connect(write=True) as db:
                db.execute("UPDATE sources SET availability='unavailable',issue='missing_or_changed' WHERE id=?", (sid,))
            raise ValueError("Source unavailable or changed") from None
        return bundle

    def audit_sources(self):
        with self.connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM sources ORDER BY id")]
        result = {}
        for sid in ids:
            try:
                self.check_source(sid)
                result[sid] = "available"
            except ValueError:
                result[sid] = "unavailable"
        return result

    def search(self, query, *, filters=None, limit=3, budget=6000, scope="production", maintenance=False):
        """Read-only snapshot; no model calls, queue writes, or source file scanning."""
        filters = filters or {}
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("Query must be 1..2000 characters")
        if type(limit) is not int or not 1 <= limit <= (10 if maintenance else 5) or budget < 1:
            raise ValueError("Invalid result budget")
        if scope not in ("experiment", "production") or not isinstance(filters, dict) or set(filters) - TAG_FIELDS:
            raise ValueError("Invalid scope or filters")
        if any(not isinstance(v, str) or not v.strip() for v in filters.values()):
            raise ValueError("Filters require string values")
        terms = tokens(query)[:128]
        if not terms:
            return []
        match = " OR ".join('"'+term+'"' for term in terms)
        sql = """SELECT v.experience_id,v.version,bm25(search_index) AS score
          FROM search_index JOIN versions v ON v.experience_id=search_index.experience_id AND v.version=search_index.version
          JOIN experiences e ON e.id=v.experience_id
          WHERE search_index MATCH ?"""
        params = [match]
        if not maintenance:
            sql += """ AND e.state='published' AND e.published_version=v.version
              AND EXISTS (SELECT 1 FROM evidence_links l WHERE l.experience_id=v.experience_id AND l.version=v.version)
              AND NOT EXISTS (SELECT 1 FROM evidence_links l JOIN sources s ON s.id=l.source_id
                WHERE l.experience_id=v.experience_id AND l.version=v.version AND (s.scope<>? OR s.availability<>'available'))"""
            params.append(scope)
        for dimension, value in filters.items():
            # Unknown compatibility is returned explicitly; known mismatch is excluded.
            sql += """ AND (NOT EXISTS (SELECT 1 FROM tags t WHERE t.experience_id=v.experience_id AND t.version=v.version AND t.dimension=?)
                OR EXISTS (SELECT 1 FROM tags t WHERE t.experience_id=v.experience_id AND t.version=v.version AND t.dimension=? AND t.value=?))"""
            params.extend((dimension, dimension, value))
        sql += " ORDER BY score,v.experience_id,v.version DESC"
        results, seen, size = [], set(), 0
        with self.connect() as db:
            for row in db.execute(sql, params):
                card = self._get(db, row[0], row[1])
                key = digest({k: card[k] for k in FIELDS})
                if not maintenance and key in seen:
                    continue
                card = {k: card[k] for k in ("id", "version", "title", *FIELDS, "tags", "evidence")}
                card["unknown_conditions"] = [k for k in filters if k not in card["tags"]]
                card["conflicts"] = [dict(r) for r in db.execute("SELECT origin,destination,kind FROM relations WHERE kind='conflict' AND (origin=? OR destination=?)", (row[0], row[0]))]
                length = len(dumps(card))
                if size + length > budget:
                    continue
                results.append(card)
                seen.add(key)
                size += length
                if len(results) >= limit:
                    break
        return results

    def set_state(self, eid, state, *, actor, reason):
        if state not in ("withdrawn", "archived") or not actor.strip() or not reason.strip():
            raise ValueError("Retirement requires state, human actor and reason")
        with self.connect(write=True) as db:
            card = self._get(db, eid)
            db.execute("UPDATE experiences SET state=? WHERE id=?", (state, eid))
            self._review(db, eid, card["version"], state, reason, actor, {})

    def enqueue(self, ids, *, request_key=None):
        """One bounded maintenance job per target; caller key supports retry-safe batches."""
        result = []
        for eid in sorted(set(ids)):
            self.get(eid)
            dedup = digest({"key": request_key or "active", "id": eid})
            with self.connect(write=True) as db:
                if request_key is None:
                    row = db.execute("SELECT id FROM jobs WHERE dedup=? AND state IN ('pending','running')", (dedup,)).fetchone()
                    if row:
                        result.append(row[0])
                        continue
                    db.execute("UPDATE jobs SET dedup=id WHERE dedup=?", (dedup,))
                jid = uuid.uuid4().hex
                db.execute("INSERT OR IGNORE INTO jobs(id,dedup,payload_json,created) VALUES (?,?,?,?)", (jid, dedup, dumps({"id": eid}), time.time()))
                result.append(db.execute("SELECT id FROM jobs WHERE dedup=?", (dedup,)).fetchone()[0])
        return result

    def jobs(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM jobs ORDER BY created,id")]

    def proposals(self):
        with self.connect() as db:
            return [{**dict(r), "change": json.loads(r["change_json"]), "review": json.loads(r["review_json"])}
                for r in db.execute("SELECT * FROM proposals ORDER BY created,id")]
