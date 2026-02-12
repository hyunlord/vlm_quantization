from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from monitor.server.models import EvalMetric, SystemMetric, TrainingMetric

DB_PATH = Path(
    os.environ.get(
        "MONITOR_DB_PATH",
        str(Path(__file__).parent.parent / "metrics.db"),
    )
)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL DEFAULT '',
            step INTEGER NOT NULL,
            epoch INTEGER NOT NULL,
            loss_total REAL,
            loss_contrastive REAL,
            loss_quantization REAL,
            loss_balance REAL,
            loss_consistency REAL,
            loss_ortho REAL DEFAULT 0.0,
            loss_lcs REAL DEFAULT 0.0,
            loss_distillation REAL DEFAULT 0.0,
            loss_adapter_align REAL DEFAULT 0.0,
            lr REAL,
            temperature REAL,
            timestamp REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS eval_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL DEFAULT '',
            epoch INTEGER NOT NULL,
            step INTEGER,
            map_i2t REAL,
            map_t2i REAL,
            backbone_map_i2t REAL,
            backbone_map_t2i REAL,
            p1 REAL,
            p5 REAL,
            p10 REAL,
            backbone_p1 REAL,
            backbone_p5 REAL,
            backbone_p10 REAL,
            bit_entropy REAL,
            quant_error REAL,
            val_loss_total REAL,
            val_loss_contrastive REAL,
            val_loss_quantization REAL,
            val_loss_balance REAL,
            val_loss_consistency REAL,
            val_loss_ortho REAL,
            val_loss_lcs REAL,
            val_loss_distillation REAL,
            val_loss_adapter_align REAL,
            timestamp REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gpu_util REAL,
            gpu_mem_used REAL,
            gpu_mem_total REAL,
            gpu_temp REAL,
            gpu_name TEXT,
            cpu_util REAL,
            ram_used REAL,
            ram_total REAL,
            timestamp REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS hash_analysis_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL DEFAULT '',
            epoch INTEGER NOT NULL,
            step INTEGER NOT NULL,
            data TEXT NOT NULL,
            timestamp REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_train_step ON training_metrics(step);
        CREATE INDEX IF NOT EXISTS idx_train_epoch ON training_metrics(epoch);
        CREATE INDEX IF NOT EXISTS idx_eval_epoch ON eval_metrics(epoch);
        CREATE INDEX IF NOT EXISTS idx_sys_ts ON system_metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_hash_epoch ON hash_analysis_snapshots(epoch);

        -- Runs table for tracking training runs
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            started_at REAL NOT NULL,
            ended_at REAL,
            status TEXT DEFAULT 'running',
            config_json TEXT,
            total_epochs INTEGER,
            total_steps INTEGER,
            best_checkpoint_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);

        -- Checkpoints table for tracking saved model checkpoints
        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            step INTEGER,
            path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            size_mb REAL,
            val_loss REAL,
            created_at REAL NOT NULL,
            hparams_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ckpt_run ON checkpoints(run_id);
        CREATE INDEX IF NOT EXISTS idx_ckpt_run_epoch ON checkpoints(run_id, epoch);
    """)
    # Migrate: add columns for existing DBs
    for col in ("loss_ortho", "loss_lcs", "loss_distillation"):
        try:
            conn.execute(
                f"ALTER TABLE training_metrics ADD COLUMN {col} REAL DEFAULT 0.0"
            )
        except sqlite3.OperationalError:
            pass  # column already exists
    for col in ("temperature",):
        try:
            conn.execute(
                f"ALTER TABLE training_metrics ADD COLUMN {col} REAL"
            )
        except sqlite3.OperationalError:
            pass
    for col in ("loss_adapter_align",):
        try:
            conn.execute(
                f"ALTER TABLE training_metrics ADD COLUMN {col} REAL DEFAULT 0.0"
            )
        except sqlite3.OperationalError:
            pass
    for col in ("step", "val_loss_total", "val_loss_contrastive",
                "val_loss_quantization", "val_loss_balance",
                "val_loss_consistency", "val_loss_ortho", "val_loss_lcs",
                "val_loss_distillation", "val_loss_adapter_align",
                "backbone_map_i2t", "backbone_map_t2i",
                "backbone_p1", "backbone_p5", "backbone_p10"):
        try:
            conn.execute(
                f"ALTER TABLE eval_metrics ADD COLUMN {col} REAL"
            )
        except sqlite3.OperationalError:
            pass
    # Migrate: add run_id column to existing tables
    for table in ("training_metrics", "eval_metrics", "hash_analysis_snapshots"):
        try:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN run_id TEXT DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass
    # Create run_id indexes (after migration ensures the column exists)
    for idx, table in (
        ("idx_train_run", "training_metrics"),
        ("idx_eval_run", "eval_metrics"),
        ("idx_hash_run", "hash_analysis_snapshots"),
    ):
        try:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {idx} ON {table}(run_id)"
            )
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def insert_training_metric(m: TrainingMetric) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO training_metrics
           (run_id, step, epoch, loss_total, loss_contrastive, loss_quantization,
            loss_balance, loss_consistency, loss_ortho, loss_lcs,
            loss_distillation, loss_adapter_align, lr, temperature, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            m.run_id, m.step, m.epoch, m.loss_total, m.loss_contrastive,
            m.loss_quantization, m.loss_balance, m.loss_consistency,
            m.loss_ortho, m.loss_lcs, m.loss_distillation,
            m.loss_adapter_align, m.lr, m.temperature, time.time(),
        ),
    )
    conn.commit()
    conn.close()


def insert_eval_metric(m: EvalMetric) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO eval_metrics
           (run_id, epoch, step, map_i2t, map_t2i,
            backbone_map_i2t, backbone_map_t2i,
            p1, p5, p10, backbone_p1, backbone_p5, backbone_p10,
            bit_entropy, quant_error,
            val_loss_total, val_loss_contrastive, val_loss_quantization,
            val_loss_balance, val_loss_consistency, val_loss_ortho,
            val_loss_lcs, val_loss_distillation, val_loss_adapter_align,
            timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            m.run_id, m.epoch, m.step, m.map_i2t, m.map_t2i,
            m.backbone_map_i2t, m.backbone_map_t2i,
            m.p1, m.p5, m.p10, m.backbone_p1, m.backbone_p5, m.backbone_p10,
            m.bit_entropy, m.quant_error,
            m.val_loss_total, m.val_loss_contrastive, m.val_loss_quantization,
            m.val_loss_balance, m.val_loss_consistency, m.val_loss_ortho,
            m.val_loss_lcs, m.val_loss_distillation, m.val_loss_adapter_align,
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


def insert_system_metric(m: SystemMetric) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO system_metrics
           (gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp, gpu_name,
            cpu_util, ram_used, ram_total, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            m.gpu_util, m.gpu_mem_used, m.gpu_mem_total, m.gpu_temp,
            m.gpu_name, m.cpu_util, m.ram_used, m.ram_total, time.time(),
        ),
    )
    conn.commit()
    conn.close()


def clear_all_metrics() -> None:
    """Delete all training, eval, and hash analysis metrics (full reset)."""
    conn = get_connection()
    conn.execute("DELETE FROM training_metrics")
    conn.execute("DELETE FROM eval_metrics")
    conn.execute("DELETE FROM hash_analysis_snapshots")
    conn.commit()
    conn.close()


def get_runs() -> list[dict]:
    """List all distinct training runs with summary stats."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            run_id,
            MIN(timestamp) as started_at,
            MAX(epoch) as epochs,
            COUNT(*) as num_training_points
        FROM training_metrics
        WHERE run_id != ''
        GROUP BY run_id
        ORDER BY run_id DESC
    """).fetchall()
    runs = []
    for r in rows:
        run_id = r["run_id"]
        eval_count = conn.execute(
            "SELECT COUNT(*) FROM eval_metrics WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        hash_count = conn.execute(
            "SELECT COUNT(*) FROM hash_analysis_snapshots WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
        runs.append({
            "run_id": run_id,
            "started_at": r["started_at"],
            "epochs": r["epochs"],
            "num_training_points": r["num_training_points"],
            "num_eval_points": eval_count,
            "has_hash_analysis": hash_count > 0,
        })
    conn.close()
    return runs


def _latest_run_id(conn: sqlite3.Connection, table: str) -> str | None:
    """Auto-detect the most recent run_id from a table."""
    row = conn.execute(
        f"SELECT run_id FROM {table} WHERE run_id != '' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["run_id"] if row else None


def get_training_metrics(
    start_step: int = 0,
    end_step: int | None = None,
    run_id: str | None = None,
) -> list[dict]:
    conn = get_connection()
    # Auto-select latest run when no run_id specified
    if not run_id:
        run_id = _latest_run_id(conn, "training_metrics")
    conditions = ["step >= ?"]
    params: list = [start_step]
    if end_step is not None:
        conditions.append("step <= ?")
        params.append(end_step)
    if run_id:
        conditions.append("run_id = ?")
        params.append(run_id)
    where = " AND ".join(conditions)
    rows = conn.execute(
        f"SELECT * FROM training_metrics WHERE {where} ORDER BY step",
        params,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_eval_metrics(run_id: str | None = None) -> list[dict]:
    conn = get_connection()
    # Auto-select latest run when no run_id specified
    if not run_id:
        run_id = _latest_run_id(conn, "eval_metrics")
    if run_id:
        rows = conn.execute(
            "SELECT * FROM eval_metrics WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM eval_metrics ORDER BY id"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_system_metrics(limit: int = 100) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM system_metrics ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


# --- Hash Analysis Snapshots ---


def insert_hash_analysis(epoch: int, step: int, data: dict,
                         run_id: str = "") -> int:
    """Insert a hash analysis snapshot. Returns the new row id."""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO hash_analysis_snapshots (run_id, epoch, step, data, timestamp) VALUES (?, ?, ?, ?, ?)",
        (run_id, epoch, step, json.dumps(data), time.time()),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_hash_analysis_list(run_id: str | None = None) -> list[dict]:
    """Return lightweight list of snapshots (no data blob)."""
    conn = get_connection()
    if run_id:
        rows = conn.execute(
            "SELECT id, run_id, epoch, step, timestamp FROM hash_analysis_snapshots WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, run_id, epoch, step, timestamp FROM hash_analysis_snapshots ORDER BY id"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_hash_analysis_by_id(snapshot_id: int) -> dict | None:
    """Return a specific snapshot's full data by id."""
    conn = get_connection()
    row = conn.execute(
        "SELECT data FROM hash_analysis_snapshots WHERE id = ?",
        (snapshot_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return json.loads(row["data"])


def get_latest_hash_analysis() -> dict | None:
    """Return the most recent snapshot's full data."""
    conn = get_connection()
    row = conn.execute(
        "SELECT data FROM hash_analysis_snapshots ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return json.loads(row["data"])


def get_poll_watermarks() -> dict:
    """Get current max IDs for DB polling watermarks."""
    conn = get_connection()
    train_id = conn.execute(
        "SELECT COALESCE(MAX(id), 0) FROM training_metrics"
    ).fetchone()[0]
    eval_id = conn.execute(
        "SELECT COALESCE(MAX(id), 0) FROM eval_metrics"
    ).fetchone()[0]
    hash_count = conn.execute(
        "SELECT COUNT(*) FROM hash_analysis_snapshots"
    ).fetchone()[0]
    conn.close()
    return {
        "train_id": train_id,
        "eval_id": eval_id,
        "hash_count": hash_count,
    }


def get_new_training_metrics(after_id: int) -> list[dict]:
    """Get training metrics with id > after_id."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM training_metrics WHERE id > ? ORDER BY id",
        (after_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_new_eval_metrics(after_id: int) -> list[dict]:
    """Get eval metrics with id > after_id."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM eval_metrics WHERE id > ? ORDER BY id",
        (after_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Run Management ---


def register_run(run_id: str, config: dict | None = None) -> int:
    """Register a new training run. Returns the run's row id."""
    conn = get_connection()
    cur = conn.execute(
        """INSERT OR IGNORE INTO runs (run_id, started_at, status, config_json)
           VALUES (?, ?, 'running', ?)""",
        (run_id, time.time(), json.dumps(config) if config else None),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def update_run_status(run_id: str, status: str, total_epochs: int | None = None,
                      total_steps: int | None = None) -> None:
    """Update a run's status and optionally total epochs/steps."""
    conn = get_connection()
    updates = ["status = ?"]
    params: list = [status]
    if status in ("completed", "failed"):
        updates.append("ended_at = ?")
        params.append(time.time())
    if total_epochs is not None:
        updates.append("total_epochs = ?")
        params.append(total_epochs)
    if total_steps is not None:
        updates.append("total_steps = ?")
        params.append(total_steps)
    params.append(run_id)
    conn.execute(
        f"UPDATE runs SET {', '.join(updates)} WHERE run_id = ?",
        params,
    )
    conn.commit()
    conn.close()


def get_run_details(run_id: str) -> dict | None:
    """Get detailed info for a specific run including checkpoint count."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        conn.close()
        return None
    result = dict(row)
    # Get checkpoint count
    ckpt_count = conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    result["checkpoint_count"] = ckpt_count
    # Get best checkpoint
    best = conn.execute(
        """SELECT id, epoch, val_loss, path FROM checkpoints
           WHERE run_id = ? AND val_loss IS NOT NULL
           ORDER BY val_loss ASC LIMIT 1""",
        (run_id,),
    ).fetchone()
    if best:
        result["best_checkpoint"] = dict(best)
    conn.close()
    return result


def get_runs_with_checkpoints() -> list[dict]:
    """List all runs with checkpoint info."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT r.*,
               (SELECT COUNT(*) FROM checkpoints c WHERE c.run_id = r.run_id) as checkpoint_count,
               (SELECT MIN(val_loss) FROM checkpoints c WHERE c.run_id = r.run_id) as best_val_loss
        FROM runs r
        ORDER BY r.started_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Checkpoint Management ---


def register_checkpoint(
    run_id: str,
    epoch: int,
    path: str,
    val_loss: float | None = None,
    step: int | None = None,
    size_mb: float | None = None,
    hparams: dict | None = None,
) -> int:
    """Register a checkpoint. Returns the checkpoint id."""
    conn = get_connection()
    filename = Path(path).name
    cur = conn.execute(
        """INSERT OR REPLACE INTO checkpoints
           (run_id, epoch, step, path, filename, size_mb, val_loss, created_at, hparams_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, epoch, step, path, filename, size_mb, val_loss, time.time(),
         json.dumps(hparams) if hparams else None),
    )
    ckpt_id = cur.lastrowid
    conn.commit()
    conn.close()
    return ckpt_id


def get_checkpoints_for_run(run_id: str) -> list[dict]:
    """Get all checkpoints for a specific run, ordered by epoch."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT * FROM checkpoints WHERE run_id = ?
           ORDER BY epoch, step""",
        (run_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_checkpoint_by_id(checkpoint_id: int) -> dict | None:
    """Get a checkpoint by its ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_checkpoint_by_path(path: str) -> dict | None:
    """Get a checkpoint by its file path."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE path = ?", (path,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_epochs_for_run(run_id: str) -> list[dict]:
    """Get epoch summaries for a run with checkpoint and metrics info."""
    conn = get_connection()
    # Get epochs from training metrics
    epochs = conn.execute("""
        SELECT epoch,
               MIN(step) as start_step,
               MAX(step) as end_step,
               COUNT(*) as num_steps,
               MIN(timestamp) as started_at,
               MAX(timestamp) as ended_at
        FROM training_metrics
        WHERE run_id = ?
        GROUP BY epoch
        ORDER BY epoch
    """, (run_id,)).fetchall()

    result = []
    for e in epochs:
        epoch_data = dict(e)
        # Check for checkpoint at this epoch
        ckpt = conn.execute(
            """SELECT id, path, val_loss, size_mb FROM checkpoints
               WHERE run_id = ? AND epoch = ?""",
            (run_id, e["epoch"]),
        ).fetchone()
        if ckpt:
            epoch_data["checkpoint"] = dict(ckpt)
        # Get eval metrics for this epoch
        eval_row = conn.execute(
            """SELECT map_i2t, map_t2i, val_loss_total FROM eval_metrics
               WHERE run_id = ? AND epoch = ?""",
            (run_id, e["epoch"]),
        ).fetchone()
        if eval_row:
            epoch_data["eval"] = dict(eval_row)
        result.append(epoch_data)
    conn.close()
    return result


def get_eval_metrics_for_checkpoints(checkpoints: list[dict]) -> dict[str, dict]:
    """Look up eval metrics (mAP, P@k) for a list of checkpoints.

    Returns a dict keyed by (run_id, epoch, step) tuple-string with metric values.
    Matches by (run_id, epoch) with optional step refinement.
    """
    if not checkpoints:
        return {}

    conn = get_connection()
    result: dict[str, dict] = {}

    for ckpt in checkpoints:
        run_id = ckpt.get("run_dir") or ckpt.get("run_id", "")
        epoch = ckpt.get("epoch")
        step = ckpt.get("step")
        if epoch is None:
            continue

        # Try exact match on (run_id, epoch, step) first, then fall back to (run_id, epoch)
        row = None
        if step is not None:
            row = conn.execute(
                """SELECT map_i2t, map_t2i, p1, p5, p10
                   FROM eval_metrics
                   WHERE run_id = ? AND epoch = ? AND step = ?""",
                (run_id, epoch, step),
            ).fetchone()
        if row is None:
            row = conn.execute(
                """SELECT map_i2t, map_t2i, p1, p5, p10
                   FROM eval_metrics
                   WHERE run_id = ? AND epoch = ?
                   ORDER BY step DESC LIMIT 1""",
                (run_id, epoch),
            ).fetchone()

        if row:
            key = ckpt.get("path", f"{run_id}_{epoch}_{step}")
            result[key] = dict(row)

    conn.close()
    return result


def get_all_checkpoints() -> list[dict]:
    """Get all checkpoints across all runs."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT c.*, r.status as run_status
           FROM checkpoints c
           LEFT JOIN runs r ON c.run_id = r.run_id
           ORDER BY c.created_at DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def sync_checkpoints_from_disk(checkpoint_dir: str, run_id: str | None = None) -> int:
    """Scan checkpoint directory and register any untracked checkpoints.
    Returns the number of newly registered checkpoints."""
    import re
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return 0

    conn = get_connection()
    count = 0

    # Pattern to extract epoch from filename
    epoch_pattern = re.compile(r"epoch[=_](\d+)")
    val_loss_pattern = re.compile(r"val[/_]total[=_]?([\d.]+)")

    for ckpt_file in ckpt_path.rglob("*.ckpt"):
        path_str = str(ckpt_file)
        # Skip if already registered
        existing = conn.execute(
            "SELECT id FROM checkpoints WHERE path = ?", (path_str,)
        ).fetchone()
        if existing:
            continue

        # Extract run_id from directory structure
        rel_path = ckpt_file.relative_to(ckpt_path)
        parts = rel_path.parts
        detected_run_id = parts[0] if len(parts) > 1 else (run_id or "unknown")

        # Extract epoch from filename
        epoch_match = epoch_pattern.search(ckpt_file.name)
        epoch = int(epoch_match.group(1)) if epoch_match else 0

        # Extract val_loss from filename
        val_match = val_loss_pattern.search(ckpt_file.name)
        val_loss = float(val_match.group(1)) if val_match else None

        # Get file size
        size_mb = ckpt_file.stat().st_size / (1024 * 1024)

        conn.execute(
            """INSERT INTO checkpoints
               (run_id, epoch, path, filename, size_mb, val_loss, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (detected_run_id, epoch, path_str, ckpt_file.name, size_mb, val_loss,
             ckpt_file.stat().st_mtime),
        )
        count += 1

    conn.commit()
    conn.close()
    return count
