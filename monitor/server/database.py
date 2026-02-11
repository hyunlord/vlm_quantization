from __future__ import annotations

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
            step INTEGER NOT NULL,
            epoch INTEGER NOT NULL,
            loss_total REAL,
            loss_contrastive REAL,
            loss_quantization REAL,
            loss_balance REAL,
            loss_consistency REAL,
            loss_ortho REAL DEFAULT 0.0,
            loss_lcs REAL DEFAULT 0.0,
            lr REAL,
            timestamp REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS eval_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER NOT NULL,
            step INTEGER,
            map_i2t REAL,
            map_t2i REAL,
            map_i2i REAL,
            map_t2t REAL,
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

        CREATE INDEX IF NOT EXISTS idx_train_step ON training_metrics(step);
        CREATE INDEX IF NOT EXISTS idx_train_epoch ON training_metrics(epoch);
        CREATE INDEX IF NOT EXISTS idx_eval_epoch ON eval_metrics(epoch);
        CREATE INDEX IF NOT EXISTS idx_sys_ts ON system_metrics(timestamp);
    """)
    # Migrate: add columns for existing DBs
    for col in ("loss_ortho", "loss_lcs"):
        try:
            conn.execute(
                f"ALTER TABLE training_metrics ADD COLUMN {col} REAL DEFAULT 0.0"
            )
        except sqlite3.OperationalError:
            pass  # column already exists
    for col in ("step", "val_loss_total", "val_loss_contrastive",
                "val_loss_quantization", "val_loss_balance",
                "val_loss_consistency", "val_loss_ortho", "val_loss_lcs",
                "backbone_map_i2t", "backbone_map_t2i",
                "backbone_p1", "backbone_p5", "backbone_p10"):
        try:
            conn.execute(
                f"ALTER TABLE eval_metrics ADD COLUMN {col} REAL"
            )
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def insert_training_metric(m: TrainingMetric) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO training_metrics
           (step, epoch, loss_total, loss_contrastive, loss_quantization,
            loss_balance, loss_consistency, loss_ortho, loss_lcs, lr, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            m.step, m.epoch, m.loss_total, m.loss_contrastive,
            m.loss_quantization, m.loss_balance, m.loss_consistency,
            m.loss_ortho, m.loss_lcs,
            m.lr, time.time(),
        ),
    )
    conn.commit()
    conn.close()


def insert_eval_metric(m: EvalMetric) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO eval_metrics
           (epoch, step, map_i2t, map_t2i, map_i2i, map_t2t,
            backbone_map_i2t, backbone_map_t2i,
            p1, p5, p10, backbone_p1, backbone_p5, backbone_p10,
            bit_entropy, quant_error,
            val_loss_total, val_loss_contrastive, val_loss_quantization,
            val_loss_balance, val_loss_consistency, val_loss_ortho,
            val_loss_lcs, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            m.epoch, m.step, m.map_i2t, m.map_t2i, m.map_i2i, m.map_t2t,
            m.backbone_map_i2t, m.backbone_map_t2i,
            m.p1, m.p5, m.p10, m.backbone_p1, m.backbone_p5, m.backbone_p10,
            m.bit_entropy, m.quant_error,
            m.val_loss_total, m.val_loss_contrastive, m.val_loss_quantization,
            m.val_loss_balance, m.val_loss_consistency, m.val_loss_ortho,
            m.val_loss_lcs, time.time(),
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


def clear_training_metrics() -> None:
    """Delete training metrics from previous run (called on new training run).

    Eval metrics are preserved so baseline validation data (collected before
    training) survives the on_train_start clear.
    """
    conn = get_connection()
    conn.execute("DELETE FROM training_metrics")
    conn.commit()
    conn.close()


def clear_all_metrics() -> None:
    """Delete all training and eval metrics (full reset)."""
    conn = get_connection()
    conn.execute("DELETE FROM training_metrics")
    conn.execute("DELETE FROM eval_metrics")
    conn.commit()
    conn.close()


def get_training_metrics(
    start_step: int = 0, end_step: int | None = None
) -> list[dict]:
    conn = get_connection()
    if end_step is not None:
        rows = conn.execute(
            "SELECT * FROM training_metrics WHERE step >= ? AND step <= ? ORDER BY step",
            (start_step, end_step),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM training_metrics WHERE step >= ? ORDER BY step",
            (start_step,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_eval_metrics() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM eval_metrics ORDER BY epoch"
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
