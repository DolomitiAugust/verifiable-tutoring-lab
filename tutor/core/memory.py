"""SQLite evidence memory. Only verified student predictions count as attempts."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


LEVELS = ("基础", "进阶", "挑战")


class EpisodeStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_id TEXT UNIQUE NOT NULL, student_id TEXT NOT NULL,
                    topic TEXT NOT NULL, difficulty TEXT NOT NULL,
                    created_at TEXT NOT NULL, status TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS related_episodes
                    ON episodes(student_id, topic, sequence DESC);
                CREATE TABLE IF NOT EXISTS topic_summary (
                    student_id TEXT NOT NULL, topic TEXT NOT NULL,
                    difficulty TEXT NOT NULL, turns INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    correct INTEGER NOT NULL DEFAULT 0,
                    incorrect INTEGER NOT NULL DEFAULT 0,
                    unanswered INTEGER NOT NULL DEFAULT 0,
                    unverified INTEGER NOT NULL DEFAULT 0,
                    last_turn_id TEXT NOT NULL,
                    PRIMARY KEY(student_id, topic, difficulty)
                );
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def snapshot(self, student_id):
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM topic_summary WHERE student_id=? ORDER BY topic,difficulty",
                (student_id,)).fetchall()
        return {"student_id": student_id, "topics": [dict(row) for row in rows],
                "interpretation": "已核验学生预测的描述统计；不是掌握度或学习效果测量。"}

    def retrieve(self, student_id, topic, limit=3):
        if not 1 <= limit <= 100:
            raise ValueError("Memory retrieval limit must be in 1..100.")
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM episodes WHERE student_id=? AND topic=? "
                "ORDER BY sequence DESC LIMIT ?", (student_id, topic, limit)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def episodes(self, student_id=None):
        """Administrative export; tutoring retrieval always requires student + topic."""
        with self._connect() as db:
            if student_id is None:
                rows = db.execute("SELECT payload FROM episodes ORDER BY sequence").fetchall()
            else:
                rows = db.execute("SELECT payload FROM episodes WHERE student_id=? ORDER BY sequence",
                                  (student_id,)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def recommend(self, student_id, topic, current_difficulty, current=None):
        if current_difficulty not in LEVELS:
            raise ValueError("Unknown program difficulty.")
        history = self.retrieve(student_id, topic, limit=100)
        if current is not None:
            history.insert(0, current)
        evidence = [r for r in history if r["student_evidence"]["status"] in ("correct", "incorrect")
                    and r["tool"]["main"]["difficulty"] == current_difficulty]
        recent = evidence[:3]
        position = LEVELS.index(current_difficulty)
        action, target = "keep", position
        reason = "尚无连续三次正确或连续两次错误的已核验学生预测，保持当前结构难度。"
        if len(recent) >= 3 and all(r["student_evidence"]["status"] == "correct" for r in recent):
            target = min(position + 1, len(LEVELS) - 1)
            action = "increase" if target > position else "keep"
            reason = "最近三次已核验学生预测均正确，可建议提高一级；最高级保持。"
        elif len(recent) >= 2 and all(r["student_evidence"]["status"] == "incorrect" for r in recent[:2]):
            target = max(position - 1, 0)
            action = "decrease" if target < position else "keep"
            reason = "最近两次已核验学生预测均错误，可建议降低一级；最低级保持。"
        used = recent if action != "decrease" else recent[:2]
        return {"action": action, "current_difficulty": current_difficulty,
                "suggested_difficulty": LEVELS[target],
                "evidence_episode_ids": [r["turn_id"] for r in used], "reason": reason,
                "source": "rule_from_verified_student_predictions",
                "evidence_scope": "same_student_same_topic_same_structural_difficulty",
                "mastery_claim": False}

    def save(self, episode):
        """Insert raw evidence and update counters in one transaction; no LLM scores."""
        status = episode["student_evidence"]["status"]
        if status not in ("correct", "incorrect", "unanswered", "unverified"):
            raise ValueError("Unknown student evidence status.")
        main = episode["tool"]["main"]
        supplied = episode["input"]["student_answer"]
        if status in ("correct", "incorrect"):
            if not main["ok"] or supplied is None:
                raise ValueError("Cannot count a prediction without verified program facts.")
            actual = "correct" if supplied == main["output"] else "incorrect"
            if status != actual:
                raise ValueError("Memory evidence disagrees with the student's actual prediction.")
        values = (episode["student_id"], episode["input"]["topic"], main["difficulty"])
        counters = [int(status in ("correct", "incorrect"))] + [
            int(status == key) for key in ("correct", "incorrect", "unanswered", "unverified")]
        with self._connect() as db:
            db.execute("INSERT INTO episodes(turn_id,student_id,topic,difficulty,created_at,status,payload) "
                       "VALUES(?,?,?,?,?,?,?)", (episode["turn_id"], *values, episode["created_at"],
                                                  status, json.dumps(episode, ensure_ascii=False)))
            db.execute("""
                INSERT INTO topic_summary(student_id,topic,difficulty,turns,attempts,correct,
                                          incorrect,unanswered,unverified,last_turn_id)
                VALUES(?,?,?,1,?,?,?,?,?,?)
                ON CONFLICT(student_id,topic,difficulty) DO UPDATE SET
                    turns=turns+1, attempts=attempts+excluded.attempts,
                    correct=correct+excluded.correct, incorrect=incorrect+excluded.incorrect,
                    unanswered=unanswered+excluded.unanswered, unverified=unverified+excluded.unverified,
                    last_turn_id=excluded.last_turn_id
            """, (*values, *counters, episode["turn_id"]))

    def close(self):
        # Every operation closes its own connection, so re-instantiation is safe.
        pass
