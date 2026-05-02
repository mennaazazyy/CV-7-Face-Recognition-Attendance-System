import tempfile
import os
import numpy as np
import pytest

# Redirect DB to a temp file so tests don't touch the real database
os.environ.setdefault("_CV7_TEST", "1")


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    import src.config as cfg
    monkeypatch.setattr(cfg, "DB_PATH", tmp_path / "test.db")
    from src.database import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    yield


def test_add_and_query_student():
    from src.database.db import add_student, query_attendance
    add_student("001", "Alice")
    rows = query_attendance("CS101-2025-01-01")
    assert rows == []


def test_mark_present_dedup():
    from src.database.db import add_student, mark_present, query_attendance
    add_student("002", "Bob")
    first = mark_present("002", "CS101-2025-01-01", 0.95)
    second = mark_present("002", "CS101-2025-01-01", 0.95)
    assert first is True
    assert second is False
    rows = query_attendance("CS101-2025-01-01")
    assert len(rows) == 1


def test_store_and_retrieve_embedding():
    from src.database.db import add_student, store_embedding, get_all_embeddings
    add_student("003", "Carol")
    emb = np.random.rand(512).astype(np.float32)
    store_embedding("003", "arcface", emb)
    results = get_all_embeddings("arcface")
    assert len(results) == 1
    sid, loaded = results[0]
    assert sid == "003"
    np.testing.assert_allclose(loaded, emb, rtol=1e-5)


def test_hard_delete_cascades():
    from src.database.db import add_student, store_embedding, delete_student, get_all_embeddings
    add_student("004", "Dave")
    store_embedding("004", "arcface", np.ones(512, dtype=np.float32))
    delete_student("004")
    assert get_all_embeddings("arcface") == []
