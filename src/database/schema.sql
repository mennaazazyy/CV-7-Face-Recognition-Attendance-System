PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT,
    created_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS face_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    template BLOB NOT NULL,
    template_dim INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    UNIQUE(student_id, model_name)
);

CREATE TABLE IF NOT EXISTS attendance_sessions (
    session_id TEXT PRIMARY KEY,
    course_code TEXT NOT NULL,
    session_date TEXT NOT NULL,
    model_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    marked_at TEXT NOT NULL,
    confidence REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES attendance_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    UNIQUE(session_id, student_id)
);

CREATE TABLE IF NOT EXISTS recognition_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    student_id TEXT,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    event_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES attendance_sessions(session_id) ON DELETE SET NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_face_templates_student ON face_templates(student_id);
CREATE INDEX IF NOT EXISTS idx_face_templates_model ON face_templates(model_name);
CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance_records(session_id);
CREATE INDEX IF NOT EXISTS idx_recognition_events_session ON recognition_events(session_id);
