import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="quiz_reports.db"):
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quiz_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    quiz_config TEXT,
                    model_used TEXT,
                    total_questions INTEGER,
                    answered_questions INTEGER,
                    correct_answers INTEGER,
                    completion_rate REAL,
                    score REAL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quiz_answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    question_id INTEGER,
                    question_text TEXT,
                    question_type TEXT,
                    user_answer TEXT,
                    correct_answer TEXT,
                    is_correct INTEGER,
                    FOREIGN KEY(session_id) REFERENCES quiz_sessions(id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quiz_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    insights TEXT,
                    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES quiz_sessions(id)
                )
            ''')
            conn.commit()

    def save_quiz_session(self, quiz_data, results):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO quiz_sessions (
                    quiz_config, 
                    model_used,
                    total_questions,
                    answered_questions,
                    correct_answers,
                    completion_rate,
                    score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(quiz_data.get('config', {})),
                quiz_data.get('model_used', ''),
                results.get('total_questions', 0),
                results.get('answered_questions', 0),
                results.get('correct_answers', 0),
                results.get('completion_rate', 0),
                results.get('score', 0)
            ))
            session_id = cursor.lastrowid
            conn.commit()
            return session_id

    def save_quiz_answers(self, session_id, questions, user_answers):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, q in enumerate(questions):
                q_key = f"q_{i}"
                if q_key in user_answers:
                    user_answer = user_answers[q_key]
                    correct_answer = q['answer']
                    
                    # Determine correctness based on question type
                    is_correct = 0
                    if q['type'] == "Multiple Choice":
                        correct_option = next((opt for opt in q['options'] if opt['letter'] == correct_answer), None)
                        is_correct = 1 if correct_option and user_answer == correct_option['text'] else 0
                    elif q['type'] == "True/False":
                        is_correct = 1 if str(user_answer).lower() == str(correct_answer).lower() else 0
                    elif q['type'] == "Fill-in-the-Blank":
                        is_correct = 1 if str(user_answer).strip().lower() == str(correct_answer).strip().lower() else 0
                    
                    cursor.execute('''
                        INSERT INTO quiz_answers (
                            session_id,
                            question_id,
                            question_text,
                            question_type,
                            user_answer,
                            correct_answer,
                            is_correct
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id,
                        i,
                        q['text'],
                        q['type'],
                        user_answer,
                        correct_answer,
                        is_correct
                    ))
            conn.commit()

    def save_quiz_report(self, session_id, insights):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO quiz_reports (session_id, insights)
                VALUES (?, ?)
            ''', (session_id, insights))
            conn.commit()
