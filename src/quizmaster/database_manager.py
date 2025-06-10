import os
import json
import psycopg2
from psycopg2 import sql
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.conn_params = {
            'host': 'localhost' if os.getenv('RUNNING_IN_DOCKER') != 'true' else 'db',
            'port': os.getenv('DB_PORT'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }
        try:
            self._create_tables()
        except psycopg2.OperationalError as e:
            if "could not translate host name" in str(e):
                print("Warning: Could not connect to database. Is the PostgreSQL container running?")
                print("Run 'docker-compose up -d' to start the database container")
            raise

    def _get_connection(self):
        return psycopg2.connect(**self.conn_params)

    def _create_tables(self):
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS quiz_sessions (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
                        id SERIAL PRIMARY KEY,
                        session_id INTEGER REFERENCES quiz_sessions(id),
                        question_id INTEGER,
                        question_text TEXT,
                        question_type TEXT,
                        user_answer TEXT,
                        correct_answer TEXT,
                        is_correct INTEGER
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS quiz_reports (
                        id SERIAL PRIMARY KEY,
                        session_id INTEGER REFERENCES quiz_sessions(id),
                        insights TEXT,
                        generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # New table for saved quizzes
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS saved_quizzes (
                        id SERIAL PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT,
                        quiz_data JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        tags TEXT[]
                    )
                ''')

                # Ensure pgvector extension is available
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()

    def save_quiz_session(self, quiz_data, results):
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO quiz_sessions (
                        quiz_config, 
                        model_used,
                        total_questions,
                        answered_questions,
                        correct_answers,
                        completion_rate,
                        score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (
                    str(quiz_data.get('config', {})),
                    quiz_data.get('model_used', ''),
                    results.get('total_questions', 0),
                    results.get('answered_questions', 0),
                    results.get('correct_answers', 0),
                    results.get('completion_rate', 0),
                    results.get('score', 0)
                ))
                session_id = cursor.fetchone()[0]
                conn.commit()
                return session_id

    def save_quiz_answers(self, session_id, questions, user_answers):
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
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
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
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
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO quiz_reports (session_id, insights)
                    VALUES (%s, %s)
                ''', (session_id, insights))
                conn.commit()
    
    def save_quiz(self, title, quiz_data, description=None, tags=None):
        """
        Save a generated quiz to the database for later use.
        
        Args:
            title (str): Title of the quiz
            quiz_data (dict): The quiz data to save
            description (str, optional): Description of the quiz
            tags (list, optional): List of tags for categorizing the quiz
            
        Returns:
            int: ID of the saved quiz
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO saved_quizzes (title, description, quiz_data, tags)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                ''', (
                    title,
                    description,
                    json.dumps(quiz_data),
                    tags if tags else []
                ))
                quiz_id = cursor.fetchone()[0]
                conn.commit()
                return quiz_id
    
    def get_saved_quiz(self, quiz_id):
        """
        Retrieve a saved quiz by ID.
        
        Args:
            quiz_id (int): ID of the quiz to retrieve
            
        Returns:
            dict: The saved quiz data or None if not found
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT id, title, description, quiz_data, created_at, tags
                    FROM saved_quizzes
                    WHERE id = %s
                ''', (quiz_id,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        'id': result[0],
                        'title': result[1],
                        'description': result[2],
                        'quiz_data': json.loads(result[3]),
                        'created_at': result[4],
                        'tags': result[5]
                    }
                return None
    
    def list_saved_quizzes(self):
        """
        List all saved quizzes.
        
        Returns:
            list: List of saved quiz metadata (without full quiz_data)
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT id, title, description, created_at, tags
                    FROM saved_quizzes
                    ORDER BY created_at DESC
                ''')
                results = cursor.fetchall()
                
                quizzes = []
                for row in results:
                    quizzes.append({
                        'id': row[0],
                        'title': row[1],
                        'description': row[2],
                        'created_at': row[3],
                        'tags': row[4]
                    })
                return quizzes
    
    def delete_saved_quiz(self, quiz_id):
        """
        Delete a saved quiz.
        
        Args:
            quiz_id (int): ID of the quiz to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    DELETE FROM saved_quizzes
                    WHERE id = %s
                    RETURNING id
                ''', (quiz_id,))
                result = cursor.fetchone()
                conn.commit()
                return result is not None
