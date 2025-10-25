#!/usr/bin/env python3
"""
Face Database

SQLite-based database for storing and retrieving face encodings
with efficient similarity search capabilities.
"""

import sqlite3
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from src.recognition.recognition_result import PersonMatch

logger = logging.getLogger(__name__)


class FaceDatabase:
    """SQLite-based face encoding database with similarity search."""
    
    def __init__(self, db_path: str, config: Dict[str, Any]):
        """
        Initialize face database.
        
        Args:
            db_path: Path to SQLite database file
            config: Configuration dictionary with database settings
        """
        self.db_path = db_path
        self.config = config
        self.max_faces_per_person = config.get('max_faces_per_person', 10)
        self.backup_enabled = config.get('backup_enabled', True)
        
        # Ensure database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Database connection
        self.conn: Optional[sqlite3.Connection] = None
        self._initialized = False
        
        # Performance metrics
        self.query_count = 0
        self.insert_count = 0
        self.total_query_time = 0.0
        
        logger.info(f"FaceDatabase initialized with path: {db_path}")
    
    def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return
        
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            # Create tables
            self._create_tables()
            
            self._initialized = True
            logger.info("Face database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _create_tables(self) -> None:
        """Create database tables."""
        cursor = self.conn.cursor()
        
        # Persons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                person_id TEXT PRIMARY KEY,
                person_name TEXT,
                is_blacklisted INTEGER DEFAULT 0,
                created_at REAL,
                updated_at REAL,
                metadata TEXT
            )
        ''')
        
        # Face encodings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_encodings (
                encoding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                encoding_data TEXT NOT NULL,
                encoding_version TEXT DEFAULT 'v1',
                quality_score REAL DEFAULT 0.0,
                created_at REAL,
                match_count INTEGER DEFAULT 0,
                last_matched_at REAL,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_person_id 
            ON face_encodings(person_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_blacklisted 
            ON persons(is_blacklisted)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_match_count 
            ON face_encodings(match_count DESC)
        ''')
        
        self.conn.commit()
    
    def is_initialized(self) -> bool:
        """Check if database is initialized."""
        return self._initialized
    
    def add_person(self, person_id: str, person_name: Optional[str] = None, 
                   is_blacklisted: bool = False, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a person to the database.
        
        Args:
            person_id: Unique person identifier
            person_name: Optional person name
            is_blacklisted: Whether person is blacklisted
            metadata: Additional metadata
            
        Returns:
            True if added successfully, False otherwise
        """
        if not self._initialized:
            self.initialize()
        
        try:
            cursor = self.conn.cursor()
            current_time = time.time()
            
            cursor.execute('''
                INSERT OR REPLACE INTO persons 
                (person_id, person_name, is_blacklisted, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (person_id, person_name, 1 if is_blacklisted else 0, 
                  current_time, current_time, json.dumps(metadata or {})))
            
            self.conn.commit()
            self.insert_count += 1
            
            logger.debug(f"Added person: {person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add person {person_id}: {e}")
            return False
    
    def add_face_encoding(self, person_id: str, encoding: Any, 
                         quality_score: float = 0.0) -> Optional[int]:
        """
        Add face encoding for a person.
        
        Args:
            person_id: Person identifier
            encoding: Face encoding (numpy array)
            quality_score: Quality score of the face
            
        Returns:
            Encoding ID if added successfully, None otherwise
        """
        if not self._initialized:
            self.initialize()
        
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, cannot add face encoding")
            return None
        
        try:
            # Convert encoding to JSON string
            encoding_str = json.dumps(encoding.tolist() if hasattr(encoding, 'tolist') else encoding)
            
            cursor = self.conn.cursor()
            current_time = time.time()
            
            # Check if person exists
            cursor.execute('SELECT person_id FROM persons WHERE person_id = ?', (person_id,))
            if not cursor.fetchone():
                # Add person if doesn't exist
                self.add_person(person_id)
            
            # Check face count for person
            cursor.execute('''
                SELECT COUNT(*) FROM face_encodings WHERE person_id = ?
            ''', (person_id,))
            count = cursor.fetchone()[0]
            
            if count >= self.max_faces_per_person:
                # Remove oldest encoding
                cursor.execute('''
                    DELETE FROM face_encodings 
                    WHERE encoding_id = (
                        SELECT encoding_id FROM face_encodings 
                        WHERE person_id = ? 
                        ORDER BY created_at ASC 
                        LIMIT 1
                    )
                ''', (person_id,))
            
            # Insert new encoding
            cursor.execute('''
                INSERT INTO face_encodings 
                (person_id, encoding_data, quality_score, created_at)
                VALUES (?, ?, ?, ?)
            ''', (person_id, encoding_str, quality_score, current_time))
            
            encoding_id = cursor.lastrowid
            self.conn.commit()
            self.insert_count += 1
            
            logger.debug(f"Added face encoding {encoding_id} for person {person_id}")
            return encoding_id
            
        except Exception as e:
            logger.error(f"Failed to add face encoding for {person_id}: {e}")
            return None
    
    def find_known_matches(self, encoding: Any, tolerance: float = 0.6) -> List[PersonMatch]:
        """
        Find matching known persons for a face encoding.
        
        Args:
            encoding: Face encoding to match
            tolerance: Similarity threshold
            
        Returns:
            List of PersonMatch objects sorted by confidence
        """
        return self._find_matches(encoding, tolerance, is_blacklisted=False)
    
    def find_blacklist_matches(self, encoding: Any, tolerance: float = 0.5) -> List[PersonMatch]:
        """
        Find matching blacklisted persons for a face encoding.
        
        Args:
            encoding: Face encoding to match
            tolerance: Similarity threshold (stricter for blacklist)
            
        Returns:
            List of PersonMatch objects sorted by confidence
        """
        return self._find_matches(encoding, tolerance, is_blacklisted=True)
    
    def _find_matches(self, encoding: Any, tolerance: float, 
                     is_blacklisted: Optional[bool] = None) -> List[PersonMatch]:
        """
        Find matching persons for a face encoding.
        
        Args:
            encoding: Face encoding to match
            tolerance: Similarity threshold
            is_blacklisted: Filter by blacklist status (None = no filter)
            
        Returns:
            List of PersonMatch objects sorted by confidence
        """
        if not self._initialized:
            self.initialize()
        
        if not NUMPY_AVAILABLE:
            return []
        
        start_time = time.time()
        matches = []
        
        try:
            cursor = self.conn.cursor()
            
            # Build query
            query = '''
                SELECT 
                    fe.encoding_id,
                    fe.person_id,
                    fe.encoding_data,
                    fe.match_count,
                    p.person_name,
                    p.is_blacklisted
                FROM face_encodings fe
                JOIN persons p ON fe.person_id = p.person_id
            '''
            
            params = []
            if is_blacklisted is not None:
                query += ' WHERE p.is_blacklisted = ?'
                params.append(1 if is_blacklisted else 0)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Compare encodings
            unknown_encoding = np.array(encoding) if not isinstance(encoding, np.ndarray) else encoding
            
            for row in rows:
                try:
                    # Parse stored encoding
                    stored_encoding = np.array(json.loads(row['encoding_data']))
                    
                    # Compute distance
                    distance = np.linalg.norm(unknown_encoding - stored_encoding)
                    
                    if distance <= tolerance:
                        # Compute confidence
                        confidence = max(0.0, 1.0 - (distance / (2 * tolerance)))
                        
                        matches.append(PersonMatch(
                            person_id=row['person_id'],
                            person_name=row['person_name'],
                            confidence=confidence,
                            similarity_score=1.0 - distance,
                            match_count=row['match_count'],
                            is_blacklisted=bool(row['is_blacklisted']),
                            metadata={'encoding_id': row['encoding_id']}
                        ))
                        
                        # Update match count
                        cursor.execute('''
                            UPDATE face_encodings 
                            SET match_count = match_count + 1, last_matched_at = ?
                            WHERE encoding_id = ?
                        ''', (time.time(), row['encoding_id']))
                    
                except Exception as e:
                    logger.warning(f"Failed to compare encoding: {e}")
                    continue
            
            self.conn.commit()
            
            # Sort by confidence (descending)
            matches.sort(key=lambda m: m.confidence, reverse=True)
            
            # Update metrics
            query_time = time.time() - start_time
            self.query_count += 1
            self.total_query_time += query_time
            
            logger.debug(f"Found {len(matches)} matches in {query_time*1000:.2f}ms")
            
            return matches
            
        except Exception as e:
            logger.error(f"Face matching query failed: {e}")
            return []
    
    def get_person_encodings(self, person_id: str) -> List[Any]:
        """
        Get all face encodings for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            List of face encodings
        """
        if not self._initialized:
            self.initialize()
        
        if not NUMPY_AVAILABLE:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT encoding_data FROM face_encodings 
                WHERE person_id = ?
                ORDER BY quality_score DESC
            ''', (person_id,))
            
            encodings = []
            for row in cursor.fetchall():
                encoding = np.array(json.loads(row['encoding_data']))
                encodings.append(encoding)
            
            return encodings
            
        except Exception as e:
            logger.error(f"Failed to get encodings for {person_id}: {e}")
            return []
    
    def get_frequent_faces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get frequently matched faces for cache warming.
        
        Args:
            limit: Maximum number of faces to return
            
        Returns:
            List of face data dictionaries
        """
        if not self._initialized:
            self.initialize()
        
        if not NUMPY_AVAILABLE:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT 
                    fe.person_id,
                    fe.encoding_data,
                    p.is_blacklisted,
                    fe.match_count
                FROM face_encodings fe
                JOIN persons p ON fe.person_id = p.person_id
                ORDER BY fe.match_count DESC
                LIMIT ?
            ''', (limit,))
            
            faces = []
            for row in cursor.fetchall():
                encoding = np.array(json.loads(row['encoding_data']))
                faces.append({
                    'person_id': row['person_id'],
                    'encoding': encoding,
                    'is_blacklisted': bool(row['is_blacklisted']),
                    'match_count': row['match_count']
                })
            
            return faces
            
        except Exception as e:
            logger.error(f"Failed to get frequent faces: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self._initialized:
            return {}
        
        try:
            cursor = self.conn.cursor()
            
            # Count persons
            cursor.execute('SELECT COUNT(*) FROM persons')
            person_count = cursor.fetchone()[0]
            
            # Count blacklisted
            cursor.execute('SELECT COUNT(*) FROM persons WHERE is_blacklisted = 1')
            blacklisted_count = cursor.fetchone()[0]
            
            # Count encodings
            cursor.execute('SELECT COUNT(*) FROM face_encodings')
            encoding_count = cursor.fetchone()[0]
            
            avg_query_time = self.total_query_time / max(1, self.query_count)
            
            return {
                'person_count': person_count,
                'blacklisted_count': blacklisted_count,
                'encoding_count': encoding_count,
                'query_count': self.query_count,
                'insert_count': self.insert_count,
                'avg_query_time': avg_query_time,
                'database_path': self.db_path
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self._initialized = False
            logger.info("Face database connection closed")
