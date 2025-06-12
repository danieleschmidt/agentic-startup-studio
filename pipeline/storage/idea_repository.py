"""
PostgreSQL repository for idea persistence with pgvector similarity search.

This module provides database operations for startup ideas including
CRUD operations, similarity search, and audit trail management.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4

import asyncpg
import numpy as np
from asyncpg import Connection, Pool

from pipeline.models.idea import (
    Idea, IdeaDraft, IdeaStatus, PipelineStage, IdeaCategory,
    QueryParams, AuditEntry, DuplicateCheckResult
)
from pipeline.config.settings import get_db_config, DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Database connection related errors."""
    pass


class QueryError(DatabaseError):
    """Database query related errors."""
    pass


class EmbeddingService:
    """Service for generating and managing text embeddings."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate vector embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding as numpy array
            
        Note:
            In production, this would call an external embedding service
            (OpenAI, Hugging Face, etc.). For now, using mock implementation.
        """
        # Mock embedding generation - replace with actual service call
        text_hash = hash(text)
        
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        # Generate deterministic mock embedding for testing
        np.random.seed(abs(text_hash) % (2**32))
        embedding = np.random.normal(0, 1, self.config.vector_dimensions)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        self._embedding_cache[text_hash] = embedding
        
        logger.debug(f"Generated embedding for text (length: {len(text)})")
        return embedding
    
    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool: Optional[Pool] = None
        self.embedding_service = EmbeddingService(config)
    
    async def initialize(self) -> None:
        """Initialize database connection pool and schema."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.timeout
            )
            
            await self._ensure_schema_exists()
            logger.info("Database connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise ConnectionError(f"Database initialization failed: {e}")
    
    async def close(self) -> None:
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        if not self.pool:
            raise ConnectionError("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            try:
                yield conn
            except Exception as e:
                logger.error(f"Database operation failed: {e}")
                raise
    
    async def _ensure_schema_exists(self) -> None:
        """Ensure required database schema exists."""
        schema_sql = """
        -- Enable pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;
        
        -- Ideas table
        CREATE TABLE IF NOT EXISTS ideas (
            idea_id UUID PRIMARY KEY,
            title VARCHAR(200) NOT NULL,
            description TEXT NOT NULL,
            category VARCHAR(50) DEFAULT 'uncategorized',
            status VARCHAR(20) DEFAULT 'DRAFT',
            current_stage VARCHAR(20) DEFAULT 'IDEATE',
            stage_progress FLOAT DEFAULT 0.0,
            problem_statement TEXT,
            solution_description TEXT,
            target_market VARCHAR(500),
            evidence_links TEXT[], -- Array of URLs
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_by VARCHAR(100),
            deck_path VARCHAR(500),
            research_data JSONB DEFAULT '{}',
            investor_scores JSONB DEFAULT '{}'
        );
        
        -- Idea embeddings table for vector similarity search
        CREATE TABLE IF NOT EXISTS idea_embeddings (
            idea_id UUID PRIMARY KEY REFERENCES ideas(idea_id) ON DELETE CASCADE,
            description_embedding vector(%s),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Audit trail table
        CREATE TABLE IF NOT EXISTS idea_audit (
            entry_id UUID PRIMARY KEY,
            idea_id UUID REFERENCES ideas(idea_id) ON DELETE CASCADE,
            action VARCHAR(50) NOT NULL,
            changes JSONB DEFAULT '{}',
            user_id VARCHAR(100),
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            correlation_id VARCHAR(100)
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status);
        CREATE INDEX IF NOT EXISTS idx_ideas_stage ON ideas(current_stage);
        CREATE INDEX IF NOT EXISTS idx_ideas_category ON ideas(category);
        CREATE INDEX IF NOT EXISTS idx_ideas_created_at ON ideas(created_at);
        CREATE INDEX IF NOT EXISTS idx_audit_idea_id ON idea_audit(idea_id);
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON idea_audit(timestamp);
        
        -- Vector similarity index
        CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
        ON idea_embeddings USING ivfflat (description_embedding vector_cosine_ops);
        """
        
        async with self.get_connection() as conn:
            await conn.execute(schema_sql, self.config.vector_dimensions)
            logger.info("Database schema verification completed")


class IdeaRepository:
    """Repository for idea CRUD operations and similarity search."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.embedding_service = db_manager.embedding_service

    def _get_enum_value(self, enum_or_str):
        """Helper method to get enum value, handling both enum objects and strings."""
        return enum_or_str.value if hasattr(enum_or_str, 'value') else enum_or_str
    
    async def save_idea(self, idea: Idea, correlation_id: Optional[str] = None) -> UUID:
        """
        Save new idea to database with embedding generation.
        
        Args:
            idea: Idea entity to save
            correlation_id: Optional correlation ID for audit trail
            
        Returns:
            Saved idea UUID
            
        Raises:
            DatabaseError: If save operation fails
        """
        try:
            async with self.db.get_connection() as conn:
                async with conn.transaction():
                    # Insert idea
                    await conn.execute("""
                        INSERT INTO ideas (
                            idea_id, title, description, category, status, current_stage,
                            stage_progress, problem_statement, solution_description,
                            target_market, evidence_links, created_at, updated_at,
                            created_by, deck_path, research_data, investor_scores
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                    """, 
                        idea.idea_id, idea.title, idea.description, self._get_enum_value(idea.category),
                        self._get_enum_value(idea.status), self._get_enum_value(idea.current_stage), idea.stage_progress,
                        idea.problem_statement, idea.solution_description, 
                        idea.target_market, idea.evidence_links, idea.created_at,
                        idea.updated_at, idea.created_by, idea.deck_path,
                        idea.research_data, idea.investor_scores
                    )
                    
                    # Generate and save embedding
                    if self.db.config.enable_vector_search:
                        embedding = await self.embedding_service.generate_embedding(idea.description)
                        await conn.execute("""
                            INSERT INTO idea_embeddings (idea_id, description_embedding)
                            VALUES ($1, $2)
                        """, idea.idea_id, embedding.tolist())
                    
                    # Create audit entry
                    audit_entry = AuditEntry(
                        idea_id=idea.idea_id,
                        action="idea_created",
                        changes={"title": idea.title, "status": idea.status.value},
                        user_id=idea.created_by,
                        correlation_id=correlation_id
                    )
                    await self._save_audit_entry(conn, audit_entry)
                    
                    logger.info(
                        f"Idea saved successfully",
                        extra={
                            "idea_id": str(idea.idea_id),
                            "title": idea.title,
                            "correlation_id": correlation_id
                        }
                    )
                    
                    return idea.idea_id
                    
        except Exception as e:
            logger.error(f"Failed to save idea: {e}")
            raise DatabaseError(f"Failed to save idea: {e}")
    
    async def find_by_id(self, idea_id: UUID) -> Optional[Idea]:
        """
        Find idea by ID.
        
        Args:
            idea_id: Idea UUID
            
        Returns:
            Idea entity or None if not found
        """
        try:
            async with self.db.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM ideas WHERE idea_id = $1
                """, idea_id)
                
                if row:
                    return self._row_to_idea(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to find idea by ID {idea_id}: {e}")
            raise QueryError(f"Failed to find idea: {e}")
    
    async def find_with_filters(self, params: QueryParams) -> List[Idea]:
        """
        Find ideas with filtering, sorting, and pagination.
        
        Args:
            params: Query parameters
            
        Returns:
            List of matching ideas
        """
        try:
            where_conditions = []
            query_args = []
            arg_counter = 1
            
            # Build WHERE clause
            if params.status_filter:
                placeholders = ",".join([f"${i}" for i in range(arg_counter, arg_counter + len(params.status_filter))])
                where_conditions.append(f"status IN ({placeholders})")
                query_args.extend([self._get_enum_value(status) for status in params.status_filter])
                arg_counter += len(params.status_filter)

            if params.stage_filter:
                placeholders = ",".join([f"${i}" for i in range(arg_counter, arg_counter + len(params.stage_filter))])
                where_conditions.append(f"current_stage IN ({placeholders})")
                query_args.extend([self._get_enum_value(stage) for stage in params.stage_filter])
                arg_counter += len(params.stage_filter)

            if params.category_filter:
                placeholders = ",".join([f"${i}" for i in range(arg_counter, arg_counter + len(params.category_filter))])
                where_conditions.append(f"category IN ({placeholders})")
                query_args.extend([self._get_enum_value(cat) for cat in params.category_filter])
                arg_counter += len(params.category_filter)
            
            if params.created_after:
                where_conditions.append(f"created_at >= ${arg_counter}")
                query_args.append(params.created_after)
                arg_counter += 1
            
            if params.created_before:
                where_conditions.append(f"created_at <= ${arg_counter}")
                query_args.append(params.created_before)
                arg_counter += 1
            
            if params.search_text:
                where_conditions.append(f"(title ILIKE ${arg_counter} OR description ILIKE ${arg_counter})")
                query_args.append(f"%{params.search_text}%")
                arg_counter += 1
            
            # Build complete query
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            order_direction = "DESC" if params.sort_desc else "ASC"
            
            query = f"""
                SELECT * FROM ideas 
                {where_clause}
                ORDER BY {params.sort_by} {order_direction}
                LIMIT ${arg_counter} OFFSET ${arg_counter + 1}
            """
            
            query_args.extend([params.limit, params.offset])
            
            async with self.db.get_connection() as conn:
                rows = await conn.fetch(query, *query_args)
                return [self._row_to_idea(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to query ideas with filters: {e}")
            raise QueryError(f"Failed to query ideas: {e}")
    
    async def find_similar_by_embedding(
        self, 
        description: str, 
        threshold: float = 0.8,
        exclude_statuses: Optional[List[IdeaStatus]] = None,
        limit: int = 10
    ) -> List[Tuple[UUID, float]]:
        """
        Find similar ideas using vector similarity search.
        
        Args:
            description: Description text to find similar ideas for
            threshold: Minimum similarity threshold (0.0 to 1.0)
            exclude_statuses: Statuses to exclude from search
            limit: Maximum number of results
            
        Returns:
            List of (idea_id, similarity_score) tuples
        """
        if not self.db.config.enable_vector_search:
            return []
        
        try:
            # Generate embedding for input description
            query_embedding = await self.embedding_service.generate_embedding(description)
            
            # Build exclusion clause
            exclude_clause = ""
            exclude_args = []
            if exclude_statuses:
                exclude_clause = "AND i.status NOT IN (" + ",".join([f"${i+2}" for i in range(len(exclude_statuses))]) + ")"
                exclude_args = [status.value for status in exclude_statuses]
            
            query = f"""
                SELECT i.idea_id, 
                       1 - (e.description_embedding <=> $1) as similarity_score
                FROM idea_embeddings e
                JOIN ideas i ON e.idea_id = i.idea_id
                WHERE 1 - (e.description_embedding <=> $1) >= $2
                {exclude_clause}
                ORDER BY similarity_score DESC
                LIMIT ${len(exclude_args) + 3}
            """
            
            query_args = [query_embedding.tolist(), threshold] + exclude_args + [limit]
            
            async with self.db.get_connection() as conn:
                rows = await conn.fetch(query, *query_args)
                
                results = [(row['idea_id'], float(row['similarity_score'])) for row in rows]
                
                logger.debug(
                    f"Found {len(results)} similar ideas above threshold {threshold}",
                    extra={"threshold": threshold, "result_count": len(results)}
                )
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to find similar ideas: {e}")
            raise QueryError(f"Vector similarity search failed: {e}")
    
    async def find_by_title_exact(self, title: str) -> List[UUID]:
        """
        Find ideas with exact title match.
        
        Args:
            title: Exact title to search for
            
        Returns:
            List of matching idea UUIDs
        """
        try:
            async with self.db.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT idea_id FROM ideas WHERE title = $1
                """, title)
                
                return [row['idea_id'] for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to find ideas by exact title: {e}")
            raise QueryError(f"Title search failed: {e}")
    
    async def update_idea(
        self, 
        idea: Idea, 
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """
        Update existing idea.
        
        Args:
            idea: Updated idea entity
            user_id: User making the update
            correlation_id: Optional correlation ID
            
        Raises:
            DatabaseError: If update fails
        """
        try:
            async with self.db.get_connection() as conn:
                async with conn.transaction():
                    # Update idea
                    result = await conn.execute("""
                        UPDATE ideas SET
                            title = $2, description = $3, category = $4, status = $5,
                            current_stage = $6, stage_progress = $7, problem_statement = $8,
                            solution_description = $9, target_market = $10, evidence_links = $11,
                            updated_at = $12, deck_path = $13, research_data = $14,
                            investor_scores = $15
                        WHERE idea_id = $1
                    """,
                        idea.idea_id, idea.title, idea.description, idea.category.value,
                        idea.status.value, idea.current_stage.value, idea.stage_progress,
                        idea.problem_statement, idea.solution_description,
                        idea.target_market, idea.evidence_links, idea.updated_at,
                        idea.deck_path, idea.research_data, idea.investor_scores
                    )
                    
                    if result == "UPDATE 0":
                        raise DatabaseError(f"Idea {idea.idea_id} not found for update")
                    
                    # Update embedding if description changed
                    if self.db.config.enable_vector_search:
                        embedding = await self.embedding_service.generate_embedding(idea.description)
                        await conn.execute("""
                            INSERT INTO idea_embeddings (idea_id, description_embedding)
                            VALUES ($1, $2)
                            ON CONFLICT (idea_id) DO UPDATE SET
                                description_embedding = EXCLUDED.description_embedding,
                                created_at = NOW()
                        """, idea.idea_id, embedding.tolist())
                    
                    # Create audit entry
                    audit_entry = AuditEntry(
                        idea_id=idea.idea_id,
                        action="idea_updated",
                        changes={"status": idea.status.value, "stage": idea.current_stage.value},
                        user_id=user_id,
                        correlation_id=correlation_id
                    )
                    await self._save_audit_entry(conn, audit_entry)
                    
                    logger.info(
                        f"Idea updated successfully",
                        extra={
                            "idea_id": str(idea.idea_id),
                            "status": idea.status.value,
                            "correlation_id": correlation_id
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Failed to update idea {idea.idea_id}: {e}")
            raise DatabaseError(f"Failed to update idea: {e}")
    
    async def delete_idea(self, idea_id: UUID, user_id: Optional[str] = None) -> bool:
        """
        Delete idea by ID.
        
        Args:
            idea_id: Idea UUID to delete
            user_id: User performing deletion
            
        Returns:
            True if deleted, False if not found
        """
        try:
            async with self.db.get_connection() as conn:
                async with conn.transaction():
                    # Create audit entry before deletion
                    audit_entry = AuditEntry(
                        idea_id=idea_id,
                        action="idea_deleted",
                        user_id=user_id
                    )
                    await self._save_audit_entry(conn, audit_entry)
                    
                    # Delete idea (cascades to embeddings and audit entries)
                    result = await conn.execute("""
                        DELETE FROM ideas WHERE idea_id = $1
                    """, idea_id)
                    
                    deleted = result == "DELETE 1"
                    
                    if deleted:
                        logger.info(f"Idea {idea_id} deleted successfully")
                    
                    return deleted
                    
        except Exception as e:
            logger.error(f"Failed to delete idea {idea_id}: {e}")
            raise DatabaseError(f"Failed to delete idea: {e}")
    
    async def get_audit_trail(self, idea_id: UUID, limit: int = 50) -> List[AuditEntry]:
        """
        Get audit trail for an idea.
        
        Args:
            idea_id: Idea UUID
            limit: Maximum number of audit entries
            
        Returns:
            List of audit entries ordered by timestamp DESC
        """
        try:
            async with self.db.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM idea_audit 
                    WHERE idea_id = $1 
                    ORDER BY timestamp DESC 
                    LIMIT $2
                """, idea_id, limit)
                
                return [self._row_to_audit_entry(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get audit trail for {idea_id}: {e}")
            raise QueryError(f"Failed to get audit trail: {e}")
    
    async def _save_audit_entry(self, conn: Connection, entry: AuditEntry) -> None:
        """Save audit entry to database."""
        await conn.execute("""
            INSERT INTO idea_audit (entry_id, idea_id, action, changes, user_id, timestamp, correlation_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, 
            entry.entry_id, entry.idea_id, entry.action, entry.changes,
            entry.user_id, entry.timestamp, entry.correlation_id
        )
    
    def _row_to_idea(self, row) -> Idea:
        """Convert database row to Idea entity."""
        return Idea(
            idea_id=row['idea_id'],
            title=row['title'],
            description=row['description'],
            category=IdeaCategory(row['category']),
            status=IdeaStatus(row['status']),
            current_stage=PipelineStage(row['current_stage']),
            stage_progress=row['stage_progress'],
            problem_statement=row['problem_statement'],
            solution_description=row['solution_description'],
            target_market=row['target_market'],
            evidence_links=row['evidence_links'] or [],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            created_by=row['created_by'],
            deck_path=row['deck_path'],
            research_data=row['research_data'] or {},
            investor_scores=row['investor_scores'] or {}
        )
    
    def _row_to_audit_entry(self, row) -> AuditEntry:
        """Convert database row to AuditEntry entity."""
        return AuditEntry(
            entry_id=row['entry_id'],
            idea_id=row['idea_id'],
            action=row['action'],
            changes=row['changes'] or {},
            user_id=row['user_id'],
            timestamp=row['timestamp'],
            correlation_id=row['correlation_id']
        )


# Factory function for repository creation
async def create_idea_repository(config: Optional[DatabaseConfig] = None) -> IdeaRepository:
    """
    Create and initialize idea repository.
    
    Args:
        config: Optional database configuration
        
    Returns:
        Initialized IdeaRepository instance
    """
    db_config = config or get_db_config()
    db_manager = DatabaseManager(db_config)
    await db_manager.initialize()
    
    return IdeaRepository(db_manager)