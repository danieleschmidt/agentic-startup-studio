"""
Advanced Base Repository with Enhanced Data Operations.

This module provides a comprehensive base repository pattern with advanced
database operations, connection pooling, caching, and monitoring.
"""

import json
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

import asyncpg
from asyncpg import Connection, Pool, Record

from pipeline.config.settings import get_settings
from pipeline.infrastructure.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Generic type for model entities


class RepositoryError(Exception):
    """Base exception for repository operations."""
    pass


class ConnectionError(RepositoryError):
    """Database connection related errors."""
    pass


class QueryError(RepositoryError):
    """Database query related errors."""
    pass


class ValidationError(RepositoryError):
    """Data validation errors."""
    pass


class TransactionContext:
    """Context manager for database transactions with automatic rollback."""

    def __init__(self, connection: Connection):
        self.connection = connection
        self.transaction = None
        self._savepoints = []

    async def __aenter__(self):
        self.transaction = self.connection.transaction()
        await self.transaction.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.transaction.rollback()
            logger.warning(f"Transaction rolled back due to {exc_type.__name__}: {exc_val}")
        else:
            await self.transaction.commit()
        return False

    async def create_savepoint(self, name: str = None) -> str:
        """Create a savepoint within the transaction."""
        if not name:
            name = f"sp_{len(self._savepoints) + 1}"

        await self.connection.execute(f"SAVEPOINT {name}")
        self._savepoints.append(name)
        return name

    async def rollback_to_savepoint(self, name: str):
        """Rollback to a specific savepoint."""
        await self.connection.execute(f"ROLLBACK TO SAVEPOINT {name}")
        # Remove savepoints created after this one
        try:
            index = self._savepoints.index(name)
            self._savepoints = self._savepoints[:index + 1]
        except ValueError:
            pass


class QueryBuilder:
    """Advanced SQL query builder with security and performance optimization."""

    def __init__(self, table_name: str):
        self.table_name = table_name
        self._select_fields = ["*"]
        self._where_conditions = []
        self._joins = []
        self._order_by = []
        self._group_by = []
        self._having = []
        self._limit = None
        self._offset = None
        self._parameters = []

    def select(self, *fields: str) -> "QueryBuilder":
        """Specify fields to select."""
        self._select_fields = list(fields)
        return self

    def where(self, condition: str, *params) -> "QueryBuilder":
        """Add WHERE condition with parameters."""
        self._where_conditions.append(condition)
        self._parameters.extend(params)
        return self

    def where_in(self, field: str, values: list[Any]) -> "QueryBuilder":
        """Add WHERE IN condition."""
        if not values:
            return self

        placeholders = ",".join([f"${len(self._parameters) + i + 1}" for i in range(len(values))])
        self._where_conditions.append(f"{field} IN ({placeholders})")
        self._parameters.extend(values)
        return self

    def where_between(self, field: str, start: Any, end: Any) -> "QueryBuilder":
        """Add WHERE BETWEEN condition."""
        param_start = len(self._parameters) + 1
        param_end = len(self._parameters) + 2
        self._where_conditions.append(f"{field} BETWEEN ${param_start} AND ${param_end}")
        self._parameters.extend([start, end])
        return self

    def join(self, table: str, condition: str) -> "QueryBuilder":
        """Add INNER JOIN."""
        self._joins.append(f"INNER JOIN {table} ON {condition}")
        return self

    def left_join(self, table: str, condition: str) -> "QueryBuilder":
        """Add LEFT JOIN."""
        self._joins.append(f"LEFT JOIN {table} ON {condition}")
        return self

    def order_by(self, field: str, direction: str = "ASC") -> "QueryBuilder":
        """Add ORDER BY clause."""
        self._order_by.append(f"{field} {direction.upper()}")
        return self

    def group_by(self, *fields: str) -> "QueryBuilder":
        """Add GROUP BY clause."""
        self._group_by.extend(fields)
        return self

    def having(self, condition: str, *params) -> "QueryBuilder":
        """Add HAVING clause."""
        self._having.append(condition)
        self._parameters.extend(params)
        return self

    def limit(self, count: int) -> "QueryBuilder":
        """Add LIMIT clause."""
        self._limit = count
        return self

    def offset(self, count: int) -> "QueryBuilder":
        """Add OFFSET clause."""
        self._offset = count
        return self

    def build_select(self) -> tuple[str, list]:
        """Build SELECT query with parameters."""
        query_parts = [
            f"SELECT {', '.join(self._select_fields)}",
            f"FROM {self.table_name}"
        ]

        if self._joins:
            query_parts.extend(self._joins)

        if self._where_conditions:
            query_parts.append(f"WHERE {' AND '.join(self._where_conditions)}")

        if self._group_by:
            query_parts.append(f"GROUP BY {', '.join(self._group_by)}")

        if self._having:
            query_parts.append(f"HAVING {' AND '.join(self._having)}")

        if self._order_by:
            query_parts.append(f"ORDER BY {', '.join(self._order_by)}")

        if self._limit:
            query_parts.append(f"LIMIT {self._limit}")

        if self._offset:
            query_parts.append(f"OFFSET {self._offset}")

        return " ".join(query_parts), self._parameters

    def build_count(self) -> tuple[str, list]:
        """Build COUNT query."""
        query_parts = [
            "SELECT COUNT(*)",
            f"FROM {self.table_name}"
        ]

        if self._joins:
            query_parts.extend(self._joins)

        if self._where_conditions:
            query_parts.append(f"WHERE {' AND '.join(self._where_conditions)}")

        if self._group_by:
            query_parts.append(f"GROUP BY {', '.join(self._group_by)}")

        if self._having:
            query_parts.append(f"HAVING {' AND '.join(self._having)}")

        return " ".join(query_parts), self._parameters


class BaseRepository(Generic[T], ABC):
    """
    Advanced base repository with comprehensive data operations.
    
    Features:
    - Connection pooling and management
    - Circuit breaker pattern for resilience
    - Query optimization and caching
    - Audit trail integration
    - Performance monitoring
    - Transaction management
    """

    def __init__(self, table_name: str, pool: Pool):
        self.table_name = table_name
        self.pool = pool
        self.settings = get_settings()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout_seconds=30,
            recovery_timeout=60
        )
        self._query_cache = {}
        self._cache_ttl = 300  # 5 minutes

    @abstractmethod
    def _map_record_to_entity(self, record: Record) -> T:
        """Map database record to domain entity."""
        pass

    @abstractmethod
    def _map_entity_to_dict(self, entity: T) -> dict[str, Any]:
        """Map domain entity to dictionary for database operations."""
        pass

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool with circuit breaker."""
        async with self.circuit_breaker:
            connection = await self.pool.acquire()
            try:
                yield connection
            finally:
                await self.pool.release(connection)

    @asynccontextmanager
    async def transaction(self):
        """Create database transaction context."""
        async with self.get_connection() as conn:
            async with TransactionContext(conn) as tx:
                yield tx

    async def execute_query(
        self,
        query: str,
        *params,
        fetch_one: bool = False,
        fetch_all: bool = False,
        fetch_val: bool = False
    ) -> list[Record] | Record | Any | None:
        """
        Execute query with error handling and monitoring.
        
        Args:
            query: SQL query to execute
            *params: Query parameters
            fetch_one: Return single record
            fetch_all: Return all records
            fetch_val: Return single value
            
        Returns:
            Query results based on fetch mode
        """
        start_time = datetime.now(UTC)

        try:
            async with self.get_connection() as conn:
                if fetch_one:
                    result = await conn.fetchrow(query, *params)
                elif fetch_all:
                    result = await conn.fetch(query, *params)
                elif fetch_val:
                    result = await conn.fetchval(query, *params)
                else:
                    result = await conn.execute(query, *params)

                # Log slow queries
                execution_time = (datetime.now(UTC) - start_time).total_seconds()
                if execution_time > 1.0:  # Log queries taking more than 1 second
                    logger.warning(
                        f"Slow query detected: {execution_time:.2f}s",
                        extra={
                            "query": query[:200] + "..." if len(query) > 200 else query,
                            "execution_time": execution_time,
                            "table": self.table_name
                        }
                    )

                return result

        except asyncpg.PostgresError as e:
            logger.error(
                f"Database error in {self.table_name}: {e}",
                extra={
                    "error_code": e.sqlstate,
                    "query": query[:200] + "..." if len(query) > 200 else query,
                    "table": self.table_name
                }
            )
            raise QueryError(f"Database query failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in {self.table_name}: {e}")
            raise RepositoryError(f"Repository operation failed: {e}") from e

    def query(self) -> QueryBuilder:
        """Create a new query builder for this table."""
        return QueryBuilder(self.table_name)

    async def find_by_id(self, entity_id: UUID) -> T | None:
        """Find entity by ID with caching."""
        cache_key = f"{self.table_name}:id:{entity_id}"

        # Check cache first
        if cache_key in self._query_cache:
            cached_result, cached_time = self._query_cache[cache_key]
            if (datetime.now(UTC) - cached_time).total_seconds() < self._cache_ttl:
                return cached_result

        # Query database
        query = f"SELECT * FROM {self.table_name} WHERE {self._get_id_column()} = $1"
        record = await self.execute_query(query, entity_id, fetch_one=True)

        if record:
            entity = self._map_record_to_entity(record)
            # Cache result
            self._query_cache[cache_key] = (entity, datetime.now(UTC))
            return entity

        return None

    async def find_all(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = None,
        order_direction: str = "ASC"
    ) -> list[T]:
        """Find all entities with pagination and ordering."""
        builder = self.query().limit(limit).offset(offset)

        if order_by:
            builder = builder.order_by(order_by, order_direction)

        query, params = builder.build_select()
        records = await self.execute_query(query, *params, fetch_all=True)

        return [self._map_record_to_entity(record) for record in records]

    async def count(self, conditions: dict[str, Any] = None) -> int:
        """Count entities with optional conditions."""
        builder = self.query()

        if conditions:
            for field, value in conditions.items():
                builder = builder.where(f"{field} = ${len(builder._parameters) + 1}", value)

        query, params = builder.build_count()
        return await self.execute_query(query, *params, fetch_val=True)

    async def create(self, entity: T, user_id: str = None) -> T:
        """Create new entity with audit trail."""
        entity_dict = self._map_entity_to_dict(entity)

        # Add metadata
        entity_dict['created_at'] = datetime.now(UTC)
        entity_dict['updated_at'] = datetime.now(UTC)
        if user_id:
            entity_dict['created_by'] = user_id

        # Build insert query
        fields = list(entity_dict.keys())
        placeholders = [f"${i+1}" for i in range(len(fields))]
        values = list(entity_dict.values())

        query = f"""
            INSERT INTO {self.table_name} ({', '.join(fields)})
            VALUES ({', '.join(placeholders)})
            RETURNING *
        """

        async with self.transaction() as tx:
            # Insert entity
            record = await self.execute_query(query, *values, fetch_one=True)
            created_entity = self._map_record_to_entity(record)

            # Create audit trail
            await self._create_audit_entry(
                tx.connection,
                action="CREATE",
                entity_id=getattr(created_entity, self._get_id_field()),
                new_values=entity_dict,
                user_id=user_id
            )

            return created_entity

    async def update(self, entity: T, user_id: str = None) -> T:
        """Update entity with audit trail."""
        entity_id = getattr(entity, self._get_id_field())

        # Get current entity for audit trail
        current_entity = await self.find_by_id(entity_id)
        if not current_entity:
            raise ValidationError(f"Entity {entity_id} not found")

        entity_dict = self._map_entity_to_dict(entity)
        old_dict = self._map_entity_to_dict(current_entity)

        # Add metadata
        entity_dict['updated_at'] = datetime.now(UTC)
        if user_id:
            entity_dict['last_modified_by'] = user_id

        # Remove ID from update fields
        id_field = self._get_id_field()
        entity_dict.pop(id_field, None)

        # Build update query
        fields = list(entity_dict.keys())
        set_clauses = [f"{field} = ${i+1}" for i, field in enumerate(fields)]
        values = list(entity_dict.values())
        values.append(entity_id)  # For WHERE clause

        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(set_clauses)}
            WHERE {self._get_id_column()} = ${len(values)}
            RETURNING *
        """

        async with self.transaction() as tx:
            # Update entity
            record = await self.execute_query(query, *values, fetch_one=True)
            if not record:
                raise ValidationError(f"Entity {entity_id} not found for update")

            updated_entity = self._map_record_to_entity(record)

            # Create audit trail with changes
            changes = self._calculate_changes(old_dict, entity_dict)
            await self._create_audit_entry(
                tx.connection,
                action="UPDATE",
                entity_id=entity_id,
                old_values=old_dict,
                new_values=entity_dict,
                changes=changes,
                user_id=user_id
            )

            # Invalidate cache
            cache_key = f"{self.table_name}:id:{entity_id}"
            self._query_cache.pop(cache_key, None)

            return updated_entity

    async def delete(self, entity_id: UUID, user_id: str = None) -> bool:
        """Soft delete entity with audit trail."""
        # Get current entity for audit trail
        current_entity = await self.find_by_id(entity_id)
        if not current_entity:
            return False

        old_dict = self._map_entity_to_dict(current_entity)

        # Check if table supports soft delete
        if await self._has_column('deleted_at'):
            # Soft delete
            query = f"""
                UPDATE {self.table_name}
                SET deleted_at = $1, updated_at = $1
                WHERE {self._get_id_column()} = $2
            """

            async with self.transaction() as tx:
                result = await self.execute_query(
                    query,
                    datetime.now(UTC),
                    entity_id
                )

                # Create audit trail
                await self._create_audit_entry(
                    tx.connection,
                    action="SOFT_DELETE",
                    entity_id=entity_id,
                    old_values=old_dict,
                    user_id=user_id
                )
        else:
            # Hard delete
            query = f"DELETE FROM {self.table_name} WHERE {self._get_id_column()} = $1"

            async with self.transaction() as tx:
                result = await self.execute_query(query, entity_id)

                # Create audit trail
                await self._create_audit_entry(
                    tx.connection,
                    action="DELETE",
                    entity_id=entity_id,
                    old_values=old_dict,
                    user_id=user_id
                )

        # Invalidate cache
        cache_key = f"{self.table_name}:id:{entity_id}"
        self._query_cache.pop(cache_key, None)

        return True

    async def exists(self, entity_id: UUID) -> bool:
        """Check if entity exists."""
        query = f"SELECT 1 FROM {self.table_name} WHERE {self._get_id_column()} = $1 LIMIT 1"
        result = await self.execute_query(query, entity_id, fetch_val=True)
        return result is not None

    async def find_by_field(self, field: str, value: Any) -> list[T]:
        """Find entities by field value."""
        query = f"SELECT * FROM {self.table_name} WHERE {field} = $1"
        records = await self.execute_query(query, value, fetch_all=True)
        return [self._map_record_to_entity(record) for record in records]

    async def find_one_by_field(self, field: str, value: Any) -> T | None:
        """Find single entity by field value."""
        query = f"SELECT * FROM {self.table_name} WHERE {field} = $1 LIMIT 1"
        record = await self.execute_query(query, value, fetch_one=True)
        return self._map_record_to_entity(record) if record else None

    async def bulk_create(self, entities: list[T], user_id: str = None) -> list[T]:
        """Bulk create entities efficiently."""
        if not entities:
            return []

        entity_dicts = []
        for entity in entities:
            entity_dict = self._map_entity_to_dict(entity)
            entity_dict['created_at'] = datetime.now(UTC)
            entity_dict['updated_at'] = datetime.now(UTC)
            if user_id:
                entity_dict['created_by'] = user_id
            entity_dicts.append(entity_dict)

        # Use COPY for large bulk inserts (>100 records)
        if len(entities) > 100:
            return await self._bulk_copy_create(entity_dicts, user_id)
        return await self._bulk_insert_create(entity_dicts, user_id)

    async def execute_raw_query(
        self,
        query: str,
        *params,
        fetch_mode: str = "all"
    ) -> list[Record] | Record | Any | None:
        """Execute raw SQL query with parameters."""
        if fetch_mode == "one":
            return await self.execute_query(query, *params, fetch_one=True)
        if fetch_mode == "val":
            return await self.execute_query(query, *params, fetch_val=True)
        return await self.execute_query(query, *params, fetch_all=True)

    # Private helper methods
    def _get_id_column(self) -> str:
        """Get the ID column name for this table."""
        return f"{self.table_name.rstrip('s')}_id"  # Remove 's' and add '_id'

    def _get_id_field(self) -> str:
        """Get the ID field name for the entity."""
        return f"{self.table_name.rstrip('s')}_id"

    async def _has_column(self, column_name: str) -> bool:
        """Check if table has specific column."""
        query = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = $1 AND column_name = $2
            )
        """
        return await self.execute_query(query, self.table_name, column_name, fetch_val=True)

    async def _create_audit_entry(
        self,
        connection: Connection,
        action: str,
        entity_id: UUID,
        old_values: dict[str, Any] = None,
        new_values: dict[str, Any] = None,
        changes: dict[str, Any] = None,
        user_id: str = None
    ):
        """Create audit trail entry."""
        audit_data = {
            'audit_id': uuid4(),
            'idea_id': entity_id,  # For now, all audits are idea-related
            'action': action,
            'entity_type': self.table_name.rstrip('s'),
            'entity_id': entity_id,
            'old_values': json.dumps(old_values) if old_values else '{}',
            'new_values': json.dumps(new_values) if new_values else '{}',
            'changes': json.dumps(changes) if changes else '{}',
            'user_id': user_id,
            'timestamp': datetime.now(UTC)
        }

        audit_query = """
            INSERT INTO audit_trail (
                audit_id, idea_id, action, entity_type, entity_id,
                old_values, new_values, changes, user_id, timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """

        await connection.execute(
            audit_query,
            audit_data['audit_id'],
            audit_data['idea_id'],
            audit_data['action'],
            audit_data['entity_type'],
            audit_data['entity_id'],
            audit_data['old_values'],
            audit_data['new_values'],
            audit_data['changes'],
            audit_data['user_id'],
            audit_data['timestamp']
        )

    def _calculate_changes(
        self,
        old_values: dict[str, Any],
        new_values: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate changes between old and new values."""
        changes = {}

        for key, new_value in new_values.items():
            old_value = old_values.get(key)
            if old_value != new_value:
                changes[key] = {
                    'old': old_value,
                    'new': new_value
                }

        return changes

    async def _bulk_insert_create(self, entity_dicts: list[dict], user_id: str) -> list[T]:
        """Bulk create using INSERT statements."""
        created_entities = []

        async with self.transaction() as tx:
            for entity_dict in entity_dicts:
                fields = list(entity_dict.keys())
                placeholders = [f"${i+1}" for i in range(len(fields))]
                values = list(entity_dict.values())

                query = f"""
                    INSERT INTO {self.table_name} ({', '.join(fields)})
                    VALUES ({', '.join(placeholders)})
                    RETURNING *
                """

                record = await self.execute_query(query, *values, fetch_one=True)
                created_entity = self._map_record_to_entity(record)
                created_entities.append(created_entity)

                # Create audit trail
                await self._create_audit_entry(
                    tx.connection,
                    action="BULK_CREATE",
                    entity_id=getattr(created_entity, self._get_id_field()),
                    new_values=entity_dict,
                    user_id=user_id
                )

        return created_entities

    async def _bulk_copy_create(self, entity_dicts: list[dict], user_id: str) -> list[T]:
        """Bulk create using COPY for large datasets."""
        # This is a simplified version - full implementation would use asyncpg COPY
        # For now, fall back to bulk insert
        return await self._bulk_insert_create(entity_dicts, user_id)


class CachedRepository(BaseRepository[T]):
    """Repository with advanced caching capabilities."""

    def __init__(self, table_name: str, pool: Pool, cache_ttl: int = 300):
        super().__init__(table_name, pool)
        self._cache_ttl = cache_ttl
        self._entity_cache = {}
        self._query_result_cache = {}

    async def find_by_id(self, entity_id: UUID) -> T | None:
        """Find by ID with enhanced caching."""
        cache_key = f"entity:{entity_id}"

        # Check entity cache
        if cache_key in self._entity_cache:
            entity, cached_time = self._entity_cache[cache_key]
            if (datetime.now(UTC) - cached_time).total_seconds() < self._cache_ttl:
                return entity

        # Fallback to base implementation
        entity = await super().find_by_id(entity_id)

        if entity:
            self._entity_cache[cache_key] = (entity, datetime.now(UTC))

        return entity

    def invalidate_cache(self, entity_id: UUID = None):
        """Invalidate cache for specific entity or all entities."""
        if entity_id:
            cache_key = f"entity:{entity_id}"
            self._entity_cache.pop(cache_key, None)
        else:
            self._entity_cache.clear()
            self._query_result_cache.clear()

    async def update(self, entity: T, user_id: str = None) -> T:
        """Update entity and invalidate cache."""
        updated_entity = await super().update(entity, user_id)
        entity_id = getattr(updated_entity, self._get_id_field())
        self.invalidate_cache(entity_id)
        return updated_entity

    async def delete(self, entity_id: UUID, user_id: str = None) -> bool:
        """Delete entity and invalidate cache."""
        result = await super().delete(entity_id, user_id)
        self.invalidate_cache(entity_id)
        return result
