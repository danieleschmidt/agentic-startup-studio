"""Database test fixtures for Agentic Startup Studio."""

import asyncio
from typing import Generator, AsyncGenerator
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from pipeline.models.idea import Base


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """Test database URL for in-memory SQLite."""
    return "sqlite:///:memory:"


@pytest.fixture(scope="session")
def async_test_db_url() -> str:
    """Async test database URL for in-memory SQLite."""
    return "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def sync_engine(test_db_url: str):
    """Create synchronous test database engine."""
    engine = create_engine(
        test_db_url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def async_engine(async_test_db_url: str):
    """Create asynchronous test database engine."""
    engine = create_async_engine(
        async_test_db_url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def sync_session(sync_engine) -> Generator[Session, None, None]:
    """Create synchronous database session for testing."""
    SessionLocal = sessionmaker(bind=sync_engine)
    session = SessionLocal()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest_asyncio.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create asynchronous database session for testing."""
    async_session_local = sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session_local() as session:
        try:
            yield session
        finally:
            await session.rollback()


@pytest.fixture
def clean_db(sync_session: Session) -> Generator[Session, None, None]:
    """Provide a clean database session that rolls back after each test."""
    transaction = sync_session.begin()
    
    try:
        yield sync_session
    finally:
        transaction.rollback()


@pytest_asyncio.fixture
async def clean_async_db(async_session: AsyncSession) -> AsyncGenerator[AsyncSession, None]:
    """Provide a clean async database session that rolls back after each test."""
    async with async_session.begin():
        try:
            yield async_session
        finally:
            await async_session.rollback()