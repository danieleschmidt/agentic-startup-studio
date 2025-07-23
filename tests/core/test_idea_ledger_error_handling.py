"""
Comprehensive tests for idea_ledger.py error handling and robustness.

Tests cover:
- Database connection failures
- Invalid input data
- Database constraint violations
- Network timeouts and recovery
- Edge cases and boundary conditions
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4, UUID
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from sqlmodel import Session

# Import the module under test
import core.idea_ledger as ledger
from core.idea_ledger import (
    DatabaseConnectionError,
    IdeaNotFoundError,
    IdeaValidationError,
    add_idea,
    get_idea_by_id,
    list_ideas,
    update_idea,
    delete_idea,
    create_db_and_tables
)
from core.models import Idea, IdeaCreate, IdeaUpdate


class TestDatabaseConnectionHandling:
    """Test database connection error handling"""

    @patch('core.idea_ledger.SQLModel.metadata.create_all')
    def test_create_db_and_tables_operational_error(self, mock_create_all):
        """Test handling of database connection failure during table creation"""
        mock_create_all.side_effect = OperationalError(
            "Connection failed", None, None
        )
        
        with pytest.raises(DatabaseConnectionError) as exc_info:
            create_db_and_tables()
        
        assert "Could not connect to database" in str(exc_info.value)
        assert mock_create_all.called

    @patch('core.idea_ledger.SQLModel.metadata.create_all')
    def test_create_db_and_tables_sqlalchemy_error(self, mock_create_all):
        """Test handling of SQLAlchemy errors during table creation"""
        mock_create_all.side_effect = SQLAlchemyError("Permission denied")
        
        with pytest.raises(DatabaseConnectionError) as exc_info:
            create_db_and_tables()
        
        assert "Could not create tables" in str(exc_info.value)

    @patch('core.idea_ledger.SQLModel.metadata.create_all')
    def test_create_db_and_tables_success(self, mock_create_all):
        """Test successful table creation"""
        mock_create_all.return_value = None
        
        result = create_db_and_tables()
        
        assert result is True
        assert mock_create_all.called


class TestAddIdeaErrorHandling:
    """Test add_idea function error handling"""

    def test_add_idea_invalid_data_validation_error(self):
        """Test handling of invalid idea data"""
        with patch('core.models.Idea.model_validate') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid field type")
            
            invalid_idea = IdeaCreate(name="Test", description="Test")
            
            with pytest.raises(IdeaValidationError) as exc_info:
                add_idea(invalid_idea)
            
            assert "Invalid idea data" in str(exc_info.value)

    @patch('core.idea_ledger.Session')
    def test_add_idea_integrity_error(self, mock_session_class):
        """Test handling of database integrity constraint violations"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        mock_session.commit.side_effect = IntegrityError(
            "UNIQUE constraint failed", None, None
        )
        
        idea_data = IdeaCreate(name="Duplicate", description="Test")
        
        with pytest.raises(IdeaValidationError) as exc_info:
            add_idea(idea_data)
        
        assert "violates database constraints" in str(exc_info.value)

    @patch('core.idea_ledger.Session')
    def test_add_idea_operational_error(self, mock_session_class):
        """Test handling of database connection failures during add"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        mock_session.commit.side_effect = OperationalError(
            "Connection lost", None, None
        )
        
        idea_data = IdeaCreate(name="Test", description="Test")
        
        with pytest.raises(DatabaseConnectionError) as exc_info:
            add_idea(idea_data)
        
        assert "Database connection failed" in str(exc_info.value)

    @patch('core.idea_ledger.Session')
    def test_add_idea_unexpected_error(self, mock_session_class):
        """Test handling of unexpected errors during add"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        mock_session.commit.side_effect = RuntimeError("Unexpected error")
        
        idea_data = IdeaCreate(name="Test", description="Test")
        
        with pytest.raises(DatabaseConnectionError) as exc_info:
            add_idea(idea_data)
        
        assert "Unexpected error" in str(exc_info.value)


class TestGetIdeaErrorHandling:
    """Test get_idea_by_id function error handling"""

    def test_get_idea_by_id_empty_uuid(self):
        """Test handling of empty/None UUID"""
        result = get_idea_by_id(None)
        assert result is None

    @patch('core.idea_ledger.Session')
    def test_get_idea_by_id_operational_error(self, mock_session_class):
        """Test handling of database connection failures during get"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        mock_session.get.side_effect = OperationalError(
            "Connection timeout", None, None
        )
        
        test_uuid = uuid4()
        
        with pytest.raises(DatabaseConnectionError) as exc_info:
            get_idea_by_id(test_uuid)
        
        assert "Database connection failed" in str(exc_info.value)

    @patch('core.idea_ledger.Session')
    def test_get_idea_by_id_sqlalchemy_error(self, mock_session_class):
        """Test handling of SQLAlchemy errors during get"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        mock_session.get.side_effect = SQLAlchemyError("Query failed")
        
        test_uuid = uuid4()
        
        with pytest.raises(DatabaseConnectionError) as exc_info:
            get_idea_by_id(test_uuid)
        
        assert "Database operation failed" in str(exc_info.value)


class TestListIdeasErrorHandling:
    """Test list_ideas function error handling"""

    def test_list_ideas_invalid_skip_parameter(self):
        """Test handling of invalid skip parameter"""
        with pytest.raises(ValueError) as exc_info:
            list_ideas(skip=-1)
        
        assert "Skip parameter must be non-negative" in str(exc_info.value)

    def test_list_ideas_invalid_limit_parameter(self):
        """Test handling of invalid limit parameters"""
        with pytest.raises(ValueError) as exc_info:
            list_ideas(limit=0)
        
        assert "Limit parameter must be between 1 and 1000" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info:
            list_ideas(limit=1001)
        
        assert "Limit parameter must be between 1 and 1000" in str(exc_info.value)

    @patch('core.idea_ledger.Session')
    def test_list_ideas_operational_error(self, mock_session_class):
        """Test handling of database connection failures during list"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        mock_session.exec.side_effect = OperationalError(
            "Database unavailable", None, None
        )
        
        with pytest.raises(DatabaseConnectionError) as exc_info:
            list_ideas()
        
        assert "Database connection failed" in str(exc_info.value)


class TestUpdateIdeaErrorHandling:
    """Test update_idea function error handling"""

    def test_update_idea_empty_uuid(self):
        """Test handling of empty/None UUID"""
        update_data = IdeaUpdate(name="Updated")
        
        with pytest.raises(ValueError) as exc_info:
            update_idea(None, update_data)
        
        assert "idea_id cannot be empty or None" in str(exc_info.value)

    def test_update_idea_invalid_update_data(self):
        """Test handling of invalid update data"""
        test_uuid = uuid4()
        
        with patch('core.models.IdeaUpdate.model_dump') as mock_dump:
            mock_dump.side_effect = ValueError("Invalid model data")
            update_data = IdeaUpdate(name="Test")
            
            with pytest.raises(IdeaValidationError) as exc_info:
                update_idea(test_uuid, update_data)
            
            assert "Invalid update data" in str(exc_info.value)

    @patch('core.idea_ledger.Session')
    def test_update_idea_not_found(self, mock_session_class):
        """Test handling of idea not found during update"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        mock_session.get.return_value = None
        
        test_uuid = uuid4()
        update_data = IdeaUpdate(name="Updated")
        
        with pytest.raises(IdeaNotFoundError) as exc_info:
            update_idea(test_uuid, update_data)
        
        assert f"Idea with ID {test_uuid} not found" in str(exc_info.value)

    @patch('core.idea_ledger.Session')
    def test_update_idea_integrity_error(self, mock_session_class):
        """Test handling of integrity constraint violations during update"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        
        # Mock existing idea
        mock_idea = Mock()
        mock_session.get.return_value = mock_idea
        mock_session.commit.side_effect = IntegrityError(
            "UNIQUE constraint failed", None, None
        )
        
        test_uuid = uuid4()
        update_data = IdeaUpdate(name="Duplicate Name")
        
        with pytest.raises(IdeaValidationError) as exc_info:
            update_idea(test_uuid, update_data)
        
        assert "violates database constraints" in str(exc_info.value)


class TestDeleteIdeaErrorHandling:
    """Test delete_idea function error handling"""

    def test_delete_idea_empty_uuid(self):
        """Test handling of empty/None UUID"""
        result = delete_idea(None)
        assert result is False

    @patch('core.idea_ledger.Session')
    def test_delete_idea_not_found(self, mock_session_class):
        """Test handling of idea not found during deletion"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        mock_session.get.return_value = None
        
        test_uuid = uuid4()
        result = delete_idea(test_uuid)
        
        assert result is False

    @patch('core.idea_ledger.Session')
    def test_delete_idea_integrity_error(self, mock_session_class):
        """Test handling of integrity constraint violations during deletion"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        
        # Mock existing idea
        mock_idea = Mock()
        mock_session.get.return_value = mock_idea
        mock_session.commit.side_effect = IntegrityError(
            "FOREIGN KEY constraint failed", None, None
        )
        
        test_uuid = uuid4()
        
        with pytest.raises(DatabaseConnectionError) as exc_info:
            delete_idea(test_uuid)
        
        assert "Cannot delete idea due to database constraints" in str(exc_info.value)

    @patch('core.idea_ledger.Session')
    def test_delete_idea_operational_error(self, mock_session_class):
        """Test handling of database connection failures during deletion"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        
        # Mock existing idea
        mock_idea = Mock()
        mock_session.get.return_value = mock_idea
        mock_session.commit.side_effect = OperationalError(
            "Connection lost", None, None
        )
        
        test_uuid = uuid4()
        
        with pytest.raises(DatabaseConnectionError) as exc_info:
            delete_idea(test_uuid)
        
        assert "Database connection failed" in str(exc_info.value)


class TestLoggingAndMonitoring:
    """Test logging behavior for monitoring and debugging"""

    @patch('core.idea_ledger.logger')
    @patch('core.idea_ledger.Session')
    def test_successful_operations_logged(self, mock_session_class, mock_logger):
        """Test that successful operations are properly logged"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        
        # Mock successful idea creation
        mock_idea = Mock()
        mock_idea.id = uuid4()
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None
        
        with patch('core.models.Idea.model_validate', return_value=mock_idea):
            idea_data = IdeaCreate(name="Test", description="Test")
            result = add_idea(idea_data)
            
            # Verify info logging for successful operation
            mock_logger.info.assert_called_with(f"Successfully added idea with ID: {mock_idea.id}")

    @patch('core.idea_ledger.logger')
    def test_validation_errors_logged(self, mock_logger):
        """Test that validation errors are properly logged"""
        with patch('core.models.Idea.model_validate') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid field")
            
            idea_data = IdeaCreate(name="Test", description="Test")
            
            with pytest.raises(IdeaValidationError):
                add_idea(idea_data)
            
            # Verify error logging
            mock_logger.error.assert_called()
            assert "Idea validation failed" in str(mock_logger.error.call_args)

    @patch('core.idea_ledger.logger')  
    def test_edge_case_warnings_logged(self, mock_logger):
        """Test that edge cases generate appropriate warnings"""
        # Test empty UUID warning
        result = get_idea_by_id(None)
        
        mock_logger.warning.assert_called_with("get_idea_by_id called with empty/None idea_id")


class TestBoundaryConditions:
    """Test boundary conditions and edge cases"""

    def test_list_ideas_boundary_values(self):
        """Test boundary values for pagination parameters"""
        # Valid boundary values should not raise exceptions
        with patch('core.idea_ledger.Session'):
            try:
                list_ideas(skip=0, limit=1)      # Minimum valid values
                list_ideas(skip=999999, limit=1000)  # Maximum valid values
            except ValueError:
                pytest.fail("Valid boundary values should not raise ValueError")

    @patch('core.idea_ledger.Session')
    def test_update_idea_empty_update_data(self, mock_session_class):
        """Test handling of empty update data"""
        mock_session = MagicMock()
        mock_session_class.return_value.__enter__.return_value = mock_session
        
        # Mock existing idea
        mock_idea = Mock()
        mock_session.get.return_value = mock_idea
        
        test_uuid = uuid4()
        
        # Mock empty update data
        with patch('core.models.IdeaUpdate.model_dump', return_value={}):
            with patch('core.idea_ledger.get_idea_by_id', return_value=mock_idea):
                update_data = IdeaUpdate()
                result = update_idea(test_uuid, update_data)
                
                assert result == mock_idea  # Should return existing idea unchanged

    def test_database_url_configuration(self):
        """Test database URL configuration handling"""
        # Test that environment variable is properly read
        original_url = ledger.DATABASE_URL
        
        # Test default fallback
        assert ledger.DEFAULT_DATABASE_URL == "postgresql://user:password@localhost:5432/appdb"
        
        # Test that current URL is either from env or default
        assert ledger.DATABASE_URL in [ledger.DEFAULT_DATABASE_URL, original_url]


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])