"""
Comprehensive test suite for command-line interface functionality.

Tests user input validation, output formatting, error scenarios,
and help text validation with proper mocking of dependencies.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import json
from io import StringIO
import sys

from pipeline.cli.ingestion_cli import (
    IdeaIngestionCLI, CLIError, UserInputError, OutputFormatError,
    create_cli_interface, main
)
from pipeline.models.idea import (
    Idea, IdeaDraft, IdeaStatus, PipelineStage, IdeaCategory, IdeaSummary
)
from pipeline.ingestion.idea_manager import (
    IdeaManager, IdeaManagementError, DuplicateIdeaError, ValidationError, StorageError
)


class TestIdeaIngestionCLI:
    """Test command-line interface functionality."""
    
    @pytest.fixture
    def mock_idea_manager(self):
        """Provide mock idea manager."""
        return Mock(spec=IdeaManager)
    
    @pytest.fixture
    def cli(self, mock_idea_manager):
        """Provide CLI with mocked dependencies."""
        return IdeaIngestionCLI(mock_idea_manager)
    
    @pytest.fixture
    def sample_idea_data(self) -> Dict[str, Any]:
        """Provide sample idea data for testing."""
        return {
            "title": "AI-powered productivity tool",
            "description": "Revolutionary solution using artificial intelligence to boost workplace productivity",
            "category": "ai_ml",
            "tags": ["ai", "productivity", "automation"],
            "evidence": "Market research shows 40% productivity increase"
        }
    
    @pytest.fixture
    def sample_idea(self) -> Idea:
        """Provide sample idea for testing."""
        return Idea(
            idea_id=uuid4(),
            title="AI-powered productivity tool",
            description="Revolutionary solution using AI",
            status=IdeaStatus.DRAFT,
            current_stage=PipelineStage.IDEATE,
            created_at=datetime.now(timezone.utc),
            stage_progress=0.3
        )
    
    @pytest.mark.asyncio
    async def test_when_create_idea_success_then_displays_success_message(
        self, cli, mock_idea_manager, sample_idea_data, capsys
    ):
        """Given valid idea data, when creating idea, then displays success message with ID."""
        test_id = uuid4()
        warnings = ["Consider adding more evidence"]
        mock_idea_manager.create_idea.return_value = (test_id, warnings)
        
        result = await cli.create_idea(
            title=sample_idea_data["title"],
            description=sample_idea_data["description"],
            category=sample_idea_data["category"],
            tags=sample_idea_data["tags"],
            evidence=sample_idea_data["evidence"],
            force=False,
            user_id="test_user"
        )
        
        assert result is True
        
        # Verify manager was called correctly
        mock_idea_manager.create_idea.assert_called_once()
        call_args = mock_idea_manager.create_idea.call_args[1]
        assert call_args["force_create"] is False
        assert call_args["user_id"] == "test_user"
        
        # Check output contains success message and ID
        captured = capsys.readouterr()
        assert "successfully created" in captured.out.lower()
        assert str(test_id) in captured.out
        assert "Consider adding more evidence" in captured.out
    
    @pytest.mark.asyncio
    async def test_when_duplicate_found_without_force_then_displays_error_and_suggestions(
        self, cli, mock_idea_manager, sample_idea_data, capsys
    ):
        """Given duplicate ideas without force flag, when creating idea, then displays error with suggestions."""
        duplicate_error = DuplicateIdeaError("Similar ideas found")
        mock_idea_manager.create_idea.side_effect = duplicate_error
        
        result = await cli.create_idea(
            title=sample_idea_data["title"],
            description=sample_idea_data["description"],
            category=sample_idea_data["category"]
        )
        
        assert result is False
        
        captured = capsys.readouterr()
        assert "similar ideas" in captured.out.lower()
        assert "--force" in captured.out
        assert len(duplicate_error.exact_matches) > 0  # Should show match count
    
    @pytest.mark.asyncio
    async def test_when_validation_error_then_displays_validation_messages(
        self, cli, mock_idea_manager, sample_idea_data, capsys
    ):
        """Given validation error, when creating idea, then displays validation messages."""
        validation_error = ValidationError("Validation failed: Title too short; Description missing evidence")
        mock_idea_manager.create_idea.side_effect = validation_error
        
        result = await cli.create_idea(
            title="AI",  # Too short
            description="Brief desc"
        )
        
        assert result is False
        
        captured = capsys.readouterr()
        assert "validation failed" in captured.out.lower()
        assert "Title too short" in captured.out
        assert "Description missing evidence" in captured.out
    
    @pytest.mark.asyncio
    async def test_when_storage_error_then_displays_system_error(
        self, cli, mock_idea_manager, sample_idea_data, capsys
    ):
        """Given storage error, when creating idea, then displays system error message."""
        storage_error = StorageError("Database connection failed")
        mock_idea_manager.create_idea.side_effect = storage_error
        
        result = await cli.create_idea(
            title=sample_idea_data["title"],
            description=sample_idea_data["description"]
        )
        
        assert result is False
        
        captured = capsys.readouterr()
        assert "system error" in captured.out.lower()
        assert "try again later" in captured.out.lower()
    
    @pytest.mark.asyncio
    async def test_when_get_idea_success_then_displays_formatted_details(
        self, cli, mock_idea_manager, sample_idea, capsys
    ):
        """Given existing idea, when getting idea, then displays formatted details."""
        mock_idea_manager.get_idea.return_value = sample_idea
        
        result = await cli.get_idea(str(sample_idea.idea_id))
        
        assert result is True
        
        captured = capsys.readouterr()
        assert sample_idea.title in captured.out
        assert sample_idea.description in captured.out
        assert sample_idea.status.value in captured.out
        assert sample_idea.current_stage.value in captured.out
        assert "30%" in captured.out  # Progress percentage
    
    @pytest.mark.asyncio
    async def test_when_get_nonexistent_idea_then_displays_not_found(
        self, cli, mock_idea_manager, capsys
    ):
        """Given non-existent idea ID, when getting idea, then displays not found message."""
        mock_idea_manager.get_idea.return_value = None
        
        result = await cli.get_idea(str(uuid4()))
        
        assert result is False
        
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()
    
    @pytest.mark.asyncio
    async def test_when_invalid_uuid_then_displays_format_error(
        self, cli, mock_idea_manager, capsys
    ):
        """Given invalid UUID format, when getting idea, then displays format error."""
        result = await cli.get_idea("invalid-uuid")
        
        assert result is False
        
        captured = capsys.readouterr()
        assert "invalid id format" in captured.out.lower()
    
    @pytest.mark.asyncio
    async def test_when_list_ideas_success_then_displays_formatted_table(
        self, cli, mock_idea_manager, capsys
    ):
        """Given ideas in repository, when listing ideas, then displays formatted table."""
        # Setup mock ideas
        ideas = [
            IdeaSummary(
                id=uuid4(),
                title="AI Tool",
                status=IdeaStatus.DRAFT,
                stage=PipelineStage.IDEATE,
                progress=0.3,
                created_at=datetime.now(timezone.utc)
            ),
            IdeaSummary(
                id=uuid4(),
                title="Mobile App",
                status=IdeaStatus.RESEARCHING,
                stage=PipelineStage.RESEARCH,
                progress=0.7,
                created_at=datetime.now(timezone.utc)
            )
        ]
        mock_idea_manager.list_ideas.return_value = ideas
        
        result = await cli.list_ideas()
        
        assert result is True
        
        captured = capsys.readouterr()
        # Should display table headers
        assert "ID" in captured.out
        assert "Title" in captured.out
        assert "Status" in captured.out
        assert "Stage" in captured.out
        assert "Progress" in captured.out
        
        # Should display idea data
        assert "AI Tool" in captured.out
        assert "Mobile App" in captured.out
        assert "30%" in captured.out
        assert "70%" in captured.out
    
    @pytest.mark.asyncio
    async def test_when_list_ideas_with_filters_then_applies_filters(
        self, cli, mock_idea_manager
    ):
        """Given filter parameters, when listing ideas, then applies filters to manager call."""
        mock_idea_manager.list_ideas.return_value = []
        
        await cli.list_ideas(
            status="draft",
            category="ai_ml",
            limit=10
        )
        
        mock_idea_manager.list_ideas.assert_called_once()
        call_args = mock_idea_manager.list_ideas.call_args[1]
        assert call_args["status"] == IdeaStatus.DRAFT
        assert call_args["category"] == IdeaCategory.AI_ML
        assert call_args["limit"] == 10
    
    @pytest.mark.asyncio
    async def test_when_empty_list_then_displays_no_ideas_message(
        self, cli, mock_idea_manager, capsys
    ):
        """Given empty idea list, when listing ideas, then displays no ideas message."""
        mock_idea_manager.list_ideas.return_value = []
        
        result = await cli.list_ideas()
        
        assert result is True
        
        captured = capsys.readouterr()
        assert "no ideas found" in captured.out.lower()
    
    @pytest.mark.asyncio
    async def test_when_advance_stage_success_then_displays_success_message(
        self, cli, mock_idea_manager, capsys
    ):
        """Given valid stage advancement, when advancing stage, then displays success message."""
        test_id = uuid4()
        mock_idea_manager.advance_stage.return_value = True
        
        result = await cli.advance_stage(
            idea_id=str(test_id),
            next_stage="research",
            user_id="test_user"
        )
        
        assert result is True
        
        mock_idea_manager.advance_stage.assert_called_once_with(
            idea_id=test_id,
            next_stage=PipelineStage.RESEARCH,
            user_id="test_user"
        )
        
        captured = capsys.readouterr()
        assert "stage advanced" in captured.out.lower()
        assert "research" in captured.out.lower()
    
    @pytest.mark.asyncio
    async def test_when_advance_stage_fails_then_displays_error(
        self, cli, mock_idea_manager, capsys
    ):
        """Given stage advancement failure, when advancing stage, then displays error message."""
        mock_idea_manager.advance_stage.side_effect = IdeaManagementError("Invalid transition")
        
        result = await cli.advance_stage(
            idea_id=str(uuid4()),
            next_stage="research"
        )
        
        assert result is False
        
        captured = capsys.readouterr()
        assert "failed to advance" in captured.out.lower()
        assert "Invalid transition" in captured.out
    
    def test_when_format_idea_details_then_returns_structured_output(self, cli, sample_idea):
        """Given idea object, when formatting details, then returns structured output."""
        output = cli._format_idea_details(sample_idea)
        
        assert sample_idea.title in output
        assert sample_idea.description in output
        assert sample_idea.status.value in output
        assert sample_idea.current_stage.value in output
        assert "30%" in output  # Progress formatting
        assert "Created:" in output
    
    def test_when_format_ideas_table_then_returns_tabular_output(self, cli):
        """Given list of idea summaries, when formatting table, then returns tabular output."""
        ideas = [
            IdeaSummary(
                id=uuid4(),
                title="Short Title",
                status=IdeaStatus.DRAFT,
                stage=PipelineStage.IDEATE,
                progress=0.5,
                created_at=datetime.now(timezone.utc)
            )
        ]
        
        output = cli._format_ideas_table(ideas)
        
        # Should contain headers
        assert "ID" in output
        assert "Title" in output
        assert "Status" in output
        
        # Should contain data
        assert "Short Title" in output
        assert "DRAFT" in output
        assert "50%" in output
    
    def test_when_parse_stage_name_then_converts_to_enum(self, cli):
        """Given stage name string, when parsing stage, then converts to enum."""
        assert cli._parse_stage_name("ideate") == PipelineStage.IDEATE
        assert cli._parse_stage_name("research") == PipelineStage.RESEARCH
        assert cli._parse_stage_name("building") == PipelineStage.BUILDING
        assert cli._parse_stage_name("testing") == PipelineStage.TESTING
        assert cli._parse_stage_name("complete") == PipelineStage.COMPLETE
        
        # Case insensitive
        assert cli._parse_stage_name("RESEARCH") == PipelineStage.RESEARCH
        assert cli._parse_stage_name("Research") == PipelineStage.RESEARCH
    
    def test_when_parse_invalid_stage_then_raises_error(self, cli):
        """Given invalid stage name, when parsing stage, then raises UserInputError."""
        with pytest.raises(UserInputError, match="Invalid stage"):
            cli._parse_stage_name("invalid_stage")
    
    def test_when_parse_status_name_then_converts_to_enum(self, cli):
        """Given status name string, when parsing status, then converts to enum."""
        assert cli._parse_status_name("draft") == IdeaStatus.DRAFT
        assert cli._parse_status_name("researching") == IdeaStatus.RESEARCHING
        assert cli._parse_status_name("building") == IdeaStatus.BUILDING
        
        # Case insensitive
        assert cli._parse_status_name("DRAFT") == IdeaStatus.DRAFT
    
    def test_when_parse_invalid_status_then_raises_error(self, cli):
        """Given invalid status name, when parsing status, then raises UserInputError."""
        with pytest.raises(UserInputError, match="Invalid status"):
            cli._parse_status_name("invalid_status")
    
    def test_when_parse_category_name_then_converts_to_enum(self, cli):
        """Given category name string, when parsing category, then converts to enum."""
        assert cli._parse_category_name("ai_ml") == IdeaCategory.AI_ML
        assert cli._parse_category_name("web_app") == IdeaCategory.WEB_APP
        assert cli._parse_category_name("mobile") == IdeaCategory.MOBILE
        
        # Case insensitive
        assert cli._parse_category_name("AI_ML") == IdeaCategory.AI_ML
    
    def test_when_validate_uuid_format_then_returns_uuid_object(self, cli):
        """Given valid UUID string, when validating format, then returns UUID object."""
        test_uuid = uuid4()
        result = cli._validate_uuid_format(str(test_uuid))
        assert result == test_uuid
    
    def test_when_validate_invalid_uuid_then_raises_error(self, cli):
        """Given invalid UUID string, when validating format, then raises UserInputError."""
        with pytest.raises(UserInputError, match="Invalid ID format"):
            cli._validate_uuid_format("not-a-uuid")


class TestCreateCLIInterface:
    """Test CLI factory function."""
    
    @pytest.mark.asyncio
    async def test_when_create_cli_interface_then_initializes_with_manager(self):
        """Given factory call, when creating CLI interface, then initializes with idea manager."""
        with patch('pipeline.cli.ingestion_cli.create_idea_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            
            result = await create_cli_interface()
            
            assert isinstance(result, IdeaIngestionCLI)
            assert result.idea_manager == mock_manager
            mock_create_manager.assert_called_once()


class TestMainFunction:
    """Test main CLI entry point."""
    
    @pytest.fixture
    def mock_cli(self):
        """Provide mock CLI interface."""
        return Mock(spec=IdeaIngestionCLI)
    
    def test_when_main_with_create_command_then_calls_create_idea(self, mock_cli):
        """Given create command arguments, when running main, then calls create_idea method."""
        # Mock sys.argv
        test_args = [
            "ingestion_cli.py", "create",
            "--title", "Test Idea",
            "--description", "Test description",
            "--category", "ai_ml",
            "--user-id", "test_user"
        ]
        
        with patch('sys.argv', test_args), \
             patch('pipeline.cli.ingestion_cli.create_cli_interface', return_value=mock_cli), \
             patch('asyncio.run') as mock_asyncio_run:
            
            # Mock the async create_idea method
            mock_cli.create_idea = AsyncMock(return_value=True)
            
            main()
            
            # Verify asyncio.run was called (main function should be async)
            assert mock_asyncio_run.called
    
    def test_when_main_with_get_command_then_calls_get_idea(self, mock_cli):
        """Given get command arguments, when running main, then calls get_idea method."""
        test_args = [
            "ingestion_cli.py", "get",
            "--id", str(uuid4())
        ]
        
        with patch('sys.argv', test_args), \
             patch('pipeline.cli.ingestion_cli.create_cli_interface', return_value=mock_cli), \
             patch('asyncio.run') as mock_asyncio_run:
            
            mock_cli.get_idea = AsyncMock(return_value=True)
            
            main()
            
            assert mock_asyncio_run.called
    
    def test_when_main_with_list_command_then_calls_list_ideas(self, mock_cli):
        """Given list command arguments, when running main, then calls list_ideas method."""
        test_args = [
            "ingestion_cli.py", "list",
            "--limit", "10"
        ]
        
        with patch('sys.argv', test_args), \
             patch('pipeline.cli.ingestion_cli.create_cli_interface', return_value=mock_cli), \
             patch('asyncio.run') as mock_asyncio_run:
            
            mock_cli.list_ideas = AsyncMock(return_value=True)
            
            main()
            
            assert mock_asyncio_run.called
    
    def test_when_main_with_advance_command_then_calls_advance_stage(self, mock_cli):
        """Given advance command arguments, when running main, then calls advance_stage method."""
        test_args = [
            "ingestion_cli.py", "advance",
            "--id", str(uuid4()),
            "--stage", "research"
        ]
        
        with patch('sys.argv', test_args), \
             patch('pipeline.cli.ingestion_cli.create_cli_interface', return_value=mock_cli), \
             patch('asyncio.run') as mock_asyncio_run:
            
            mock_cli.advance_stage = AsyncMock(return_value=True)
            
            main()
            
            assert mock_asyncio_run.called
    
    def test_when_main_with_help_then_displays_usage(self):
        """Given help argument, when running main, then displays usage information."""
        test_args = ["ingestion_cli.py", "--help"]
        
        with patch('sys.argv', test_args), \
             pytest.raises(SystemExit):  # argparse exits on --help
            
            main()


class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""
    
    @pytest.fixture
    def cli(self):
        """Provide CLI with mock manager for error testing."""
        mock_manager = Mock(spec=IdeaManager)
        return IdeaIngestionCLI(mock_manager)
    
    @pytest.mark.asyncio
    async def test_when_unexpected_error_then_displays_generic_message(
        self, cli, capsys
    ):
        """Given unexpected error, when executing command, then displays generic error message."""
        cli.idea_manager.create_idea.side_effect = Exception("Unexpected error")
        
        result = await cli.create_idea(
            title="Test",
            description="Test description"
        )
        
        assert result is False
        
        captured = capsys.readouterr()
        assert "unexpected error" in captured.out.lower()
    
    def test_when_output_too_long_for_terminal_then_truncates_gracefully(self, cli):
        """Given very long output, when formatting, then truncates gracefully."""
        # Create idea with very long description
        long_description = "A" * 10000  # Very long string
        idea = Idea(
            idea_id=uuid4(),
            title="Test",
            description=long_description,
            status=IdeaStatus.DRAFT,
            current_stage=PipelineStage.IDEATE
        )
        
        output = cli._format_idea_details(idea)
        
        # Should not crash and should be reasonable length
        assert len(output) < 50000  # Reasonable upper bound
        assert "Test" in output
    
    def test_when_special_characters_in_output_then_handles_safely(self, cli):
        """Given special characters in idea data, when formatting, then handles safely."""
        idea = Idea(
            idea_id=uuid4(),
            title="Test with émojis 🚀 and ñ special chars",
            description="Description with quotes 'single' and \"double\" and \n newlines",
            status=IdeaStatus.DRAFT,
            current_stage=PipelineStage.IDEATE
        )
        
        output = cli._format_idea_details(idea)
        
        # Should not crash and should preserve important content
        assert "émojis 🚀" in output
        assert "ñ special" in output
        assert "quotes" in output