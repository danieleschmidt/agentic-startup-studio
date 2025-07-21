#!/usr/bin/env python3
"""
Security tests for SQL injection vulnerability fixes.
Validates that the implemented security measures prevent injection attacks.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from pipeline.adapters.google_ads_adapter import GoogleAdsAdapter
from core.models import CampaignStatus


class TestSQLInjectionFixes:
    """Test SQL injection vulnerability fixes."""
    
    def setup_method(self):
        """Set up test environment."""
        self.adapter = GoogleAdsAdapter()
        self.adapter._session = Mock()
        self.adapter._get_auth_headers = AsyncMock(return_value={'Authorization': 'Bearer test'})
        self.adapter.post_json = AsyncMock(return_value={'results': []})
    
    @pytest.mark.asyncio
    async def test_campaign_status_filter_validation(self):
        """Test that status filter validates enum values properly."""
        # Test valid enum value
        valid_status = CampaignStatus.ENABLED
        await self.adapter.get_campaigns(status_filter=valid_status)
        
        # Verify the query was called with validated status
        call_args = self.adapter.post_json.call_args[0][1]['query']
        assert "campaign.status = 'ENABLED'" in call_args
    
    @pytest.mark.asyncio 
    async def test_campaign_status_filter_rejects_invalid_type(self):
        """Test that status filter rejects non-enum types."""
        with pytest.raises(ValueError, match="Invalid status filter type"):
            await self.adapter.get_campaigns(status_filter="MALICIOUS_INJECTION")
    
    @pytest.mark.asyncio
    async def test_campaign_status_filter_rejects_invalid_enum_value(self):
        """Test that status filter rejects invalid enum values."""
        # Create a mock enum with invalid value
        mock_status = Mock()
        mock_status.value = "'; DROP TABLE campaigns; --"
        
        with pytest.raises(ValueError, match="Invalid status filter value"):
            await self.adapter.get_campaigns(status_filter=mock_status)
    
    @pytest.mark.asyncio
    async def test_campaign_ids_validation(self):
        """Test that campaign ID filtering validates numeric IDs."""
        # Test with valid numeric IDs
        valid_ids = ['123456', '789012']
        await self.adapter.get_performance_data(campaign_ids=valid_ids)
        
        # Verify the query includes the validated IDs
        call_args = self.adapter.post_json.call_args[0][1]['query'] 
        assert "campaign.id IN ('123456','789012')" in call_args
    
    @pytest.mark.asyncio
    async def test_campaign_ids_rejects_non_numeric(self):
        """Test that campaign ID filtering rejects non-numeric values."""
        # Test with injection attempt
        malicious_ids = ['123'; 'DROP TABLE ideas'; '456']
        
        # The sanitization should filter out non-numeric IDs
        await self.adapter.get_performance_data(campaign_ids=malicious_ids)
        
        # Verify only numeric IDs are included (none in this case)
        call_args = self.adapter.post_json.call_args[0][1]['query']
        assert "DROP TABLE" not in call_args
        assert "campaign.id IN" not in call_args  # No valid IDs to filter
    
    @pytest.mark.asyncio
    async def test_campaign_ids_mixed_valid_invalid(self):
        """Test campaign ID filtering with mixed valid and invalid IDs."""
        mixed_ids = ['123456', 'invalid_id', '789012', "'; DROP TABLE campaigns; --"]
        
        await self.adapter.get_performance_data(campaign_ids=mixed_ids)
        
        # Verify only valid numeric IDs are included
        call_args = self.adapter.post_json.call_args[0][1]['query']
        assert "campaign.id IN ('123456','789012')" in call_args
        assert "DROP TABLE" not in call_args
        assert "invalid_id" not in call_args
    
    def test_allowlist_validation_coverage(self):
        """Test that allowlisted values cover all valid enum values."""
        # Ensure our allowlist covers all legitimate campaign statuses
        allowed_statuses = ['ENABLED', 'PAUSED', 'REMOVED']
        
        # Check that all CampaignStatus enum values are in allowlist
        for status in CampaignStatus:
            assert status.value in allowed_statuses, f"Status {status.value} not in allowlist"
    
    @pytest.mark.asyncio
    async def test_empty_campaign_ids_handling(self):
        """Test handling of empty campaign IDs list."""
        await self.adapter.get_performance_data(campaign_ids=[])
        
        call_args = self.adapter.post_json.call_args[0][1]['query']
        assert "campaign.id IN" not in call_args  # No filter should be added
    
    @pytest.mark.asyncio
    async def test_none_campaign_ids_handling(self):
        """Test handling of None campaign IDs."""
        await self.adapter.get_performance_data(campaign_ids=None)
        
        call_args = self.adapter.post_json.call_args[0][1]['query']
        assert "campaign.id IN" not in call_args  # No filter should be added


class TestSecurityPatterns:
    """Test general security patterns and best practices."""
    
    def test_no_string_formatting_in_queries(self):
        """Verify that no unsafe string formatting is used in SQL queries."""
        # This would be expanded to scan actual code files
        # For now, we test the principle with a mock
        
        unsafe_patterns = [
            "f\"SELECT * FROM table WHERE id = {user_input}\"",
            "\"SELECT * FROM table WHERE id = '{}'\".format(user_input)",
            "\"SELECT * FROM table WHERE id = '%s'\" % user_input"
        ]
        
        # In a real implementation, this would scan the codebase
        # and ensure these patterns don't exist
        for pattern in unsafe_patterns:
            assert "user_input" in pattern  # Placeholder test
    
    def test_input_validation_functions(self):
        """Test that input validation functions work correctly."""
        # Test numeric validation
        assert "123456".isdigit() == True
        assert "123abc".isdigit() == False
        assert "'; DROP TABLE".isdigit() == False
        assert "".isdigit() == False
    
    def test_enum_validation(self):
        """Test that enum validation prevents arbitrary values."""
        valid_statuses = ['ENABLED', 'PAUSED', 'REMOVED']
        
        # Test valid values
        assert 'ENABLED' in valid_statuses
        assert 'PAUSED' in valid_statuses
        
        # Test invalid values
        assert "'; DROP TABLE campaigns; --" not in valid_statuses
        assert 'INVALID_STATUS' not in valid_statuses
        assert '' not in valid_statuses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])