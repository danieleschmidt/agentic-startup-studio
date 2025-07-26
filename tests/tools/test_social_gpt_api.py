# tests/tools/test_social_gpt_api.py
import pytest
import os
from unittest.mock import patch, Mock
import responses
import requests
from tools.social_gpt_api import TwitterV2Client, run


class TestTwitterV2Client:
    
    def setup_method(self):
        """Setup test environment"""
        self.bearer_token = "test_bearer_token"
        self.client = TwitterV2Client(bearer_token=self.bearer_token)
        
    def test_client_initialization_with_token(self):
        """Test client initializes with bearer token"""
        assert self.client.bearer_token == self.bearer_token
        assert self.client.base_url == "https://api.twitter.com/2"
        
    def test_client_initialization_without_token_raises_error(self):
        """Test client raises error when no bearer token provided"""
        with pytest.raises(ValueError, match="Twitter Bearer token is required"):
            TwitterV2Client(bearer_token=None)
            
    @responses.activate
    def test_post_tweet_success(self):
        """Test successful tweet posting"""
        tweet_text = "Hello World! #test"
        mock_response = {
            "data": {
                "id": "1234567890",
                "text": tweet_text
            }
        }
        
        responses.add(
            responses.POST,
            "https://api.twitter.com/2/tweets",
            json=mock_response,
            status=201
        )
        
        result = self.client.post_tweet(tweet_text)
        
        assert result["success"] is True
        assert result["tweet_id"] == "1234567890"
        assert result["text"] == tweet_text
        
    @responses.activate 
    def test_post_tweet_with_rate_limit(self):
        """Test tweet posting handles rate limit"""
        responses.add(
            responses.POST,
            "https://api.twitter.com/2/tweets",
            status=429,
            headers={"x-rate-limit-reset": "1234567890"}
        )
        
        result = self.client.post_tweet("Test tweet")
        
        assert result["success"] is False
        assert "rate limit" in result["error"].lower()
        
    @responses.activate
    def test_post_tweet_authentication_error(self):
        """Test tweet posting handles authentication error"""
        responses.add(
            responses.POST,
            "https://api.twitter.com/2/tweets",
            status=401,
            json={"errors": [{"message": "Unauthorized"}]}
        )
        
        result = self.client.post_tweet("Test tweet")
        
        assert result["success"] is False
        assert "authentication" in result["error"].lower()
        
    @responses.activate
    def test_post_tweet_validation_error(self):
        """Test tweet posting handles validation errors"""
        long_tweet = "a" * 281  # Exceeds Twitter's character limit
        
        responses.add(
            responses.POST,
            "https://api.twitter.com/2/tweets",
            status=400,
            json={"errors": [{"message": "Tweet text too long"}]}
        )
        
        result = self.client.post_tweet(long_tweet)
        
        assert result["success"] is False
        assert "validation" in result["error"].lower()
        
    @responses.activate
    def test_post_thread_success(self):
        """Test successful thread posting"""
        thread = ["Tweet 1", "Tweet 2", "Tweet 3"]
        
        # Mock each tweet response
        for i, tweet in enumerate(thread):
            tweet_id = f"123456789{i}"
            mock_response = {
                "data": {
                    "id": tweet_id,
                    "text": tweet
                }
            }
            responses.add(
                responses.POST,
                "https://api.twitter.com/2/tweets",
                json=mock_response,
                status=201
            )
            
        result = self.client.post_thread(thread)
        
        assert result["success"] is True
        assert len(result["tweets"]) == 3
        assert result["tweets"][0]["tweet_id"] == "1234567890"
        
    def test_post_thread_empty_thread(self):
        """Test thread posting with empty thread"""
        result = self.client.post_thread([])
        
        assert result["success"] is False
        assert "empty" in result["error"].lower()
        
    @responses.activate
    def test_post_thread_partial_failure(self):
        """Test thread posting with partial failure"""
        thread = ["Tweet 1", "Tweet 2"]
        
        # First tweet succeeds
        responses.add(
            responses.POST,
            "https://api.twitter.com/2/tweets",
            json={"data": {"id": "1234567890", "text": "Tweet 1"}},
            status=201
        )
        
        # Second tweet fails
        responses.add(
            responses.POST,
            "https://api.twitter.com/2/tweets",
            status=400,
            json={"errors": [{"message": "Duplicate content"}]}
        )
        
        result = self.client.post_thread(thread)
        
        assert result["success"] is False
        assert len(result["tweets"]) == 1  # Only first tweet posted
        assert "partial" in result["error"].lower()


class TestRunFunction:
    
    @patch.dict(os.environ, {"TWITTER_BEARER": "test_token"})
    @patch('tools.social_gpt_api.TwitterV2Client')
    def test_run_with_valid_environment(self, mock_client_class):
        """Test run function with valid environment setup"""
        mock_client = Mock()
        mock_client.post_thread.return_value = {
            "success": True,
            "tweets": [{"tweet_id": "123", "text": "Test"}]
        }
        mock_client_class.return_value = mock_client
        
        thread = ["Test tweet"]
        result = run(thread)
        
        mock_client_class.assert_called_once_with("test_token")
        mock_client.post_thread.assert_called_once_with(thread)
        assert result["success"] is True
        
    @patch.dict(os.environ, {}, clear=True)
    def test_run_without_bearer_token(self):
        """Test run function without TWITTER_BEARER environment variable"""
        thread = ["Test tweet"]
        result = run(thread)
        
        assert result["success"] is False
        assert "environment" in result["error"].lower()
        
    @patch.dict(os.environ, {"TWITTER_BEARER": ""})
    def test_run_with_empty_bearer_token(self):
        """Test run function with empty TWITTER_BEARER"""
        thread = ["Test tweet"]
        result = run(thread)
        
        assert result["success"] is False
        assert "environment" in result["error"].lower()


class TestSecurityValidation:
    
    def test_tweet_text_sanitization(self):
        """Test that tweet text is properly sanitized"""
        client = TwitterV2Client("test_token")
        
        # Test with potentially malicious content
        malicious_text = "<script>alert('xss')</script>Normal text"
        sanitized = client._sanitize_tweet_text(malicious_text)
        
        assert "<script>" not in sanitized
        assert "Normal text" in sanitized
        
    def test_character_limit_validation(self):
        """Test tweet character limit validation"""
        client = TwitterV2Client("test_token")
        
        # Test with text exceeding 280 characters
        long_text = "a" * 281
        is_valid, error = client._validate_tweet_text(long_text)
        
        assert is_valid is False
        assert "character limit" in error.lower()
        
    def test_empty_text_validation(self):
        """Test empty tweet text validation"""
        client = TwitterV2Client("test_token")
        
        is_valid, error = client._validate_tweet_text("")
        
        assert is_valid is False
        assert "empty" in error.lower()