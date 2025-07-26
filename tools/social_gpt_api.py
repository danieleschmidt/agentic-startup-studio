# tools/social_gpt_api.py - Twitter API v2 Integration
import requests
import os
import time
import re
import html
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class TwitterV2Client:
    """Twitter API v2 client with rate limiting and error handling"""
    
    def __init__(self, bearer_token: str):
        if not bearer_token:
            raise ValueError("Twitter Bearer token is required")
        
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        })
        
    def _sanitize_tweet_text(self, text: str) -> str:
        """Sanitize tweet text to prevent XSS and ensure safety"""
        # HTML decode first
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove potentially dangerous scripts
        text = re.sub(r'(javascript:|data:|vbscript:)', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def _validate_tweet_text(self, text: str) -> Tuple[bool, str]:
        """Validate tweet text against Twitter's requirements"""
        if not text or not text.strip():
            return False, "Tweet text cannot be empty"
            
        if len(text) > 280:
            return False, f"Tweet exceeds 280 character limit (current: {len(text)})"
            
        return True, ""
        
    def post_tweet(self, text: str, reply_to_id: str = None) -> Dict[str, Any]:
        """Post a single tweet with proper error handling"""
        try:
            # Sanitize and validate
            text = self._sanitize_tweet_text(text)
            is_valid, error_msg = self._validate_tweet_text(text)
            
            if not is_valid:
                return {"success": False, "error": f"Validation error: {error_msg}"}
            
            # Prepare request payload
            payload = {"text": text}
            if reply_to_id:
                payload["reply"] = {"in_reply_to_tweet_id": reply_to_id}
            
            # Make API request
            response = self.session.post(
                f"{self.base_url}/tweets",
                json=payload,
                timeout=30
            )
            
            # Handle different response codes
            if response.status_code == 201:
                data = response.json()["data"]
                logger.info(f"Tweet posted successfully: {data['id']}")
                return {
                    "success": True,
                    "tweet_id": data["id"],
                    "text": data["text"]
                }
                
            elif response.status_code == 429:
                reset_time = response.headers.get("x-rate-limit-reset", "unknown")
                error_msg = f"Rate limit exceeded. Reset time: {reset_time}"
                logger.warning(error_msg)
                return {"success": False, "error": error_msg}
                
            elif response.status_code == 401:
                error_msg = "Authentication failed. Check bearer token."
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
            elif response.status_code == 400:
                error_data = response.json().get("errors", [])
                error_msg = f"Validation error: {error_data[0].get('message', 'Unknown error')}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
            else:
                error_msg = f"Unexpected error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except requests.exceptions.Timeout:
            error_msg = "Request timeout. Twitter API may be slow."
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
        except requests.exceptions.ConnectionError:
            error_msg = "Network connection error."
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def post_thread(self, thread: List[str]) -> Dict[str, Any]:
        """Post a thread of tweets with proper chaining"""
        if not thread:
            return {"success": False, "error": "Thread cannot be empty"}
        
        posted_tweets = []
        last_tweet_id = None
        
        for i, tweet_text in enumerate(thread):
            logger.info(f"Posting tweet {i+1}/{len(thread)}")
            
            # Post tweet (reply to previous if not first)
            result = self.post_tweet(tweet_text, reply_to_id=last_tweet_id)
            
            if result["success"]:
                posted_tweets.append(result)
                last_tweet_id = result["tweet_id"]
                
                # Rate limiting: wait between tweets
                if i < len(thread) - 1:  # Don't wait after last tweet
                    time.sleep(1)  # 1 second between tweets
            else:
                # Thread posting failed
                error_msg = f"Thread posting failed at tweet {i+1}: {result['error']}"
                logger.error(error_msg)
                
                if posted_tweets:
                    error_msg += f". Partial thread posted ({len(posted_tweets)} tweets)."
                
                return {
                    "success": False,
                    "error": error_msg,
                    "tweets": posted_tweets
                }
        
        logger.info(f"Thread posted successfully: {len(posted_tweets)} tweets")
        return {
            "success": True,
            "tweets": posted_tweets,
            "thread_length": len(posted_tweets)
        }


def run(thread: List[str]) -> Dict[str, Any]:
    """
    Post a Twitter thread using the v2 API
    
    Args:
        thread: List of tweet texts to post as a thread
        
    Returns:
        Dict with success status and results/errors
    """
    # Get bearer token from environment
    bearer_token = os.getenv("TWITTER_BEARER")
    
    if not bearer_token or not bearer_token.strip():
        error_msg = "TWITTER_BEARER environment variable not set or empty"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    try:
        # Initialize client and post thread
        client = TwitterV2Client(bearer_token.strip())
        result = client.post_thread(thread)
        
        # Log results
        if result["success"]:
            tweet_count = len(result["tweets"])
            logger.info(f"Successfully posted thread with {tweet_count} tweets")
            
            # Print results for backwards compatibility
            for i, tweet in enumerate(result["tweets"], 1):
                print(f"[tweet {i}] {tweet['text']} (ID: {tweet['tweet_id']})")
        else:
            logger.error(f"Thread posting failed: {result['error']}")
            print(f"[ERROR] {result['error']}")
        
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error in run(): {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
