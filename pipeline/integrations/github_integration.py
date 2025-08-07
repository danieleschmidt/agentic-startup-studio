"""
GitHub Integration Service for Agentic Startup Studio.

Provides comprehensive GitHub API integration for:
- Repository management and creation
- Code analysis and metrics collection
- Issue and PR management
- Release automation
- Deployment status tracking
"""

import asyncio
import base64
import logging
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin

import aiohttp

from pipeline.config.settings import get_settings
from pipeline.infrastructure.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class GitHubAPIError(Exception):
    """Base exception for GitHub API errors."""
    pass


class GitHubRateLimitError(GitHubAPIError):
    """Exception raised when GitHub API rate limit is exceeded."""
    pass


class GitHubAuthenticationError(GitHubAPIError):
    """Exception raised for GitHub authentication failures."""
    pass


class GitHubRepository:
    """Represents a GitHub repository with metadata."""

    def __init__(self, data: dict[str, Any]):
        self.id = data.get('id')
        self.name = data.get('name')
        self.full_name = data.get('full_name')
        self.description = data.get('description')
        self.private = data.get('private', False)
        self.html_url = data.get('html_url')
        self.clone_url = data.get('clone_url')
        self.ssh_url = data.get('ssh_url')
        self.created_at = data.get('created_at')
        self.updated_at = data.get('updated_at')
        self.pushed_at = data.get('pushed_at')
        self.size = data.get('size', 0)
        self.stargazers_count = data.get('stargazers_count', 0)
        self.watchers_count = data.get('watchers_count', 0)
        self.forks_count = data.get('forks_count', 0)
        self.open_issues_count = data.get('open_issues_count', 0)
        self.language = data.get('language')
        self.topics = data.get('topics', [])
        self.license = data.get('license', {}).get('name') if data.get('license') else None

    def to_dict(self) -> dict[str, Any]:
        """Convert repository to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'full_name': self.full_name,
            'description': self.description,
            'private': self.private,
            'html_url': self.html_url,
            'clone_url': self.clone_url,
            'ssh_url': self.ssh_url,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'pushed_at': self.pushed_at,
            'size': self.size,
            'stargazers_count': self.stargazers_count,
            'watchers_count': self.watchers_count,
            'forks_count': self.forks_count,
            'open_issues_count': self.open_issues_count,
            'language': self.language,
            'topics': self.topics,
            'license': self.license
        }


class GitHubIntegration:
    """
    Comprehensive GitHub API integration service.
    
    Features:
    - Repository management and creation
    - Code analysis and metrics
    - Issue and PR operations
    - Release automation
    - Webhook management
    - Rate limiting and error handling
    """

    def __init__(self, token: str = None):
        self.settings = get_settings()
        self.token = token or getattr(self.settings, 'github_token', None)
        self.base_url = "https://api.github.com"

        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout_seconds=30,
            recovery_timeout=60
        )

        # Rate limiting tracking
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = None

        if not self.token:
            logger.warning("GitHub token not provided - some features will be limited")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] = None,
        params: dict[str, Any] = None,
        headers: dict[str, str] = None
    ) -> dict[str, Any]:
        """Make authenticated request to GitHub API with error handling."""

        url = urljoin(self.base_url, endpoint.lstrip('/'))

        # Prepare headers
        request_headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Agentic-Startup-Studio/2.0.0'
        }

        if self.token:
            request_headers['Authorization'] = f'token {self.token}'

        if headers:
            request_headers.update(headers)

        # Check rate limiting
        if self.rate_limit_remaining < 10:
            if self.rate_limit_reset:
                wait_time = self.rate_limit_reset - datetime.now(UTC).timestamp()
                if wait_time > 0:
                    logger.warning(f"Rate limit approaching, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)

        try:
            async with self.circuit_breaker:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=method,
                        url=url,
                        json=data,
                        params=params,
                        headers=request_headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:

                        # Update rate limiting info
                        self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                        if 'X-RateLimit-Reset' in response.headers:
                            self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])

                        if response.status == 401:
                            raise GitHubAuthenticationError("GitHub authentication failed")
                        if response.status == 403:
                            if 'rate limit' in (await response.text()).lower():
                                raise GitHubRateLimitError("GitHub API rate limit exceeded")
                            raise GitHubAPIError(f"GitHub API access forbidden: {response.status}")
                        if response.status >= 400:
                            error_text = await response.text()
                            raise GitHubAPIError(f"GitHub API error {response.status}: {error_text}")

                        if response.content_type == 'application/json':
                            return await response.json()
                        return {'content': await response.text()}

        except aiohttp.ClientError as e:
            logger.error(f"GitHub API request failed: {e}")
            raise GitHubAPIError(f"Request failed: {e}")

    async def get_rate_limit_status(self) -> dict[str, Any]:
        """Get current rate limit status."""
        try:
            result = await self._make_request('GET', '/rate_limit')
            return result
        except Exception as e:
            logger.error(f"Failed to get rate limit status: {e}")
            return {
                'rate': {
                    'remaining': self.rate_limit_remaining,
                    'reset': self.rate_limit_reset
                }
            }

    async def get_user_info(self, username: str = None) -> dict[str, Any]:
        """Get GitHub user information."""
        endpoint = f'/users/{username}' if username else '/user'
        return await self._make_request('GET', endpoint)

    async def create_repository(
        self,
        name: str,
        description: str = None,
        private: bool = True,
        auto_init: bool = True,
        gitignore_template: str = "Python",
        license_template: str = "mit",
        organization: str = None
    ) -> GitHubRepository:
        """Create a new GitHub repository."""

        data = {
            'name': name,
            'description': description or f"Repository for {name} startup idea",
            'private': private,
            'auto_init': auto_init,
            'gitignore_template': gitignore_template,
            'license_template': license_template
        }

        endpoint = f'/orgs/{organization}/repos' if organization else '/user/repos'

        try:
            result = await self._make_request('POST', endpoint, data=data)
            repo = GitHubRepository(result)

            logger.info(f"Created GitHub repository: {repo.full_name}")
            return repo

        except GitHubAPIError as e:
            logger.error(f"Failed to create repository {name}: {e}")
            raise

    async def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        """Get repository information."""
        endpoint = f'/repos/{owner}/{repo}'
        result = await self._make_request('GET', endpoint)
        return GitHubRepository(result)

    async def list_repositories(
        self,
        organization: str = None,
        user: str = None,
        type_filter: str = "all",
        sort: str = "updated",
        per_page: int = 30
    ) -> list[GitHubRepository]:
        """List repositories for user or organization."""

        if organization:
            endpoint = f'/orgs/{organization}/repos'
        elif user:
            endpoint = f'/users/{user}/repos'
        else:
            endpoint = '/user/repos'

        params = {
            'type': type_filter,
            'sort': sort,
            'per_page': per_page
        }

        result = await self._make_request('GET', endpoint, params=params)
        return [GitHubRepository(repo_data) for repo_data in result]

    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str = None,
        labels: list[str] = None,
        assignees: list[str] = None
    ) -> dict[str, Any]:
        """Create a new issue in repository."""

        data = {
            'title': title,
            'body': body or ""
        }

        if labels:
            data['labels'] = labels
        if assignees:
            data['assignees'] = assignees

        endpoint = f'/repos/{owner}/{repo}/issues'
        return await self._make_request('POST', endpoint, data=data)

    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: str = None,
        draft: bool = False
    ) -> dict[str, Any]:
        """Create a new pull request."""

        data = {
            'title': title,
            'head': head,
            'base': base,
            'body': body or "",
            'draft': draft
        }

        endpoint = f'/repos/{owner}/{repo}/pulls'
        return await self._make_request('POST', endpoint, data=data)

    async def create_release(
        self,
        owner: str,
        repo: str,
        tag_name: str,
        name: str = None,
        body: str = None,
        draft: bool = False,
        prerelease: bool = False
    ) -> dict[str, Any]:
        """Create a new release."""

        data = {
            'tag_name': tag_name,
            'name': name or tag_name,
            'body': body or "",
            'draft': draft,
            'prerelease': prerelease
        }

        endpoint = f'/repos/{owner}/{repo}/releases'
        return await self._make_request('POST', endpoint, data=data)

    async def get_repository_metrics(self, owner: str, repo: str) -> dict[str, Any]:
        """Get comprehensive repository metrics."""

        try:
            # Get basic repository info
            repo_info = await self.get_repository(owner, repo)

            # Get additional metrics
            endpoints = {
                'commits': f'/repos/{owner}/{repo}/commits',
                'contributors': f'/repos/{owner}/{repo}/contributors',
                'languages': f'/repos/{owner}/{repo}/languages',
                'topics': f'/repos/{owner}/{repo}/topics',
                'releases': f'/repos/{owner}/{repo}/releases'
            }

            metrics = {'repository': repo_info.to_dict()}

            for metric_name, endpoint in endpoints.items():
                try:
                    params = {'per_page': 100} if metric_name in ['commits', 'contributors', 'releases'] else None
                    result = await self._make_request('GET', endpoint, params=params)
                    metrics[metric_name] = result
                except Exception as e:
                    logger.warning(f"Failed to get {metric_name} for {owner}/{repo}: {e}")
                    metrics[metric_name] = []

            # Calculate derived metrics
            metrics['health_score'] = self._calculate_repository_health_score(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Failed to get repository metrics for {owner}/{repo}: {e}")
            raise

    def _calculate_repository_health_score(self, metrics: dict[str, Any]) -> float:
        """Calculate a health score for the repository based on various factors."""

        repo = metrics['repository']
        commits = metrics.get('commits', [])
        contributors = metrics.get('contributors', [])
        releases = metrics.get('releases', [])

        score = 0.0

        # Activity score (0-30 points)
        if commits:
            recent_commits = len([c for c in commits[:10]])  # Last 10 commits
            score += min(recent_commits * 3, 30)

        # Community score (0-25 points)
        score += min(repo['stargazers_count'] * 0.1, 15)
        score += min(len(contributors) * 2, 10)

        # Maintenance score (0-25 points)
        if releases:
            score += min(len(releases), 10)
        if repo['open_issues_count'] < 20:
            score += 15
        elif repo['open_issues_count'] < 50:
            score += 10
        elif repo['open_issues_count'] < 100:
            score += 5

        # Documentation score (0-20 points)
        if repo['description']:
            score += 10
        if 'readme' in str(repo).lower():
            score += 10

        return min(score, 100.0)

    async def upload_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: str = "main",
        encoding: str = "utf-8"
    ) -> dict[str, Any]:
        """Upload a file to repository."""

        # Encode content
        if encoding == "utf-8":
            encoded_content = base64.b64encode(content.encode('utf-8')).decode('ascii')
        else:
            encoded_content = base64.b64encode(content).decode('ascii')

        data = {
            'message': message,
            'content': encoded_content,
            'branch': branch
        }

        endpoint = f'/repos/{owner}/{repo}/contents/{path}'
        return await self._make_request('PUT', endpoint, data=data)

    async def setup_webhook(
        self,
        owner: str,
        repo: str,
        url: str,
        events: list[str] = None,
        secret: str = None
    ) -> dict[str, Any]:
        """Set up a webhook for repository events."""

        if events is None:
            events = ['push', 'pull_request', 'issues', 'release']

        config = {
            'url': url,
            'content_type': 'json'
        }

        if secret:
            config['secret'] = secret

        data = {
            'name': 'web',
            'active': True,
            'events': events,
            'config': config
        }

        endpoint = f'/repos/{owner}/{repo}/hooks'
        return await self._make_request('POST', endpoint, data=data)

    async def search_repositories(
        self,
        query: str,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 30
    ) -> dict[str, Any]:
        """Search for repositories on GitHub."""

        params = {
            'q': query,
            'sort': sort,
            'order': order,
            'per_page': per_page
        }

        endpoint = '/search/repositories'
        result = await self._make_request('GET', endpoint, params=params)

        # Convert items to GitHubRepository objects
        if 'items' in result:
            result['repositories'] = [GitHubRepository(repo_data) for repo_data in result['items']]

        return result

    async def analyze_competition(self, keywords: list[str]) -> dict[str, Any]:
        """Analyze competitive landscape on GitHub based on keywords."""

        try:
            # Search for repositories with relevant keywords
            query = " OR ".join(keywords)
            search_results = await self.search_repositories(
                query=query,
                sort="stars",
                per_page=50
            )

            repositories = search_results.get('repositories', [])

            # Analyze the competitive landscape
            analysis = {
                'total_repositories': len(repositories),
                'top_repositories': [repo.to_dict() for repo in repositories[:10]],
                'language_distribution': {},
                'average_stars': 0,
                'total_stars': 0,
                'market_saturation_score': 0.0
            }

            if repositories:
                # Calculate language distribution
                for repo in repositories:
                    if repo.language:
                        analysis['language_distribution'][repo.language] = \
                            analysis['language_distribution'].get(repo.language, 0) + 1

                # Calculate average stars
                total_stars = sum(repo.stargazers_count for repo in repositories)
                analysis['total_stars'] = total_stars
                analysis['average_stars'] = total_stars / len(repositories)

                # Calculate market saturation score (0-100)
                # Higher score means more saturated market
                max_stars = max(repo.stargazers_count for repo in repositories)
                if max_stars > 1000:
                    analysis['market_saturation_score'] = min(total_stars / 10000 * 100, 100)
                else:
                    analysis['market_saturation_score'] = min(total_stars / 1000 * 100, 100)

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze competition for keywords {keywords}: {e}")
            raise

    async def get_trending_repositories(
        self,
        language: str = None,
        since: str = "daily"
    ) -> list[GitHubRepository]:
        """Get trending repositories (requires external trending API or scraping)."""

        # Note: GitHub doesn't have an official trending API
        # This is a simplified implementation using search with recent filters

        try:
            # Create a query for recent popular repositories
            date_filter = "created:>2023-01-01" if since == "yearly" else "created:>2024-01-01"
            query = f"stars:>10 {date_filter}"

            if language:
                query += f" language:{language}"

            result = await self.search_repositories(
                query=query,
                sort="stars",
                order="desc",
                per_page=30
            )

            return result.get('repositories', [])

        except Exception as e:
            logger.error(f"Failed to get trending repositories: {e}")
            return []


# Usage example and testing utilities

async def test_github_integration():
    """Test function for GitHub integration."""

    github = GitHubIntegration()

    try:
        # Test rate limit status
        rate_limit = await github.get_rate_limit_status()
        print(f"Rate limit: {rate_limit}")

        # Test user info
        user_info = await github.get_user_info()
        print(f"User: {user_info.get('login', 'Unknown')}")

        # Test search
        search_results = await github.search_repositories("machine learning")
        print(f"Found {len(search_results.get('repositories', []))} ML repositories")

        # Test competitive analysis
        competition = await github.analyze_competition(["fintech", "payments", "banking"])
        print(f"Fintech competition: {competition['total_repositories']} repositories")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_github_integration())
