#!/usr/bin/env python3

"""
Release automation script for Agentic Startup Studio.
Handles semantic versioning, changelog generation, and release packaging.
"""

import os
import sys
import subprocess
import json
import re
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ReleaseInfo:
    """Information about a release."""
    version: str
    previous_version: str
    changelog: str
    commit_hash: str
    build_artifacts: List[str]
    release_notes: str


class SemanticVersionManager:
    """Manages semantic versioning and version bumping."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def get_current_version(self) -> str:
        """Get current version from git tags."""
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip().lstrip('v')
        except subprocess.CalledProcessError:
            return "0.0.0"
    
    def parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse semantic version string."""
        version = version.lstrip('v')
        parts = version.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version}")
        return tuple(int(part) for part in parts)
    
    def bump_version(self, current_version: str, bump_type: str) -> str:
        """Bump version according to semantic versioning."""
        major, minor, patch = self.parse_version(current_version)
        
        if bump_type == "major":
            return f"{major + 1}.0.0"
        elif bump_type == "minor":
            return f"{major}.{minor + 1}.0"
        elif bump_type == "patch":
            return f"{major}.{minor}.{patch + 1}"
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
    
    def determine_bump_type(self, commits: List[str]) -> str:
        """Determine version bump type from commit messages."""
        has_breaking = False
        has_feature = False
        has_fix = False
        
        for commit in commits:
            commit_lower = commit.lower()
            
            # Check for breaking changes
            if "breaking change" in commit_lower or "!" in commit:
                has_breaking = True
            
            # Check for features
            if commit_lower.startswith(("feat:", "feature:")):
                has_feature = True
            
            # Check for fixes
            if commit_lower.startswith(("fix:", "bugfix:", "hotfix:")):
                has_fix = True
        
        if has_breaking:
            return "major"
        elif has_feature:
            return "minor"
        elif has_fix:
            return "patch"
        else:
            return "patch"  # Default to patch for other changes


class ChangelogGenerator:
    """Generates changelog from git commits."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def get_commits_since_tag(self, tag: str) -> List[Dict[str, str]]:
        """Get all commits since a specific tag."""
        try:
            result = subprocess.run(
                ["git", "log", f"{tag}..HEAD", "--pretty=format:%H|%s|%an|%ad", "--date=short"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        commits.append({
                            "hash": parts[0],
                            "message": parts[1],
                            "author": parts[2],
                            "date": parts[3],
                        })
            return commits
        except subprocess.CalledProcessError:
            return []
    
    def categorize_commits(self, commits: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """Categorize commits by type."""
        categories = {
            "breaking": [],
            "features": [],
            "fixes": [],
            "performance": [],
            "documentation": [],
            "refactor": [],
            "test": [],
            "chore": [],
            "other": [],
        }
        
        for commit in commits:
            message = commit["message"].lower()
            
            if "breaking change" in message or "!" in commit["message"]:
                categories["breaking"].append(commit)
            elif message.startswith(("feat:", "feature:")):
                categories["features"].append(commit)
            elif message.startswith(("fix:", "bugfix:", "hotfix:")):
                categories["fixes"].append(commit)
            elif message.startswith(("perf:", "performance:")):
                categories["performance"].append(commit)
            elif message.startswith(("docs:", "doc:")):
                categories["documentation"].append(commit)
            elif message.startswith(("refactor:", "style:")):
                categories["refactor"].append(commit)
            elif message.startswith(("test:", "tests:")):
                categories["test"].append(commit)
            elif message.startswith(("chore:", "build:", "ci:")):
                categories["chore"].append(commit)
            else:
                categories["other"].append(commit)
        
        return categories
    
    def generate_changelog_section(self, version: str, categorized_commits: Dict[str, List[Dict[str, str]]]) -> str:
        """Generate changelog section for a version."""
        date = datetime.now().strftime("%Y-%m-%d")
        changelog = f"## [{version}] - {date}\n\n"
        
        sections = [
            ("breaking", "💥 BREAKING CHANGES"),
            ("features", "✨ Features"),
            ("fixes", "🐛 Bug Fixes"),
            ("performance", "⚡ Performance Improvements"),
            ("documentation", "📚 Documentation"),
            ("refactor", "♻️ Code Refactoring"),
            ("test", "✅ Tests"),
            ("chore", "🔧 Chore"),
        ]
        
        for category, title in sections:
            commits = categorized_commits.get(category, [])
            if commits:
                changelog += f"### {title}\n\n"
                for commit in commits:
                    # Clean up commit message
                    message = commit["message"]
                    # Remove conventional commit prefixes
                    message = re.sub(r'^(feat|fix|docs|style|refactor|perf|test|chore|build|ci)(\(.+\))?!?:\s*', '', message)
                    changelog += f"- {message} ([{commit['hash'][:8]}](../../commit/{commit['hash']}))\n"
                changelog += "\n"
        
        # Add other commits if any
        other_commits = categorized_commits.get("other", [])
        if other_commits:
            changelog += "### Other Changes\n\n"
            for commit in other_commits:
                changelog += f"- {commit['message']} ([{commit['hash'][:8]}](../../commit/{commit['hash']}))\n"
            changelog += "\n"
        
        return changelog
    
    def update_changelog_file(self, new_section: str) -> None:
        """Update the CHANGELOG.md file with new section."""
        changelog_file = self.project_root / "CHANGELOG.md"
        
        if changelog_file.exists():
            with open(changelog_file, 'r') as f:
                existing_content = f.read()
        else:
            existing_content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\nThe format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),\nand this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).\n\n"
        
        # Insert new section after the header
        lines = existing_content.split('\n')
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith('## [') or i == len(lines) - 1:
                header_end = i
                break
        
        new_lines = lines[:header_end] + new_section.split('\n') + lines[header_end:]
        
        with open(changelog_file, 'w') as f:
            f.write('\n'.join(new_lines))


class ReleaseManager:
    """Manages the complete release process."""
    
    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.version_manager = SemanticVersionManager(project_root)
        self.changelog_generator = ChangelogGenerator(project_root)
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        if level == "DEBUG" and not self.verbose:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "🚀" if level == "INFO" else "🐛" if level == "DEBUG" else "⚠️" if level == "WARN" else "❌"
        print(f"{prefix} [{timestamp}] {message}")
    
    def run_command(self, cmd: List[str], description: str) -> subprocess.CompletedProcess:
        """Run a command and handle errors."""
        self.log(f"Running: {description}")
        if self.verbose:
            self.log(f"Command: {' '.join(cmd)}", "DEBUG")
            
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            self.log(f"✅ {description} completed successfully")
            return result
        except subprocess.CalledProcessError as e:
            self.log(f"❌ {description} failed: {e}", "ERROR")
            if e.stdout:
                self.log(f"STDOUT: {e.stdout}", "DEBUG")
            if e.stderr:
                self.log(f"STDERR: {e.stderr}", "DEBUG")
            raise
    
    def validate_repository_state(self) -> bool:
        """Validate that the repository is in a good state for release."""
        self.log("Validating repository state...")
        
        # Check if we're on main/master branch
        result = self.run_command(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            "Getting current branch"
        )
        current_branch = result.stdout.strip()
        
        if current_branch not in ["main", "master"]:
            self.log(f"Not on main/master branch (currently on {current_branch})", "ERROR")
            return False
        
        # Check for uncommitted changes
        result = self.run_command(
            ["git", "status", "--porcelain"],
            "Checking for uncommitted changes"
        )
        
        if result.stdout.strip():
            self.log("Repository has uncommitted changes", "ERROR")
            return False
        
        # Pull latest changes
        try:
            self.run_command(
                ["git", "pull", "origin", current_branch],
                "Pulling latest changes"
            )
        except subprocess.CalledProcessError:
            self.log("Failed to pull latest changes", "WARN")
        
        return True
    
    def run_pre_release_tests(self) -> bool:
        """Run comprehensive tests before release."""
        self.log("Running pre-release tests...")
        
        try:
            self.run_command(
                ["python", "scripts/run_comprehensive_tests.py", "--types", "lint", "unit", "integration"],
                "Pre-release test validation"
            )
            return True
        except subprocess.CalledProcessError:
            self.log("Pre-release tests failed", "ERROR")
            return False
    
    def create_release(self, bump_type: Optional[str] = None, version: Optional[str] = None) -> ReleaseInfo:
        """Create a new release."""
        self.log("Creating new release...")
        
        # Get current version
        current_version = self.version_manager.get_current_version()
        self.log(f"Current version: {current_version}")
        
        # Determine new version
        if version:
            new_version = version.lstrip('v')
        else:
            # Get commits since last tag
            commits = self.changelog_generator.get_commits_since_tag(f"v{current_version}")
            commit_messages = [commit["message"] for commit in commits]
            
            if not bump_type:
                bump_type = self.version_manager.determine_bump_type(commit_messages)
            
            new_version = self.version_manager.bump_version(current_version, bump_type)
        
        self.log(f"New version: {new_version}")
        
        # Generate changelog
        commits = self.changelog_generator.get_commits_since_tag(f"v{current_version}")
        categorized_commits = self.changelog_generator.categorize_commits(commits)
        changelog_section = self.changelog_generator.generate_changelog_section(new_version, categorized_commits)
        
        # Update CHANGELOG.md
        self.changelog_generator.update_changelog_file(changelog_section)
        
        # Update version in pyproject.toml if it exists
        self._update_project_version(new_version)
        
        # Commit changes
        self.run_command(
            ["git", "add", "CHANGELOG.md", "pyproject.toml"],
            "Adding release files"
        )
        
        self.run_command(
            ["git", "commit", "-m", f"chore: release v{new_version}"],
            "Committing release changes"
        )
        
        # Create tag
        self.run_command(
            ["git", "tag", "-a", f"v{new_version}", "-m", f"Release v{new_version}"],
            "Creating release tag"
        )
        
        # Get commit hash
        result = self.run_command(
            ["git", "rev-parse", "HEAD"],
            "Getting commit hash"
        )
        commit_hash = result.stdout.strip()
        
        return ReleaseInfo(
            version=new_version,
            previous_version=current_version,
            changelog=changelog_section,
            commit_hash=commit_hash,
            build_artifacts=[],
            release_notes=self._generate_release_notes(new_version, categorized_commits)
        )
    
    def _update_project_version(self, version: str) -> None:
        """Update version in pyproject.toml."""
        pyproject_file = self.project_root / "pyproject.toml"
        
        if pyproject_file.exists():
            with open(pyproject_file, 'r') as f:
                content = f.read()
            
            # Update version in pyproject.toml
            content = re.sub(
                r'version\s*=\s*["\'][\d.]+["\']',
                f'version = "{version}"',
                content
            )
            
            with open(pyproject_file, 'w') as f:
                f.write(content)
            
            self.log(f"Updated version in pyproject.toml to {version}")
    
    def _generate_release_notes(self, version: str, categorized_commits: Dict[str, List[Dict[str, str]]]) -> str:
        """Generate release notes."""
        notes = f"# Release v{version}\n\n"
        
        # Summary
        total_commits = sum(len(commits) for commits in categorized_commits.values())
        notes += f"This release includes {total_commits} commits with the following changes:\n\n"
        
        # Highlights
        if categorized_commits.get("breaking"):
            notes += "## ⚠️ Breaking Changes\n\n"
            notes += "This release contains breaking changes. Please review the changelog carefully before upgrading.\n\n"
        
        if categorized_commits.get("features"):
            notes += "## ✨ New Features\n\n"
            for commit in categorized_commits["features"][:5]:  # Top 5 features
                message = re.sub(r'^feat(\(.+\))?!?:\s*', '', commit["message"])
                notes += f"- {message}\n"
            notes += "\n"
        
        if categorized_commits.get("fixes"):
            notes += "## 🐛 Bug Fixes\n\n"
            for commit in categorized_commits["fixes"][:5]:  # Top 5 fixes
                message = re.sub(r'^fix(\(.+\))?!?:\s*', '', commit["message"])
                notes += f"- {message}\n"
            notes += "\n"
        
        # Installation instructions
        notes += "## Installation\n\n"
        notes += "```bash\n"
        notes += f"# Docker\n"
        notes += f"docker pull agentic-startup-studio:{version}\n\n"
        notes += f"# Git\n"
        notes += f"git checkout v{version}\n"
        notes += "```\n\n"
        
        # Links
        notes += "## Links\n\n"
        notes += f"- [Full Changelog](CHANGELOG.md)\n"
        notes += f"- [Documentation](docs/)\n"
        notes += f"- [Issues](https://github.com/terragonlabs/agentic-startup-studio/issues)\n"
        
        return notes
    
    def push_release(self, push_tags: bool = True) -> None:
        """Push release to remote repository."""
        self.log("Pushing release to remote...")
        
        # Push commits
        self.run_command(
            ["git", "push", "origin", "HEAD"],
            "Pushing commits"
        )
        
        # Push tags
        if push_tags:
            self.run_command(
                ["git", "push", "origin", "--tags"],
                "Pushing tags"
            )
    
    def build_release_artifacts(self, release_info: ReleaseInfo) -> List[str]:
        """Build release artifacts."""
        self.log("Building release artifacts...")
        
        # Build Docker image
        try:
            self.run_command(
                ["python", "scripts/build.py", "--target", "production"],
                "Building production Docker image"
            )
            release_info.build_artifacts.append(f"Docker image: agentic-startup-studio:{release_info.version}")
        except subprocess.CalledProcessError:
            self.log("Failed to build Docker image", "WARN")
        
        return release_info.build_artifacts
    
    def full_release(self, bump_type: Optional[str] = None, version: Optional[str] = None,
                     push: bool = True, skip_tests: bool = False) -> bool:
        """Run complete release process."""
        self.log("🚀 Starting full release process...")
        
        try:
            # 1. Validate repository state
            if not self.validate_repository_state():
                return False
            
            # 2. Run pre-release tests
            if not skip_tests and not self.run_pre_release_tests():
                return False
            
            # 3. Create release
            release_info = self.create_release(bump_type, version)
            
            # 4. Build artifacts
            self.build_release_artifacts(release_info)
            
            # 5. Push to remote
            if push:
                self.push_release()
            
            # 6. Display summary
            self.log("🎉 Release completed successfully!")
            self.log(f"Version: {release_info.version}")
            self.log(f"Previous version: {release_info.previous_version}")
            self.log(f"Commit: {release_info.commit_hash}")
            
            if release_info.build_artifacts:
                self.log("Build artifacts:")
                for artifact in release_info.build_artifacts:
                    self.log(f"  - {artifact}")
            
            return True
            
        except Exception as e:
            self.log(f"Release failed: {e}", "ERROR")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Release Agentic Startup Studio")
    parser.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        help="Version bump type (auto-detected if not specified)"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Explicit version to release (overrides bump type)"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push to remote repository"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pre-release tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    release_manager = ReleaseManager(project_root, verbose=args.verbose)
    
    success = release_manager.full_release(
        bump_type=args.bump,
        version=args.version,
        push=not args.no_push,
        skip_tests=args.skip_tests
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()