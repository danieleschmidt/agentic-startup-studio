#!/usr/bin/env python3
"""
Version update script for semantic-release integration.

This script updates the version in all relevant files when a new release is created.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


class VersionUpdater:
    """Handles version updates across multiple files."""
    
    def __init__(self, new_version: str):
        self.new_version = new_version
        self.files_to_update = [
            ("pyproject.toml", self._update_pyproject_toml),
            ("src/agentic_startup_studio/__init__.py", self._update_init_py),
            ("docs/api-documentation.md", self._update_api_docs),
            ("docker-compose.yml", self._update_docker_compose),
        ]
    
    def _update_pyproject_toml(self, content: str) -> str:
        """Update version in pyproject.toml."""
        pattern = r'version\s*=\s*"[^"]+\"'
        replacement = f'version = "{self.new_version}"'
        return re.sub(pattern, replacement, content)
    
    def _update_init_py(self, content: str) -> str:
        """Update version in __init__.py."""
        pattern = r'__version__\s*=\s*"[^"]+\"'
        replacement = f'__version__ = "{self.new_version}"'
        return re.sub(pattern, replacement, content)
    
    def _update_api_docs(self, content: str) -> str:
        """Update version references in API documentation."""
        pattern = r'version:\s*[^\n]+'
        replacement = f'version: {self.new_version}'
        return re.sub(pattern, replacement, content)
    
    def _update_docker_compose(self, content: str) -> str:
        """Update image tags in docker-compose.yml."""
        pattern = r'image:\s*agentic-startup-studio:[^\n]+'
        replacement = f'image: agentic-startup-studio:{self.new_version}'
        return re.sub(pattern, replacement, content)
    
    def update_all_files(self) -> List[Tuple[str, bool]]:
        """Update version in all configured files."""
        results = []
        
        for file_path, update_func in self.files_to_update:
            path = Path(file_path)
            
            if not path.exists():
                print(f"⚠️  File not found: {file_path}")
                results.append((file_path, False))
                continue
            
            try:
                # Read current content
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update content
                updated_content = update_func(content)
                
                # Write back if changed
                if updated_content != content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    print(f"✅ Updated version in {file_path}")
                    results.append((file_path, True))
                else:
                    print(f"📝 No changes needed in {file_path}")
                    results.append((file_path, True))
                    
            except Exception as e:
                print(f"❌ Error updating {file_path}: {str(e)}")
                results.append((file_path, False))
        
        return results
    
    def create_version_file(self) -> bool:
        """Create a VERSION file with the current version."""
        try:
            version_file = Path("VERSION")
            with open(version_file, 'w') as f:
                f.write(f"{self.new_version}\n")
            print(f"✅ Created VERSION file with {self.new_version}")
            return True
        except Exception as e:
            print(f"❌ Error creating VERSION file: {str(e)}")
            return False


def validate_version(version: str) -> bool:
    """Validate that the version follows semantic versioning."""
    pattern = r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?(?:\+[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?$'
    return bool(re.match(pattern, version))


def main():
    """Main function to update version across files."""
    parser = argparse.ArgumentParser(description="Update version across project files")
    parser.add_argument("version", help="New version to set (semver format)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without making changes")
    parser.add_argument("--create-version-file", action="store_true", help="Create a VERSION file")
    
    args = parser.parse_args()
    
    # Validate version format
    if not validate_version(args.version):
        print(f"❌ Invalid version format: {args.version}")
        print("Version must follow semantic versioning (e.g., 1.2.3, 1.2.3-beta.1)")
        sys.exit(1)
    
    print(f"🚀 Updating version to {args.version}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
    
    updater = VersionUpdater(args.version)
    
    if args.dry_run:
        print("\nFiles that would be updated:")
        for file_path, _ in updater.files_to_update:
            if Path(file_path).exists():
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (not found)")
    else:
        # Update all files
        results = updater.update_all_files()
        
        # Create VERSION file if requested
        if args.create_version_file:
            updater.create_version_file()
        
        # Summary
        successful_updates = sum(1 for _, success in results if success)
        total_files = len(results)
        
        print(f"\n📊 Summary:")
        print(f"   Files updated: {successful_updates}/{total_files}")
        
        if successful_updates == total_files:
            print(f"✅ All files updated successfully to version {args.version}")
            sys.exit(0)
        else:
            print(f"❌ Some files failed to update")
            sys.exit(1)


if __name__ == "__main__":
    main()