#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) Generator

Generates comprehensive SBOM for the Agentic Startup Studio application
including all dependencies, licenses, and security information.
"""

import json
import subprocess
import sys
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import pkg_resources
import platform


class SBOMGenerator:
    """Generate Software Bill of Materials for the application."""
    
    def __init__(self, output_format: str = "spdx"):
        self.output_format = output_format
        self.project_root = Path(__file__).parent.parent
        self.sbom_data = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": "agentic-startup-studio-sbom",
            "documentNamespace": f"https://terragonlabs.com/spdx/agentic-startup-studio-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "creationInfo": {
                "created": datetime.now(timezone.utc).isoformat(),
                "creators": ["Tool: Terragon SBOM Generator"],
                "licenseListVersion": "3.20"
            },
            "packages": []
        }
    
    def get_git_info(self) -> Dict[str, str]:
        """Get Git repository information."""
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                cwd=self.project_root
            ).decode().strip()
            
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root
            ).decode().strip()
            
            return {
                "commit": commit_hash,
                "branch": branch
            }
        except subprocess.CalledProcessError:
            return {
                "commit": "unknown",
                "branch": "unknown"
            }
    
    def get_python_dependencies(self) -> List[Dict[str, Any]]:
        """Get Python package dependencies with version and license info."""
        dependencies = []
        
        # Get installed packages
        for dist in pkg_resources.working_set:
            package_info = {
                "SPDXID": f"SPDXRef-Package-{dist.project_name}",
                "name": dist.project_name,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "versionInfo": dist.version,
                "supplier": "NOASSERTION",
                "originator": "NOASSERTION",
                "homepage": "NOASSERTION",
                "sourceInfo": f"Python package {dist.project_name} version {dist.version}",
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION"
            }
            
            # Try to get more detailed package information
            try:
                metadata = dist.get_metadata('METADATA')
                if metadata:
                    # Parse metadata for license, homepage, etc.
                    for line in metadata.split('\n'):
                        if line.startswith('Home-page:'):
                            package_info["homepage"] = line.split(':', 1)[1].strip()
                        elif line.startswith('License:'):
                            license_info = line.split(':', 1)[1].strip()
                            if license_info and license_info != "UNKNOWN":
                                package_info["licenseDeclared"] = license_info
                                package_info["licenseConcluded"] = license_info
                        elif line.startswith('Author:'):
                            author = line.split(':', 1)[1].strip()
                            if author:
                                package_info["originator"] = f"Person: {author}"
            except Exception:
                pass
            
            dependencies.append(package_info)
        
        return dependencies
    
    def get_system_dependencies(self) -> List[Dict[str, Any]]:
        """Get system-level dependencies and runtime information."""
        system_deps = []
        
        # Operating system
        system_deps.append({
            "SPDXID": "SPDXRef-Package-OS",
            "name": f"{platform.system()}-{platform.release()}",
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "versionInfo": platform.version(),
            "supplier": "NOASSERTION",
            "sourceInfo": f"Operating System: {platform.platform()}",
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION"
        })
        
        # Python runtime
        system_deps.append({
            "SPDXID": "SPDXRef-Package-Python",
            "name": "Python",
            "downloadLocation": "https://www.python.org/downloads/",
            "filesAnalyzed": False,
            "versionInfo": platform.python_version(),
            "supplier": "Organization: Python Software Foundation",
            "homepage": "https://www.python.org/",
            "sourceInfo": f"Python runtime version {platform.python_version()}",
            "licenseConcluded": "Python-2.0",
            "licenseDeclared": "Python-2.0",
            "copyrightText": "Copyright (c) 2001-2024 Python Software Foundation"
        })
        
        return system_deps
    
    def get_application_info(self) -> Dict[str, Any]:
        """Get main application package information."""
        git_info = self.get_git_info()
        
        # Calculate source hash
        source_hash = self.calculate_source_hash()
        
        return {
            "SPDXID": "SPDXRef-Package-AgenticStartupStudio",
            "name": "agentic-startup-studio",
            "downloadLocation": "https://github.com/terragonlabs/agentic-startup-studio",
            "filesAnalyzed": True,
            "versionInfo": "2.0.0",
            "supplier": "Organization: Terragon Labs",
            "originator": "Organization: Terragon Labs",
            "homepage": "https://docs.terragonlabs.com/agentic-startup-studio",
            "sourceInfo": f"Git commit {git_info['commit']} on branch {git_info['branch']}",
            "licenseConcluded": "MIT",
            "licenseDeclared": "MIT",
            "copyrightText": "Copyright (c) 2025 Terragon Labs",
            "checksums": [{
                "algorithm": "SHA256",
                "checksumValue": source_hash
            }],
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "git",
                    "referenceLocator": f"https://github.com/terragonlabs/agentic-startup-studio.git@{git_info['commit']}"
                }
            ]
        }
    
    def calculate_source_hash(self) -> str:
        """Calculate SHA256 hash of source files."""
        hasher = hashlib.sha256()
        
        # Hash main source files
        source_patterns = [
            "**/*.py",
            "**/pyproject.toml", 
            "**/requirements.txt",
            "**/Dockerfile"
        ]
        
        files_to_hash = []
        for pattern in source_patterns:
            files_to_hash.extend(self.project_root.glob(pattern))
        
        # Sort for consistent hashing
        files_to_hash.sort(key=lambda x: str(x))
        
        for file_path in files_to_hash:
            if file_path.is_file():
                try:
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
                except Exception:
                    continue
        
        return hasher.hexdigest()
    
    def get_security_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Get known security vulnerabilities using safety check."""
        vulnerabilities = []
        
        try:
            # Run safety check
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    vulnerabilities.append({
                        "package": vuln.get("package"),
                        "vulnerability_id": vuln.get("vulnerability_id"),
                        "advisory": vuln.get("advisory"),
                        "severity": vuln.get("severity", "unknown")
                    })
        except Exception as e:
            print(f"Warning: Could not run security check: {e}")
        
        return vulnerabilities
    
    def generate_sbom(self) -> Dict[str, Any]:
        """Generate complete SBOM."""
        # Add main application
        self.sbom_data["packages"].append(self.get_application_info())
        
        # Add Python dependencies
        python_deps = self.get_python_dependencies()
        self.sbom_data["packages"].extend(python_deps)
        
        # Add system dependencies
        system_deps = self.get_system_dependencies()
        self.sbom_data["packages"].extend(system_deps)
        
        # Add security information
        vulnerabilities = self.get_security_vulnerabilities()
        if vulnerabilities:
            self.sbom_data["vulnerabilities"] = vulnerabilities
        
        # Add relationships
        self.sbom_data["relationships"] = []
        
        # Main package depends on all others
        for package in self.sbom_data["packages"][1:]:  # Skip main package
            self.sbom_data["relationships"].append({
                "spdxElementId": "SPDXRef-Package-AgenticStartupStudio",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": package["SPDXID"]
            })
        
        return self.sbom_data
    
    def save_sbom(self, output_path: Optional[Path] = None) -> Path:
        """Save SBOM to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = self.project_root / f"sbom-{timestamp}.spdx.json"
        
        sbom_data = self.generate_sbom()
        
        with open(output_path, 'w') as f:
            json.dump(sbom_data, f, indent=2, sort_keys=True)
        
        print(f"✅ SBOM generated: {output_path}")
        print(f"📦 Packages catalogued: {len(sbom_data['packages'])}")
        print(f"🔗 Relationships mapped: {len(sbom_data['relationships'])}")
        
        if 'vulnerabilities' in sbom_data:
            print(f"🚨 Security vulnerabilities found: {len(sbom_data['vulnerabilities'])}")
        
        return output_path
    
    def validate_sbom(self, sbom_path: Path) -> bool:
        """Validate generated SBOM."""
        try:
            with open(sbom_path) as f:
                sbom_data = json.load(f)
            
            # Basic validation
            required_fields = ["spdxVersion", "dataLicense", "SPDXID", "name"]
            for field in required_fields:
                if field not in sbom_data:
                    print(f"❌ Missing required field: {field}")
                    return False
            
            # Validate packages
            if not sbom_data.get("packages"):
                print("❌ No packages found in SBOM")
                return False
            
            for package in sbom_data["packages"]:
                if not package.get("SPDXID") or not package.get("name"):
                    print("❌ Package missing required fields")
                    return False
            
            print("✅ SBOM validation passed")
            return True
            
        except Exception as e:
            print(f"❌ SBOM validation failed: {e}")
            return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SBOM for Agentic Startup Studio")
    parser.add_argument("--output", "-o", type=Path, help="Output file path")
    parser.add_argument("--format", choices=["spdx", "cyclonedx"], default="spdx", 
                       help="SBOM format")
    parser.add_argument("--validate", action="store_true", help="Validate generated SBOM")
    
    args = parser.parse_args()
    
    generator = SBOMGenerator(output_format=args.format)
    
    try:
        output_path = generator.save_sbom(args.output)
        
        if args.validate:
            if not generator.validate_sbom(output_path):
                sys.exit(1)
        
        print(f"\n📋 SBOM Summary:")
        print(f"   Format: {args.format.upper()}")
        print(f"   Output: {output_path}")
        print(f"   Size: {output_path.stat().st_size:,} bytes")
        
    except Exception as e:
        print(f"❌ Error generating SBOM: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()