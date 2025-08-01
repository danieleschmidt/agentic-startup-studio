#!/bin/bash

# =============================================================================
# Release Automation Script for Agentic Startup Studio
# Handles semantic versioning, changelog generation, and release publishing
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage information
usage() {
    cat << EOF
Release Automation Script for Agentic Startup Studio

Usage: $0 [OPTIONS] VERSION_TYPE

VERSION_TYPE:
    major           Increment major version (x.0.0)
    minor           Increment minor version (x.y.0)
    patch           Increment patch version (x.y.z)
    prerelease      Create prerelease version (x.y.z-alpha.n)
    custom VERSION  Use custom version number

OPTIONS:
    --dry-run       Show what would be done without making changes
    --skip-tests    Skip running tests before release
    --skip-build    Skip building Docker image
    --push          Push release to remote repository
    --publish       Publish packages to registry
    -h, --help      Display this help message

EXAMPLES:
    # Create patch release with dry run
    $0 --dry-run patch

    # Create minor release and push to remote
    $0 --push minor

    # Create custom version
    $0 custom v2.1.0-beta.1

EOF
}

# Parse command line arguments
VERSION_TYPE=""
CUSTOM_VERSION=""
DRY_RUN=false
SKIP_TESTS=false
SKIP_BUILD=false
PUSH=false
PUBLISH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --publish)
            PUBLISH=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        major|minor|patch|prerelease)
            VERSION_TYPE="$1"
            shift
            ;;
        custom)
            VERSION_TYPE="custom"
            CUSTOM_VERSION="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ -z "${VERSION_TYPE}" ]]; then
    log_error "Version type is required"
    usage
    exit 1
fi

if [[ "${VERSION_TYPE}" == "custom" ]] && [[ -z "${CUSTOM_VERSION}" ]]; then
    log_error "Custom version is required when using 'custom' type"
    usage
    exit 1
fi

# Get current version from pyproject.toml
get_current_version() {
    if [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
        grep '^version = ' "${PROJECT_ROOT}/pyproject.toml" | cut -d'"' -f2
    else
        echo "0.0.0"
    fi
}

# Calculate next version
calculate_next_version() {
    local current_version="$1"
    local version_type="$2"
    
    # Remove 'v' prefix if present
    current_version="${current_version#v}"
    
    # Parse version components
    local major minor patch prerelease
    if [[ "${current_version}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-.*)?$ ]]; then
        major="${BASH_REMATCH[1]}"
        minor="${BASH_REMATCH[2]}"
        patch="${BASH_REMATCH[3]}"
        prerelease="${BASH_REMATCH[4]}"
    else
        log_error "Invalid version format: ${current_version}"
        exit 1
    fi
    
    case "${version_type}" in
        major)
            echo "$((major + 1)).0.0"
            ;;
        minor)
            echo "${major}.$((minor + 1)).0"
            ;;
        patch)
            echo "${major}.${minor}.$((patch + 1))"
            ;;
        prerelease)
            if [[ -n "${prerelease}" ]]; then
                # Increment existing prerelease
                if [[ "${prerelease}" =~ ^-(.+)\.([0-9]+)$ ]]; then
                    local pre_type="${BASH_REMATCH[1]}"
                    local pre_num="${BASH_REMATCH[2]}"
                    echo "${major}.${minor}.${patch}-${pre_type}.$((pre_num + 1))"
                else
                    echo "${major}.${minor}.${patch}${prerelease}.1"
                fi
            else
                echo "${major}.${minor}.$((patch + 1))-alpha.1"
            fi
            ;;
        custom)
            echo "${CUSTOM_VERSION#v}"
            ;;
        *)
            log_error "Unknown version type: ${version_type}"
            exit 1
            ;;
    esac
}

# Validate working directory
validate_working_directory() {
    log_info "Validating working directory..."
    
    cd "${PROJECT_ROOT}"
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Check for uncommitted changes
    if [[ -n "$(git status --porcelain)" ]]; then
        log_error "Working directory has uncommitted changes"
        log_info "Please commit or stash changes before creating a release"
        exit 1
    fi
    
    # Check if we're on main branch
    local current_branch
    current_branch="$(git rev-parse --abbrev-ref HEAD)"
    if [[ "${current_branch}" != "main" ]] && [[ "${current_branch}" != "master" ]]; then
        log_warning "Not on main/master branch (current: ${current_branch})"
        if [[ "${DRY_RUN}" == false ]]; then
            read -p "Continue anyway? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    log_success "Working directory validation passed"
}

# Run tests
run_tests() {
    if [[ "${SKIP_TESTS}" == true ]]; then
        log_info "Skipping tests (--skip-tests flag)"
        return
    fi
    
    log_info "Running test suite..."
    
    cd "${PROJECT_ROOT}"
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would run pytest with coverage"
        return
    fi
    
    # Run tests with coverage
    if ! make ci-test; then
        log_error "Tests failed. Cannot proceed with release."
        exit 1
    fi
    
    log_success "All tests passed"
}

# Run quality checks
run_quality_checks() {
    log_info "Running quality checks..."
    
    cd "${PROJECT_ROOT}"
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would run quality checks (lint, type-check, security)"
        return
    fi
    
    # Run linting and security checks
    if ! make ci-quality; then
        log_error "Quality checks failed. Cannot proceed with release."
        exit 1
    fi
    
    if ! make ci-security; then
        log_error "Security checks failed. Cannot proceed with release."
        exit 1
    fi
    
    log_success "Quality checks passed"
}

# Update version in files
update_version_files() {
    local new_version="$1"
    
    log_info "Updating version to ${new_version}..."
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would update version in pyproject.toml and other files"
        return
    fi
    
    # Update pyproject.toml
    if [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
        sed -i.bak "s/^version = .*/version = \"${new_version}\"/" "${PROJECT_ROOT}/pyproject.toml"
        rm "${PROJECT_ROOT}/pyproject.toml.bak"
    fi
    
    # Update __init__.py files if they exist
    find "${PROJECT_ROOT}" -name "__init__.py" -exec grep -l "__version__" {} \; | while read -r file; do
        sed -i.bak "s/__version__ = .*/__version__ = \"${new_version}\"/" "$file"
        rm "${file}.bak"
    done
    
    # Update Docker labels if Dockerfile exists
    if [[ -f "${PROJECT_ROOT}/Dockerfile" ]]; then
        sed -i.bak "s/ARG VERSION=.*/ARG VERSION=${new_version}/" "${PROJECT_ROOT}/Dockerfile"
        rm "${PROJECT_ROOT}/Dockerfile.bak" 2>/dev/null || true
    fi
    
    log_success "Version updated in project files"
}

# Generate changelog
generate_changelog() {
    local new_version="$1"
    local current_version="$2"
    
    log_info "Generating changelog for version ${new_version}..."
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would generate changelog entry"
        return
    fi
    
    local changelog_file="${PROJECT_ROOT}/CHANGELOG.md"
    local temp_file="${PROJECT_ROOT}/CHANGELOG.tmp"
    
    # Get commits since last version
    local commits
    if git rev-parse "v${current_version}" >/dev/null 2>&1; then
        commits=$(git log --oneline "v${current_version}"..HEAD --pretty=format:"- %s")
    else
        commits=$(git log --oneline --pretty=format:"- %s")
    fi
    
    # Create new changelog entry
    cat > "${temp_file}" << EOF
# Changelog

## [${new_version}] - $(date +%Y-%m-%d)

### Changes
${commits}

EOF
    
    # Append existing changelog if it exists
    if [[ -f "${changelog_file}" ]]; then
        # Skip the first line if it's just "# Changelog"
        tail -n +2 "${changelog_file}" >> "${temp_file}" 2>/dev/null || true
    fi
    
    mv "${temp_file}" "${changelog_file}"
    
    log_success "Changelog updated"
}

# Build Docker image
build_docker_image() {
    local new_version="$1"
    
    if [[ "${SKIP_BUILD}" == true ]]; then
        log_info "Skipping Docker build (--skip-build flag)"
        return
    fi
    
    log_info "Building Docker image for version ${new_version}..."
    
    cd "${PROJECT_ROOT}"
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would build Docker image with tag v${new_version}"
        return
    fi
    
    # Build Docker image using our build script
    if ! ./scripts/build.sh --tag "v${new_version}" --environment prod; then
        log_error "Docker build failed"
        exit 1
    fi
    
    log_success "Docker image built successfully"
}

# Create git tag and commit
create_git_release() {
    local new_version="$1"
    
    log_info "Creating git commit and tag for version ${new_version}..."
    
    cd "${PROJECT_ROOT}"
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would create commit and tag v${new_version}"
        return
    fi
    
    # Add all version-related changes
    git add pyproject.toml CHANGELOG.md
    find . -name "__init__.py" -exec git add {} \; 2>/dev/null || true
    
    # Create release commit
    git commit -m "chore: release v${new_version}

- Update version to ${new_version}
- Update changelog with latest changes
- Prepare for release

🤖 Generated with Claude Code (https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # Create annotated tag
    git tag -a "v${new_version}" -m "Release version ${new_version}

$(git log --oneline $(git describe --tags --abbrev=0 2>/dev/null || echo "HEAD~10")..HEAD --pretty=format:"- %s" | head -10)
"
    
    log_success "Git commit and tag created"
}

# Push release
push_release() {
    local new_version="$1"
    
    if [[ "${PUSH}" == false ]]; then
        log_info "Skipping push (use --push flag to push to remote)"
        return
    fi
    
    log_info "Pushing release to remote repository..."
    
    cd "${PROJECT_ROOT}"
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would push commit and tag v${new_version} to origin"
        return
    fi
    
    # Push commit and tag
    git push origin HEAD
    git push origin "v${new_version}"
    
    log_success "Release pushed to remote repository"
}

# Publish packages
publish_packages() {
    local new_version="$1"
    
    if [[ "${PUBLISH}" == false ]]; then
        log_info "Skipping package publishing (use --publish flag to publish)"
        return
    fi
    
    log_info "Publishing packages..."
    
    cd "${PROJECT_ROOT}"
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would build and publish Python packages"
        return
    fi
    
    # Build Python packages
    make build
    
    # Publish to PyPI (would require authentication setup)
    log_info "Package built. Manual PyPI publishing required with:"
    log_info "  twine upload dist/*"
    
    log_success "Packages prepared for publishing"
}

# Generate release summary
generate_release_summary() {
    local new_version="$1"
    local current_version="$2"
    
    log_info "Release Summary"
    log_info "==============="
    log_info "Previous version: ${current_version}"
    log_info "New version: ${new_version}"
    log_info "Release type: ${VERSION_TYPE}"
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_warning "This was a DRY RUN - no changes were made"
    else
        log_success "Release v${new_version} completed successfully!"
        
        log_info ""
        log_info "Next steps:"
        if [[ "${PUSH}" == false ]]; then
            log_info "  git push origin HEAD && git push origin v${new_version}"
        fi
        if [[ "${PUBLISH}" == false ]]; then
            log_info "  make build && twine upload dist/*"
        fi
        log_info "  Create GitHub release from tag v${new_version}"
    fi
}

# Main execution
main() {
    local current_version
    local new_version
    
    current_version="$(get_current_version)"
    
    if [[ "${VERSION_TYPE}" == "custom" ]]; then
        new_version="${CUSTOM_VERSION#v}"
    else
        new_version="$(calculate_next_version "${current_version}" "${VERSION_TYPE}")"
    fi
    
    log_info "Starting release process for Agentic Startup Studio"
    log_info "Current version: ${current_version}"
    log_info "Target version: ${new_version}"
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_warning "DRY RUN MODE - No changes will be made"
    fi
    
    validate_working_directory
    run_tests
    run_quality_checks
    update_version_files "${new_version}"
    generate_changelog "${new_version}" "${current_version}"
    build_docker_image "${new_version}"
    create_git_release "${new_version}"
    push_release "${new_version}"
    publish_packages "${new_version}"
    generate_release_summary "${new_version}" "${current_version}"
}

# Execute main function
main "$@"