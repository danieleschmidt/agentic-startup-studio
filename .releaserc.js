module.exports = {
  branches: [
    'main',
    {
      name: 'dev',
      prerelease: 'beta'
    },
    {
      name: 'release/*',
      prerelease: 'rc'
    }
  ],
  plugins: [
    // Analyze commits to determine release type
    [
      '@semantic-release/commit-analyzer',
      {
        preset: 'conventionalcommits',
        releaseRules: [
          { type: 'feat', release: 'minor' },
          { type: 'fix', release: 'patch' },
          { type: 'perf', release: 'patch' },
          { type: 'docs', release: false },
          { type: 'style', release: false },
          { type: 'refactor', release: 'patch' },
          { type: 'test', release: false },
          { type: 'build', release: 'patch' },
          { type: 'ci', release: false },
          { type: 'chore', release: false },
          { type: 'revert', release: 'patch' },
          { scope: 'security', release: 'patch' },
          { breaking: true, release: 'major' }
        ],
        parserOpts: {
          noteKeywords: ['BREAKING CHANGE', 'BREAKING CHANGES']
        }
      }
    ],

    // Generate release notes
    [
      '@semantic-release/release-notes-generator',
      {
        preset: 'conventionalcommits',
        presetConfig: {
          types: [
            { type: 'feat', section: '🚀 Features' },
            { type: 'fix', section: '🐛 Bug Fixes' },
            { type: 'perf', section: '⚡ Performance Improvements' },
            { type: 'revert', section: '⏪ Reverts' },
            { type: 'docs', section: '📚 Documentation', hidden: false },
            { type: 'style', section: '💎 Styles', hidden: true },
            { type: 'chore', section: '🔧 Miscellaneous Chores', hidden: true },
            { type: 'refactor', section: '♻️ Code Refactoring' },
            { type: 'test', section: '🧪 Tests', hidden: true },
            { type: 'build', section: '🏗️ Build System' },
            { type: 'ci', section: '👷 CI/CD', hidden: true },
            { type: 'security', section: '🔒 Security' }
          ]
        },
        writerOpts: {
          commitsSort: ['subject', 'scope']
        }
      }
    ],

    // Update CHANGELOG.md
    [
      '@semantic-release/changelog',
      {
        changelogFile: 'CHANGELOG.md',
        changelogTitle: '# Changelog\n\nAll notable changes to the Agentic Startup Studio will be documented in this file.\n\nThe format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),\nand this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).'
      }
    ],

    // Execute shell commands for version updates
    [
      '@semantic-release/exec',
      {
        verifyReleaseCmd: 'echo "Verifying release ${nextRelease.version}"',
        generateNotesCmd: 'echo "Generating notes for ${nextRelease.version}"',
        prepareCmd: [
          // Update pyproject.toml version
          'sed -i "s/^version = .*/version = \\"${nextRelease.version}\\"/" pyproject.toml',
          // Update __init__.py version
          'echo "__version__ = \\"${nextRelease.version}\\"" > src/agentic_startup_studio/__init__.py',
          // Run tests to ensure version update doesn't break anything
          'python -m pytest tests/ -x --tb=short',
          // Update documentation version references
          'find docs/ -name "*.md" -exec sed -i "s/Version: [0-9]\\+\\.[0-9]\\+\\.[0-9]\\+/Version: ${nextRelease.version}/g" {} +'
        ].join(' && '),
        publishCmd: 'echo "Publishing version ${nextRelease.version}"',
        successCmd: [
          'echo "Successfully released ${nextRelease.version}"',
          // Optionally trigger deployment
          'echo "Triggering deployment pipeline..."'
        ].join(' && '),
        failCmd: 'echo "Release failed for version ${nextRelease.version}"'
      }
    ],

    // Commit version updates
    [
      '@semantic-release/git',
      {
        assets: [
          'CHANGELOG.md',
          'pyproject.toml',
          'src/agentic_startup_studio/__init__.py',
          'docs/**/*.md'
        ],
        message: 'chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}'
      }
    ],

    // Create GitHub release
    [
      '@semantic-release/github',
      {
        assets: [
          {
            path: 'dist/*.whl',
            label: 'Python Wheel Distribution'
          },
          {
            path: 'dist/*.tar.gz',
            label: 'Source Distribution'
          }
        ],
        discussionCategoryName: 'Releases',
        addReleases: 'bottom',
        labels: false,
        assignees: ['terragon-labs/core-team'],
        releasedLabels: ['released']
      }
    ]
  ],

  // Global options
  preset: 'conventionalcommits',
  tagFormat: 'v${version}',
  repositoryUrl: 'https://github.com/terragonlabs/agentic-startup-studio',
  
  // Success/failure conditions
  success: [
    '@semantic-release/github'
  ],
  fail: [
    '@semantic-release/github'
  ]
};