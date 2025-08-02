# Support

Thank you for using Agentic Startup Studio! This document outlines how to get support for the project.

## Getting Help

### 📚 Documentation
- **Getting Started**: [docs/guides/GETTING_STARTED.md](docs/guides/GETTING_STARTED.md)
- **Developer Guide**: [docs/guides/DEVELOPER_GUIDE.md](docs/guides/DEVELOPER_GUIDE.md)
- **API Documentation**: [docs/api-documentation.md](docs/api-documentation.md)
- **Deployment Guide**: [docs/deployment-guide.md](docs/deployment-guide.md)

### 🐛 Reporting Issues
When reporting issues, please:

1. **Search existing issues** first to avoid duplicates
2. **Use issue templates** provided in `.github/ISSUE_TEMPLATE/`
3. **Provide detailed information**:
   - Version of Agentic Startup Studio
   - Operating system and version
   - Python version
   - Complete error messages
   - Steps to reproduce

### 🚨 Security Vulnerabilities
**DO NOT** create public issues for security vulnerabilities. Instead:

1. Email security issues to: **security@terragonlabs.com**
2. Encrypt sensitive communications using our PGP key (see SECURITY.md)
3. Include detailed vulnerability information
4. Allow 90 days for response before public disclosure

## Support Channels

### Community Support
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Comprehensive guides and API references

### Enterprise Support
For organizations requiring enterprise-level support:

- **Priority Support**: Dedicated support channel with SLA
- **Professional Services**: Implementation assistance and consulting
- **Custom Development**: Feature development and customization
- **Training**: Team training and onboarding

Contact: **enterprise@terragonlabs.com**

## Response Times

### Community Support
- **Bug Reports**: Best effort, typically 1-3 business days
- **Feature Requests**: Evaluated during sprint planning
- **Documentation Issues**: 1-2 business days
- **Security Issues**: Within 24 hours

### Enterprise Support
- **Critical Issues**: 4 hours or less
- **High Priority**: 1 business day
- **Normal Priority**: 3 business days
- **Low Priority**: 5 business days

## Contributing
Want to help improve the project? See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Coding standards
- Pull request process
- Issue triage guidelines

## Community Guidelines
All interactions must follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Frequently Asked Questions

### Installation Issues
**Q: I'm getting installation errors with Python dependencies**
A: Ensure you're using Python 3.11+ and consider using virtual environments. See [SETUP.md](SETUP.md) for detailed instructions.

**Q: Docker containers fail to start**
A: Check that all required environment variables are set in `.env`. See [.env.example](.env.example) for reference.

### Configuration Issues
**Q: How do I configure external API keys?**
A: Copy `.env.example` to `.env` and fill in your API keys. For production, use Google Cloud Secret Manager.

**Q: Database connection errors**
A: Verify PostgreSQL is running and connection details in `.env` are correct. For Docker, ensure the database container is healthy.

### Performance Issues
**Q: Pipeline processing is slow**
A: Check the performance monitoring dashboard and consider adjusting batch sizes in configuration.

**Q: High memory usage**
A: Review the resource limits in your deployment configuration and consider scaling options.

---

For additional support resources and updates, visit our [documentation site](https://docs.terragonlabs.com/agentic-startup-studio).

*Last Updated: 2025-08-02*