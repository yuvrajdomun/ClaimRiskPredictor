# Contributing to Insurance Claim Risk Predictor

We welcome contributions to the Insurance Claim Risk Predictor project! This document provides guidelines for contributing to the project.

## 🎯 How to Contribute

### Reporting Bugs

1. **Check existing issues** first to avoid duplicates
2. **Use the bug report template** when creating new issues
3. **Include detailed information**:
   - Python version
   - Dependencies versions
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages/logs

### Suggesting Features

1. **Check existing feature requests** to avoid duplicates
2. **Use the feature request template**
3. **Provide detailed description**:
   - Use case and motivation
   - Proposed solution
   - Alternative solutions considered
   - Additional context

### Code Contributions

#### Setting up Development Environment

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/ClaimRiskPredictor.git
   cd ClaimRiskPredictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. **Install pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

#### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests**
   ```bash
   python -m pytest tests/
   ```

4. **Run linting**
   ```bash
   flake8 .
   black .
   isort .
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add feature: brief description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Use the PR template
   - Link related issues
   - Provide detailed description

## 📝 Coding Standards

### Python Style Guide

- **Follow PEP 8** for Python code style
- **Use Black** for code formatting
- **Use isort** for import sorting
- **Maximum line length**: 88 characters (Black default)

### Documentation

- **Add docstrings** to all functions, classes, and modules
- **Use Google style** docstrings
- **Update README.md** for significant changes
- **Add inline comments** for complex logic

Example docstring:
```python
def predict_fraud(claim_data: dict) -> float:
    """Predict fraud probability for insurance claim.
    
    Args:
        claim_data: Dictionary containing claim features.
        
    Returns:
        Fraud probability between 0 and 1.
        
    Raises:
        ValueError: If required features are missing.
    """
    pass
```

### Testing

- **Write unit tests** for new functions
- **Use pytest** as testing framework
- **Aim for >80% code coverage**
- **Test edge cases** and error conditions

Example test:
```python
def test_predict_fraud():
    """Test fraud prediction functionality."""
    predictor = FraudPredictor()
    claim = {
        'age': 35,
        'vehicle_type': 'sedan',
        # ... other required fields
    }
    
    result = predictor.predict_single(**claim)
    
    assert 0 <= result <= 1
    assert isinstance(result, float)
```

### Git Commit Messages

Use conventional commit format:

```
type(scope): brief description

Detailed explanation if needed

Fixes #issue_number
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

## 🔧 Development Workflow

### Adding New Features

1. **Plan the feature**
   - Create or comment on related issue
   - Discuss implementation approach
   - Consider impact on existing code

2. **Implement incrementally**
   - Break large features into smaller commits
   - Test each increment
   - Update documentation continuously

3. **Consider backwards compatibility**
   - Avoid breaking existing API
   - Provide migration path if needed
   - Update version appropriately

### Working with Machine Learning Models

- **Document model assumptions** and limitations
- **Include validation metrics** in tests
- **Provide example usage** in docstrings
- **Consider computational efficiency**
- **Test with different data distributions**

### API Changes

- **Maintain backward compatibility** when possible
- **Version API endpoints** for breaking changes
- **Update OpenAPI documentation**
- **Test with realistic payloads**

## 📊 Performance Guidelines

- **Optimize for prediction latency** (target <100ms)
- **Memory efficiency** for large datasets
- **Batch processing** capabilities
- **Profile code** for bottlenecks
- **Cache expensive operations**

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests**: Individual function testing
2. **Integration Tests**: Component interaction testing
3. **End-to-End Tests**: Full workflow testing
4. **Performance Tests**: Latency and throughput
5. **Bias Tests**: Fairness validation

### Test Data

- **Use synthetic data** for reproducibility
- **Test edge cases** (missing values, outliers)
- **Validate model outputs** are reasonable
- **Test different demographic groups**

## 📋 Pull Request Process

### PR Requirements

- [ ] **Tests pass** locally and in CI
- [ ] **Code follows** style guidelines
- [ ] **Documentation updated** if needed
- [ ] **No breaking changes** without discussion
- [ ] **Performance impact** considered
- [ ] **Bias implications** evaluated

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in different environments
4. **Documentation review**
5. **Final approval** and merge

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## 🎯 Areas for Contribution

### High Priority

- **Model improvements**: New algorithms, hyperparameter tuning
- **Performance optimization**: Latency reduction, memory efficiency
- **Bias mitigation**: Fairness improvements, new metrics
- **Documentation**: Examples, tutorials, API docs

### Medium Priority

- **Testing**: More comprehensive test coverage
- **Monitoring**: Logging, metrics, alerting
- **Deployment**: Docker, Kubernetes, cloud platforms
- **Visualization**: New charts, interactive features

### Good First Issues

- **Code cleanup**: Refactoring, type hints
- **Documentation**: Fix typos, add examples
- **Tests**: Add missing test cases
- **Configuration**: Environment variable support

## 💬 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and ideas
- **Code Review**: Tag maintainers for review help
- **Documentation**: Check README and inline docs first

## 🏆 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Insurance Claim Risk Predictor! 🎉