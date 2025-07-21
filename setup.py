"""
Setup script for Insurance Fraud Detector package.

This script provides the package installation configuration and metadata
for the insurance fraud detection system.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
    "jupyter>=1.0.0",
    "notebook>=6.5.0",
    "jupyterlab>=3.6.0",
]

# Optional requirements for different use cases
extras_require = {
    "dev": dev_requirements,
    "test": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "notebook>=6.5.0",
        "jupyterlab>=3.6.0",
        "ipywidgets>=8.0.0",
    ],
    "viz": [
        "plotly>=5.14.0",
        "seaborn>=0.12.0",
        "matplotlib>=3.6.0",
    ],
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=1.0.0",
    ],
    "api": [
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "gunicorn>=20.1.0",
    ],
    "web": [
        "streamlit>=1.22.0",
    ],
    "all": dev_requirements + [
        "jupyter>=1.0.0",
        "notebook>=6.5.0",
        "jupyterlab>=3.6.0",
        "ipywidgets>=8.0.0",
        "plotly>=5.14.0",
        "seaborn>=0.12.0",
        "matplotlib>=3.6.0",
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=1.0.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "gunicorn>=20.1.0",
        "streamlit>=1.22.0",
    ]
}

setup(
    name="insurance-fraud-detector",
    version="1.0.0",
    author="Insurance Analytics Team",
    author_email="contact@insurancefraud.ai",
    description="AI-powered insurance fraud detection system with ML, causal analysis, and bias detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/insurance-fraud-detector",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/insurance-fraud-detector/issues",
        "Source": "https://github.com/yourusername/insurance-fraud-detector",
        "Documentation": "https://insurance-fraud-detector.readthedocs.io/",
        "Changelog": "https://github.com/yourusername/insurance-fraud-detector/blob/main/CHANGELOG.md",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "insurance_fraud_detector": [
            "config/*.yaml",
            "config/*.yml",
            "data/*.csv",
            "models/*.pkl",
            "models/*.joblib",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "fraud-detector=insurance_fraud_detector.cli:main",
            "fraud-train=insurance_fraud_detector.scripts.train:main",
            "fraud-predict=insurance_fraud_detector.scripts.predict:main",
            "fraud-evaluate=insurance_fraud_detector.scripts.evaluate:main",
        ],
    },
    keywords=[
        "insurance",
        "fraud-detection",
        "machine-learning",
        "artificial-intelligence",
        "causal-inference",
        "bias-detection",
        "fairness",
        "ensemble-learning",
        "explainable-ai",
        "fintech",
    ],
    zip_safe=False,
    test_suite="tests",
)