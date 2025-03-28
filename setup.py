from setuptools import setup, find_packages

setup(
    name="inventory_management",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0,<2.0.0",
        "pandas>=2.0.0",
        "scipy>=1.11.3",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.2",
        "plotly==5.18.0",
        "dash==2.14.1",
        "dash-bootstrap-components==1.5.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "tensorflow>=2.14.0",
        "prophet>=1.1.4"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.1",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.1"
        ]
    },
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="Inventory Management System with ML-based optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="inventory, optimization, machine-learning, forecasting",
    url="https://github.com/yourusername/inventory_management",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
) 