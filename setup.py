from setuptools import setup, find_packages

setup(
    name="quant_platform",
    version="0.1.0",
    packages=find_packages(where=".", include=["src", "src.*"]),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "streamlit>=1.28.0",
        "yfinance>=0.2.28",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "cvxpy>=1.3.0",
        "sqlalchemy>=2.0.0",
        "loguru>=0.7.0",
        "openai>=1.30.0",
        "statsmodels>=0.14.0",
        "arch>=6.2.0",
    ],
    python_requires=">=3.10",
)
