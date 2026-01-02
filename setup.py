"""
Setup script for volatility-surface-lab package.
"""

from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding='utf-8')

setup(
    name="volatility-surface-lab",
    version="0.1.0",
    description="Arbitrage-free volatility surface construction and calibration toolkit",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Alexander Robbins",
    author_email="your.email@ufl.edu",  # Update this
    url="https://github.com/yourusername/volatility-surface-lab",  # Update this
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    packages=find_packages(exclude=["tests", "notebooks", "data"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "scipy>=1.7.0,<2.0.0",
        "pandas>=1.3.0,<3.0.0",
        "matplotlib>=3.4.0,<4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "notebook>=6.4.0",
        ],
        "fast": [
            "numba>=0.55.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="finance options volatility surface SVI Heston calibration quantitative",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/volatility-surface-lab/issues",
        "Source": "https://github.com/yourusername/volatility-surface-lab",
    },
)