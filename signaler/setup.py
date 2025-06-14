"""
Setup configuration for Trading Signal System.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="trading-signal-system",
    version="0.1.0",
    author="Trading System Team",
    author_email="team@tradingsystem.com",
    description="GNN-based trading signal generation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/trading-signal-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "trading-daily-ingestion=src.jobs.daily_ingestion:main",
            "trading-backfill=src.jobs.backfill_job:main",
            "trading-train=src.jobs.training_job:main",
            "trading-predict=src.training.prediction_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.yaml", "*.yml"],
    },
)