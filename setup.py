"""
Setup script for the NestAI library.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nestai",
    version="1.0.0",
    author="NestAI Team",
    author_email="info@nestai.example.com",
    description="A simplified interface for working with AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nestai/nestai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "cryptography>=36.0.0",
        "requests>=2.25.0",
        "aiohttp>=3.7.4",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.15.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
        ],
    },
)

