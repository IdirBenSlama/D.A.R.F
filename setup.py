#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="darf-framework",
    version="1.0.0",
    author="DARF Team",
    author_email="info@darf-framework.org",
    description="Decentralized Autonomous Reaction Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/darf-framework/darf",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "darf=run:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.html"],
    },
)
