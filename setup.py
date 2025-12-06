from setuptools import setup, find_packages

setup(
    name="tfcm",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "timm",
        "pillow",
        "requests",
        "tqdm"
    ],
    entry_points={
        "console_scripts": [
            "tfcm-cli=tfcm.cli:main",
        ],
    },
)