from setuptools import setup, find_packages

setup(
    name="evolutia",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["evolutia_cli"],
    install_requires=[
        "PyYAML",
        "requests",
        "python-dotenv",
        "openai",
        "anthropic",
        "google-generativeai",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "evolutia=evolutia_cli:main",
        ],
    },
)
