from setuptools import setup, find_packages

# Leer el README para la descripción larga del paquete
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="evolutia",
    version="0.1.5",
    include_package_data=True,
    author="Gerardo Lacy-Mora",
    author_email="gerardolacymora@gmail.com",
    description="Sistema automatizado para generar preguntas de examen desafiantes basadas en materiales didácticos existentes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/glacy/evolutIA",
    packages=find_packages(),
    py_modules=["evolutia_cli"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "PyYAML",
        "requests",
        "python-dotenv",
        "openai",
        "anthropic",
        "google-generativeai",
        "tqdm",
    ],
    extras_require={
        "rag": [
            "chromadb",
            "tiktoken",
            "sentence-transformers",
        ],
    },
    entry_points={
        "console_scripts": [
            "evolutia=evolutia_cli:main",
        ],
    },
)
