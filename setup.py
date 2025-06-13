from setuptools import setup

README_FILE = "README.md"
VERSION_FILE = "nalcos/_version.py"

with open(README_FILE, "r", encoding="utf-8") as f:
    long_description = f.read()

exec(open(VERSION_FILE).read())

setup(
    name="nalcos",  # This is the name of the package
    version=__version__,  # The initial release version
    author="Pushkar Patel",  # Full name of the author
    author_email="thepushkarp+pypi@gmail.com",  # Email address of the author
    url="https://github.com/thepushkarp/nalcos",  # URL of the project
    description="Search Git commits in natural language",
    long_description=long_description,  # Long description read from the the readme file
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="commit information-retrieval sentence-transformers natural-language huggingface",
    packages=["nalcos"],  # List of all python modules to be installed
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],  # Information to filter the project on PyPi website
    python_requires=">=3.6",  # Minimum version requirement of the package
    entry_points={"console_scripts": ["nalcos = nalcos.nalcos:main"]},
    install_requires=[
        "requests==2.32.4",
        "torch==2.7.1",
        "sentence_transformers==2.0.0",
        "appdirs==1.4.4",
        "transformers==4.50.0",
        "tqdm==4.66.3",
        "GitPython==3.1.41",
        "rich==10.9.0",
        "black==24.3.0",
    ],
)
