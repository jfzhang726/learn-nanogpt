from setuptools import setup, find_packages

setup(
    name="learn-nanogpt",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "matplotlib",
        "pytz",
        "tiktoken",
        "datasets",
    ],
    author="JF Zhang",
    author_email="jfzhang726@gmail.com",
    description="A course work for implementing NanoGPT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jfzhang726/learn-nanogpt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
