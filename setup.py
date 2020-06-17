from setuptools import setup, find_packages

dependencies = [
    "torch",
    "jupyterlab",
    "pytest",
    "pytest-cov",
    "gensim",
    "pytorch-nlp",
]

setup(
    name="bald",
    version="0.1",
    author="ActuallyOpenAI",
    packages=find_packages(include="bald"),
    install_requires=dependencies,
)
