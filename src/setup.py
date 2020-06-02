from setuptools import setup, find_packages

DEPENDENCIES = [
    "torch",
    "jupyterlab",
]

setup(
    name="bald",
    version="0.1",
    author="ActuallyOpenAI",
    packages=find_packages(),
    install_requires=DEPENDENCIES,
)