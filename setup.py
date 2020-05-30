from setuptools import setup

DEPENDENCIES = [
    "torch",
    "jupyterlab",
]

setup(
    name="bald",
    version="0.1",
    author="ActuallyOpenAI",
    packages=["bald"],
    package_dir={
        "":"./bald",
    },
    install_requires=DEPENDENCIES,
)