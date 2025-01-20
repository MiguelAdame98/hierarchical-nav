from setuptools import setup, find_packages

with open("requirements.txt", "r") as req:
    requirements = req.read().splitlines()

setup(
    name="dommel_library",
    version="0.4.0",
    packages=find_packages(),
    install_requires=requirements,
)
