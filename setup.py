from setuptools import setup, find_packages

setup(
    name="embedders",
    version="0.1",
    packages=find_packages(where="src/embedders"),
    package_dir={"": "src"},
    requires=[],  # TODO: Add dependencies
)
