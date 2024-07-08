from setuptools import setup, find_packages

setup(
    name="embedders",
    version="0.1",
    packages=find_packages(where="src/embedders"),
    package_dir={"": "src"},
    requires=[
        "torch",
        "geoopt",
        "networkx",
        "numpy ",
        "pandas",
        "matplotlib",
        "scipy",
        "torchtyping",
    ]
)
