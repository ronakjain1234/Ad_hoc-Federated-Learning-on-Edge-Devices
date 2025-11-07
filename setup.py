from setuptools import setup, find_packages

setup(
    name="adhocfl",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "networkx",
        "pyyaml",
        "tqdm",
        "matplotlib",
    ],
    python_requires=">=3.7",
)

