from setuptools import setup, find_packages

setup(
    name="aell",
    version="0.1",
    author="MetaAI Group, CityU",
    description="Evolutionary Computation + Large Language Model for automatic algorithm design",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "numpy"
    ],
    test_suite="tests"
)