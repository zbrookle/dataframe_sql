from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

from versioneer import get_cmdclass, get_version

CODE_DIRECTORY = Path(__file__).parent


def read_file(file_path: Path):
    """Source the contents of a file"""
    with open(str(file_path), encoding="utf-8") as file:
        return file.read()


setup(
    name="dataframe_sql",
    version=get_version(),
    cmdclass=get_cmdclass(),
    packages=find_packages(),
    long_description=read_file(CODE_DIRECTORY / "README.rst"),
    maintainer="Zach Brookler",
    maintainer_email="zachb1996@yahoo.com",
    description="A package for querying dataframes using SQL",
    python_requires=">=3.7.0",
    install_requires=["sql-to-ibis==0.4.0"],
    project_urls={
        "Source Code": "https://github.com/zbrookle/dataframe_sql",
        "Bug Tracker": "https://github.com/zbrookle/dataframe_sql/issues",
        "Documentation": "https://github.com/zbrookle/dataframe_sql",
    },
    url="https://github.com/zbrookle/dataframe_sql",
    download_url="https://github.com/zbrookle/dataframe_sql/archive/master.zip",
    keywords=["pandas", "data", "dataframe", "sql"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    long_description_content_type="text/x-rst",
    include_package_data=True,
)
