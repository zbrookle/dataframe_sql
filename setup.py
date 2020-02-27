from distutils.core import setup
from pathlib import Path

from versioneer import get_cmdclass, get_version

CODE_DIRECTORY = Path(__file__).parent

setup(
    name="dataframe_sql",
    version=get_version(),
    cmdclass=get_cmdclass(),
    maintainer="Zach Brookler",
    maintainer_email="zachb1996@yahoo.com",
    description="A package for querying dataframes using SQL",
    python_requires=">=3.6.1",
    install_requires=["lark-parser == 0.8.1", "pandas == 1.0.1"],
    url="https://github.com/zbrookle/dataframe_sql",
    download_url="https://github.com/zbrookle/dataframe_sql/archive/master.zip",
    keywords=["pandas", "data", "dataframe", "sql"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
    ],
)
