import os

from distutils.core import setup
from Cython.Build import cythonize

CODE_DIRECTORY = "dataframe_sql"

setup(
    ext_modules=cythonize(os.path.join(CODE_DIRECTORY, "parsers.py"), annotate=True)
)
