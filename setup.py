from distutils.core import setup
from distutils.extension import Extension
import os
from pathlib import Path
import re

from Cython.Distutils import build_ext

CODE_DIRECTORY = Path(__file__).parent



def get_cython_files():
    ext_modules = []
    for root, subdirs, files in os.walk(CODE_DIRECTORY):
        if re.match(r"\./dataframe_sql.*", root) \
                and not re.match(r".*(tests|__pycache__).*", root):
            for file in files:
                match = re.match(r"(?P<file_name>.*)\.pyx", file)
                if match:
                    file_name = match.group("file_name")
                    extension_name = f"{root}/{file_name}"[2:].replace("/", ".")
                    ext_modules.append(Extension(extension_name,
                                                 sources=[f"{root}/{file}"]))

    return ext_modules

setup(
  name='DataFrameSql',
  cmdclass={'build_ext': build_ext},
  ext_modules=get_cython_files(),
)
