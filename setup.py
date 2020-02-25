from distutils.core import setup
from pathlib import Path
from versioneer import get_cmdclass, get_version

CODE_DIRECTORY = Path(__file__).parent


setup(name="dataframe_sql", version=get_version(), cmdclass=get_cmdclass())
