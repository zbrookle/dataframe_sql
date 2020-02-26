# flake8: noqa
from dataframe_sql.sql_select_query import query, register_temp_table, remove_temp_table

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
