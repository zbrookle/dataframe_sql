"""
Convert dataframe_sql statement to run on pandas dataframes
"""
import os
from pathlib import Path
import re
from typing import Any, Dict

from lark import Lark, UnexpectedToken
from lark.exceptions import VisitError
from pandas.core.frame import DataFrame

from dataframe_sql.exceptions.sql_exception import (
    DataFrameDoesNotExist,
    InvalidQueryException,
)
from dataframe_sql.parsing.sql_parser import SQLTransformer
from dataframe_sql.sql_objects import AmbiguousColumn

SHOW_TREE = False
SHOW_DF = False

_ROOT = Path(__file__).parent
GRAMMAR_PATH = os.path.join(_ROOT, "grammar", "sql.grammar")
with open(file=GRAMMAR_PATH) as sql_grammar_file:
    _GRAMMAR_TEXT = sql_grammar_file.read()


def register_temp_table(frame: DataFrame, table_name: str):
    """
    Registers related metadata from a :class: ~`pandas.DataFrame` for use with SQL

    Parameters
    ----------
    frame : :class: ~`pandas.DataFrame`
        :class: ~`pandas.DataFrame` object to register
    table_name : str
        String that will be used to represent the :class: ~`pandas.DataFrame` in SQL

    See Also
    --------
    remove_temp_table : Removes all registered metadata related to a table name
    query : Query a registered :class: ~`pandas.DataFrame` using an SQL interface

    Examples
    --------
    >>> df = pd.read_csv("a_csv_file.csv")
    >>> register_temp_table(df, "my_table_name")
    """
    table_info = TableInfo()
    table_info.register_temporary_table(frame, table_name)


def remove_temp_table(table_name: str):
    """
    Removes all registered metadata related to a table name

    Parameters
    ----------
    table_name : str
        Name of the table to be removed

    See Also
    --------
    register_temp_table : Registers related metadata from a :class: ~`pandas.DataFrame`
                          for use with SQL
    query : Query a registered :class: ~`pandas.DataFrame` using an SQL interface

    Examples
    --------
    >>> remove_temp_table("my_table_name")
    """
    table_info = TableInfo()
    table_info.remove_temp_table(table_name)


def query(sql: str, show_execution_plan: bool = False):
    """
    Query a registered :class: ~`pandas.DataFrame` using an SQL interface

    Query a registered :class: ~`pandas.DataFrame` using the following interface based
    on the following general syntax:
    SELECT
    col_name | expr [, col_name | expr] ...
    [FROM table_reference [, table_reference | join_expr]]
    [WHERE where_condition]
    [GROUP BY {col_name | expr }, ... ]
    [HAVING where_condition]
    [ORDER BY {col_name | expr | position}
      [ASC | DESC], ... ]
    [LIMIT {[offset,] row_count | row_count OFFSET offset}]
    [ (UNION ( [DISTINCT] | ALL ) | INTERSECT ( [DISTINCT] | ALL ) |
      EXCEPT ( [DISTINCT] | ALL ) ]
    select_expr


    Parameters
    ----------
    sql : str
        SQL string querying the :class: ~`pandas.DataFrame`

    Returns
    -------
    :class: ~`pandas.DataFrame`
        The :class: ~`pandas.DataFrame` resulting from the SQL query provided


    """
    return SqlToDataFrame(sql, show_execution_plan).data_frame


class SqlToDataFrame:
    parser = Lark(_GRAMMAR_TEXT, parser="lalr")

    def __init__(self, sql: str, show_exececution_plan: bool = False):
        self.sql = sql
        self.execution_plan = ""
        self._show_execution_plan = show_exececution_plan

        self.ast = self.parse_sql()
        self.data_frame: DataFrame = self.ast

    def parse_sql(self):
        try:
            tree = self.parser.parse(self.sql)

            table_info = TableInfo()

            return SQLTransformer(
                table_info.dataframe_name_map.copy(),
                table_info.dataframe_map.copy(),
                table_info.column_name_map.copy(),
                table_info.column_to_dataframe_name.copy(),
                self._show_execution_plan,
            ).transform(tree)
        except UnexpectedToken as err:
            message = (
                f"Expected one of the following input(s): {err.expected}\n"
                f"Unexpected input at line {err.line}, column {err.column}\n"
                f"{err.get_context(self.sql)}"
            )
            raise InvalidQueryException(message)
        except VisitError as err:
            match = re.match(
                r"(\n|.)*DataFrame\s(?P<table>.*)\shas\snot\sbeen\sdefined(\n|.)*",
                str(err),
                re.MULTILINE,
            )
            if match:
                raise DataFrameDoesNotExist(table_name=match.group("table"))
            else:
                raise err


class TableInfo:
    column_to_dataframe_name: Dict[str, Any] = {}
    column_name_map: Dict[str, Dict[str, str]] = {}
    dataframe_name_map: Dict[str, str] = {}
    dataframe_map: Dict[str, DataFrame] = {}

    def add_column_to_column_to_dataframe_name_map(self, column, table):
        if self.column_to_dataframe_name.get(column) is None:
            self.column_to_dataframe_name[column] = table
        elif isinstance(self.column_to_dataframe_name[column], AmbiguousColumn):
            self.column_to_dataframe_name[column].tables.append(table)
        else:
            original_table = self.column_to_dataframe_name[column]
            self.column_to_dataframe_name[column] = AmbiguousColumn(
                [original_table, table]
            )

    def register_temporary_table(self, frame: DataFrame, table_name: str):
        if table_name.lower() in self.dataframe_name_map:
            raise Exception(
                f"A table {table_name.lower()} has already been registered. Keep in "
                f"mind that table names are case insensitive"
            )

        self.dataframe_name_map[table_name.lower()] = table_name
        self.dataframe_map[table_name] = frame
        self.column_name_map[table_name] = {}
        for column in frame.columns:
            lower_column = column.lower()
            self.column_name_map[table_name][lower_column] = column
            self.add_column_to_column_to_dataframe_name_map(lower_column, table_name)

    def remove_temp_table(self, table_name: str):
        if table_name.lower() not in self.dataframe_name_map:
            raise Exception(f"Table {table_name.lower()} is not registered")
        real_table_name = self.dataframe_name_map[table_name.lower()]

        columns = self.dataframe_map[real_table_name].columns.to_list()
        for column in columns:
            lower_column = column.lower()
            value = self.column_to_dataframe_name[lower_column]
            if isinstance(value, AmbiguousColumn):
                value.tables.remove(real_table_name)
                if len(value.tables) == 1:
                    self.column_to_dataframe_name[lower_column] = value.tables[0]
            else:
                del self.column_to_dataframe_name[lower_column]

        del self.dataframe_name_map[table_name.lower()]
        del self.dataframe_map[real_table_name]
        del self.column_name_map[real_table_name]
