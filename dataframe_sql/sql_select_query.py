"""
Convert dataframe_sql statement to run on pandas dataframes
"""
import ibis
from pandas import DataFrame
from sql_to_ibis import (
    query as ibis_query,
    register_temp_table as ibis_register,
    remove_temp_table as ibis_remove,
)

IBIS_PANDAS_CLIENT = ibis.pandas.PandasClient({})


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
    ibis_register(
        ibis.pandas.from_dataframe(frame, name=table_name, client=IBIS_PANDAS_CLIENT),
        table_name,
    )


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
    ibis_remove(table_name)


def query(sql: str) -> DataFrame:
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
    return ibis_query(sql).execute()
