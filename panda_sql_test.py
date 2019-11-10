"""
Test cases for panda to sql
"""
from pandas import read_csv
from sql_to_pandas import SqlToPandas, MultipleQueriesException, InvalidQueryException

forest_Fires = read_csv('~/PycharmProjects/sql_to_pandas/forestfires.csv') # Name is weird intentionally


def lower_case_globals():
    """
    Returns all globals but in lower case
    :param global_vars:
    :return:
    """
    return {global_var.lower(): globals()[global_var] for global_var in globals()}


def sql_to_pandas_with_vars(sql: str):
    """
    Preset with data in SqlToPandas class
    :param sql: Sql query
    :return: SqlToPandasClass with
    """
    return SqlToPandas(sql, lower_case_globals())


def test_for_multiple_statements():
    """
    Test that exception is raised when there are multiple queries
    :return:
    """
    sql = 'select * from foo; select * from bar;'
    try:
        sql_to_pandas_with_vars(sql)
    except Exception as err:
        assert isinstance(err, MultipleQueriesException)


# TODO Expand on these errors for invalid queries
def test_for_valid_query():
    """
    Test that exception is raised for invalid query
    :return:
    """
    sql = "hello world!"
    sql_to_pandas_with_vars(sql)
    try:
        sql_to_pandas_with_vars(sql)
    except InvalidQueryException as err:
        assert isinstance(err, InvalidQueryException)


def test_select_star():
    """
    Tests the simple select * case
    :return:
    """
    sql_to_pandas_with_vars("select * from forest_fires").execute_sql()

test_select_star()

def test_case_insensitivity():
    """
    Tests to ensure that the sql is case insensitive for table names
    :return:
    """
    pass


def test_select_specific_fields():
    """
    Tests selecting specific fields
    :return:
    """
    pass


def test_type_conversion():
    """
    Tests sql as statements
    :return:
    """
    pass


def test_distinct():
    """
    Test use of the distinct keyword
    :return:
    """
    pass


def test_subquery():
    """
    Test ability to perform subqueries
    :return:
    """
    pass


def test_joins():
    """
    Test right, left, inner, and outer joins
    :return:
    """
    pass


def test_for_non_existent_table():
    """
    Check that exception is raised if table does not exist
    :return:
    """
    pass
