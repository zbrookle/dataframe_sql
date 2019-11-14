"""
Test cases for panda to sql
"""
# pylint: disable=broad-except

from pandas import read_csv
from sql_to_pandas import SqlToPandas
from sql_exception import MultipleQueriesException, InvalidQueryException, DataFrameDoesNotExist

forest_Fires = read_csv('~/PycharmProjects/sql_to_pandas/data/forestfires.csv') # Name is weird intentionally

digimon_mon_list = read_csv('~/PycharmProjects/sql_to_pandas/data/DigiDB_digimonlist.csv')
digimon_move_list = read_csv('~/PycharmProjects/sql_to_pandas/data/DigiDB_movelist.csv')
digimon_support_list = read_csv('~/PycharmProjects/sql_to_pandas/data/DigiDB_supportlist.csv')


def lower_case_globals():
    """
    Returns all globals but in lower case
    :return:
    """
    return {global_var: globals()[global_var] for global_var in globals()}


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


def test_for_valid_query():
    """
    Test that exception is raised for invalid query
    :return:
    """
    sql = "hello world!"
    try:
        sql_to_pandas_with_vars(sql)
    except InvalidQueryException as err:
        assert isinstance(err, InvalidQueryException)


def test_select_star():
    """
    Tests the simple select * case
    :return:
    """
    myframe = sql_to_pandas_with_vars("select * from forest_fires").data_frame
    assert forest_Fires.equals(myframe)


def test_case_insensitivity():
    """
    Tests to ensure that the sql is case insensitive for table names
    :return:
    """
    assert forest_Fires.equals(sql_to_pandas_with_vars("select * from FOREST_fires").data_frame)


def test_select_specific_fields():
    """
    Tests selecting specific fields
    :return:
    """
    myframe = sql_to_pandas_with_vars("select temp,RH,wind,rain as water,area from forest_fires").data_frame
    pandas_frame = forest_Fires[['temp', 'RH', 'wind', 'rain', 'area']].rename(columns={'rain': 'water'})
    assert myframe.equals(pandas_frame)


def test_type_conversion():
    """
    Tests sql as statements
    :return:
    """
    myframe = sql_to_pandas_with_vars("select cast(temp as int64),cast(RH as float64) my_rh,wind,rain,area , cast(2 as int64) my_num from forest_fires").data_frame
    fire_frame = forest_Fires[['temp', 'RH', 'wind', 'rain', 'area']].rename(columns={'RH': 'my_rh'})
    fire_frame["my_num"] = 2
    pandas_frame = fire_frame.astype({'temp': 'int64', 'my_rh': 'float64', 'my_num': 'int64'})
    assert pandas_frame.equals(myframe)

def test_for_non_existent_table():
    """
    Check that exception is raised if table does not exist
    :return:
    """
    try:
        sql_to_pandas_with_vars("select * from a_table_thats_not_here")
    except Exception as err:
        assert isinstance(err, DataFrameDoesNotExist)

def test_using_math():
    """
    Test the mathematical operations and order of operations
    :return:
    """
    my_frame = sql_to_pandas_with_vars("select temp, 1 + 2 * 3 as my_number from forest_fires").data_frame
    pandas_frame = forest_Fires[['temp']].copy()
    pandas_frame['my_number'] = 1 + 2 * 3
    print(pandas_frame)
    assert pandas_frame.equals(my_frame)

def test_distinct():
    """
    Test use of the distinct keyword
    :return:
    """
    sql_to_pandas_with_vars("select distinct area, rain from forest_fires")


def test_subquery():
    """
    Test ability to perform subqueries
    :return:
    """
    sql_to_pandas_with_vars("select * from (select distinct area, rain from forest_fires) subquery")


def test_joins():
    """
    Test right, left, inner, and outer joins
    :return:
    """
    sql_to_pandas_with_vars(
        """select * from digimon_mon_list inner join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type""")


def test_sum():
    """
    Test the sum
    :return:
    """
    sql_to_pandas_with_vars("select sum(temp) from forest fires")


def test_avg():
    """
    Test the avg
    :return:
    """
    sql_to_pandas_with_vars("select avg(temp) from forest fires")


def test_max():
    """
    Test the max
    :return:
    """
    sql_to_pandas_with_vars("select max(temp) from forest fires")


def test_min():
    """
    Test the min
    :return:
    """
    sql_to_pandas_with_vars("select min(temp) from forest fires")


if __name__ == "__main__":
    # sql_to_pandas_with_vars("select * from (select distinct area, rain from forest_fires) subquery")
    test_distinct()