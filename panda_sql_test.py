"""
Test cases for panda to sql
"""
# pylint: disable=broad-except
import numpy as np
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
    return SqlToPandas(sql, lower_case_globals()).data_frame


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
    myframe = sql_to_pandas_with_vars("select * from forest_fires")
    assert forest_Fires.equals(myframe)


def test_case_insensitivity():
    """
    Tests to ensure that the sql is case insensitive for table names
    :return:
    """
    assert forest_Fires.equals(sql_to_pandas_with_vars("select * from FOREST_fires"))


def test_select_specific_fields():
    """
    Tests selecting specific fields
    :return:
    """
    myframe = sql_to_pandas_with_vars("select temp,RH,wind,rain as water,area from forest_fires")
    pandas_frame = forest_Fires[['temp', 'RH', 'wind', 'rain', 'area']].rename(columns={'rain': 'water'})
    assert myframe.equals(pandas_frame)


def test_type_conversion():
    """
    Tests sql as statements
    :return:
    """
    myframe = sql_to_pandas_with_vars("select cast(temp as int64),cast(RH as float64) my_rh,wind,rain,area , cast(2 as int64) my_num from forest_fires")
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
    my_frame = sql_to_pandas_with_vars("select temp, 1 + 2 * 3 as my_number from forest_fires")
    pandas_frame = forest_Fires[['temp']].copy()
    pandas_frame['my_number'] = 1 + 2 * 3
    print(pandas_frame)
    assert pandas_frame.equals(my_frame)


def test_distinct():
    """
    Test use of the distinct keyword
    :return:
    """
    my_frame = sql_to_pandas_with_vars("select distinct area, rain from forest_fires")
    pandas_frame = forest_Fires[['area', 'rain']].copy()
    pandas_frame.drop_duplicates(keep='first', inplace=True)
    pandas_frame.reset_index(inplace=True)
    pandas_frame.drop(columns='index', inplace=True)
    assert pandas_frame.equals(my_frame)


def test_subquery():
    """
    Test ability to perform subqueries
    :return:
    """
    my_frame = sql_to_pandas_with_vars("select * from (select area, rain from forest_fires) rain_area")
    pandas_frame = forest_Fires[['area', 'rain']].copy()
    assert pandas_frame.equals(my_frame)


def test_join_no_inner():
    """
    Test join
    :return:
    """
    my_frame = sql_to_pandas_with_vars(
        """select * from digimon_mon_list join
            digimon_move_list
            on digimon_mon_list.attribute = digimon_move_list.attribute""")
    pandas_frame1 = digimon_mon_list
    pandas_frame2 = digimon_move_list
    merged_frame = pandas_frame1.merge(pandas_frame2, on="Attribute")
    assert merged_frame.equals(my_frame)


def test_join_w_inner():
    """
    Test join
    :return:
    """
    my_frame = sql_to_pandas_with_vars(
        """select * from digimon_mon_list inner join
            digimon_move_list
            on digimon_mon_list.attribute = digimon_move_list.attribute""")
    pandas_frame1 = digimon_mon_list
    pandas_frame2 = digimon_move_list
    merged_frame = pandas_frame1.merge(pandas_frame2, on="Attribute")
    assert merged_frame.equals(my_frame)


def test_outer_join_no_outer():
    """
    Test outer join
    :return:
    """
    my_frame = sql_to_pandas_with_vars(
        """select * from digimon_mon_list full outer join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type""")
    pandas_frame1 = digimon_mon_list
    pandas_frame2 = digimon_move_list
    merged_frame = pandas_frame1.merge(pandas_frame2, how="outer", on="Type")
    assert merged_frame.equals(my_frame)


def test_outer_join_w_outer():
    """
    Test outer join
    :return:
    """
    my_frame = sql_to_pandas_with_vars(
        """select * from digimon_mon_list full join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type""")
    pandas_frame1 = digimon_mon_list
    pandas_frame2 = digimon_move_list
    merged_frame = pandas_frame1.merge(pandas_frame2, how="outer", on="Type")
    assert merged_frame.equals(my_frame)


def test_left_joins():
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_frame = sql_to_pandas_with_vars(
        """select * from digimon_mon_list left join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type""")
    pandas_frame1 = digimon_mon_list
    pandas_frame2 = digimon_move_list
    merged_frame = pandas_frame1.merge(pandas_frame2, how="left", on="Type")
    assert merged_frame.equals(my_frame)


def test_left_outer_joins():
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_frame = sql_to_pandas_with_vars(
        """select * from digimon_mon_list left outer join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type""")
    pandas_frame1 = digimon_mon_list
    pandas_frame2 = digimon_move_list
    merged_frame = pandas_frame1.merge(pandas_frame2, how="left", on="Type")
    assert merged_frame.equals(my_frame)


def test_right_joins():
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_frame = sql_to_pandas_with_vars(
        """select * from digimon_mon_list right join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type""")
    pandas_frame1 = digimon_mon_list
    pandas_frame2 = digimon_move_list
    merged_frame = pandas_frame1.merge(pandas_frame2, how="right", on="Type")
    assert merged_frame.equals(my_frame)


def test_right_outer_joins():
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_frame = sql_to_pandas_with_vars(
        """select * from digimon_mon_list right outer join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type""")
    pandas_frame1 = digimon_mon_list
    pandas_frame2 = digimon_move_list
    merged_frame = pandas_frame1.merge(pandas_frame2, how="right", on="Type")
    assert merged_frame.equals(my_frame)


def test_cross_joins():
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_frame = sql_to_pandas_with_vars(
        """select * from digimon_mon_list cross join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type""")
    pandas_frame1 = digimon_mon_list
    pandas_frame2 = digimon_move_list
    merged_frame = pandas_frame1.merge(pandas_frame2, how="outer", on="Type")
    assert merged_frame.equals(my_frame)


def test_group_by():
    """
    Test group by constraint
    :return:
    """
    my_frame = sql_to_pandas_with_vars("""select month, day from forest_fires group by month, day""")
    pandas_frame = forest_Fires.groupby(["month", "day"]).size().to_frame('size').reset_index().drop(columns=['size'])
    assert pandas_frame.equals(my_frame)


def test_avg():
    """
    Test the avg
    :return:
    """
    my_frame = sql_to_pandas_with_vars("select avg(temp) from forest_fires")
    pandas_frame = forest_Fires.agg({'temp': np.mean}).to_frame('mean_temp').reset_index().drop(columns=['index'])
    assert pandas_frame.equals(my_frame)

def test_sum():
    """
    Test the sum
    :return:
    """
    my_frame = sql_to_pandas_with_vars("select sum(temp) from forest_fires")
    pandas_frame = forest_Fires.agg({'temp': np.sum}).to_frame('sum_temp').reset_index().drop(columns=['index'])
    assert pandas_frame.equals(my_frame)


def test_max():
    """
    Test the max
    :return:
    """
    my_frame = sql_to_pandas_with_vars("select max(temp) from forest_fires")
    pandas_frame = forest_Fires.agg({'temp': np.max}).to_frame('max_temp').reset_index().drop(columns=['index'])
    assert pandas_frame.equals(my_frame)

def test_min():
    """
    Test the min
    :return:
    """
    my_frame = sql_to_pandas_with_vars("select min(temp) from forest_fires")
    pandas_frame = forest_Fires.agg({'temp': np.min}).to_frame('min_temp').reset_index().drop(columns=['index'])
    assert pandas_frame.equals(my_frame)

def test_multiple_aggs():
    """
    Test multiple aggregations
    :return:
    """
    my_frame = sql_to_pandas_with_vars("select min(temp), max(temp), avg(temp), max(wind) from forest_fires")
    pandas_frame = forest_Fires['min_temp'] = forest_Fires.temp.copy()
    pandas_frame = forest_Fires['max_temp'] = forest_Fires.temp.copy()
    pandas_frame = forest_Fires['mean_temp'] = forest_Fires.temp.copy()
    pandas_frame = forest_Fires.agg({'min_temp': np.min, 'max_temp': np.max, 'mean_temp': np.mean, 'wind': np.max})
    pandas_frame.rename({'wind': 'max_wind'}, inplace=True)
    pandas_frame = pandas_frame.to_frame().transpose()
    assert pandas_frame.equals(my_frame)

def test_agg_w_groupby():
    """
    Test using aggregates and group by together
    :return:
    """
    my_frame = sql_to_pandas_with_vars("select day, min(temp), max(temp) group by day")


def test_where_clause():
    """
    Test where clause
    :return:
    """
    my_frame = sql_to_pandas_with_vars("""select * from forest_fires where month = 'mar'""")

def test_having():
    """
    Test having clause
    :return:
    """
    sql_to_pandas_with_vars("select min(temp) from forest_fires having max(temp) > 20")

if __name__ == "__main__":
    test_multiple_aggs()