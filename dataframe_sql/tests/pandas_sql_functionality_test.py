"""
Test cases for panda to sql
"""
# pylint: disable=broad-except
from copy import deepcopy
from datetime import date, datetime
from functools import wraps
from types import FunctionType

from freezegun import freeze_time
import numpy as np
from pandas import concat, merge
import pandas.testing as tm
import pytest

from dataframe_sql import query, register_temp_table, remove_temp_table
from dataframe_sql.exceptions.sql_exception import (
    DataFrameDoesNotExist,
    InvalidQueryException,
)
from dataframe_sql.sql_objects import AmbiguousColumn
from dataframe_sql.sql_select_query import TableInfo
from dataframe_sql.tests.utils import (
    AVOCADO,
    DIGIMON_MON_LIST,
    DIGIMON_MOVE_LIST,
    FOREST_FIRES,
    register_env_tables,
    remove_env_tables,
)


@pytest.fixture(autouse=True, scope="module")
def module_setup_teardown():
    register_env_tables()
    yield
    remove_env_tables()


def display_dict_difference(before_dict: dict, after_dict: dict, name: str):
    dict_diff_report = f"Dictionary Difference Report for {name}:\n"
    for key in before_dict:
        after_value = after_dict.get(key)
        before_value = before_dict[key]
        if after_value != before_value:
            dict_diff_report += (
                f"Value at key '{key}' was {before_value} and now is "
                f"{after_value}\n"
            )
    for key in after_dict:
        if before_dict.get(key) is None:
            dict_diff_report += (
                f"There is now a value {after_dict[key]} at '{key}', "
                f"but there was nothing there before\n"
            )

    raise Exception(dict_diff_report)


def assert_state_not_change(func: FunctionType):
    @wraps(func)
    def new_func():
        table_state = {}
        for key in TableInfo.dataframe_map:
            table_state[key] = TableInfo.dataframe_map[key].copy()
        column_to_dataframe_name = deepcopy(TableInfo.column_to_dataframe_name)
        column_name_map = deepcopy(TableInfo.column_name_map)
        dataframe_name_map = deepcopy(TableInfo.dataframe_name_map)

        func()

        for key in TableInfo.dataframe_map:
            tm.assert_frame_equal(table_state[key], TableInfo.dataframe_map[key])
        if column_to_dataframe_name != TableInfo.column_to_dataframe_name:
            display_dict_difference(
                column_to_dataframe_name,
                TableInfo.column_to_dataframe_name,
                "column_to_dataframe_name",
            )
        if column_name_map != TableInfo.column_name_map:
            display_dict_difference(
                column_name_map, TableInfo.column_name_map, "column_name_map"
            )
        if dataframe_name_map != TableInfo.dataframe_name_map:
            display_dict_difference(
                dataframe_name_map, TableInfo.dataframe_name_map, "dataframe_name_map"
            )

    return new_func


def test_add_remove_temp_table():
    """
    Tests registering and removing temp tables
    :return:
    """
    frame_name = "digimon_mon_list"
    real_frame_name = TableInfo.dataframe_name_map[frame_name]
    remove_temp_table(frame_name)
    tables_present_in_column_to_dataframe = set()
    for column in TableInfo.column_to_dataframe_name:
        table = TableInfo.column_to_dataframe_name[column]
        if isinstance(table, AmbiguousColumn):
            for table_name in table.tables:
                tables_present_in_column_to_dataframe.add(table_name)
        else:
            tables_present_in_column_to_dataframe.add(table)

    # Ensure column metadata is removed correctly
    assert (
        frame_name not in TableInfo.dataframe_name_map
        and real_frame_name not in TableInfo.dataframe_map
        and real_frame_name not in TableInfo.column_name_map
        and real_frame_name not in tables_present_in_column_to_dataframe
    )

    registered_frame_name = real_frame_name
    register_temp_table(DIGIMON_MON_LIST, registered_frame_name)

    assert (
        TableInfo.dataframe_name_map.get(frame_name.lower()) == registered_frame_name
        and real_frame_name in TableInfo.column_name_map
    )

    tm.assert_frame_equal(
        TableInfo.dataframe_map[registered_frame_name], DIGIMON_MON_LIST
    )

    # Ensure column metadata is added correctly
    for column in DIGIMON_MON_LIST.columns:
        assert column == TableInfo.column_name_map[registered_frame_name].get(
            column.lower()
        )
        lower_column = column.lower()
        assert lower_column in TableInfo.column_to_dataframe_name
        table = TableInfo.column_to_dataframe_name.get(lower_column)
        if isinstance(table, AmbiguousColumn):
            assert registered_frame_name in table.tables
        else:
            assert registered_frame_name == table


@assert_state_not_change
def test_for_valid_query():
    """
    Test that exception is raised for invalid query
    :return:
    """
    sql = "hello world!"
    try:
        query(sql)
    except InvalidQueryException as err:
        assert isinstance(err, InvalidQueryException)


@assert_state_not_change
def test_select_star():
    """
    Tests the simple select * case
    :return:
    """
    my_frame = query("select * from forest_fires")
    pandas_frame = FOREST_FIRES
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_case_insensitivity():
    """
    Tests to ensure that the sql is case insensitive for table names
    :return:
    """
    my_frame = query("select * from FOREST_fires")
    pandas_frame = FOREST_FIRES
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_select_specific_fields():
    """
    Tests selecting specific fields
    :return:
    """
    my_frame = query("select temp, RH, wind, rain as water, area from forest_fires")
    pandas_frame = FOREST_FIRES[["temp", "RH", "wind", "rain", "area"]].rename(
        columns={"rain": "water"}
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_type_conversion():
    """
    Tests sql as statements
    :return:
    """
    my_frame = query(
        """select cast(temp as int64),
        cast(RH as float64) my_rh, wind, rain, area,
        cast(2.0 as int64) my_int,
        cast(3 as float64) as my_float,
        cast(7 as object) as my_object,
        cast(0 as bool) as my_bool from forest_fires"""
    )
    fire_frame = FOREST_FIRES[["temp", "RH", "wind", "rain", "area"]].rename(
        columns={"RH": "my_rh"}
    )
    fire_frame["my_int"] = 2
    fire_frame["my_float"] = 3
    fire_frame["my_object"] = str(7)
    fire_frame["my_bool"] = 0
    pandas_frame = fire_frame.astype(
        {
            "temp": "int64",
            "my_rh": "float64",
            "my_int": "int64",
            "my_float": "float64",
            "my_bool": "bool",
        }
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_for_non_existent_table():
    """
    Check that exception is raised if table does not exist
    :return:
    """
    try:
        query("select * from a_table_that_is_not_here")
    except Exception as err:
        assert isinstance(err, DataFrameDoesNotExist)


@assert_state_not_change
def test_using_math():
    """
    Test the mathematical operations and order of operations
    :return:
    """
    my_frame = query("select temp, 1 + 2 * 3 as my_number from forest_fires")
    pandas_frame = FOREST_FIRES[["temp"]].copy()
    pandas_frame["my_number"] = 1 + 2 * 3
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_distinct():
    """
    Test use of the distinct keyword
    :return:
    """
    my_frame = query("select distinct area, rain from forest_fires")
    pandas_frame = FOREST_FIRES[["area", "rain"]].copy()
    pandas_frame.drop_duplicates(keep="first", inplace=True)
    pandas_frame.reset_index(inplace=True)
    pandas_frame.drop(columns="index", inplace=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_subquery():
    """
    Test ability to perform subqueries
    :return:
    """
    my_frame = query("select * from (select area, rain from forest_fires) rain_area")
    pandas_frame = FOREST_FIRES[["area", "rain"]].copy()
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_join_no_inner():
    """
    Test join
    :return:
    """
    my_frame = query(
        """select * from digimon_mon_list join
            digimon_move_list
            on digimon_mon_list.attribute = digimon_move_list.attribute"""
    )
    pandas_frame1 = DIGIMON_MON_LIST
    pandas_frame2 = DIGIMON_MOVE_LIST
    pandas_frame = pandas_frame1.merge(pandas_frame2, on="Attribute")
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_join_wo_specifying_table():
    """
    Test join where table isn't specified in join
    :return:
    """
    my_frame = query(
        """
        select * from digimon_mon_list join
        digimon_move_list
        on mon_attribute = move_attribute
        """
    )
    pandas_frame1 = DIGIMON_MON_LIST
    pandas_frame2 = DIGIMON_MOVE_LIST
    pandas_frame = pandas_frame1.merge(
        pandas_frame2, left_on="mon_attribute", right_on="move_attribute"
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_join_w_inner():
    """
    Test join
    :return:
    """
    my_frame = query(
        """select * from digimon_mon_list inner join
            digimon_move_list
            on digimon_mon_list.attribute = digimon_move_list.attribute"""
    )
    pandas_frame1 = DIGIMON_MON_LIST
    pandas_frame2 = DIGIMON_MOVE_LIST
    pandas_frame = pandas_frame1.merge(pandas_frame2, on="Attribute")
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_outer_join_no_outer():
    """
    Test outer join
    :return:
    """
    my_frame = query(
        """select * from digimon_mon_list full outer join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type"""
    )
    pandas_frame1 = DIGIMON_MON_LIST
    pandas_frame2 = DIGIMON_MOVE_LIST
    pandas_frame = pandas_frame1.merge(pandas_frame2, how="outer", on="Type")
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_outer_join_w_outer():
    """
    Test outer join
    :return:
    """
    my_frame = query(
        """select * from digimon_mon_list full join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type"""
    )
    pandas_frame1 = DIGIMON_MON_LIST
    pandas_frame2 = DIGIMON_MOVE_LIST
    pandas_frame = pandas_frame1.merge(pandas_frame2, how="outer", on="Type")
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_left_joins():
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_frame = query(
        """select * from digimon_mon_list left join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type"""
    )
    pandas_frame1 = DIGIMON_MON_LIST
    pandas_frame2 = DIGIMON_MOVE_LIST
    pandas_frame = pandas_frame1.merge(pandas_frame2, how="left", on="Type")
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_left_outer_joins():
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_frame = query(
        """select * from digimon_mon_list left outer join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type"""
    )
    pandas_frame1 = DIGIMON_MON_LIST
    pandas_frame2 = DIGIMON_MOVE_LIST
    pandas_frame = pandas_frame1.merge(pandas_frame2, how="left", on="Type")
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_right_joins():
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_frame = query(
        """select * from digimon_mon_list right join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type"""
    )
    pandas_frame1 = DIGIMON_MON_LIST
    pandas_frame2 = DIGIMON_MOVE_LIST
    pandas_frame = pandas_frame1.merge(pandas_frame2, how="right", on="Type")
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_right_outer_joins():
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_frame = query(
        """select * from digimon_mon_list right outer join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type"""
    )
    pandas_frame1 = DIGIMON_MON_LIST
    pandas_frame2 = DIGIMON_MOVE_LIST
    pandas_frame = pandas_frame1.merge(pandas_frame2, how="right", on="Type")
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_cross_joins():
    """
    Test right, left, inner, and outer joins
    :return:
    """
    my_frame = query(
        """select * from digimon_mon_list cross join
            digimon_move_list
            on digimon_mon_list.type = digimon_move_list.type"""
    )
    pandas_frame1 = DIGIMON_MON_LIST
    pandas_frame2 = DIGIMON_MOVE_LIST
    pandas_frame = pandas_frame1.merge(pandas_frame2, how="outer", on="Type")
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_group_by():
    """
    Test group by constraint
    :return:
    """
    my_frame = query("""select month, day from forest_fires group by month, day""")
    pandas_frame = (
        FOREST_FIRES[["month", "day"]].drop_duplicates().reset_index(drop=True)
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_avg():
    """
    Test the avg
    :return:
    """
    my_frame = query("select avg(temp) from forest_fires")

    pandas_frame = (
        FOREST_FIRES.agg({"temp": np.mean})
        .to_frame("_col0")
        .reset_index()
        .drop(columns=["index"])
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_sum():
    """
    Test the sum
    :return:
    """
    my_frame = query("select sum(temp) from forest_fires")
    pandas_frame = (
        FOREST_FIRES.agg({"temp": np.sum})
        .to_frame("_col0")
        .reset_index()
        .drop(columns=["index"])
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_max():
    """
    Test the max
    :return:
    """
    my_frame = query("select max(temp) from forest_fires")
    pandas_frame = (
        FOREST_FIRES.agg({"temp": np.max})
        .to_frame("_col0")
        .reset_index()
        .drop(columns=["index"])
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_min():
    """
    Test the min
    :return:
    """
    my_frame = query("select min(temp) from forest_fires")
    pandas_frame = (
        FOREST_FIRES.agg({"temp": np.min})
        .to_frame("_col0")
        .reset_index()
        .drop(columns=["index"])
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_multiple_aggs():
    """
    Test multiple aggregations
    :return:
    """
    my_frame = query(
        "select min(temp), max(temp), avg(temp), max(wind) from forest_fires"
    )
    pandas_frame = FOREST_FIRES.copy()
    pandas_frame["_col0"] = FOREST_FIRES.temp.copy()
    pandas_frame["_col1"] = FOREST_FIRES.temp.copy()
    pandas_frame["_col2"] = FOREST_FIRES.temp.copy()
    pandas_frame = pandas_frame.agg(
        {"_col0": np.min, "_col1": np.max, "_col2": np.mean, "wind": np.max}
    )
    pandas_frame.rename({"wind": "_col3"}, inplace=True)
    pandas_frame = pandas_frame.to_frame().transpose()
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_agg_w_groupby():
    """
    Test using aggregates and group by together
    :return:
    """
    my_frame = query(
        "select day, month, min(temp), max(temp) from forest_fires group by day, month"
    )
    pandas_frame = FOREST_FIRES.copy()
    pandas_frame["_col0"] = pandas_frame.temp
    pandas_frame["_col1"] = pandas_frame.temp
    pandas_frame = (
        pandas_frame.groupby(["day", "month"])
        .aggregate({"_col0": np.min, "_col1": np.max})
        .reset_index()
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_where_clause():
    """
    Test where clause
    :return:
    """
    my_frame = query("""select * from forest_fires where month = 'mar'""")
    pandas_frame = FOREST_FIRES.copy()
    pandas_frame = pandas_frame[pandas_frame.month == "mar"].reset_index(drop=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_all_boolean_ops_clause():
    """
    Test where clause
    :return:
    """
    my_frame = query(
        """select * from forest_fires where month = 'mar' and temp > 8 and rain >= 0
        and area != 0 and dc < 100 and ffmc <= 90.1
        """
    )

    pandas_frame = FOREST_FIRES.copy()
    pandas_frame = pandas_frame[
        (pandas_frame.month == "mar")
        & (pandas_frame.temp > 8.0)
        & (pandas_frame.rain >= 0)
        & (pandas_frame.area != 0)
        & (pandas_frame.DC < 100)
        & (pandas_frame.FFMC <= 90.1)
    ].reset_index(drop=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_order_by():
    """
    Test order by clause
    :return:
    """
    my_frame = query(
        """select * from forest_fires order by temp desc, wind asc, area"""
    )
    pandas_frame = FOREST_FIRES.copy()
    pandas_frame.sort_values(
        by=["temp", "wind", "area"], ascending=[0, 1, 1], inplace=True
    )
    pandas_frame.reset_index(drop=True, inplace=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_limit():
    """
    Test limit clause
    :return:
    """
    my_frame = query("""select * from forest_fires limit 10""")
    pandas_frame = FOREST_FIRES.copy().head(10)
    tm.assert_frame_equal(pandas_frame, my_frame)


# # TODO Add in parentheses support for Order of ops
# @assert_state_not_change
# def test_having_multiple_conditions():
#     """
#     Test having clause
#     :return:
#     """
#     my_frame = query(
#         "select min(temp) from forest_fires having min(temp) > 2 and "
#         "max(dc) < 200 or month = 'oct'"
#     )
#     pandas_frame = FOREST_FIRES.copy()
#     pandas_frame["_col0"] = FOREST_FIRES["temp"]
#     aggregated_df = pandas_frame.aggregate({"_col0": "min"}).to_frame().transpose()
#     pandas_frame = aggregated_df[aggregated_df["_col0"] > 2]
#     tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_having_one_condition():
    """
    Test having clause
    :return:
    """
    my_frame = query("select min(temp) from forest_fires having min(temp) > 2")
    pandas_frame = FOREST_FIRES.copy()
    pandas_frame["_col0"] = FOREST_FIRES["temp"]
    aggregated_df = pandas_frame.aggregate({"_col0": "min"}).to_frame().transpose()
    pandas_frame = aggregated_df[aggregated_df["_col0"] > 2]
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_having_with_group_by():
    """
    Test having clause
    :return:
    """
    my_frame = query(
        "select day, min(temp) from forest_fires group by day having min(temp) > 5"
    )
    pandas_frame = FOREST_FIRES.copy()
    pandas_frame["_col0"] = FOREST_FIRES["temp"]
    pandas_frame = (
        pandas_frame[["day", "_col0"]].groupby("day").aggregate({"_col0": np.min})
    )
    pandas_frame = pandas_frame[pandas_frame["_col0"] > 5].reset_index()
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_operations_between_columns_and_numbers():
    """
    Tests operations between columns
    :return:
    """
    my_frame = query("""select temp * wind + rain / dmc + 37 from forest_fires""")
    pandas_frame = FOREST_FIRES.copy()
    pandas_frame["_col0"] = (
        pandas_frame["temp"] * pandas_frame["wind"]
        + pandas_frame["rain"] / pandas_frame["DMC"]
        + 37
    )
    pandas_frame = pandas_frame["_col0"].to_frame()
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_select_star_from_multiple_tables():
    """
    Test selecting from two different tables
    :return:
    """
    my_frame = query("""select * from forest_fires, digimon_mon_list""")
    forest_fires = FOREST_FIRES.copy()
    digimon_mon_list_new = DIGIMON_MON_LIST.copy()
    forest_fires["_temp_id"] = 1
    digimon_mon_list_new["_temp_id"] = 1
    pandas_frame = merge(forest_fires, digimon_mon_list_new, on="_temp_id").drop(
        columns=["_temp_id"]
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_select_columns_from_two_tables_with_same_column_name():
    """
    Test selecting tables
    :return:
    """
    my_frame = query("""select * from forest_fires table1, forest_fires table2""")
    table1 = FOREST_FIRES.copy()
    table2 = FOREST_FIRES.copy()
    table1["_temp_id"] = 1
    table2["_temp_id"] = 1
    pandas_frame = merge(table1, table2, on="_temp_id").drop(columns=["_temp_id"])
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_maintain_case_in_query():
    """
    Test nested subqueries
    :return:
    """
    my_frame = query("""select wind, rh from forest_fires""")
    pandas_frame = FOREST_FIRES.copy()[["wind", "RH"]].rename(columns={"RH": "rh"})
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_nested_subquery():
    """
    Test nested subqueries
    :return:
    """
    my_frame = query(
        """select * from
            (select wind, rh from
              (select * from forest_fires) fires) wind_rh"""
    )
    pandas_frame = FOREST_FIRES.copy()[["wind", "RH"]].rename(columns={"RH": "rh"})
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_union():
    """
    Test union in queries
    :return:
    """
    my_frame = query(
        """
    select * from forest_fires order by wind desc limit 5
    union
    select * from forest_fires order by wind asc limit 5
    """
    )
    pandas_frame1 = (
        FOREST_FIRES.copy().sort_values(by=["wind"], ascending=[False]).head(5)
    )
    pandas_frame2 = (
        FOREST_FIRES.copy().sort_values(by=["wind"], ascending=[True]).head(5)
    )
    pandas_frame = (
        concat([pandas_frame1, pandas_frame2], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_union_distinct():
    """
    Test union distinct in queries
    :return:
    """
    my_frame = query(
        """
        select * from forest_fires order by wind desc limit 5
         union distinct
        select * from forest_fires order by wind asc limit 5
        """
    )
    pandas_frame1 = (
        FOREST_FIRES.copy().sort_values(by=["wind"], ascending=[False]).head(5)
    )
    pandas_frame2 = (
        FOREST_FIRES.copy().sort_values(by=["wind"], ascending=[True]).head(5)
    )
    pandas_frame = (
        concat([pandas_frame1, pandas_frame2], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_union_all():
    """
    Test union distinct in queries
    :return:
    """
    my_frame = query(
        """
        select * from forest_fires order by wind desc limit 5
         union all
        select * from forest_fires order by wind asc limit 5
        """
    )
    pandas_frame1 = (
        FOREST_FIRES.copy().sort_values(by=["wind"], ascending=[False]).head(5)
    )
    pandas_frame2 = (
        FOREST_FIRES.copy().sort_values(by=["wind"], ascending=[True]).head(5)
    )
    pandas_frame = concat(
        [pandas_frame1, pandas_frame2], ignore_index=True
    ).reset_index(drop=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_intersect_distinct():
    """
    Test union distinct in queries
    :return:
    """
    my_frame = query(
        """
            select * from forest_fires order by wind desc limit 5
             intersect distinct
            select * from forest_fires order by wind desc limit 3
            """
    )
    pandas_frame1 = (
        FOREST_FIRES.copy().sort_values(by=["wind"], ascending=[False]).head(5)
    )
    pandas_frame2 = (
        FOREST_FIRES.copy().sort_values(by=["wind"], ascending=[False]).head(3)
    )
    pandas_frame = merge(
        left=pandas_frame1,
        right=pandas_frame2,
        how="inner",
        on=list(pandas_frame1.columns),
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_except_distinct():
    """
    Test except distinct in queries
    :return:
    """
    my_frame = query(
        """
                select * from forest_fires order by wind desc limit 5
                 except distinct
                select * from forest_fires order by wind desc limit 3
                """
    )
    pandas_frame1 = (
        FOREST_FIRES.copy().sort_values(by=["wind"], ascending=[False]).head(5)
    )
    pandas_frame2 = (
        FOREST_FIRES.copy().sort_values(by=["wind"], ascending=[False]).head(3)
    )
    pandas_frame = (
        pandas_frame1[~pandas_frame1.isin(pandas_frame2).all(axis=1)]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_except_all():
    """
    Test except distinct in queries
    :return:
    """
    my_frame = query(
        """
                select * from forest_fires order by wind desc limit 5
                 except all
                select * from forest_fires order by wind desc limit 3
                """
    )
    pandas_frame1 = (
        FOREST_FIRES.copy().sort_values(by=["wind"], ascending=[False]).head(5)
    )
    pandas_frame2 = (
        FOREST_FIRES.copy().sort_values(by=["wind"], ascending=[False]).head(3)
    )
    pandas_frame = pandas_frame1[
        ~pandas_frame1.isin(pandas_frame2).all(axis=1)
    ].reset_index(drop=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_between_operator():
    """
    Test using between operator
    :return:
    """
    my_frame = query(
        """
    select * from forest_fires
    where wind between 5 and 6
    """
    )
    pandas_frame = FOREST_FIRES.copy()
    pandas_frame = pandas_frame[
        (pandas_frame.wind >= 5) & (pandas_frame.wind <= 6)
    ].reset_index(drop=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_in_operator():
    """
    Test using in operator in a sql query
    :return:
    """
    my_frame = query(
        """
    select * from forest_fires where day in ('fri', 'sun')
    """
    )
    pandas_frame = FOREST_FIRES.copy()
    pandas_frame = pandas_frame[pandas_frame.day.isin(("fri", "sun"))].reset_index(
        drop=True
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_in_operator_expression_numerical():
    """
    Test using in operator in a sql query
    :return:
    """
    my_frame = query(
        """
    select * from forest_fires where X in (5, 9)
    """
    )
    pandas_frame = FOREST_FIRES.copy()
    pandas_frame = pandas_frame[(pandas_frame["X"]).isin((5, 9))].reset_index(drop=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_not_in_operator():
    """
    Test using in operator in a sql query
    :return:
    """
    my_frame = query(
        """
    select * from forest_fires where day not in ('fri', 'sun')
    """
    )
    pandas_frame = FOREST_FIRES.copy()
    pandas_frame = pandas_frame[~pandas_frame.day.isin(("fri", "sun"))].reset_index(
        drop=True
    )
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_case_statement_w_name():
    """
    Test using case statements
    :return:
    """
    my_frame = query(
        """
        select case when wind > 5 then 'strong'
        when wind = 5 then 'mid'
        else 'weak' end as wind_strength
        from
        forest_fires
        """
    )
    pandas_frame = FOREST_FIRES.copy()[["wind"]]
    pandas_frame.loc[pandas_frame.wind > 5, "wind_strength"] = "strong"
    pandas_frame.loc[pandas_frame.wind == 5, "wind_strength"] = "mid"
    pandas_frame.loc[pandas_frame.wind < 5, "wind_strength"] = "weak"
    pandas_frame.drop(columns=["wind"], inplace=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_case_statement_w_no_name():
    """
    Test using case statements
    :return:
    """
    my_frame = query(
        """
        select case when wind > 5 then 'strong' when wind = 5 then 'mid' else 'weak' end
        from forest_fires
        """
    )
    pandas_frame = FOREST_FIRES.copy()[["wind"]]
    pandas_frame.loc[pandas_frame.wind > 5, "_col0"] = "strong"
    pandas_frame.loc[pandas_frame.wind == 5, "_col0"] = "mid"
    pandas_frame.loc[
        ~((pandas_frame.wind == 5) | (pandas_frame.wind > 5)), "_col0"
    ] = "weak"
    pandas_frame.drop(columns=["wind"], inplace=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_case_statement_w_other_columns_as_result():
    """
    Test using case statements
    :return:
    """
    my_frame = query(
        """
        select case when wind > 5 then month when wind = 5 then 'mid' else day end
        from forest_fires
        """
    )
    pandas_frame = FOREST_FIRES.copy()[["wind"]]
    pandas_frame.loc[pandas_frame.wind > 5, "_col0"] = FOREST_FIRES["month"]
    pandas_frame.loc[pandas_frame.wind == 5, "_col0"] = "mid"
    pandas_frame.loc[
        ~((pandas_frame.wind == 5) | (pandas_frame.wind > 5)), "_col0"
    ] = FOREST_FIRES["day"]
    pandas_frame.drop(columns=["wind"], inplace=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_rank_statement_one_column():
    """
    Test rank statement
    :return:
    """
    my_frame = query(
        """
    select wind, rank() over(order by wind) as wind_rank
    from forest_fires
    """
    )
    pandas_frame = FOREST_FIRES.copy()[["wind"]]
    pandas_frame["wind_rank"] = pandas_frame.wind.rank(method="min").astype("int")
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_rank_statement_many_columns():
    """
    Test rank statement
    :return:
    """
    my_frame = query(
        """
    select wind, rain, month, rank() over(order by wind desc, rain asc, month) as rank
    from forest_fires
    """
    )
    pandas_frame = FOREST_FIRES.copy()[["wind", "rain", "month"]]
    pandas_frame.sort_values(
        by=["wind", "rain", "month"], ascending=[False, True, True], inplace=True
    )
    pandas_frame.reset_index(inplace=True)
    rank_map = {}
    rank_counter = 1
    rank_offset = 0
    pandas_frame["rank"] = 0
    rank_series = pandas_frame["rank"].copy()
    for row_num, row in enumerate(pandas_frame.iterrows()):
        key = "".join(map(str, list(list(row)[1])[1:4]))
        if rank_map.get(key):
            rank_offset += 1
            rank = rank_map[key]
        else:
            rank = rank_counter + rank_offset
            rank_map[key] = rank
            rank_counter += 1
        rank_series[row_num] = rank
    pandas_frame["rank"] = rank_series
    pandas_frame.sort_values(by="index", ascending=True, inplace=True)
    pandas_frame.drop(columns=["index"], inplace=True)
    pandas_frame.reset_index(drop=True, inplace=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_dense_rank_statement_many_columns():
    """
    Test dense_rank statement
    :return:
    """
    my_frame = query(
        """
    select wind, rain, month,
    dense_rank() over(order by wind desc, rain asc, month) as rank
    from forest_fires
    """
    )
    pandas_frame = FOREST_FIRES.copy()[["wind", "rain", "month"]]
    pandas_frame.sort_values(
        by=["wind", "rain", "month"], ascending=[False, True, True], inplace=True
    )
    pandas_frame.reset_index(inplace=True)
    rank_map = {}
    rank_counter = 1
    pandas_frame["rank"] = 0
    rank_series = pandas_frame["rank"].copy()
    for row_num, row in enumerate(pandas_frame.iterrows()):
        key = "".join(map(str, list(list(row)[1])[1:4]))
        if rank_map.get(key):
            rank = rank_map[key]
        else:
            rank = rank_counter
            rank_map[key] = rank
            rank_counter += 1
        rank_series[row_num] = rank
    pandas_frame["rank"] = rank_series
    pandas_frame.sort_values(by="index", ascending=True, inplace=True)
    pandas_frame.drop(columns=["index"], inplace=True)
    pandas_frame.reset_index(drop=True, inplace=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_rank_over_partition_by():
    """
    Test rank partition by statement
    :return:
    """
    my_frame = query(
        """
    select wind, rain, month, day,
    rank() over(partition by day order by wind desc, rain asc, month) as rank
    from forest_fires
    """
    )
    pandas_frame = FOREST_FIRES.copy()[["wind", "rain", "month", "day"]]
    partition_slice = 4
    rank_map = {}
    partition_rank_counter = {}
    partition_rank_offset = {}
    pandas_frame.sort_values(
        by=["wind", "rain", "month"], ascending=[False, True, True], inplace=True
    )
    pandas_frame.reset_index(inplace=True)
    pandas_frame["rank"] = 0
    rank_series = pandas_frame["rank"].copy()
    for row_num, series_tuple in enumerate(pandas_frame.iterrows()):
        row = series_tuple[1]
        row_list = list(row)[1:partition_slice]
        partition_list = list(row)[partition_slice:5]
        key = str(row_list)
        partition_key = str(partition_list)
        if rank_map.get(partition_key):
            if rank_map[partition_key].get(key):
                partition_rank_counter[partition_key] += 1
                rank = rank_map[partition_key][key]
            else:
                partition_rank_counter[partition_key] += 1
                rank = (
                    partition_rank_counter[partition_key]
                    + partition_rank_offset[partition_key]
                )
                rank_map[partition_key][key] = rank
        else:
            rank = 1
            rank_map[partition_key] = {}
            partition_rank_counter[partition_key] = 1
            partition_rank_offset[partition_key] = 0
            rank_map[partition_key][key] = rank
        rank_series[row_num] = rank
    pandas_frame["rank"] = rank_series
    pandas_frame.sort_values(by="index", ascending=True, inplace=True)
    pandas_frame.drop(columns=["index"], inplace=True)
    pandas_frame.reset_index(drop=True, inplace=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_dense_rank_over_partition_by():
    """
    Test rank partition by statement
    :return:
    """
    my_frame = query(
        """
    select wind, rain, month, day,
    dense_rank() over(partition by day order by wind desc, rain asc, month) as rank
    from forest_fires
    """
    )
    pandas_frame = FOREST_FIRES.copy()[["wind", "rain", "month", "day"]]
    partition_slice = 4
    rank_map = {}
    partition_rank_counter = {}
    pandas_frame.sort_values(
        by=["wind", "rain", "month"], ascending=[False, True, True], inplace=True
    )
    pandas_frame.reset_index(inplace=True)
    pandas_frame["rank"] = 0
    rank_series = pandas_frame["rank"].copy()
    for row_num, series_tuple in enumerate(pandas_frame.iterrows()):
        row = series_tuple[1]
        row_list = list(row)[1:partition_slice]
        partition_list = list(row)[partition_slice:]
        key = str(row_list)
        partition_key = str(partition_list)
        if rank_map.get(partition_key):
            if rank_map[partition_key].get(key):
                rank = rank_map[partition_key][key]
            else:
                partition_rank_counter[partition_key] += 1
                rank = partition_rank_counter[partition_key]
                rank_map[partition_key][key] = rank
        else:
            rank = 1
            rank_map[partition_key] = {}
            partition_rank_counter[partition_key] = 1
            rank_map[partition_key][key] = rank
        rank_series[row_num] = rank
    pandas_frame["rank"] = rank_series
    pandas_frame.sort_values(by="index", ascending=True, inplace=True)
    pandas_frame.drop(columns=["index"], inplace=True)
    pandas_frame.reset_index(drop=True, inplace=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_set_string_value_as_column_value():
    """
    Select a string like 'Yes' as a column value
    :return:
    """
    my_frame = query(
        """
    select wind, 'yes' as wind_yes from forest_fires"""
    )
    pandas_frame = FOREST_FIRES.copy()
    pandas_frame["wind_yes"] = "yes"
    pandas_frame = pandas_frame[["wind", "wind_yes"]]
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_date_cast():
    """
    Select casting a string as a date
    :return:
    """
    my_frame = query(
        """
    select wind, cast('2019-01-01' as datetime64) as my_date from forest_fires"""
    )
    pandas_frame = FOREST_FIRES.copy()
    pandas_frame["my_date"] = datetime.strptime("2019-01-01", "%Y-%m-%d")
    pandas_frame = pandas_frame[["wind", "my_date"]]
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_timestamps():
    """
    Select now() as date
    :return:
    """
    with freeze_time(datetime.now()):
        my_frame = query(
            """
        select wind, now(), today(), timestamp('2019-01-31', '23:20:32')
        from forest_fires"""
        )
        pandas_frame = FOREST_FIRES.copy()[["wind"]]
        pandas_frame["now()"] = datetime.now()
        pandas_frame["today()"] = date.today()
        pandas_frame["_literal0"] = datetime(2019, 1, 31, 23, 20, 32)
        tm.assert_frame_equal(pandas_frame, my_frame)


# TODO Add in more having and boolean tests
# TODO Add in parentheses for order of operations


@assert_state_not_change
def test_case_statement_with_same_conditions():
    """
    Test using case statements
    :return:
    """
    my_frame = query(
        """
        select case when wind > 5 then month when wind > 5 then 'mid' else day end
        from forest_fires
        """
    )
    pandas_frame = FOREST_FIRES.copy()[["wind"]]
    pandas_frame.loc[pandas_frame.wind > 5, "_col0"] = FOREST_FIRES["month"]
    pandas_frame.loc[~(pandas_frame.wind > 5), "_col0"] = FOREST_FIRES["day"]
    pandas_frame.drop(columns=["wind"], inplace=True)
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_multiple_aliases_same_column():
    """
    Test multiple aliases on the same column
    :return:
    """
    my_frame = query(
        """
        select wind as my_wind, wind as also_the_wind, wind as yes_wind
        from
        forest_fires
        """
    )

    pandas_frame = FOREST_FIRES[["wind"]].copy()
    pandas_frame.loc[:, "my_wind"] = FOREST_FIRES["wind"].copy()
    pandas_frame.loc[:, "also_the_wind"] = FOREST_FIRES["wind"]
    pandas_frame.loc[:, "yes_wind"] = FOREST_FIRES["wind"]
    pandas_frame = pandas_frame.drop(columns=["wind"])
    tm.assert_frame_equal(pandas_frame, my_frame)


@assert_state_not_change
def test_sql_data_types():
    """
    Tests sql data types
    :return:
    """
    my_frame = query(
        """
        select
            cast(avocado_id as object) as avocado_id_object,
            cast(avocado_id as int16) as avocado_id_int16,
            cast(avocado_id as smallint) as avocado_id_smallint,
            cast(avocado_id as int32) as avocado_id_int32,
            cast(avocado_id as int) as avocado_id_int,
            cast(avocado_id as int64) as avocado_id_int64,
            cast(avocado_id as bigint) as avocado_id_bigint,
            cast(avocado_id as float) as avocado_id_float,
            cast(avocado_id as float16) as avocado_id_float16,
            cast(avocado_id as float32) as avocado_id_float32,
            cast(avocado_id as float64) as avocado_id_float64,
            cast(avocado_id as bool) as avocado_id_bool,
            cast(avocado_id as category) as avocado_id_category,
            cast(date as datetime64) as date,
            cast(date as timestamp) as time,
            cast(region as varchar) as region_varchar,
            cast(region as string) as region_string
        from avocado
        """
    )

    pandas_frame = AVOCADO.copy()[["avocado_id", "Date", "region"]]
    pandas_frame["avocado_id_object"] = pandas_frame["avocado_id"].astype("object")
    pandas_frame["avocado_id_int16"] = pandas_frame["avocado_id"].astype("int16")
    pandas_frame["avocado_id_smallint"] = pandas_frame["avocado_id"].astype("int16")
    pandas_frame["avocado_id_int32"] = pandas_frame["avocado_id"].astype("int32")
    pandas_frame["avocado_id_int"] = pandas_frame["avocado_id"].astype("int32")
    pandas_frame["avocado_id_int64"] = pandas_frame["avocado_id"].astype("int64")
    pandas_frame["avocado_id_bigint"] = pandas_frame["avocado_id"].astype("int64")
    pandas_frame["avocado_id_float"] = pandas_frame["avocado_id"].astype("float")
    pandas_frame["avocado_id_float16"] = pandas_frame["avocado_id"].astype("float16")
    pandas_frame["avocado_id_float32"] = pandas_frame["avocado_id"].astype("float32")
    pandas_frame["avocado_id_float64"] = pandas_frame["avocado_id"].astype("float64")
    pandas_frame["avocado_id_bool"] = pandas_frame["avocado_id"].astype("bool")
    pandas_frame["avocado_id_category"] = pandas_frame["avocado_id"].astype("category")
    pandas_frame["date"] = pandas_frame["Date"].astype("datetime64")
    pandas_frame["time"] = pandas_frame["Date"].astype("datetime64")
    pandas_frame["region_varchar"] = pandas_frame["region"].astype("string")
    pandas_frame["region_string"] = pandas_frame["region"].astype("string")
    pandas_frame = pandas_frame.drop(columns=["avocado_id", "Date", "region"])

    tm.assert_frame_equal(pandas_frame, my_frame)


def test_math_order_of_operations_no_parens():
    """
    Test math parentheses
    :return:
    """

    my_frame = query("select 20 * avocado_id + 3 / 20 as my_math from avocado")

    pandas_frame = AVOCADO.copy()[["avocado_id"]]
    pandas_frame["my_math"] = 20 * pandas_frame["avocado_id"] + 3 / 20

    pandas_frame = pandas_frame.drop(columns=["avocado_id"])

    tm.assert_frame_equal(pandas_frame, my_frame)


def test_math_order_of_operations_with_parens():
    """
    Test math parentheses
    :return:
    """

    my_frame = query(
        "select 20 * (avocado_id + 3) / (20 + avocado_id) as my_math from avocado"
    )

    pandas_frame = AVOCADO.copy()[["avocado_id"]]
    pandas_frame["my_math"] = (
        20 * (pandas_frame["avocado_id"] + 3) / (20 + pandas_frame["avocado_id"])
    )

    pandas_frame = pandas_frame.drop(columns=["avocado_id"])

    tm.assert_frame_equal(pandas_frame, my_frame)


def test_boolean_order_of_operations_with_parens():
    """
    Test boolean order of operations with parentheses
    :return:
    """
    my_frame = query(
        "select * from forest_fires "
        "where (month = 'oct' and day = 'fri') or "
        "(month = 'nov' and day = 'tue')"
    )

    pandas_frame = FOREST_FIRES.copy()
    pandas_frame = pandas_frame[
        ((pandas_frame["month"] == "oct") & (pandas_frame["day"] == "fri"))
        | ((pandas_frame["month"] == "nov") & (pandas_frame["day"] == "tue"))
    ].reset_index(drop=True)

    tm.assert_frame_equal(pandas_frame, my_frame)


if __name__ == "__main__":
    register_env_tables()

    test_sql_data_types()

    remove_env_tables()
