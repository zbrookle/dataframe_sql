"""
Shared functions among the tests like setting up test environment
"""
from pathlib import Path

from pandas import DataFrame, read_csv
import pytest

from dataframe_sql import register_temp_table, remove_temp_table

DATA_PATH = Path(__file__).parent.parent / "data"


# Import the data for testing
FOREST_FIRES = read_csv(DATA_PATH / "forestfires.csv")
DIGIMON_MON_LIST = read_csv(DATA_PATH / "DigiDB_digimonlist.csv")
DIGIMON_MOVE_LIST = read_csv(DATA_PATH / "DigiDB_movelist.csv")
DIGIMON_SUPPORT_LIST = read_csv(DATA_PATH / "DigiDB_supportlist.csv")
AVOCADO = read_csv(DATA_PATH / "avocado.csv")

# Name change is for name interference
DIGIMON_MON_LIST["mon_attribute"] = DIGIMON_MON_LIST["Attribute"]
DIGIMON_MOVE_LIST["move_attribute"] = DIGIMON_MOVE_LIST["Attribute"]


def register_env_tables():
    """
    Returns all globals but in lower case
    :return:
    """
    for variable_name in globals():
        variable = globals()[variable_name]
        if isinstance(variable, DataFrame):
            register_temp_table(frame=variable, table_name=variable_name)


def remove_env_tables():
    """
    Remove all env tables
    :return:
    """
    for variable_name in globals():
        variable = globals()[variable_name]
        if isinstance(variable, DataFrame):
            remove_temp_table(table_name=variable_name)


def fix_naming_inconsistencies(pandas_frame: DataFrame) -> DataFrame:
    pandas_frame = pandas_frame.rename(
        columns={"Type_y": "DIGIMON_MOVE_LIST.Type", "Type_x": "DIGIMON_MON_LIST.Type"}
    )
    if "Attribute_x" in pandas_frame.columns:
        pandas_frame = pandas_frame.rename(
            columns={
                "Attribute_y": "DIGIMON_MOVE_LIST.Attribute",
                "Attribute_x": "DIGIMON_MON_LIST.Attribute",
            }
        )
    if "Attribute" in pandas_frame.columns:
        pandas_frame["DIGIMON_MOVE_LIST.Attribute"] = pandas_frame["Attribute"]
        pandas_frame = pandas_frame.rename(
            columns={"Attribute": "DIGIMON_MON_LIST.Attribute"}
        )
    pandas_frame = pandas_frame[
        [
            "Number",
            "Digimon",
            "Stage",
            "DIGIMON_MON_LIST.Type",
            "DIGIMON_MON_LIST.Attribute",
            "Memory",
            "Equip Slots",
            "Lv 50 HP",
            "Lv50 SP",
            "Lv50 Atk",
            "Lv50 Def",
            "Lv50 Int",
            "Lv50 Spd",
            "mon_attribute",
            "Move",
            "SP Cost",
            "DIGIMON_MOVE_LIST.Type",
            "Power",
            "DIGIMON_MOVE_LIST.Attribute",
            "Inheritable",
            "Description",
            "move_attribute",
        ]
    ]
    return pandas_frame


join_params = pytest.mark.parametrize(
    ("sql_join", "pandas_join"),
    [
        ("", "inner"),
        ("inner", "inner"),
        ("full outer", "outer"),
        ("full", "outer"),
        ("left outer", "left"),
        ("left", "left"),
        ("right outer", "right"),
        ("right", "right"),
    ],
)
