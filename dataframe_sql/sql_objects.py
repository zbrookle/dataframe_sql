"""
Module containing all sql objects
"""
from datetime import date, datetime
from typing import Any, List, Optional, Tuple

from lark import Transformer
from pandas import Series


# pylint: disable=too-few-public-methods
class AmbiguousColumn:
    """
    Class for identifying ambiguous table names
    """

    def __init__(self, tables: List[str]) -> None:
        self.tables = tables

    def __repr__(self) -> str:
        return f"AmbiguousColumn({','.join(self.tables)})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AmbiguousColumn) and self.tables == other.tables


class Value:
    """
    Parent class for expression_count and columns
    """

    def __init__(self, value, alias="", typename=""):
        self.value = value
        self.alias = alias
        self.typename = typename
        self.final_name = alias

    def __repr__(self):
        if isinstance(self.value, Series):
            print_value = "SeriesObject"
        else:
            print_value = self.value

        display = (
            f"{type(self).__name__}(final_name={self.final_name}, value={print_value}"
        )
        if self.alias:
            display += f", alias={self.alias}"
        if self.typename:
            display += f", type={self.typename}"
        return display

    def __add__(self, other):
        return Expression(
            value=self.value + self.get_other_value(other),
            alias=self.alias,
            execution_plan=f"{self.get_plan_representation()} + "
            f"{other.get_plan_representation()}",
        )

    def __sub__(self, other):
        return Expression(
            value=self.value - self.get_other_value(other),
            alias=self.alias,
            execution_plan=f"{self.get_plan_representation()} - "
            f"{other.get_plan_representation()}",
        )

    def __mul__(self, other):
        return Expression(
            value=self.value * self.get_other_value(other),
            alias=self.alias,
            execution_plan=f"{self.get_plan_representation()} * "
            f"{other.get_plan_representation()}",
        )

    def __truediv__(self, other):
        return Expression(
            value=self.value / self.get_other_value(other),
            alias=self.alias,
            execution_plan=f"{self.get_plan_representation()} / "
            f"{other.get_plan_representation()}",
        )

    def get_table(self):
        """
        Returns the table of the current value
        :return:
        """
        return None

    def get_name(self) -> str:
        """
        Returns the name of the current value
        :return:
        """
        return self.final_name

    def get_value(self):
        """
        Returns the value of the object
        :return:
        """
        return self.value

    def get_plan_representation(self) -> str:
        """
        Return the representation that the object will have in the execution plan
        :return:
        """
        return f"{self.get_value()}"

    @staticmethod
    def get_other_name(other) -> str:
        """
        Gets the name representation for the other value
        :param other:
        :return:
        """
        if isinstance(other, Value):
            return other.get_name()
        return str(other)

    @staticmethod
    def get_other_table(other) -> Optional[str]:
        """
        Gets the name representation for the other value
        :param other:
        :return:
        """
        return other.get_table()

    @staticmethod
    def get_other_value(other):
        """
        Return the appropriate value based on the type of other
        :param other:
        :return:
        """
        if isinstance(other, Value):
            return other.get_value()
        return other

    def set_alias(self, alias):
        """
        Sets the alias and final name for the value object
        :param alias:
        :return:
        """
        self.alias = alias
        self.final_name = alias

    def __gt__(self, other):
        if isinstance(other, Value):
            return self.value > other.value
        return self.value > other

    def __lt__(self, other):
        if isinstance(other, Value):
            return self.value < other.value
        return self.value < other

    def __ge__(self, other):
        if isinstance(other, Value):
            return self.value >= other.value
        return self.value >= other

    def __le__(self, other):
        if isinstance(other, Value):
            return self.value <= other.value
        return self.value <= other


class Literal(Value):
    """
    Stores literal data
    """

    literal_count = 0

    def __init__(self, value, alias=""):
        Value.__init__(self, value, alias)
        if not alias:
            self.alias = f"_literal{self.literal_count}"
            self.literal_count += 1

    def __repr__(self):
        return Value.__repr__(self) + ")"

    def get_name(self):
        return str(self.value)

    def get_plan_representation(self) -> str:
        if isinstance(self.value, str):
            return f"'{self.value}'"
        elif isinstance(self.value, date) and not isinstance(self.value, datetime):
            return self.value.strftime("date(%Y, %-m, %-d)")
        elif isinstance(self.value, datetime):
            return self.value.strftime("datetime(%Y, %-m, %-d, %-H, %-M, %-S)")
        return f"{self.value}"


class Number(Literal):
    """
    Stores numerical data
    """

    def __init__(self, value):
        Literal.__init__(self, value)


class String(Literal):
    """
    Store information about a string literal
    """

    def __init__(self, value):
        Literal.__init__(self, value)


class Date(Literal):
    """
    Store information about a date literal
    """

    def __init__(self, value):
        Literal.__init__(self, value)


class Bool(Literal):
    """
    Store information about a date literal
    """

    def __init__(self, value):
        Literal.__init__(self, value)


class ValueWithPlan(Value):
    def __init__(self, value, execution_plan):
        Value.__init__(self, value)
        self.execution_plan = execution_plan

    def __repr__(self):
        return Value.__repr__(self) + ")"

    def __or__(self, other):
        if not isinstance(other, Value):
            raise Exception(
                f"Operator | is not supported between type {type(other)} "
                f"and type ValueWithPlan"
            )

        return ValueWithPlan(
            self.get_value() | other.get_value(),
            f"{self.get_plan_representation()} | " f"{other.get_plan_representation()}",
        )

    def __and__(self, other):
        if not isinstance(other, Value):
            raise Exception(
                f"Operator && is not supported between type {type(other)} "
                f"and type ValueWithPlan"
            )

        ValueWithPlan(
            self.get_value() & other.get_value(),
            f"{self.get_plan_representation()} & " f"{other.get_plan_representation()}",
        )

    def get_plan_representation(self) -> str:
        return self.execution_plan


class DerivedColumn(Value):
    """
    Base class for expressions and aggregates
    """

    expression_count = 0

    def __init__(self, value, alias="", typename="", function=""):
        Value.__init__(self, value, alias, typename)
        self.function = function
        if self.alias:
            self.final_name = self.alias
        else:
            if isinstance(self.value, (Series, Column)):
                self.final_name = f"_col{self.expression_count}"
                self.alias = self.final_name
                DerivedColumn.increment_expression_count()
            else:
                self.final_name = str(self.value)
        self.has_columns = True

    def __repr__(self):
        display = Value.__repr__(self)
        if self.function:
            display += f", function={self.function}"
        return display + ")"

    @classmethod
    def increment_expression_count(cls):
        cls.expression_count += 1

    @classmethod
    def reset_expression_count(cls):
        cls.expression_count = 0


class Expression(DerivedColumn):
    """
    Store information about an sql_object
    """

    def __init__(self, value, alias="", typename="", function="", execution_plan=""):
        DerivedColumn.__init__(self, value, alias, typename, function)
        self.execution_plan = execution_plan

    def evaluate(self):
        """
        Returns the value from the sql_object
        :return:
        """
        if isinstance(self.value, Column):
            return self.value.value
        return self.value

    def get_name(self) -> str:
        return self.alias

    def get_plan_representation(self) -> str:
        return self.execution_plan


class Aggregate(DerivedColumn):
    """
    Store information about aggregations
    """

    _function_map = {
        "average": "mean",
        "avg": "mean",
        "mean": "mean",
        "maximum": "max",
        "max": "max",
        "minimum": "min",
        "min": "min",
        "sum": "sum",
    }

    def __init__(self, value, function, alias="", typename=""):
        DerivedColumn.__init__(
            self, value, alias, typename, self._function_map[function.lower()]
        )


class Column(Value):
    """
    Store information about columns
    """

    def __init__(self, name: str, alias="", typename="", value=None):
        Value.__init__(self, value, alias, typename)
        self.name = name
        if self.alias:
            self.final_name = self.alias
        else:
            self.final_name = self.name
        self.table = None

    def __repr__(self):
        display = Value.__repr__(self)
        display += f", name={self.name}"
        display += f", table={self.table}"
        return display + ")"

    def __eq__(self, other):
        other = self.get_other_value(other)
        return self.value == other

    def __gt__(self, other):
        other = self.get_other_value(other)
        return self.value > other

    def __lt__(self, other):
        other = self.get_other_value(other)
        return self.value < other

    def __ge__(self, other):
        other = self.get_other_value(other)
        return self.value >= other

    def __le__(self, other):
        other = self.get_other_value(other)
        return self.value <= other

    def set_value(self, new_value: Series):
        """
        Set the value of the column to value
        :param new_value:
        :return:
        """
        self.value = new_value

    def get_name(self):
        return self.name

    def get_table(self):
        return self.table

    def get_plan_representation(self):
        return f"{self.table}['{self.name}']"


class Subquery:
    """
    Wrapper for subqueries
    """

    def __init__(self, name: str, query_info: dict, execution_plan: str):
        self.name = name
        self.query_info = query_info
        self.execution_plan = execution_plan

    def __repr__(self):
        return f"Subquery(name={self.name}, query_info={self.query_info})"


class Join:
    """
    Wrapper for join related info
    """

    def __init__(
        self,
        left_table: str,
        right_table: str,
        join_type: str,
        left_on: str,
        right_on: str,
    ):
        self.left_table = left_table
        self.right_table = right_table
        self.join_type = join_type
        self.left_on = left_on
        self.right_on = right_on


class QueryInfo:
    """
    Class that holds metadata extracted / derived from a sql query
    """

    def __init__(self):
        self.column_selected = {}
        self.columns = []
        self.expressions = []
        self.literals = []
        self.frame_names = []
        self.aliases = {}
        self.all_names = []
        self.name_order = {}
        self.conversions = {}
        self.aggregates = {}
        self.group_columns = []
        self.where_expr = None
        self.distinct = False
        self.having_expr = None
        self.internal_transformer: Optional[Transformer] = None
        self.order_by: List[Tuple[str, bool]] = []
        self.limit: Optional[int] = None
        self.having_transformer: Optional[Transformer] = None

    @staticmethod
    def set_none_var(value, default):
        return default if not value else value
