"""
Convert sql statement to run on pandas dataframes
"""
import re
from typing import Tuple, List
import numpy as np
from sqlparse import split
from lark import Lark, Transformer, v_args
from lark.lexer import Token
from lark.exceptions import UnexpectedToken
from lark.tree import Tree
from pandas import DataFrame, Series
from sql_exception import MultipleQueriesException, InvalidQueryException, DataFrameDoesNotExist

with open(file="sql.grammar") as sql_grammar_file:
    GRAMMAR_TEXT = sql_grammar_file.read()

SHOW_TREE = False
GET_TABLE_REGEX = re.compile(r"^(?P<table>[a-z_]\w*)\.(?P<column>[a-z_]\w*)$", re.IGNORECASE)


def get_child_from_list(tree: Tree):
    """
    Returns value from the first token in a list of child tokens
    """
    return tree.children[0]


def num_eval(arg):
    """
    Takes an argument that may be a string or number and outputs a number
    :param arg:
    :return:
    """
    assert isinstance(arg, (Token, float, int))
    if isinstance(arg, str):
        # pylint: disable=eval-used
        return eval(arg)
    return arg

class Value:
    """
    Parent class for expressions and columns
    """

    def __init__(self, value, alias='', typename=''):
        self.value = value
        self.alias = alias
        self.typename = typename
        self.final_name = alias

    def __repr__(self):
        display = f"{type(self).__name__}(final_name={self.final_name}, value={self.value}"
        if self.alias:
            display += f", alias={self.alias}"
        if self.typename:
            display += f", type={self.typename}"
        return display

    def set_alias(self, alias):
        self.alias = alias
        self.final_name = alias

class Number(Value):
    """
    Stores numerical data
    """

    def __init__(self, value):
        super(Number, self).__init__(value)

class Expression(Value):
    """
    Store information about an expression
    """

    def __init__(self, value, alias='', typename='', function=''):
        self.value = value
        self.alias = alias
        self.typename = typename
        self.function = function
        if self.alias:
            self.final_name = self.alias
        else:
            self.final_name = self.value
            if self.function:
                if isinstance(self.value, Column):
                    expression_name = self.value.name
                else:
                    expression_name = str(self.value)
                self.alias = self.function + "_" + expression_name
                self.final_name = self.alias
        self.has_columns = True
        # self.set_expression_type()

    def __repr__(self):
        display = super(Expression, self).__repr__()
        if self.function:
            display += f", function={self.function}"
        return display + ")"

    def evaluate(self):
        if isinstance(self.value, Column):
            return self.value.value
        return self.value

    # def set_expression_type(self):
    #     """
    #     Set the type of expression (string, number, tbd)
    #     :return:
    #     """
    #     if isinstance(self.value, (int, float)) or re.match(r"^'.*'$", self.value):
    #         self.has_columns = False


class Column(Value):
    """
    Store information about columns
    """

    def __init__(self, name: str, alias='', typename=''):
        super(Column, self).__init__(None, alias, typename)
        self.name = name
        if self.alias:
            self.final_name = self.alias
        else:
            self.final_name = self.name

    def __repr__(self):
        display = super(Column, self).__repr__()
        display += f", name={self.name}"
        return display + ")"

    def set_value(self, new_value: Series):
        """
        Set the value of the column to value
        :param new_value:
        :return:
        """
        self.value = new_value


class Subquery:
    """
    Wrapper for subqueries
    """

    def __init__(self, name: str, query_info: dict):
        self.name = name
        self.query_info = query_info

    def __repr__(self):
        return f"Subquery(name={self.name}, query_info={self.query_info})"

# pylint: disable=no-self-use, super-init-not-called
@v_args(inline=True)
class SQLTransformer(Transformer):
    """
    Transformer for the lark sql parser
    """
    def __init__(self, env):
        self.dataframe_name_map = {}
        self.dataframe_map = {}
        self.column_name_map = {}
        self._temp_dataframes_dict = {}
        for key in env:
            if isinstance(env[key], DataFrame):
                dataframe = env[key]
                self.dataframe_name_map[key.lower()] = key
                self.dataframe_map[key] = dataframe
                self.column_name_map[key] = {}
                for column in dataframe.columns:
                    self.column_name_map[key][column.lower()] = column

    def mul(self, arg1, arg2):
        """
        Returns the product two numbers
        """
        return num_eval(arg1) * num_eval(arg2)

    def add(self, arg1, arg2):
        """
        Returns the sum two numbers
        """
        return num_eval(arg1) + num_eval(arg2)

    def sub(self, arg1, arg2):
        """
        Returns the difference between two numbers
        """
        return num_eval(arg1) - num_eval(arg2)

    def div(self, arg1, arg2):
        """
        Returns the division of two numbers
        """
        return num_eval(arg1) / num_eval(arg2)

    def function_name(self, function_name):
        function_name = function_name.lower()
        return Token("function", function_name)

    # def sql_function(self, function, expression):
    #     print(expression)
    #     self._funcs_to_apply[expression] = function
    #     return Tree("column_name", )

    def table(self, table_name):
        """
        Check for existance of pandas dataframe with same name
        If not exists raise DataFrameDoesNotExist
        Otherwise return the name of the actual DataFrame
        :return:
        """
        table_name = table_name.lower()
        if table_name not in self.dataframe_name_map:
            raise DataFrameDoesNotExist(table_name)
        return self.dataframe_name_map[table_name]

    def number(self, numerical_value):
        """
        Return a number token with a numeric value as a child
        :param numerical_value:
        :return:
        """
        return Number(numerical_value)

    def column_name(self, *names: str):
        """
        Returns a column token with the name extracted
        :param name: Name of column
        :return: Tree with column token
        """
        full_name = ".".join([str(name) for name in names])
        return Column(full_name)

    def alias_string(self, name: str):
        """
        Returns an alias token with the name extracted
        :param name:
        :return:
        """
        return Tree("alias", str(name))

    def as_type(self, expression, typename):
        """
        Extracts token type and returns tree object with expression and type
        :param expression: Expression to be evaluated / the name of a column
        :param typename: Data type
        :return:
        """
        expression.typename = typename.value
        return expression

    def from_expression(self, column):
        return Token("from_expression", column)

    def select_expression(self, expression, alias: Tree = ''):
        """
        Returns the appropriate object for the given expression
        :param expression: An expression token
        :param alias: A token containing the name to be assigned to the expression
        :return:
        """
        if isinstance(expression, Tree):
            value = expression.children
            if expression.data == "sql_function":
                function = value[0]
                print(value[1])
                value = value[1]
                expression = Expression(value=value, function=function)

        if alias:
            expression.set_alias(alias.children)
        # return Expression(value=value, alias=found_alias, typename=typename, function=function)
        return expression

    def get_lower_columns(self, table_name):
        """
        Returns a list of lower case column names for a given table name
        :param column_list:
        :return:
        """
        return [column.lower() for column in list(self.get_frame(table_name).columns)]

    def determine_column_side(self, column, left_table, right_table):
        """
        Check if column table prefix is one of the two tables (if there is one) AND the column has to be in one of the
        two tables
        """
        left_columms = self.get_lower_columns(left_table)
        right_columns = self.get_lower_columns(right_table)
        column_match = GET_TABLE_REGEX.match(column)
        column_table = ""
        if column_match:
            column = column_match.group("column")
            column_table = column_match.group("table")

        if column not in left_columms and column not in right_columns:
            raise Exception("Column not found")

        if column_table:
            if column_table == left_table and column in left_columms:
                return "left", column
            elif column_table == right_table and column in right_columns:
                return "right", column
            else:
                raise Exception("Table specified in join columns not present in join")
        else:
            if column in left_columms and column in right_columns:
                raise Exception(f"Ambiguous column: {column}\nSpecify table name with table_name.{column}")
            elif column in left_columms:
                return "left", column
            elif column in right_columns:
                return "right", column
            else:
                raise Exception("Column does not exist in either table")

    def join(self, *args):
        return args[0]

    def join_expression(self, *args):
        """
        Evaluate a join into one dataframe using a merge method
        :param table1: The first dataframe
        :param join_type: The type of join ex: inner, outer, right, left
        :param table2:
        :param join_condition:
        :return:
        """
        # There will only ever be four args if a join is specified and three if a join isn't specified
        if len(args) == 3:
            join_type = "inner"
            table1 = args[0]
            table2 = args[1]
            join_condition = args[2]
        else:
            table1 = args[0]
            join_type = args[1]
            table2 = args[2]
            join_condition = args[3]
            if "outer" in join_type:
                match = re.match(r"(?P<type>.*)\souter", join_type)
                join_type = match.group("type")
            if join_type in ("full", "cross"):
                join_type = "outer"
        frame1: DataFrame = self.get_frame(table1)
        frame2: DataFrame = self.get_frame(table2)

        # Check that there is a column from both sides
        boolean_expression = join_condition.children[0]
        column_comparison = boolean_expression.children
        column1 = str(column_comparison[0].name)
        column2 = str(column_comparison[1].name)

        column1_side, column1 = self.determine_column_side(column1, table1, table2)
        column2_side, column2 = self.determine_column_side(column2, table1, table2)
        if column1_side == column2_side:
            raise Exception("Join columns must be one column from each join table!")
        else:
            column1 = self.column_name_map[table1][column1]
            column2 = self.column_name_map[table2][column2]
            if column1_side == "left":
                left_on = column1
                right_on = column2
            else:
                left_on = column2
                right_on = column1

        dictionary_name = f"{table1}x{table2}"
        self._temp_dataframes_dict[dictionary_name] = frame1.merge(right=frame2, how=join_type,
                                                                   left_on=left_on, right_on=right_on)
        return Subquery(dictionary_name, query_info="")

    def group_by(self, column):
        return Token("group", str(column.name))

    def full_query(self, query_info):
        # TODO Add in support for set operations like union
        return query_info

    def get_frame(self, frame_name) -> DataFrame:
        """
        Returns the dataframe with the name given
        :param frame_name:
        :return:
        """
        if isinstance(frame_name, Subquery):
            return self._temp_dataframes_dict[frame_name.name]
        return self.dataframe_map[frame_name]

    @staticmethod
    def has_star(column_list: List[str]):
        """
        Returns true if any columns have a star
        :param column_list:
        :return:
        """
        for column_name in column_list:
            if re.match(r"\*", column_name):
                return True
        return False

    def subquery(self, query_info, alias):
        alias_name = alias.children
        self._temp_dataframes_dict[alias_name] = self.to_dataframe(query_info)
        return Subquery(name=alias_name, query_info=query_info)

    def set_column_value(self, column: Column, dataframe_names: List[str]):
        """
        Sets the column value based on what it is in the dataframe
        :param column:
        :param dataframe_names:
        :return:
        """
        for dataframe_name in dataframe_names:
            dataframe = self.get_frame(dataframe_name)
            if column.name in dataframe.columns:
                if column.value:
                    raise Exception(f"Ambiguous column reference: {column.name}")
                column.value = dataframe[column.name]

    def select(self, *select_expressions: Tuple[Tree]):
        """
        Forms the final sequence of methods that will be executed
        :param select_expressions:
        :return:
        """
        print("Select Expressions:", select_expressions)

        distinct = False
        if isinstance(select_expressions[0], Token):
            select_constraint_token = select_expressions[0]
            select_expressions = select_expressions[1:]
            if str(select_constraint_token) == "distinct":
                distinct = True
        columns = []
        expressions = []
        numbers = []
        group_columns = []
        dataframe_names = []
        aliases = {}
        all_names = []
        name_order = {}
        conversions = {}
        aggregates = {}
        for token_pos, token in enumerate(select_expressions):
            if isinstance(token, Token):
                if token.type == "from_expression":
                    dataframe_names.append(token.value)
                elif token.type == "group":
                    group_columns.append(token.value)
            else:
                all_names.append(token.final_name)
                name_order[token.final_name] = token_pos

                if token.typename:
                    conversions[token.final_name] = token.typename

                if isinstance(token, Column):
                    columns.append(token)
                    if token.alias:
                        aliases[token.name] = token.alias

                if isinstance(token, Expression):
                    expressions.append(token)
                    if token.alias:
                        aliases[token.value] = token.alias
                    if token.function:
                        aggregates[token.alias] = token.function

                if isinstance(token, Number):
                    numbers.append(token)

            # # Determine which table columns are in
            # for column in columns:
            #     for dataframe_name in dataframe_names:
            #         if column in self.get_frame(dataframe_name).columns:
            #             print("yes!")
            #         else:
            #             print("no :(")

            for expression in expressions:
                if isinstance(expression.value, Column):
                    column = expression.value
                    self.set_column_value(column, dataframe_names)

        return {"columns": columns, "expressions": expressions, "dataframes": dataframe_names,
                "name_order": name_order, "all_names": all_names, "conversions": conversions, "distinct": distinct,
                "group_columns": group_columns, "aliases": aliases, "numbers": numbers, "aggregates": aggregates}

    def to_dataframe(self, query_info):
        """
        Returns the dataframe resulting from the SQL query
        :return:
        """
        print("Query info:", query_info)
        frame_names = query_info["dataframes"]
        conversions = query_info["conversions"]
        all_names = query_info["all_names"]
        numbers = query_info["numbers"]
        first_frame = self.get_frame(frame_names[0])

        column_names = [column.name for column in query_info["columns"]]
        if self.has_star(column_names):
            new_frame: DataFrame = first_frame.copy()
        else:
            new_frame: DataFrame = first_frame[column_names].rename(columns=query_info["aliases"])

        # Evaluate in-line expressions
        for expression in query_info["expressions"]:
            new_frame[expression.alias] = expression.evaluate()

        for number in numbers:
            new_frame[number.alias] = number.value

        if conversions:
            return new_frame.astype(conversions)

        if query_info["group_columns"]:
           new_frame = new_frame.groupby(query_info["group_columns"]).size().to_frame('size').reset_index().drop(columns=['size'])

        if query_info["aggregates"]:
            print("yes")
            new_frame = new_frame.aggregate(query_info["aggregates"])
            print(new_frame)

        if query_info["distinct"]:
            new_frame.drop_duplicates(keep='first', inplace=True)
            new_frame.reset_index(inplace=True)
            new_frame.drop(columns=['index'], inplace=True)

        return new_frame


class SqlToPandas:
    """
    Class that handles conversion from sql to pandas data frame methods.
    """
    def __init__(self, sql: str, all_global_vars):
        self.sql = sql
        self.verify_sql()
        if SHOW_TREE:
            self.parser = Lark(GRAMMAR_TEXT, parser='lalr')
        else:
            self.parser = Lark(GRAMMAR_TEXT, parser='lalr', transformer=SQLTransformer(all_global_vars))
        self.ast = self.parse_sql()
        if SHOW_TREE:
            print(self.ast.pretty())
        else:
            print(self.ast)
            self.data_frame = self.ast

    def verify_sql(self):
        """
        Verifies that the sql is ONE valid statement
        :param sql: A sql query
        :return:
        """
        split_sql = split(self.sql)
        if len(split_sql) > 1:
            raise MultipleQueriesException

    def parse_sql(self):
        """
        Splits the sql into tokens
        :return:
        """
        try:
            return self.parser.parse(self.sql)
        except UnexpectedToken as err:
            message = f"Invalid query!\n" \
                      f"Expected one of the following input(s): {err.expected}\n" \
                      f"Unexpected input at line {err.line}, column {err.column}\n" \
                      f"{err.get_context(self.sql)}"
            raise InvalidQueryException(message)
