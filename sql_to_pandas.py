"""
Convert sql statement to run on pandas dataframes
"""
import re
from typing import Tuple, List
from sqlparse import split
from lark import Lark, Transformer, v_args
from lark.lexer import Token
from lark.exceptions import UnexpectedToken
from lark.tree import Tree
from pandas import DataFrame
from sql_exception import MultipleQueriesException, InvalidQueryException, DataFrameDoesNotExist

with open(file="sql.grammar") as sql_grammar_file:
    GRAMMAR_TEXT = sql_grammar_file.read()

SHOW_TREE = False


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


class Expression:
    """
    Store information about an expression
    """
    value_name = "value"

    def __init__(self, value, alias='', typename=''):
        self.value = value
        self.alias = alias
        self.typename = typename
        if self.alias:
            self.final_name = self.alias
        else:
            self.final_name = self.value

    def __repr__(self):
        display = f"{type(self).__name__}({self.value_name}={self.value}"
        if self.alias:
            display += f", alias={self.alias}"
        if self.typename:
            display += f", type={self.typename}"
        return display + ")"


class Column(Expression):
    """
    Store information about columns
    """
    value_name = "name"

    def __init__(self, value: str, alias='', typename=''):
        super(Column, self).__init__(value, alias, typename)

# pylint: disable=no-self-use, super-init-not-called
@v_args(inline=True)
class SQLTransformer(Transformer):
    """
    Transformer for the lark sql parser
    """
    def __init__(self, env):
        self.dataframe_name_map = {key.lower(): key for key in env if isinstance(env[key], DataFrame)}
        self.dataframe_map = {key: env[key] for key in env if isinstance(env[key], DataFrame)}

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
        return Tree("number", numerical_value)

    def column_name(self, name: str):
        """
        Returns a column token with the name extracted
        :param name: Name of column
        :return: Tree with column token
        """
        return Tree("column_name", str(name))

    def alias_string(self, name: str):
        """
        Returns an alias token with the name extracted
        :param name:
        :return:
        """
        return Tree("alias", str(name))

    # def typename(self, *args):
    #     print(args)

    def as_type(self, expression, typename):
        """
        Extracts token type and returns tree object with expression and type
        :param expression: Expression to be evaluated / the name of a column
        :param typename: Data type
        :return:
        """
        return Tree("as_type", (expression, str(typename)))

    def select_expression(self, expression: Tree, alias: Tree = ''):
        """
        Returns the appropriate object for the given expression
        :param expression: An expression token
        :param alias: A token containing the name to be assigned to the expression
        :return:
        """
        typename = ''
        if expression.data == "as_type":
            typename = expression.children[1]
            expression = expression.children[0]
        found_alias = ''
        if alias:
            found_alias = alias.children
        if expression.data == "column_name":
            column_name = expression.children
            return Column(value=column_name, alias=found_alias, typename=typename)
        return Expression(value=expression.children, alias=found_alias, typename=typename)

    def select(self, *select_expressions: Tuple[Tree]):
        """
        Forms the final sequence of methods that will be executed
        :param select_expressions:
        :return:
        """
        print(select_expressions)
        dataframe_names_tree: Tree = select_expressions[-1]
        dataframe_names = dataframe_names_tree.children

        distinct = False
        non_dataframe_tokens = select_expressions[:-1]
        if isinstance(non_dataframe_tokens[0], Token):
            select_constraint_token = non_dataframe_tokens[0]
            non_dataframe_tokens = non_dataframe_tokens[1:]
            if str(select_constraint_token) == "distinct":
                distinct = True

        columns = []
        aliases = {}
        expressions = []
        all_names = []
        name_order = {}
        conversions = {}
        for token_pos, token in enumerate(non_dataframe_tokens):
            all_names.append(token.final_name)
            name_order[token.final_name] = token_pos
            if token.alias:
                aliases[token.value] = token.alias

            if isinstance(token, Expression) and token.typename:
                conversions[token.final_name] = token.typename

            if isinstance(token, Column):
                columns.append(token.value)
                continue
            if isinstance(token, Expression):
                expressions.append(token.value)

        return {"columns": columns, "expressions": expressions, "aliases": aliases, "dataframes": dataframe_names,
                "name_order": name_order, "all_names": all_names, "conversions": conversions, "distinct": distinct}

    def full_query(self, query_info):
        # TODO Add in support for set operations like union
        print(query_info)
        return query_info

    def get_frame(self, frame_name):
        """
        Returns the dataframe with the name given
        :param frame_name:
        :return:
        """
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

    def to_dataframe(self, query_info):
        """
        Returns the dataframe resulting from the SQL query
        :return:
        """
        frame_names = query_info["dataframes"]
        aliases = query_info["aliases"]
        conversions = query_info["conversions"]
        all_names = query_info["all_names"]
        first_frame = self.get_frame(frame_names[0])


        columns = query_info["columns"]
        if self.has_star(columns):
            new_frame: DataFrame = first_frame.copy()
        else:
            new_frame: DataFrame = first_frame[columns].rename(columns=query_info["aliases"])

        # Evaluate in-line expressions
        for expression in query_info["expressions"]:
            new_frame[aliases[expression]] = expression

        if conversions:
            return new_frame.astype(conversions)
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
