"""
Convert sql statement to run on pandas dataframes
"""
from sqlparse import split
from typing import Tuple
from lark import Lark, Transformer, v_args
from lark.lexer import Token
from lark.exceptions import UnexpectedCharacters
from lark.tree import Tree
from pandas import DataFrame
from sql_exception import MultipleQueriesException, InvalidQueryException, DataFrameDoesNotExist

with open(file="sql.grammar") as sql_grammar_file:
    GRAMMAR_TEXT = sql_grammar_file.read()

def num_eval(arg):
    """
    Takes an argument that may be a string or number and outputs a number
    :param arg:
    :return:
    """
    assert isinstance(arg, Token) or isinstance(arg, float) or isinstance(arg, int)
    if isinstance(arg, str):
        return eval(arg)
    return arg


@v_args(inline=True)
class SQLTransformer(Transformer):
    """
    Transformer for the lark sql parser
    """
    def __init__(self, env):
        self.all_dataframes = {key.lower(): {"DataFrame": env[key], "frame_var_name": key} for key in env
                               if isinstance(env[key], DataFrame)}

    def mul(self, arg1, arg2):
        return num_eval(arg1) * num_eval(arg2)

    def add(self, arg1, arg2):
        return num_eval(arg1) + num_eval(arg2)

    def sub(self, arg1, arg2):
        return num_eval(arg1) - num_eval(arg2)

    def div(self, arg1, arg2):
        return num_eval(arg1) / num_eval(arg2)

    def table(self, table_name):
        """
        Check for existance of pandas dataframe with same name
        If not exists raise DataFrameDoesNotExist
        Otherwise return the name of the actual DataFrame
        :return:
        """
        if table_name not in self.all_dataframes:
            raise DataFrameDoesNotExist(table_name)
        return self.all_dataframes[table_name]["frame_var_name"]

    def select(self, *select_expressions: Tuple[Tree]):
        """
        Forms the final sequence of methods that will be executed
        :param select_expressions:
        :return:
        """
        dataframe_names_tree: Tree = select_expressions[-1]
        dataframe_names = dataframe_names_tree.children
        expression_trees = select_expressions[:-1]
        return *select_expressions


    def alias_string(self, alias):
        return str(alias)

    def column_name(self, column_name):
        return str(column_name)

class SqlToPandas:
    """
    Class that handles conversion from sql to pandas data frame methods.
    """
    def __init__(self, sql: str, all_global_vars):
        self.sql = sql
        self.verify_sql()
        self.parser = Lark(GRAMMAR_TEXT, parser='lalr', start='query_expr', transformer=SQLTransformer(all_global_vars))
        self.ast = self.parse_sql()
        print(self.ast.pretty())

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
        # TODO Put more specific errors
        try:
            return self.parser.parse(self.sql)
        except UnexpectedCharacters as err:
            message_begin = "Invalid query!"
            message_reason = ""
            message_loc = f"Unexpected input at line {err.line}, column {err.column}."
            if err.allowed == {'SELECT'}:
                message_reason = "Not a select statement."
            raise InvalidQueryException(f"{message_begin} {message_reason} {message_loc} \nQuery:"
                                        f"\n{err.get_context(self.sql)}")

    def execute_sql(self):
        """
        Run the query that was pass into the sql variable
        :return:
        """
        print(self.ast)
