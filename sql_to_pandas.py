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
from pandas import DataFrame, Series, merge
from sql_exception import MultipleQueriesException, InvalidQueryException, DataFrameDoesNotExist

with open(file="sql.grammar") as sql_grammar_file:
    GRAMMAR_TEXT = sql_grammar_file.read()

SHOW_TREE = False
ORDER_TYPES = ['asc', 'desc', 'ascending', 'descending']
ORDER_TYPES_MAPPING = {'asc':'asc', 'desc':'desc', 'ascending':'asc', 'descending':'desc'}
GET_TABLE_REGEX = re.compile(r"^(?P<table>[a-z_]\w*)\.(?P<column>[a-z_]\w*)$", re.IGNORECASE)
FUNCTION_MAPPING = {'average': 'mean', 'avg': 'mean', 'mean': 'mean',
                    'maximum': 'max', 'max': 'max',
                    'minimum': 'min', 'min': 'min'}


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

# pylint: disable=too-few-public-methods
class AmbiguousColumn:
    """
    Class for identifying ambiguous table names
    """

    def __init__(self, tables):
        self.tables = tables


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
        if isinstance(self.value, Series):
            print_value = "SeriesObject"
        else:
            print_value = self.value

        display = f"{type(self).__name__}(final_name={self.final_name}, value={print_value}"
        if self.alias:
            display += f", alias={self.alias}"
        if self.typename:
            display += f", type={self.typename}"
        return display

    def __add__(self, other):
        other_name = self.get_other_name(other)
        other = self.get_other_value(other)
        return Expression(value=self.value + other, alias=f'{self.final_name}_add_{other_name}')

    def __sub__(self, other):
        other_name = self.get_other_name(other)
        other = self.get_other_value(other)
        return Expression(value=self.value - other, alias=f'{self.final_name}_sub_{other_name}')

    def __mul__(self, other):
        other_name = self.get_other_name(other)
        other = self.get_other_value(other)
        return Expression(value=self.value * other, alias=f'{self.final_name}_mul_{other_name}')

    def __truediv__(self, other):
        other_name = self.get_other_name(other)
        other = self.get_other_value(other)
        return Expression(value=self.value / other, alias=f"{self.final_name}_div_{other_name}")

    @staticmethod
    def get_other_name(other):
        """
        Gets the name representation for the other value
        :param other:
        :return:
        """
        if isinstance(other, Column):
            return other.name
        if isinstance(other, Expression):
            return other.alias
        if isinstance(other, Number):
            return str(other.value)
        return str(other)

    @staticmethod
    def get_other_value(other):
        """
        Return the appropriate value based on the type of other
        :param other:
        :return:
        """
        if isinstance(other, (Number, Column, Expression)):
            return other.value
        return other

    def set_alias(self, alias):
        """
        Sets the alias and final name for the value object
        :param alias:
        :return:
        """
        self.alias = alias
        self.final_name = alias


class Number(Value):
    """
    Stores numerical data
    """

    def __init__(self, value):
        super(Number, self).__init__(value)

    def __gt__(self, other):
        if isinstance(other, Number):
            return self.value > other.value
        return self.value > other

    def __lt__(self, other):
        if isinstance(other, Number):
            return self.value < other.value
        return self.value < other


class Expression(Value):
    """
    Store information about an expression
    """

    def __init__(self, value, alias='', typename='', function=''):
        super(Expression, self).__init__(value, alias, typename)
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

    def __repr__(self):
        display = super(Expression, self).__repr__()
        if self.function:
            display += f", function={self.function}"
        return display + ")"

    def evaluate(self):
        """
        Returns the value from the expression
        :return:
        """
        if isinstance(self.value, Column):
            return self.value.value
        return self.value


class Column(Value):
    """
    Store information about columns
    """

    def __init__(self, name: str, alias='', typename='', value=None):
        super(Column, self).__init__(value, alias, typename)
        self.name = name
        if self.alias:
            self.final_name = self.alias
        else:
            self.final_name = self.name
        self.table = None

    def __repr__(self):
        display = super(Column, self).__repr__()
        display += f", name={self.name}"
        display += f", table={self.table}"
        return display + ")"

    def __eq__(self, other):
        other = self.get_other_value(other)
        return self.value == other

    def __gt__(self, other):
        other = self.get_other_value(other)
        return self.value > other

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


class TransformerBaseClass(Transformer):
    """
    Base class for transformers
    """
    def __init__(self, dataframe_name_map=None, dataframe_map=None, column_name_map=None, column_to_dataframe_name=None,
                 _temp_dataframes_dict=None):
        super(TransformerBaseClass, self).__init__(visit_tokens=False)
        self.dataframe_name_map = dataframe_name_map
        self.dataframe_map = dataframe_map
        self.column_name_map = column_name_map
        self.column_to_dataframe_name = column_to_dataframe_name
        self._temp_dataframes_dict = _temp_dataframes_dict

    def get_frame(self, frame_name) -> DataFrame:
        """
        Returns the dataframe with the name given
        :param frame_name:
        :return:
        """
        if isinstance(frame_name, Token):
            frame_name = frame_name.value
        if isinstance(frame_name, Subquery):
            return self._temp_dataframes_dict[frame_name.name]
        return self.dataframe_map[frame_name]

    def set_column_value(self, column: Column):
        """
        Sets the column value based on what it is in the dataframe
        :param column:
        :return:
        """
        if column.name != "*":
            dataframe_name = self.column_to_dataframe_name.get(column.name.lower())
            if isinstance(dataframe_name, AmbiguousColumn):
                raise Exception(f"Ambiguous column reference: {column.name}")
            dataframe = self.get_frame(dataframe_name)
            column_true_name = self.column_name_map[dataframe_name][column.name.lower()]
            column.value = dataframe[column_true_name]
            column.table = dataframe_name

    def column_name(self, name_list_format: List[str]):
        """
        Returns a column token with the name extracted
        :param name_list_format: List formatted name
        :return: Tree with column token
        """
        name = "".join(name_list_format)
        column = Column(name="".join(name))
        self.set_column_value(column)
        return column


# pylint: disable=no-self-use, too-many-public-methods
class InternalTransformer(TransformerBaseClass):
    """
    Evaluates subtrees with knowledge of provided tables that are in the proper scope
    """
    def __init__(self, tables, dataframe_map, column_name_map, column_to_dataframe_name):
        super(InternalTransformer, self).__init__(dataframe_map=dataframe_map, column_name_map=column_name_map)
        self.tables = tables
        self.column_to_dataframe_name = {column: column_to_dataframe_name[column] for column in column_to_dataframe_name
                                         if column_to_dataframe_name[column] in self.tables}

    def mul(self, args):
        """
        Returns the product two numbers
        """
        arg1 = args[0]
        arg2 = args[1]
        return num_eval(arg1) * num_eval(arg2)

    def expression_mul(self, args):
        """
        Returns the product of two expressions
        :param args:
        :return:
        """
        arg1 = args[0]
        arg2 = args[1]
        return arg1 * arg2

    def add(self, args):
        """
        Returns the sum two numbers
        """
        arg1 = args[0]
        arg2 = args[1]
        return num_eval(arg1) + num_eval(arg2)

    def expression_add(self, args):
        """
        Returns the sum of two expressions
        :param args:
        :return:
        """
        arg1 = args[0]
        arg2 = args[1]
        return arg1 + arg2

    def sub(self, args):
        """
        Returns the difference between two numbers
        """
        arg1 = args[0]
        arg2 = args[1]
        return num_eval(arg1) - num_eval(arg2)

    def expression_sub(self, args):
        """
        Returns the difference between two expressions
        :param args:
        :return:
        """
        arg1 = args[0]
        arg2 = args[1]
        return arg1 - arg2

    def div(self, args):
        """
        Returns the division of two numbers
        """
        arg1 = args[0]
        arg2 = args[1]
        return num_eval(arg1) / num_eval(arg2)

    def expression_div(self, args):
        """
        Returns the difference between two expressions
        :param args:
        :return:
        """
        arg1 = args[0]
        arg2 = args[1]
        return arg1 / arg2

    def number(self, numerical_value):
        """
        Return a number token with a numeric value as a child
        :param numerical_value:
        :return:
        """
        return Number(num_eval(numerical_value[0]))

    def string(self, string_token):
        """
        Return value of the token associated with the string
        :param string_token:
        :return:
        """
        return string_token[0].value

    def equals(self, expressions):
        """
        Compares two expressions for equality
        :param expressions:
        :return:
        """
        return expressions[0] == expressions[1]

    def greater_than(self, expressions):
        """
        Performs a greater than expression
        :param expressions:
        :return:
        """
        return expressions[0] > expressions[1]

    def less_than(self, expressions):
        """
        Performs a less than expression
        :param expressions:
        :return:
        """
        return expressions[0] < expressions[1]

    def bool_expression(self, bool_expression):
        """
        Return the bool expression
        :param bool_expression:
        :return: boolean expression
        """
        return bool_expression[0]

    def where_expr(self, truth_value_dataframe):
        """
        Return a where token
        :param truth_value_dataframe:
        :return: Token
        """
        return Token("where_expr", truth_value_dataframe[0])

    def function_name(self, function_name):
        """
        Returns the function name tree
        :param function_name:
        :return:
        """
        function_name = function_name[0].lower()
        true_function_name = FUNCTION_MAPPING.get(function_name)
        if true_function_name:
            return Tree("aggregate", true_function_name)
        return Tree("function", function_name)

    def alias_string(self, name: str):
        """
        Returns an alias token with the name extracted
        :param name:
        :return:
        """
        return Tree("alias", str(name[0]))

    def from_expression(self, expression):
        """
        Return a from expression token
        :param expression:
        :return: Token from expression
        """
        expression = expression[0]
        if isinstance(expression, Subquery):
            value = expression
        else:
            value = expression.value
        return Token("from_expression", value)

    def select_expression(self, expression_and_alias):
        """
        Returns the appropriate object for the given expression
        :param expression_and_alias: An expression token and A token containing the name to be assigned
        :return:
        """
        expression = expression_and_alias[0]
        alias = ''
        if len(expression_and_alias) == 2:
            alias = expression_and_alias[1]
        if isinstance(expression, Tree):
            value = expression.children
            if expression.data == "sql_function":
                function = value[0].children
                value = value[1]
                expression = Expression(value=value, function=function)

        if alias:
            expression.set_alias(alias.children)
        return expression

    def join(self, *args):
        """
        Extracts the join expression
        :param args: Arguments that are passed to the join
        :return: join expression
        """
        return args[0]

    def group_by(self, column):
        """
        Returns a group token
        :param column: Column to group by
        :return: group token
        """
        column = column[0]
        return Token("group", str(column.name))

    def as_type(self, column_and_type):
        """
        Extracts token type and returns tree object with expression and type
        :param expression: Expression to be evaluated / the name of a column
        :param typename: Data type
        :return:
        """
        column = column_and_type[0]
        typename = column_and_type[1]
        column.typename = typename.value
        return column


# pylint: disable=no-self-use
class HavingTransformer(TransformerBaseClass):
    """
    Transformer for having clauses since group by needs to be applied first
    """

    # pylint: disable=too-many-arguments
    def __init__(self, tables, group_by, dataframe_map, column_name_map, column_to_dataframe_name):
        self.tables = tables
        self.group_by = group_by
        super(HavingTransformer, self).__init__(dataframe_map=dataframe_map, column_name_map=column_name_map,
                                                column_to_dataframe_name=column_to_dataframe_name)

    def aggregate(self, function_name_list_form):
        """
        Return the string representation fo aggregate function name instead of list
        :param function_name_list_form:
        :return:
        """
        return "".join(function_name_list_form)

    def function_name(self, tokens):
        """
        Extracts function name from token
        :param fucntion_name:
        :return:
        """
        return tokens[0].value

    def sql_function(self, function_expr):
        aggregate_name = function_expr[0]
        column = function_expr[1]
        table = self.dataframe_map[column.table]
        aggregates = {column.name: aggregate_name}
        if self.group_by:
            new_series = table.groupby(self.group_by).aggregate(aggregates).reset_index()
        else:
            new_series = table.aggregate(aggregates).to_frame().transpose()
        return new_series[column.name]

    def having_expr(self, having_expr):
        internal_transformer = InternalTransformer(self.tables, self.dataframe_map, self.column_name_map,
                                                   self.column_to_dataframe_name)
        having_expr = Tree("having_expr", having_expr)
        return internal_transformer.transform(having_expr)


# pylint: disable=no-self-use, super-init-not-called
@v_args(inline=True)
class SQLTransformer(TransformerBaseClass):
    """
    Transformer for the lark sql parser
    """
    def __init__(self, env):
        super(SQLTransformer, self).__init__(dataframe_name_map={}, dataframe_map={}, column_name_map={},
                                             column_to_dataframe_name={}, _temp_dataframes_dict={})
        for key in env:
            if isinstance(env[key], DataFrame):
                dataframe = env[key]
                self.dataframe_name_map[key.lower()] = key
                self.dataframe_map[key] = dataframe
                self.column_name_map[key] = {}
                for column in dataframe.columns:
                    lower_column = column.lower()
                    self.column_name_map[key][lower_column] = column
                    self.add_column_to_column_to_dataframe_name_map(lower_column, key)

    def add_column_to_column_to_dataframe_name_map(self, column, table):
        """
        Adds a column to the column_to_dataframe_name_map
        :param column:
        :param table:
        :return:
        """
        if self.column_to_dataframe_name.get(column) is None:
            self.column_to_dataframe_name[column] = table
        elif isinstance(self.column_to_dataframe_name[column], AmbiguousColumn):
            self.column_to_dataframe_name[column].tables.append(table)
        else:
            original_table = self.column_to_dataframe_name[column]
            self.column_to_dataframe_name[column] = AmbiguousColumn([original_table, table])

    def table(self, table_name, alias=''):
        """
        Check for existence of pandas dataframe with same name
        If not exists raise DataFrameDoesNotExist
        Otherwise return the name of the actual DataFrame
        :return:
        """
        table_name = table_name.lower()
        if table_name not in self.dataframe_name_map:
            raise DataFrameDoesNotExist(table_name)
        return Token("table", self.dataframe_name_map[table_name])

    def order(self, order_by_token):
        """
        Returns and validates the order name, can be asc, desc, ascending, descending
        :param order_by_name:
        :return:
        """
        name = order_by_token.value.lower()
        if name not in ORDER_TYPES:
            raise Exception(f'Order type must be one of {ORDER_TYPES}')
        return name

    def order_by_expression(self, column, asc_desc=''):
        """
        Returns the column name for the order expression
        :param column:
        :param asc_desc: Determines whether the sort is ascending or descending
        :return:
        """
        if not asc_desc:
            asc_desc = 'asc'
        else:
            asc_desc = ORDER_TYPES_MAPPING[asc_desc]
        asc_desc = asc_desc == 'asc'
        return Token("order_by", (column.children, asc_desc))

    def integer(self, integer_token):
        """
        Returns the integer value
        :param integer_token:
        :return:
        """
        integer_value = int(integer_token.value)
        return integer_value

    def limit_count(self, limit_count_value):
        """
        Returns a limit token
        :param integer_tree:
        :return:
        """
        return Token("limit", limit_count_value)

    def full_query(self, query_info, *args):
        # TODO Add in support for set operations like union
        order_by = []
        limit = None
        for token in args:
            if isinstance(token, Token):
                if token.type == 'order_by':
                    order_by.append(token.value)
                elif token.type == "limit":
                    limit = token.value
        query_info['order_by'] = order_by
        query_info['limit'] = limit
        return query_info

    def subquery(self, query_info, alias):
        alias_name = alias.children[0].value
        self._temp_dataframes_dict[alias_name] = self.to_dataframe(query_info)
        subquery = Subquery(name=alias_name, query_info=query_info)
        for column in self._temp_dataframes_dict[alias_name].columns:
            self.add_column_to_column_to_dataframe_name_map(column, alias_name)
        # TODO Fix nested subqueries
        print(self.column_to_dataframe_name)
        return subquery

    def column_name(self, *names):
        full_name = ".".join([str(name) for name in names])
        return Tree("column_name", full_name)

    def join(self, join_expression):
        return join_expression

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
        column_match = GET_TABLE_REGEX.match(column)
        column_table = ""
        if column_match:
            column = column_match.group("column").lower()
            column_table = column_match.group("table").lower()

        left_columns = self.column_name_map[left_table]
        right_columns = self.column_name_map[right_table]
        if column not in left_columns and column not in right_columns:
            raise Exception("Column not found")

        left_table = left_table.lower()
        right_table = right_table.lower()
        if column_table:
            if column_table == left_table and column in left_columns:
                return "left", column
            if column_table == right_table and column in right_columns:
                return "right", column
            raise Exception("Table specified in join columns not present in join")
        if column in left_columns and column in right_columns:
            raise Exception(f"Ambiguous column: {column}\nSpecify table name with table_name.{column}")
        if column in left_columns:
            return "left", column
        if column in right_columns:
            return "right", column
        raise Exception("Column does not exist in either table")

    def join_expression(self, *args):
        """
        Evaluate a join into one dataframe using a merge method
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

        # Check that there is a column from both sides
        column_comparison = join_condition.children[0].children
        column1 = str(column_comparison[0].children)
        column2 = str(column_comparison[1].children)

        column1_side, column1 = self.determine_column_side(column1, table1, table2)
        column2_side, column2 = self.determine_column_side(column2, table1, table2)
        if column1_side == column2_side:
            raise Exception("Join columns must be one column from each join table!")
        column1 = self.column_name_map[table1][column1]
        column2 = self.column_name_map[table2][column2]
        if column1_side == "left":
            left_on = column1
            right_on = column2
        else:
            left_on = column2
            right_on = column1

        dictionary_name = f"{table1}x{table2}"
        self._temp_dataframes_dict[dictionary_name] = self.get_frame(table1).merge(right=self.get_frame(table2),
                                                                                   how=join_type, left_on=left_on,
                                                                                   right_on=right_on)
        return Subquery(dictionary_name, query_info="")

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

    def select(self, *select_expressions: Tuple[Tree]):
        """
        Forms the final sequence of methods that will be executed
        :param select_expressions:
        :return:
        """
        print("Select Expressions:", select_expressions)

        tables = []
        having_expr = None
        for select_expression in select_expressions:
            if isinstance(select_expression, Tree):
                if select_expression.data == 'from_expression':
                    tables.append(select_expression.children[0])
                elif select_expression.data == 'having_expr':
                    having_expr = select_expression

        select_expressions_no_having = tuple([select_expression for select_expression in select_expressions if
                                              isinstance(select_expression, Tree) and select_expression.data !=
                                              'having_expr' or not isinstance(select_expression, Tree)])

        internal_transformer = InternalTransformer(tables, self.dataframe_map, self.column_name_map,
                                                   self.column_to_dataframe_name)
        select_expressions = internal_transformer.transform(Tree("select", select_expressions_no_having)).children

        distinct = False
        if isinstance(select_expressions[0], Token):
            if str(select_expressions[0]) == "distinct":
                distinct = True
            select_expressions = select_expressions[1:]

        columns = []
        expressions = []
        numbers = []
        group_columns = []
        frame_names = []
        aliases = {}
        all_names = []
        name_order = {}
        conversions = {}
        aggregates = {}
        where_expr = None
        for token_pos, token in enumerate(select_expressions):
            if isinstance(token, Token):
                if token.type == "from_expression":
                    frame_names.append(token.value)
                elif token.type == "group":
                    group_columns.append(token.value)
                elif token.type == "where_expr":
                    where_expr = token.value
            elif isinstance(token, Tree):
                if token.data == "having_expr":
                    having_expr = token
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
                        aliases[str(token.value)] = token.alias
                    if token.function:
                        aggregates[token.alias] = token.function

                if isinstance(token, Number):
                    numbers.append(token)

        if having_expr is not None:
            having_expr = HavingTransformer(tables, group_columns, self.dataframe_map, self.column_name_map,
                                            self.column_to_dataframe_name).transform(having_expr).children[0]

        return {"columns": columns, "expressions": expressions, "dataframes": frame_names, "name_order": name_order,
                "all_names": all_names, "conversions": conversions, "distinct": distinct, "aggregates": aggregates,
                "group_columns": group_columns, "aliases": aliases, "numbers": numbers, "where_expr": where_expr,
                "having_expr": having_expr}

    def cross_join(self, df1, df2):
        """
        Returns the crossjoin between two dataframes
        :param df1: Dataframe1
        :param df2: Dataframe2
        :return: Crossjoined dataframe
        """
        temp_key_name = '_cross_join_tempkey'
        df1[temp_key_name] = 1
        df2[temp_key_name] = 1
        new_frame = merge(df1, df2, on=temp_key_name).drop(columns=[temp_key_name])
        df1.drop(columns=[temp_key_name], inplace=True)
        if df1 is not df2:
            df2.drop(columns=[temp_key_name], inplace=True)
        return new_frame

    def to_dataframe(self, query_info):
        """
        Returns the dataframe resulting from the SQL query
        :return:
        """
        print("Query info:", query_info)
        frame_names = query_info["dataframes"]
        conversions = query_info["conversions"]
        numbers = query_info["numbers"]
        group_columns = query_info["group_columns"]
        aggregates = query_info["aggregates"]
        columns = query_info["columns"]
        expressions = query_info["expressions"]
        where_expr = query_info["where_expr"]
        order_by = query_info["order_by"]
        limit = query_info["limit"]
        having_expr = query_info["having_expr"]

        if not frame_names:
            raise Exception("No table specified")

        first_frame = self.get_frame(frame_names[0])
        for frame_name in frame_names[1:]:
            next_frame = self.get_frame(frame_name)
            first_frame = self.cross_join(first_frame, next_frame)

        column_names = [column.name for column in columns]
        if self.has_star(column_names):
            new_frame: DataFrame = first_frame.copy()
        else:
            new_frame: DataFrame = first_frame[column_names].rename(columns=query_info["aliases"])

        # Evaluate in-line expressions
        for expression in expressions:
            new_frame[expression.alias] = expression.evaluate()

        for number in numbers:
            new_frame[number.alias] = number.value

        if conversions:
            return new_frame.astype(conversions)

        if where_expr is not None:
            new_frame = new_frame[where_expr]

        if group_columns and not aggregates:
            new_frame = new_frame.groupby(group_columns).size().to_frame('size').reset_index().drop(columns=['size'])
        elif aggregates and not group_columns:
            new_frame = new_frame.aggregate(aggregates).to_frame().transpose()
        elif aggregates and group_columns:
            new_frame = new_frame.groupby(group_columns).aggregate(aggregates).reset_index()

        if having_expr is not None:
            new_frame = new_frame[having_expr].reset_index().drop(columns=['index'])

        if query_info["distinct"]:
            new_frame.drop_duplicates(keep='first', inplace=True)
            new_frame.reset_index(inplace=True)
            new_frame.drop(columns=['index'], inplace=True)

        if order_by:
            new_frame.sort_values(by=[pair[0] for pair in order_by], ascending=[pair[1] for pair in order_by],
                                  inplace=True)

        if limit is not None:
            new_frame = new_frame.head(limit)

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
        print("Result:")
        if SHOW_TREE:
            print(self.ast)
            print(self.ast.pretty())
        else:
            print(self.ast)
            self.data_frame = self.ast

    def verify_sql(self):
        """
        Verifies that the sql is ONE valid statement
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
