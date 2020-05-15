# dataframe_sql

![CI](https://github.com/zbrookle/dataframe_sql/workflows/CI/badge.svg)


## Installation

```bash
pip install dataframe_sql
```

## Usage

In this simple example, a DataFrame is read in from a csv and then using the query
function you can produce a new DataFrame from the sql query.

``` python
from pandas import read_csv
from dataframe_sql import register_temp_table, query

my_table = read_csv("some_file.csv")

register_temp_table(my_table, "my_table")

query("""select * from my_table""")
```

The package currently only supports pandas but there are plans to support dask and
rapids in the future.

## Execution plan

### Values of certain "random" variables
In certain places it was necessary to create functionality that pandas doesn't support
and in those cases, there may be strange variables that you come across in the
execution plan.

This is what their values are:

```python
FALSE_SERIES = Series(data=[False for _ in range(0, dataframe_size)]))
NONE_SERIES = Series(data=[None for _ in range(0, dataframe_size)]))
```

### SQL Syntax
The sql syntax for dataframe_sql is as follows:

Select statement:

```SQL
SELECT [{ ALL | DISTINCT }]
    { [ <expression> ] | <expression> [ [ AS ] <alias> ] } [, ...]
[ FROM <from_item>  [, ...] ]
[ WHERE <bool_expression> ]
[ GROUP BY { <expression> [, ...] } ]
[ HAVING <bool_expression> ]
```

Set operations:

```SQL
<select_statement1>
{UNION [DISTINCT] | UNION ALL | INTERSECT [DISTINCT] | EXCEPT [DISTINCT] | EXCEPT ALL}
<select_statment2>
```

Joins:

```SQL
INNER, CROSS, FULL OUTER, LEFT OUTER, RIGHT OUTER, FULL, LEFT, RIGHT
```

Order by and limit:

```SQL
<set>
[ORDER BY <expression>]
[LIMIT <number>]
```

Supported expressions and functions:
```SQL 
+, -, *, /
```
```SQL
CASE WHEN <condition> THEN <result> [WHEN ...] ELSE <result> END
```
```SQL
SUM, AVG, MIN, MAX
```
```SQL
{RANK | DENSE_RANK} OVER([PARTITION BY (<expresssion> [, <expression>...)])
```
```SQL
CAST (<expression> AS <data_type>)
```
*Anything in <> is meant to be some string <br>
*Anything in [] is optional <br>
*Anything in {} is grouped together

### Supported Data Types for cast expressions include:
* VARCHAR, STRING
* INT16, SMALLINT
* INT32, INT
* INT64, BIGINT
* FLOAT16
* FLOAT32
* FLOAT, FLOAT64
* BOOL
* DATETIME64, TIMESTAMP
* CATEGORY
* OBJECT

*Data types in dataframe SQL support many different name for certain datatypes becuase
popular SQL data types are not implemented with common names in pandas and other
dataframe frameworks 
<br>
**To make this less confusing all data types that are of the same size on the
backend are grouped together in this list

## Issues that come from Pandas

- No native cross join
- No rank over(order by ...)
- No straight aggregation without groupby object
- ** No pandas date object