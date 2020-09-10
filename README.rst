dataframe_sql
=============

.. image:: https://github.com/zbrookle/dataframe_sql/workflows/CI/badge.svg?branch=master
    :target: https://github.com/zbrookle/dataframe_sql/actions?query=workflow

.. image:: https://pepy.tech/badge/dataframe-sql
    :target: https://pepy.tech/project/dataframe-sql

.. image:: https://img.shields.io/pypi/l/dataframe_sql.svg
    :target: https://github.com/zbrookle/dataframe_sql/blob/master/LICENSE.txt

.. image:: https://img.shields.io/pypi/status/dataframe_sql.svg
    :target: https://pypi.python.org/pypi/dataframe_sql/

.. image:: https://img.shields.io/pypi/v/dataframe_sql.svg
    :target: https://pypi.python.org/pypi/dataframe_sql/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

``dataframe_sql`` is a Python_ package that translates SQL syntax into operations on
pandas_ DataFrames, a functionality which is not available in the central pandas
package.

.. _Python: https://www.python.org/

Installation
------------

.. code-block:: bash

    pip install dataframe_sql


Usage
-----

In this simple example, a DataFrame is read in from a csv and then using the query
function you can produce a new DataFrame from the sql query.

.. code-block:: python

    from pandas import read_csv
    from dataframe_sql import register_temp_table, query

    my_table = read_csv("some_file.csv")

    register_temp_table(my_table, "my_table")

    query("""select * from my_table""")


The package currently only supports pandas_ but there are plans to support dask_,
rapids_, and modin_ in the future.

.. _pandas: https://github.com/pandas-dev/pandas
.. _dask: https://github.com/dask/dask
.. _rapids: https://github.com/rapidsai/cudf
.. _modin: https://github.com/modin-project/modin

SQL Syntax
----------
The SQL syntax for ``dataframe_sql`` is exactly the same as the syntax in
``sql_to_ibis``, its underlying package.

You can find the full SQL syntax
`here <https://github.com/zbrookle/sql_to_ibis#sql-syntax>`__

Why use dataframe_sql?
----------------------

While there are other packages that accomplish the goal of using SQL with pandas
DataFrames, all other packages such as pandasql_ actually use a database on the
backend which defeats the purpose of using pandas_ to begin with. In the case of
pandasql_ which uses SQLite, this can result in major performance bottlenecks.
``dataframe_sql`` actually performs native pandas operations in memory on DataFrames,
which avoids conflicts that may arise from using external databases.

.. _pandasql: https://github.com/yhat/pandasql

