#!/bin/bash
#
# Run checks related to code quality.
#
# This script is intended for both the CI and to check locally that code standards are
# respected. We are currently linting (PEP-8 and similar), looking for patterns of
# common mistakes (sphinx directives with missing blank lines, old style classes,
# unwanted imports...), we run doctests here (currently some files only), and we
# validate formatting error in docstrings.
#
# Usage:
#   $ ./ci/code_checks.sh               # run all checks
#   $ ./ci/code_checks.sh lint          # run linting only
#   $ ./ci/code_checks.sh patterns      # check for patterns that should not exist
#   $ ./ci/code_checks.sh code          # checks on imported code
#   $ ./ci/code_checks.sh doctests      # run doctests
#   $ ./ci/code_checks.sh docstrings    # validate docstring errors
#   $ ./ci/code_checks.sh dependencies  # check that dependencies are consistent
#   $ ./ci/code_checks.sh typing	# run static type analysis

[[ -z "$1" || "$1" == "lint" || "$1" == "patterns" || "$1" == "code" || "$1" == "doctests" || "$1" == "docstrings" || "$1" == "dependencies" || "$1" == "typing" ]] || \
    { echo "Unknown command $1. Usage: $0 [lint|patterns|code|doctests|docstrings|dependencies|typing]"; exit 9999; }

RET=0
CHECK=$1

function invgrep {
    # grep with inverse exist status and formatting for azure-pipelines
    #
    # This function works exactly as grep, but with opposite exit status:
    # - 0 (success) when no patterns are found
    # - 1 (fail) when the patterns are found
    #
    # This is useful for the CI, as we want to fail if one of the patterns
    # that we want to avoid is found by grep.
    grep -n "$@" | sed "s/^/$INVGREP_PREPEND/" | sed "s/$/$INVGREP_APPEND/" ; EXIT_STATUS=${PIPESTATUS[0]}
    return $((! $EXIT_STATUS))
}

if [[ "$GITHUB_ACTIONS" == "true" ]]; then
    FLAKE8_FORMAT="##[error]%(path)s:%(row)s:%(col)s:%(code)s:%(text)s"
    INVGREP_PREPEND="##[error]"
else
    FLAKE8_FORMAT="default"
fi

### LINTING ###
if [[ -z "$CHECK" || "$CHECK" == "lint" ]]; then

    echo "black --version"
    black --version

    MSG='Checking black formatting' ; echo $MSG
    black . --check
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    # `setup.cfg` contains the list of error codes that are being ignored in flake8

    echo "flake8 --version"
    flake8 --version

    MSG='Linting .py code' ; echo $MSG
    flake8 --format="$FLAKE8_FORMAT" .
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    echo "isort --version-number"
    isort --version-number

    # Imports - Check formatting using isort see setup.cfg for settings
    MSG='Check import format using isort' ; echo $MSG
    ISORT_CMD="isort --recursive --check-only dataframe_sql"
    if [[ "$GITHUB_ACTIONS" == "true" ]]; then
        eval $ISORT_CMD | awk '{print "##[error]" $0}'; RET=$(($RET + ${PIPESTATUS[0]}))
    else
        eval $ISORT_CMD
    fi
    RET=$(($RET + $?)) ; echo $MSG "DONE"

fi

### PATTERNS ###
if [[ -z "$CHECK" || "$CHECK" == "patterns" ]]; then

    # Check for imports from collections.abc instead of `from collections import abc`
    MSG='Check for non-standard imports' ; echo $MSG
    invgrep -R --include="*.py*" -E "from collections.abc import" dataframe_sql
    invgrep -R --include="*.py*" -E "from numpy import nan" dataframe_sql

    # Checks for function_object suite
    # Check for imports from pandas.util.testing instead of `import pandas.util.testing as tm`
    invgrep -R --include="*.py*" -E "from pandas.util.testing import" dataframe_sql/tests
    invgrep -R --include="*.py*" -E "from pandas.util import testing as tm" dataframe_sql/tests
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for use of exec' ; echo $MSG
    invgrep -R --include="*.py*" -E "[^a-zA-Z0-9_]exec\(" --exclude performance_tests.py dataframe_sql
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for pytest warns' ; echo $MSG
    invgrep -r -E --include '*.py' 'pytest\.warns' dataframe_sql/tests/
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for pytest raises without context' ; echo $MSG
    invgrep -r -E --include '*.py' "[[:space:]] pytest.raises" dataframe_sql/tests/
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for python2-style file encodings' ; echo $MSG
    invgrep -R --include="*.py" --include="*.pyx" -E "# -\*- coding: utf-8 -\*-" dataframe_sql
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for python2-style super usage' ; echo $MSG
    invgrep -R --include="*.py" -E "super\(\w*, (self|cls)\)" dataframe_sql
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    # Check for the following code in testing: `np.testing` and `np.array_equal`
    MSG='Check for invalid testing' ; echo $MSG
    invgrep -r -E --include '*.py' --exclude testing.py '(numpy|np)(\.testing|\.array_equal)' dataframe_sql/tests/
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for deprecated messages without sphinx directive' ; echo $MSG
    invgrep -R --include="*.py" --include="*.pyx" -E "(DEPRECATED|DEPRECATE|Deprecated)(:|,|\.)" dataframe_sql
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for python2 new-style classes and for empty parentheses' ; echo $MSG
    invgrep -R --include="*.py" --include="*.pyx" -E "class\s\S*\((object)?\):" dataframe_sql
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for incorrect sphinx directives' ; echo $MSG
    invgrep -R --include="*.py" --include="*.pyx" --include="*.rst" -E "\.\. (autosummary|contents|currentmodule|deprecated|function|image|important|include|ipython|literalinclude|math|module|note|raw|seealso|toctree|versionadded|versionchanged|warning):[^:]" ./dataframe_sql
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    # Check for the following code in testing: `unittest.mock`, `mock.Mock()` or `mock.patch`
    MSG='Check that unittest.mock is not used (pytest builtin monkeypatch fixture should be used instead)' ; echo $MSG
    invgrep -r -E --include '*.py' '(unittest(\.| import )mock|mock\.Mock\(\)|mock\.patch)' dataframe_sql/tests/
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for extra blank lines after the class definition' ; echo $MSG
    invgrep -R --include="*.py" -E 'class.*:\n\n( )+"""' .
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for use of comment-based annotation syntax' ; echo $MSG
    invgrep -R --include="*.py" '# type: (?!ignore)' dataframe_sql
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for use of foo.__class__ instead of type(foo)' ; echo $MSG
    invgrep -R --include="*.py" '\.__class__' dataframe_sql
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for use of xrange instead of range' ; echo $MSG
    invgrep -R --include="*.py" 'xrange' dataframe_sql
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check that no file in the repo contains trailing whitespaces' ; echo $MSG
    INVGREP_APPEND=" <- trailing whitespaces found"
    invgrep -RI --exclude=\*.{svg,c,cpp,html,js} --exclude-dir=env "\s$" *
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    unset INVGREP_APPEND
fi

### TYPING ###
if [[ -z "$CHECK" || "$CHECK" == "typing" ]]; then

    echo "mypy --version"
    mypy --version

    MSG='Performing static analysis using mypy' ; echo $MSG
    mypy dataframe_sql
    RET=$(($RET + $?)) ; echo $MSG "DONE"
fi


exit $RET