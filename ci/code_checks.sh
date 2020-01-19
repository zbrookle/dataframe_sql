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

    # pandas/_libs/src is C code, so no need to search there.
    MSG='Linting .py code' ; echo $MSG
    flake8 --format="$FLAKE8_FORMAT" .
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    echo "isort --version-number"
    isort --version-number

    # Imports - Check formatting using isort see setup.cfg for settings
    MSG='Check import format using isort' ; echo $MSG
    ISORT_CMD="isort --recursive --check-only pandas asv_bench"
    if [[ "$GITHUB_ACTIONS" == "true" ]]; then
        eval $ISORT_CMD | awk '{print "##[error]" $0}'; RET=$(($RET + ${PIPESTATUS[0]}))
    else
        eval $ISORT_CMD
    fi
    RET=$(($RET + $?)) ; echo $MSG "DONE"

fi

### PATTERNS ###
if [[ -z "$CHECK" || "$CHECK" == "patterns" ]]; then

    # Check for imports from pandas.core.common instead of `import pandas.core.common as com`
    # Check for imports from collections.abc instead of `from collections import abc`
    MSG='Check for non-standard imports' ; echo $MSG
    invgrep -R --include="*.py*" -E "from pandas.core.common import" pandas
    invgrep -R --include="*.py*" -E "from pandas.core import common" pandas
    invgrep -R --include="*.py*" -E "from collections.abc import" pandas
    invgrep -R --include="*.py*" -E "from numpy import nan" pandas

    # Checks for test suite
    # Check for imports from pandas.util.testing instead of `import pandas.util.testing as tm`
    invgrep -R --include="*.py*" -E "from pandas.util.testing import" pandas/tests
    invgrep -R --include="*.py*" -E "from pandas.util import testing as tm" pandas/tests
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for use of exec' ; echo $MSG
    invgrep -R --include="*.py*" -E "[^a-zA-Z0-9_]exec\(" pandas
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for pytest warns' ; echo $MSG
    invgrep -r -E --include '*.py' 'pytest\.warns' pandas/tests/
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for pytest raises without context' ; echo $MSG
    invgrep -r -E --include '*.py' "[[:space:]] pytest.raises" pandas/tests/
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for python2-style file encodings' ; echo $MSG
    invgrep -R --include="*.py" --include="*.pyx" -E "# -\*- coding: utf-8 -\*-" pandas scripts
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for python2-style super usage' ; echo $MSG
    invgrep -R --include="*.py" -E "super\(\w*, (self|cls)\)" pandas
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    # Check for the following code in testing: `np.testing` and `np.array_equal`
    MSG='Check for invalid testing' ; echo $MSG
    invgrep -r -E --include '*.py' --exclude testing.py '(numpy|np)(\.testing|\.array_equal)' pandas/tests/
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    # Check for the following code in the extension array base tests: `tm.assert_frame_equal` and `tm.assert_series_equal`
    MSG='Check for invalid EA testing' ; echo $MSG
    invgrep -r -E --include '*.py' --exclude base.py 'tm.assert_(series|frame)_equal' pandas/tests/extension/base
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for deprecated messages without sphinx directive' ; echo $MSG
    invgrep -R --include="*.py" --include="*.pyx" -E "(DEPRECATED|DEPRECATE|Deprecated)(:|,|\.)" pandas
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for python2 new-style classes and for empty parentheses' ; echo $MSG
    invgrep -R --include="*.py" --include="*.pyx" -E "class\s\S*\((object)?\):" pandas asv_bench/benchmarks scripts
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for backticks incorrectly rendering because of missing spaces' ; echo $MSG
    invgrep -R --include="*.rst" -E "[a-zA-Z0-9]\`\`?[a-zA-Z0-9]" doc/source/
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for incorrect sphinx directives' ; echo $MSG
    invgrep -R --include="*.py" --include="*.pyx" --include="*.rst" -E "\.\. (autosummary|contents|currentmodule|deprecated|function|image|important|include|ipython|literalinclude|math|module|note|raw|seealso|toctree|versionadded|versionchanged|warning):[^:]" ./pandas ./doc/source
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    # Check for the following code in testing: `unittest.mock`, `mock.Mock()` or `mock.patch`
    MSG='Check that unittest.mock is not used (pytest builtin monkeypatch fixture should be used instead)' ; echo $MSG
    invgrep -r -E --include '*.py' '(unittest(\.| import )mock|mock\.Mock\(\)|mock\.patch)' pandas/tests/
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for wrong space after code-block directive and before colon (".. code-block ::" instead of ".. code-block::")' ; echo $MSG
    invgrep -R --include="*.rst" ".. code-block ::" doc/source
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for wrong space after ipython directive and before colon (".. ipython ::" instead of ".. ipython::")' ; echo $MSG
    invgrep -R --include="*.rst" ".. ipython ::" doc/source
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for extra blank lines after the class definition' ; echo $MSG
    invgrep -R --include="*.py" --include="*.pyx" -E 'class.*:\n\n( )+"""' .
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for use of comment-based annotation syntax' ; echo $MSG
    invgrep -R --include="*.py" -P '# type: (?!ignore)' pandas
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for use of foo.__class__ instead of type(foo)' ; echo $MSG
    invgrep -R --include=*.{py,pyx} '\.__class__' pandas
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for use of xrange instead of range' ; echo $MSG
    invgrep -R --include=*.{py,pyx} 'xrange' pandas
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check that no file in the repo contains trailing whitespaces' ; echo $MSG
    INVGREP_APPEND=" <- trailing whitespaces found"
    invgrep -RI --exclude=\*.{svg,c,cpp,html,js} --exclude-dir=env "\s$" *
    RET=$(($RET + $?)) ; echo $MSG "DONE"
    unset INVGREP_APPEND
fi

### CODE ###
if [[ -z "$CHECK" || "$CHECK" == "code" ]]; then

    MSG='Check import. No warnings, and blacklist some optional dependencies' ; echo $MSG
    python -W error -c "
import sys
import pandas

blacklist = {'bs4', 'gcsfs', 'html5lib', 'http', 'ipython', 'jinja2', 'hypothesis',
             'lxml', 'matplotlib', 'numexpr', 'openpyxl', 'py', 'pytest', 's3fs', 'scipy',
             'tables', 'urllib.request', 'xlrd', 'xlsxwriter', 'xlwt'}

# GH#28227 for some of these check for top-level modules, while others are
#  more specific (e.g. urllib.request)
import_mods = set(m.split('.')[0] for m in sys.modules) | set(sys.modules)
mods = blacklist & import_mods
if mods:
    sys.stderr.write('err: pandas should not import: {}\n'.format(', '.join(mods)))
    sys.exit(len(mods))
    "
    RET=$(($RET + $?)) ; echo $MSG "DONE"

fi

### DOCSTRINGS ###
if [[ -z "$CHECK" || "$CHECK" == "docstrings" ]]; then

    MSG='Validate docstrings (GL03, GL04, GL05, GL06, GL07, GL09, GL10, SS04, SS05, PR03, PR04, PR05, PR10, EX04, RT01, RT04, RT05, SA01, SA02, SA03, SA05)' ; echo $MSG
    $BASE_DIR/scripts/validate_docstrings.py --format=azure --errors=GL03,GL04,GL05,GL06,GL07,GL09,GL10,SS04,SS05,PR03,PR04,PR05,PR10,EX04,RT01,RT04,RT05,SA01,SA02,SA03,SA05
    RET=$(($RET + $?)) ; echo $MSG "DONE"

fi

### DEPENDENCIES ###
if [[ -z "$CHECK" || "$CHECK" == "dependencies" ]]; then

    MSG='Check that requirements-dev.txt has been generated from environment.yml' ; echo $MSG
    $BASE_DIR/scripts/generate_pip_deps_from_conda.py --compare --azure
    RET=$(($RET + $?)) ; echo $MSG "DONE"

fi

### TYPING ###
if [[ -z "$CHECK" || "$CHECK" == "typing" ]]; then

    echo "mypy --version"
    mypy --version

    MSG='Performing static analysis using mypy' ; echo $MSG
    mypy pandas
    RET=$(($RET + $?)) ; echo $MSG "DONE"
fi


exit $RET