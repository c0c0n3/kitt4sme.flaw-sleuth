from fipy import pyproject_version

from flawsleuth import __version__, pyproject_file


def test_project_version_same_as_lib():
    project_version = pyproject_version(pyproject_file())
    assert __version__ == project_version
