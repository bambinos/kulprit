from kulprit.families.family import Family

import pytest

from tests import KulpritTest


class MyStructure:
    def __init__(self):
        self.family = "unimplemented-family"


class MyData:
    def __init__(self):
        self.structure = MyStructure()


class TestFamily(KulpritTest):
    """Test the family module used in the procedure."""

    def test_not_implemented_family(self):
        """Test that unimplemented families raise a warning."""

        with pytest.raises(NotImplementedError):
            bad_data = MyData()
            Family(data=bad_data)
