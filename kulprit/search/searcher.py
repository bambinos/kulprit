"""Core search module."""

from typing_extensions import Literal

from kulprit.projection.projector import Projector
from kulprit.search.forward import ForwardSearchPath
from kulprit.search.l1 import L1SearchPath


class Searcher:
    def __init__(self, projector: Projector) -> None:
        """Initialise forward or l1 search class.

        Parameters:
        ----------
        projector : Projector
            A projector object.

        Raises
        ------
            UserWarning: If method is not "forward" or "l1".
        """
        # initialise projector for the search procedure
        self.projector = projector

        # define all available search heuristics
        self.method_dict = {
            "forward": ForwardSearchPath,
            "l1": L1SearchPath,
        }

        # indicator variable tracking whether or not a search has been run
        self.search_completed = False

    def __repr__(self) -> str:
        """String representation of the search path."""

        return self.path.__repr__()

    def search(self, max_terms: int, method: Literal["forward", "l1"] = "forward") -> dict:
        """Primary search method of the procedure.

        Performs forward search through the parameter space.

        Parameters:
        ----------
        max_terms : int
            Number of terms to perform the forward search up to.
        method : str
            Method to use for search.

        Returns:
        -------
        dict: A dictionary of submodels, keyed by the number of terms in the submodel.
        """

        # test valid solution method
        if method not in self.method_dict:
            raise UserWarning("Please either select either forward search or L1 search.")

        # initialise search path
        self.path = self.method_dict[method](  # pylint: disable=attribute-defined-outside-init
            self.projector
        )

        # perform the search according to the chosen heuristic
        k_submodels = self.path.search(max_terms=max_terms)

        # toggle indicator variable and return search path
        self.search_completed = True

        # feed path result through to the projector
        self.projector.path = k_submodels

        # return the final search path
        return k_submodels
