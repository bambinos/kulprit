"""Link function module. Many of the methods and classes are adapted from Bambi."""

from collections import namedtuple
import torch


def force_within_unit_interval(x):  # pragma: no cover
    """Make sure data in unit interval is in (0, 1)."""

    eps = torch.finfo(torch.float).eps
    x[x == 0] = eps
    x[x == 1] = 1 - eps
    return x


def force_greater_than_zero(x):  # pragma: no cover
    """Make sure data in positive reals is in (0, infty)"""

    eps = torch.finfo(torch.float).eps
    x[x == 0] = eps
    return x


def identity(x):  # pragma: no cover
    return x


def cloglog(mu):  # pragma: no cover
    """Cloglog function that ensures the input is greater than 0."""

    mu = force_greater_than_zero(mu)
    return torch.special.log(-torch.special.log(1 - mu))


def invcloglog(eta):  # pragma: no cover
    """Inverse of the cloglog function that ensures result is in (0, 1)."""

    result = 1 - torch.special.exp(-torch.exp(eta))
    return force_within_unit_interval(result)


def probit(mu):  # pragma: no cover
    """Probit function that ensures the input is in (0, 1)."""

    mu = force_within_unit_interval(mu)
    return 2**0.5 * torch.special.erfinv(2 * mu - 1)


def invprobit(eta):  # pragma: no cover
    """Inverse of the probit function that ensures result is in (0, 1)."""

    result = 0.5 + 0.5 * torch.special.erf(eta / 2**0.5)
    return force_within_unit_interval(result)


def expit(eta):  # pragma: no cover
    """Expit function that ensures result is in (0, 1)."""

    result = torch.special.expit(eta)
    result = force_within_unit_interval(result)
    return result


def logit(mu):  # pragma: no cover
    """Logit function that ensures the input is in (0, 1)."""

    mu = force_within_unit_interval(mu)
    return torch.special.logit(mu)


def softmax(eta, axis=None):  # pragma: no cover
    """Softmax function."""

    result = torch.special.softmax(eta, axis=axis)
    result = force_within_unit_interval(result)
    return result


def inverse_squared(mu):  # pragma: no cover
    return 1 / mu**2


def inv_inverse_squared(eta):  # pragma: no cover
    return 1 / torch.sqrt(eta)


def arctan_2(eta):  # pragma: no cover
    return 2 * torch.arctan(eta)


def tan_2(mu):  # pragma: no cover
    return torch.tan(mu / 2)


def inverse(mu):  # pragma: no cover
    return 1 / mu


def inv_inverse(eta):  # pragma: no cover
    return 1 / eta


def link_not_implemented(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("link not implemented")


# link: Known as g in the GLM literature. Maps the response to the linear
# predictor scale. linkinv: Known as g^(-1) in the GLM literature. Maps the
# linear predictor to the response scale.
LinksContainer = namedtuple("LinksContainer", ["link", "linkinv"])

LINKS = {
    "cloglog": LinksContainer(cloglog, invcloglog),
    "identity": LinksContainer(identity, identity),
    "inverse_squared": LinksContainer(inverse_squared, inv_inverse_squared),
    "inverse": LinksContainer(inverse, inv_inverse),
    "log": LinksContainer(torch.log, torch.exp),
    "logit": LinksContainer(logit, expit),
    "probit": LinksContainer(probit, invprobit),
    "softmax": LinksContainer(link_not_implemented, softmax),
    "tan_2": LinksContainer(tan_2, arctan_2),
}


class Link:  # pragma: no cover
    r"""Representation of a link function.
    This object is adapted from Bambi and contains two main functions. One is
    the link function itself, the function that maps values in the response
    scale to the linear predictor, and the other is the inverse of the link
    function, that maps values of the linear predictor to the response scale.
    Attributes:
        name (str): The name of the link function. If it is a known name, it's
            not necessary to pass any other arguments because functions are
            already defined internally. If not known, all of ``link``, ``linkinv``
            and ``linkinv_backend`` must be specified.
        link (function): A function that maps the response to the linear
            predictor. Known as the math`g` function in GLM jargon. Does not
            need to be specified when ``name`` is a known name.
        linkinv (function): A function that maps the linear predictor to the
            response. Known as the math`g^{-1}` function in GLM jargon. Does not
            need to be specified when ``name`` is a known name.
        linkinv_backend (function): Same than ``linkinv`` but must be something
            that works with PyMC backend (i.e. it must work with PyTorch tensors).
            Does not need to be specified when ``name`` is a known name.
    """

    def __init__(self, name, link=None, linkinv=None, linkinv_backend=None):
        self.name = name
        self.link = link
        self.linkinv = linkinv
        self.linkinv_backend = linkinv_backend

        if name in LINKS:
            self.link = LINKS[name].link
            self.linkinv = LINKS[name].linkinv
        else:
            if not link or not linkinv or not linkinv_backend:
                raise ValueError(
                    f"Link name '{name}' is not supported and at least one of",
                    "'link', 'linkinv' or 'linkinv_backend' are unspecified.",
                )
