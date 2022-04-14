"""Formatting utility module."""


def spacify(string):  # pragma: no cover
    return "  " + "  ".join(string.splitlines(True))


def multilinify(line, sep=","):  # pragma: no cover
    sep += "\n"
    return "\n" + sep.join(line)
