"""Explicit keyword aliases; the wrapped implementation retains its defaults."""

import functools
import inspect


def translate(function, args, kwargs, aliases, *, signature=None):
    signature = signature or inspect.signature(function)
    values = dict(kwargs)
    positional = signature.bind_partial(*args).arguments
    for alias, canonical in aliases.items():
        if alias not in values:
            continue
        value = values.pop(alias)
        if canonical in positional:
            if positional[canonical] != value:
                raise TypeError(f"conflicting {alias} and positional {canonical}")
        elif canonical in values:
            if values[canonical] != value:
                raise TypeError(f"conflicting {alias} and {canonical}")
        else:
            values[canonical] = value
    signature.bind(*args, **values)
    return values


def keyword_aliases(function, aliases):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def call(*args, **kwargs):
        if not any(alias in kwargs for alias in aliases):
            return function(*args, **kwargs)
        return function(*args, **translate(function, args, kwargs, aliases, signature=signature))

    return call


def public_keywords(function, names, *, aliases=None):
    """Publish upstream names, accepting existing implementation names as aliases.

    ``names`` maps public keywords to implementation keywords. Extra aliases
    do not affect the published signature. Values and defaults are never scaled.
    All reflection occurs once, when the wrapper is created.
    """
    signature = inspect.signature(function)
    mapping = {**(aliases or {}), **names}
    for target in mapping.values():
        if target not in signature.parameters:
            raise ValueError(f"unknown implementation parameter: {target}")
    call = keyword_aliases(function, mapping)
    published = {old: new for new, old in names.items()}
    call.__signature__ = signature.replace(
        parameters=[
            parameter.replace(name=published.get(parameter.name, parameter.name))
            for parameter in signature.parameters.values()
        ]
    )
    return call
