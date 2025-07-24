"""Module describing the high-level abstract syntax of LLFPL."""

from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass

# -- Types of the language.

Level = int
"""Type alias for `Level`'s in the linear language."""


class LLType(metaclass=ABCMeta):
    """Abstract base class of the types of the language."""

    pass


@dataclass
class LLBool(LLType):
    """Type of linear booleans."""

    level: Level


@dataclass
class LLFunc(LLType):
    """Type of linear functions."""

    rator: LLType
    rand: LLType


@dataclass
class LLTuple(LLType):
    """Type of linear tuples."""

    lhs: LLType
    rhs: LLType


@dataclass
class LLList(LLType):
    """Type of linear lists."""

    type_arg: LLType
    level: Level


@dataclass
class LLModal(LLType):
    """Modal type denoting multiple usage."""

    type_arg: LLType
    level: Level


@dataclass
class LLCredit(LLType):
    """Credit type in the linear language."""

    level: Level


# -- Terms and constants of the language.


# -- -- Constants


class LLConst(metaclass=ABCMeta):
    """Constant values within the language."""


@dataclass
class LLTrue(LLConst):
    """Boolean type of `true`."""

    level: Level


@dataclass
class LLFalse(LLConst):
    """Bolean type of `false`."""

    level: Level


@dataclass
class LLCaseBool(LLConst):
    """Boolean destructor constant."""

    type_arg: LLType
    level: Level


@dataclass
class LLCaseList(LLConst):
    """List destructor constant."""

    type_arg0: LLType
    type_arg1: LLType
    level: Level


@dataclass
class LLCons(LLConst):
    """Cons constant function."""

    type_arg: LLType
    level: Level


@dataclass
class LLNil(LLConst):
    """Nil constant."""

    type_arg: LLType
    level: Level


@dataclass
class LLDollar(LLConst):
    """Chit constant."""

    level: Level


@dataclass
class LLTupleConstr(LLConst):
    """Tuple constructor."""

    level: Level
    type_arg0: LLType
    type_arg1: LLType


@dataclass
class LLProj(LLConst):
    level: Level
    type_argument: LLType


# -- -- Terms of the language


class LLTerm(metaclass=ABCMeta):
    """Type of terms in the language."""

    pass


@dataclass
class LLVar(LLTerm):
    var: str


@dataclass
class LLAnn(LLTerm):
    """Variables in the language."""

    var: str
    type_ann: LLType


@dataclass
class LLTermConst(LLTerm):
    """Constnats that are terms."""

    const: LLConst


@dataclass
class LLLambda(LLTerm):
    """Lambda abstractions."""

    var: str
    type_ann: LLType
    body: LLTerm


@dataclass
class LLApp(LLTerm):
    rator: LLTerm
    rand: LLTerm


@dataclass
class LLHoleFill(LLTerm):
    outer_level: Level
    var: str
    value: LLTerm
    body: LLTerm


@dataclass
class LLBox(LLTerm):
    level: Level
    term: LLTerm


@dataclass
class LLHole(LLTerm):
    term: LLTerm
    level: Level


@dataclass
class LLSubst(LLTerm):
    lhs: LLTerm
    rhs: LLTerm
    x1: str
    x2: str


@dataclass
class LLLet(LLTerm):
    name: str
    ann: str
    body: LLTerm
