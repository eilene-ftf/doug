import sys

import numpy
import torch
import torchhd
import torchhd.structures as struct

from itertools import chain, combinations
from functools import reduce
from operator import mul
from typing import TypeVar
from collections.abc import Iterable

from language.lexer import KEYWORDS
from language.syntax import (
    LLBool,
    LLTerm,
    LLType,
    Level,
    LLLambda,
    LLDollar,
    LLLet,
    LLList,
    LLCaseBool,
    LLCaseList,
    LLAnn,
    LLApp,
    LLCons,
    LLConst,
    LLFalse,
    LLFunc,
    LLTuple,
    LLModal,
    LLTrue,
    LLCredit,
    LLNil,
    LLTupleConstr,
    LLProj,
    LLVar,
)

T = TypeVar('T')

def powerset(s: Iterable[T]) -> Iterable[tuple[T, ...]]:
    """Returns the powerset of s, minus the empty set.
    
    :param s: An iterable, typically a set or dict.
    :return: The powerset of s, minus {}.
    :rtype: Iterable[tuple[T, ...]]
    """
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


class EncodingEnvironment:
    """The encoding environment."""

    def __init__(self, dim: int, device=None, vsa="FHRR") -> None:
        self.vsa = vsa
        self.device = device
        self.dim = dim
        self.declarative_memory = struct.Memory()
        # TODO: add cleanup-memory
        self.cleanup_memory = None

        self.codebook: dict[str, torchhd.FHRRTensor] = {}
        self.perms: dict[str, torch.Tensor] = {}
        self.init_codebooks()
        # TODO: remove and replace with RHC embedding
        self.fractional_embed = torchhd.embeddings.FractionalPower(1, self.dim)

    def init_codebooks(self) -> None:
        """Initialize the codebook of the encoding environment."""
        symbols = [
            symbol
            for symbol in torchhd.random(len(KEYWORDS), self.dim, vsa=self.vsa)
        ]
        for key, symbol in zip(KEYWORDS, symbols):
            self.codebook[key] = symbol

        # Structural items are permutations in the perms codebook which are used
        # as roles in role-filler structure pairs in encoding the abstract syntax.
        # To see where it is used, and the particular fillers for the roles for
        # each syntactic item, see the encoding functions below.
        # Each structural item begins with the `#:`.
        structural_items = [
            "#:kind",
            "#:type",
            "#:level",
            "#:map",
            "#:dom",
            "#:codom",
            "#:left",
            "#:right",
            "#:from",
            "#:var",
            "#:to",
        ]
        structural_perms = [torch.randperm(self.dim) for _ in structural_items]
        structural_symbols = [
            symbol
            for symbol in torchhd.random(
                len(structural_items), self.dim, vsa=self.vsa
            )
        ]
        for item, perm, symbol in zip(structural_items, structural_perms, structural_symbols):
            self.codebook[item] = symbol
            self.perms[item] = perm

    # TODO: There's a smarter way to do this that involves some algebraic identity
    #       It's in the HDM paper
    def _set_product(self, s: dict[str, str|torchhd.VSATensor]) -> torchhd.VSATensor:
        """Gets the product of a set of role-filler pairs.

        :param s: Role-filler pairs from a sub-chunk.
        :return: The product of all fillers, permuted by their roles.
        :rtype: torchhd.VSATensor

        For a set {k1: v1, k2: v2, ...} returns the product 
        v1[k1] * v2[k2] * ..., where ki are permutations, and vi are vectors. 
        Takes the a dict comprised of names of role permutations in self.perms
        where values may be either codebook names or vectors.
        """
        def dependent_bind(acc: torchhd.VSATensor, role: str) -> torchhd.VSATensor:
            # If filler is a string, retrieve codebook vector
            if isinstance(s[role], str):
                s[role] = self.codebook(s[role])
            # Bind permuted filler with accumulator
            return acc * s[role][self.perms[role]]

        # Multiply all fillers, permuted by their roles
        return reduce(dependent_bind, s, torchhd.VSATensor.identity(1, self.dim, self.vsa)[0])

    def _chunk(self, rf: dict[str, str|torchhd.VSATensor]) -> torchhd.VSATensor:
        """Generates a chunk representation from c by summing the products of
        sets in the powerset of rf.

        :param rf: Set of role-filler pairs we want to chunk.
        :return: The sum of products of subsets of the powerset of rf.
        :rtype: torchhd.VSATensor
        """
        return sum(self._set_product(c) for c in powerset(rf))

    def encode_type(self, type_: LLType) -> torchhd.VSATensor:
        """Encode an `LLType` into a hypervector.

        :param LLType type_: The syntactic description of the type.
        :return: The encoded representation of the type.
        :rtype: torchhd.VSATensor
        """
        # Match on the dataclass
        match type_:
            # Boolean types
            # Returns chunk{#:level : <level>, #:kind : "bool"}
            case LLBool(level):
                encoded_level = self.encode_level(level)
                
                return self._chunk({"#:kind": "bool", "#:level": encoded_level})

            # Function types
            # Returns chunk{#:kind : "->" , #:type : chunk{#:dom : <rator>, #:codom : <rand>}}
            case LLFunc(rator, rand):
                rator_encoded = self.encode_type(rator)
                rand_encoded = self.encode_type(rand)
                # TODO: add levels

                _type = self._chunk({"#:dom": rator_encoded, "#:codom": rand_encoded})

                return self._chunk({"#kind": "->", "#:type": _type})

            # Tuple types
            # Returns
            # chunk{#:kind : "*" , #:type : chunk{#:right : <rhs>, #:left : <lhs>}}
            case LLTuple(lhs, rhs):
                lhs_encoded = self.encode_type(lhs)
                rhs_encoded = self.encode_type(rhs)
                # TODO: add levels

                _type = self._chunk({"#:left": lhs_encoded, "#:right": rhs_encoded})

                return self._chunk({"#:kind": "*", "#:type": _type})

            # List types
            # chunk{#:kind : "list", #:level : <level>, #:type : <type>}
            case LLList(type_arg, level):
                encoded_level = self.encode_level(level)
                _type = self.encode_type(type_arg)

                return self._chunk({"#:level": encoded_level, "#:kind": "list", "#:type": _type})

            # "Modal" types: marks it as non-affine.
            # Returns
            # chunk{#:kind : "!", #:level : <level>, #:type : <type>}
            case LLModal(type_arg, level):
                encoded_level = self.encode_level(level)
                _type = self.encode_type(type_arg)

                return self._chunk({"#:level": encoded_level, "#:kind": "!", "#:type": _type})

            # Chit type
            # Returns
            # chunk{#:kind : "◇", #:level : <level>}
            case LLCredit(level):
                encoded_level = self.encode_level(level)

                return self._chunk({"#:kind": "◇", "#:level": encoded_level})

            case _:
                raise TypeError("ERROR: innapropriate argument type")

    def encode_constant(self, constant: LLConst) -> torchhd.VSATensor:
        """Encode an `LLConst` into a hypervector.

        :param LLConst constant: The constant to encode.
        :return: The result of the encoding.
        :rtype: torchhd.VSATensor
        """
        match constant:
            # Boolean constants
            # Returns
            # "true" + (#:level * <level>)
            case LLTrue(level):
                encoded_level = self.encode_level(level)

                return (
                        self.codebook["true"] +
                        encoded_level.bind(self.codebook["#:level"])
                )

            # Returns
            # "false" + (#:level * <level>)
            case LLFalse(level):
                encoded_level = self.encode_level(level)

                return (
                        self.codebook["false"] +
                        encoded_level.bind(self.codebook["#:level"])
                )

            # Boolean destructor
            # Returns
            # "case-bool" + (#:level * "level")
            case LLCaseBool(type_arg, level):
                _type = self.encode_type(type_arg)
                encoded_level = self.encode_level(level)
                kind = self.codebook["case-bool"]

                return (
                    _type.bind(self.codebook["#:type"])
                    + encoded_level.bind(self.codebook["#:level"])
                    + kind
                )

            # List destructor
            # Returns
            # "case-list" + (#:type * ( (#:from * <type_arg0>) + (#:to * <type_arg1>) )) + (#:level * <level>)
            case LLCaseList(type_arg0, type_arg1, level):
                encoded_level = self.encode_level(level)
                from_ = self.encode_type(type_arg0)
                to_ = self.encode_type(type_arg1)
                kind = self.codebook["case-list"]
                _type = from_.bind(self.codebook["#:from"]) + to_.bind(
                    self.codebook["#:to"]
                )

                return (
                    kind
                    + encoded_level.bind(self.codebook["#:level"])
                    + _type.bind(self.codebook["#:type"])
                )

            # List constructor
            # Returns
            # "cons" + (#:level * <level>) + (#:type * <type>)
            case LLCons(type_arg, level):
                encoded_level = self.encode_level(level)
                _type = self.encode_type(type_arg)
                kind = self.codebook["cons"]
                return (
                    kind
                    + _type.bind(self.codebook["#:type"])
                    + encoded_level.bind(self.codebook[":level"])
                )

            # Empty list constructor
            # Returns
            # "nil" + (#:level * <level>) + (#:type * <type>)
            case LLNil(type_arg, level):
                encoded_level = self.encode_level(level)
                _type = self.encode_type(type_arg)
                kind = self.codebook["nil"]
                return (
                    kind
                    + _type.bind(self.codebook["#:type"])
                    + encoded_level.bind(self.codebook[":level"])
                )

            # Chit type constructor
            # Returns
            # "dollar" + (#:level * <level>)
            case LLDollar(level):
                encoded_level = self.encode_level(level)
                kind = self.codebook["dollar"]
                return kind + encoded_level.bind(
                    self.codebook["#:level"]
                )

            # Affine tuple type constructor
            # Returns
            # "tuple" +  (#:level * <level>) +
            #   (#:type * ( (#:left * <left>) + (#:right + <right>) ))
            case LLTupleConstr(level, type_arg0, type_arg1):
                encoded_level = self.encode_level(level)
                left = self.encode_type(type_arg0)
                right = self.encode_type(type_arg1)
                _type = left.bind(self.codebook["#:left"]) + right.bind(
                    self.codebook["#:right"]
                )
                kind = self.codebook["tuple"]
                return (
                    encoded_level.bind(self.codebook["#:level"])
                    + kind
                    + _type.bind(self.codebook["#:type"])
                )
                # this is a test

            # Tuple type destructor
            # Returns
            # (#:kind * "pi") + (#:level * <level>) + (#:type + <type_arg>)
            case LLProj(level, type_arg):
                encoded_level = self.encode_level(level)
                _type = self.encode_type(type_arg)
                kind = self.codebook["pi"]

                return (
                    kind
                    + _type.bind(self.codebook["#:type"])
                    + encoded_level.bind(self.codebook["#:level"])
                )
            case _:
                raise TypeError("ERROR: inappropriate argument type")

    def encode_term(self, term: LLTerm) -> torchhd.VSATensor:
        """Encode an `LLTerm` into a hypervector.

        :param LLTerm term: The term to encode.
        :return: The result of the encoding.
        :rtype: torchhd.VSATensor
        """
        # Match on the form of the term
        match term:
            # Annotated variables
            # Returns
            # ":" + (#:type * <type_ann>) + (#:var * self.codebook[var])
            case LLAnn(var, type_ann):
                if var not in self.codebook:
                    self.codebook[var] = torchhd.random(
                        1, self.dim, vsa=self.vsa
                    )[0]
                _type = self.encode_type(type_ann)
                kind = self.codebook[":"]
                return (
                    _type.bind(self.codebook["#:type"])
                    + self.codebook[var].bind(self.codebook["#:var"])
                    + kind
                )

            case LLVar(var):
                if var not in self.codebook:
                    self.codebook[var] = torchhd.random(1, self.dim, vsa=self.vsa)[0]
                kind = self.codebook["val"]
                return kind + self.codebook[var].bind(self.codebook["#:val"])

            case LLLambda(var, type_ann, body):
                if var not in self.codebook:
                    self.codebook[var] = torchhd.random(1, self.dim, vsa=self.vsa)[0]
                kind = self.codebook["lambda"]
                _type = self.encode_type(type_ann)
                _body = self.encode_term(body)
                
                return (
                        kind 
                        + _type.bind(self.codebook["#:type"])
                        + var.bind(self.codebook["#:var"])
                        + _body.bind(self.codebook["#:body"])
                )
            
            case LLApp(rator, rand):
                kind = self.codebook["app"]
                _rator = self.encode_term(rator)
                _rand = self.encode_term(rand)

                return (
                        kind
                        + _rator.bind(self.codebook["#:rator"]),
                        + _rand.bind(self.codebook["#:rand"])
                )
            
            # I think this may be incorrect
            case LLBox(level, term):
                pass

            case LLHole(level, term):
                kind = self.codebook["hole"]
                encoded_level = self.encode_level(level)
                _term = self.encode_term(term)

                return (
                        kind
                        + encoded_level.bind(self.codebook["#:level"])
                        + _term.bind(self.codebook["#:term"])
                )

            # Is this supposed to be brace?
            case LLHoleFill(var, outer_level, value, body):
                pass

            # I think this is wrong too
            case LLSubst(x1, x2, lhs, rhs):
                pass

            case LLLet(name, ann, body):
                pass


            case _:
                raise TypeError("ERROR; inappropriate argument type")

    def encode_level(self, level: Level) -> torchhd.VSATensor:
        """Encode a a `level` as an integer representation.

        :param Level level: The level to encode.
        :return: An encoded form of the level, using fractional binding.
        """
        return self.fractional_embed(torch.tensor([level]))
