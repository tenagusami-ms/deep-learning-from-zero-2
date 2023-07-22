"""
各種validation
"""
from __future__ import annotations

import collections.abc as abc
import inspect
from pathlib import Path
from typing import Mapping, Any, Sequence, Union, TypeVar

from .Exceptions import UsageError
from .Fundamental import dict_raiser, dictionary_entry

T = TypeVar("T")


def validate_type(
        value: Any,
        value_type: type,
        meaning: str,
        function_position: str
) -> None:
    """
    型が適切かどうか検証。
    Args:
        value(Any): 値
        value_type(type): 値の型
        meaning(str): 値の意味
        function_position(str): 関数の位置
    Raises:
        UsageError: 型が適切でない
    """
    if not isinstance(value, value_type):
        raise UsageError(f"{meaning} must be an instance of {value_type.__name__}. ({function_position})")


def validate_str(
        value: Any,
        meaning: str,
        function_position: str,
        *,
        minimum_length: int = 1,
        maximum_length: int = 1024
) -> None:
    """
    文字列かどうか検証。
    Args:
        value(Any): 値
        meaning(str): 値の意味
        function_position(str): 関数の位置
        minimum_length(int, optional): 文字列の最小長(default: 1)
        maximum_length(int, optional): 文字列の最大長(default: 1024)
    Raises:
        UsageError: 正規の文字列でない
    """
    validate_type(value, str, meaning, function_position)
    if not minimum_length <= len(value) <= maximum_length:
        raise UsageError(
            f"the length of {meaning} must be between {minimum_length} and {maximum_length} ({function_position}).")


def validate_range(
        value: T,
        value_type: type,
        meaning: str,
        function_position: str,
        *,
        minimum: T,
        maximum: T,
) -> None:
    """
    値が範囲内かどうか検証。
    Args:
        value(T): 値
        value_type(type): 値の型
        meaning(str): 値の意味
        function_position(str): 関数の位置
        minimum(T): 最小値
        maximum(T): 最大値
    Raises:
        UsageError: 範囲外の値
    """
    validate_type(value, value_type, meaning, function_position)
    validate_type(minimum, value_type, f"the minimum value of {meaning}", function_position)
    validate_type(maximum, value_type, f"the maximum value of {meaning}", function_position)
    try:
        if not minimum <= value <= maximum:
            raise UsageError(f"{meaning} must be between {minimum} and {maximum}. ({function_position})")
    except TypeError:
        raise UsageError(f"ordering is not defined for a type {value_type.__name__}. ({function_position})")


def validate_dictionary(
        dictionary: Any,
        key_type: type,
        value_type: Union[type, Sequence[type]],
        meaning: str,
        function_position: str,
        empty_is_valid: bool = False
) -> None:
    """
    適切な型のキーと値を持つ辞書かどうか検証。
    Args:
        dictionary(Any): 辞書
        key_type(type): キーの型
        value_type(type): 値の型
        meaning(str): 辞書の意味
        function_position(str): 関数の位置
        empty_is_valid(bool): 空の辞書でもOKならTrue
    Raises:
        UsageError: 適切な辞書でない
    """
    validate_type(dictionary, abc.Mapping, meaning, function_position)
    if len(dictionary) == 0:
        if not empty_is_valid:
            raise UsageError(f"{meaning} must not be an empty dictionary. ({function_position})")
        else:
            return
    if not all(isinstance(key, key_type) for key in dictionary):
        raise UsageError(f"{meaning} must have keys of {key_type.__name__}. ({function_position})")
    if isinstance(value_type, type):
        if not all(isinstance(value, value_type) for value in dictionary.values()):
            raise UsageError(f"{meaning} must have values with the type {value_type.__name__}. ({function_position})")
    elif isinstance(value_type, abc.Sequence):
        if not all(isinstance(value, tuple(value_type)) for value in dictionary.values()):
            raise UsageError(
                f"{meaning} must have one of values with types in {', '.join([v.__name__ for v in value_type])}."
                f" ({function_position})")
    else:
        raise UsageError(f"a value in {meaning} is invalid."
                         f" ({inspect.currentframe().f_code.co_name} in {__name__})")


def validate_dictionary_entry(
        dictionary: Mapping[str, Any],
        keys: str | Sequence[str],
        value_type: type,
        meaning: str,
        function_position: str
) -> None:
    """
    辞書のエントリが存在して適切な型かどうか検証。
    Args:
        dictionary(Mapping[str, Any]): 検証済辞書
        keys(str | Sequence[str]): キーまたはキーの列
        value_type(type): 値の型
        meaning(str): エントリの意味
        function_position(str): 関数の位置
    Raises:
        UsageError: エントリがない
    """
    if isinstance(keys, str):
        if keys not in dictionary:
            dict_raiser(function_position, f"keys: '{keys}'")
        if not isinstance(dictionary[keys], value_type):
            raise UsageError(f"{meaning} must be an instance of {value_type.__name__}. ({function_position})")
    elif isinstance(keys, abc.Sequence):
        if not all(isinstance(key, str) for key in keys):
            raise UsageError(f"keys must be an instance of str."
                             f" ({inspect.currentframe().f_code.co_name} in {__name__})")
        if (item := dictionary_entry(dictionary, keys)) is None:
            dict_raiser(function_position, 'keys: ["' + '"]["'.join(keys) + '"}')
        if not isinstance(item, value_type):
            raise UsageError(f"{meaning} must be an instance of {value_type.__name__}. ({function_position})")


def validate_collection(
        collection: Any,
        meaning: str,
        function_position: str,
        element_type: type = None,
        empty_is_valid: bool = False
) -> None:
    """
    適切な型の要素だけからなる集合(列を含む)かどうか検証。
    型の情報が与えられていないときは、すべての要素が最初の要素と同じ型かどうか検証。
    空集合でもOKかどうかはオプション
    Args:
        collection(Any): 集合のはずのオブジェクト
        meaning(str): オブジェクトの意味
        function_position(str): 関数の位置
        element_type(type, optional): 要素の型
        empty_is_valid(bool, optional): 空集合でもOKならTrue
    Raises:
        UsageError: 適切な集合でない
    """
    validate_type(collection, abc.Collection, meaning, function_position)
    if len(collection) == 0:
        if not empty_is_valid:
            raise UsageError(f"{meaning} must not be an empty sequence. ({function_position})")
        else:
            return
    if element_type is None:
        if not all(isinstance(item, type(collection[0])) for item in collection):
            raise UsageError(f"{meaning} must be an instance of {element_type.__name__}. ({function_position})")
    else:
        if not all(isinstance(item, element_type) for item in collection):
            raise UsageError(f"{meaning} must be an instance of {element_type.__name__}. ({function_position})")


def validate_readable_file_paths(
        paths: Path | Sequence[Path],
        meaning: str,
        function_position: str
) -> None:
    """
    読み込み可能なファイルパスかどうか検証。
    Args:
        paths(Path | Sequence[Path]): ファイルパスまたはファイルパスの列
        meaning(str): ファイルパスの意味
        function_position(str): 関数の位置
    Raises:
        UsageError: 読み出し可能でないファイルパスが存在する（ファイルが存在しない、読み出し権限がない、等）
    """
    if isinstance(paths, Path):
        try:
            with open(paths, "rb"):
                pass
        except OSError:
            raise UsageError(f"{meaning} must be a readable file path. ({function_position})")
    elif isinstance(paths, abc.Sequence):
        if not all(isinstance(path, Path) for path in paths):
            raise UsageError(f"{meaning} must be a sequence of paths. ({function_position})")
        try:
            for path in paths:
                with open(path, "rb"):
                    pass
        except OSError:
            raise UsageError(f"{meaning} must be a sequence of readable file paths. ({function_position})")


def validate_callable(
        maybe_callable: Any,
        name: str,
        function_position: str,
) -> None:
    """
    呼び出し可能オブジェクトかどうか検証。
    Args:
        maybe_callable(Any): たぶん呼び出し可能
        name(str): オブジェクトの名前
        function_position(str): 呼び出し元の位置(エラーメッセージ用)
    Raises:
        UsageError: 呼び出し可能でない
    """
    if not callable(maybe_callable):
        raise UsageError(f"{name} must be a function. ({function_position})")
