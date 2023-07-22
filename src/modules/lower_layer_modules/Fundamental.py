"""
全モジュールの基盤となる情報のクラス
例外や、抽象性・共有性の高い情報などの定義
"""
from __future__ import annotations

import collections.abc as abc
import inspect
from functools import reduce
from itertools import chain, dropwhile
from logging import Logger, getLogger, NullHandler
from pathlib import Path
from subprocess import run, CompletedProcess, SubprocessError, Popen, PIPE
from typing import (
    Any, Iterable, Sequence, TypeVar, Mapping, Union, MutableMapping, Generator, NoReturn, Optional, Callable, Iterator,
    MutableSequence)

from .Exceptions import ProcessError, DataReadError

T = TypeVar("T")
JSONWritable = Union[Mapping[str, Any], Sequence[Any], str, int, float]

logger: Logger = getLogger(__name__)
logger.addHandler(NullHandler())


# def disabled_value(data_type: Type):
#     """
#     型を指定しての無効データ値
#     Args:
#         data_type(Type): 型,またはnumpyの型クラス
#
#     Returns:
#         無効値
#     """
#     if (data_type is int
#             or data_type is np.dtype("int32")
#             or data_type is np.dtype("int64")
#             or data_type is np.int32
#             or data_type is np.int64):
#         return -9999
#     return -9999.0


#
#
# @disabled_value.register(int)
# def _(_: int) -> int:
#     """
#     整数の無効データ値
#
#     Returns:
#         無効値(int) = -9999
#     """
#     return -9999
#
#
# @disabled_value.register(numbers.Real)
# def _(_) -> float:
#     """
#     実数の無効データ値
#
#     Returns:
#         無効値(float) = -9999
#     """
#     return -9999.0
#
#
# @disabled_value.register(np.int32)
# def _(_) -> np.int32:
#     """
#     numpy.int32の無効データ値
#
#     Returns:
#         無効値(numpy.int32) = -9999
#     """
#     return np.int32(-9999)
#
#
# @disabled_value.register(np.int64)
# def _(_) -> np.int64:
#     """
#     numpy.int64の無効データ値
#
#     Returns:
#         無効値(numpy.int64) = -9999
#     """
#     return np.int64(-9999)
#
#
# @disabled_value.register(np.float64)
# def _(_) -> np.float64:
#     """
#     numpy.float64の無効データ値
#
#     Returns:
#         無効値(int) = -9999.0
#     """
#     return np.float64(-9999.0)


def execute_command(
        command_args: Sequence[str],
        working_directory=Path("..")
) -> None:
    """
    コマンド実行
    Args:
        command_args(Iterable[str]): コマンド・引数リスト
        working_directory(pathlib.Path, optional): コマンド実行ディレクトリ(default: .)

    Raises:
        ProcessError: コマンド実行失敗
    """
    nice_setting: Sequence[str] = ["nice", "-n", "19", "ionice", "-c", "2", "-n", "7"]
    command_list: Sequence[str] = list(chain(nice_setting, command_args))
    try:
        process: CompletedProcess = run(command_list, cwd=working_directory)
        process.check_returncode()
    except (OSError, ValueError, SubprocessError) as e:
        command_str: str = reduce(lambda ss, s: ss + " " + str(s), command_args, "")
        raise ProcessError(f"execution command {command_str} "
                           + f"failed (process message: {e.args[0]}, in module {__name__}).")


def execute_command_without_console_output(
        command_args: Sequence[str],
        working_directory=Path("..")
) -> None:
    """
    コマンド実行
    Args:
        command_args(Iterable[str]): コマンド・引数リスト
        working_directory(pathlib.Path, optional): コマンド実行ディレクトリ(default: .)

    Raises:
        ProcessError: コマンド実行失敗
    """
    nice_setting: Sequence[str] = ["nice", "-n", "19", "ionice", "-c", "2", "-n", "7"]
    command_list: Sequence[str] = list(chain(nice_setting, command_args))
    try:
        process: CompletedProcess = run(command_list, cwd=working_directory,
                                        capture_output=True)
        process.check_returncode()
    except (OSError, ValueError, SubprocessError) as e:
        command_str: str = reduce(lambda ss, s: ss + " " + str(s), command_args, "")
        raise ProcessError(f"execution command {command_str} "
                           + f"failed (process message: {e.args[0]}, in module {__name__}).")


def execute_command_with_input(
        command_args: Sequence[str],
        working_directory=Path(".."),
        input_string=None
) -> None:
    """
    コマンド実行（標準入力からの入力も入れる）

    Args:
        command_args(Sequence[str]): コマンド・引数リスト(List[str])
        working_directory(pathlib.Path, optional): コマンド実行ディレクトリ(default: .)
        input_string(str, optional): 標準入力に入れる文字列

    Raises:
        ProcessError: コマンド実行失敗
    """
    if input_string is None:
        input_string = "\n"
    try:
        nice_setting: Sequence[str] = ["nice", "-n", "19", "ionice", "-c", "2", "-n", "7"]
        # with Popen(list(chain(nice_setting, command_args)),
        #            cwd=working_directory, stdin=PIPE, stdout=PIPE, shell=True,
        #            encoding="utf-8") as process:
        with Popen(["echo", "-e", input_string], stdout=PIPE) as p1:
            with Popen(list(chain(nice_setting, command_args)),
                       cwd=working_directory,
                       encoding="utf-8", stdin=p1.stdout) as process:
                process.stdin.write(input_string)
                process.stdin.flush()
                process.wait()
                p1.wait()
                _, _ = process.communicate()

    except (OSError, ValueError, SubprocessError) as e:
        command_str: str = reduce(lambda ss, s: ss + " " + str(s), command_args, "")
        raise ProcessError(f"execution command {command_str} "
                           + f"failed (process message: {e.args[0]}, in module {__name__}).")


# def read_array_data(
#         file: Path,
#         data_type=np.float64
# ) -> np.ndarray:
#     """
#     CSVファイルから値の配列を読み取る
#     Args:
#         file(pathlib.Path): CSVファイル
#         data_type(Type, optional): データ型
#
#     Returns:
#         値配列(np.ndarray)
#     """
#     data: pd.DataFrame = pd.read_csv(file, dtype=data_type, header=None)
#     return np.asarray(data, dtype=data_type)
#
#
# def read_array_data_space_separated(file: Path, data_type=np.float64) -> np.ndarray:
#     """
#     複数のスペース区切りの2D配列テキストデータファイルを読んでnumpy ndarrayにする。
#     Args:
#         file(Pathlib.Path): 2D配列テキストデータファイル
#         data_type(Type, optional): データの型[default: float]
#
#     Returns:
#        ファイル内のデータ配列(numpy.ndarray)
#     """
#     data: pd.DataFrame = pd.read_csv(
#         file, dtype=data_type, delim_whitespace=True, header=None)
#     return np.asarray(data, dtype=data_type)
#
#
# def make_array_data(
#         data_array: Iterable[Iterable[Any]],
#         data_type=np.float64
# ) -> np.ndarray:
#     """
#     2D配列から配列オブジェクトにする
#     Args:
#         data_array(Iterable[Iterable[Any]]): 2D配列
#         data_type(Type, optional): データ型
#
#     Returns:
#         値配列(np.ndarray)
#     """
#     return np.asarray(pd.DataFrame(data_array, dtype=data_type), dtype=data_type)


def is_empty_sequence(sequence: Sequence[T]) -> bool:
    """
    シークエンスが空かどうか
    Args:
        sequence(Sequence): シークエンス

    Returns:
        空ならTrue
    """
    return len(sequence) == 0


def merge_mappings(dictionaries: Iterable[Mapping]) -> Mapping:
    """
    辞書の(shallow)マージ
    同じキーがあれば後の辞書の値が優先。
    Args:
        dictionaries(Iterable[Mapping]):

    Returns:
        マージした辞書(Mapping)。
    """
    return {key: value for dictionary in dictionaries for key, value in dictionary.items()}


def deep_merge_dicts(*dicts: Mapping) -> Mapping:
    """
    辞書のディープマージ
    後ろの引数の要素優先。
    Args:
        dicts(Iterable[Mapping]): 辞書の可変長引数イテラブル

    Returns:
        マージした辞書(Mapping[str, Any])
    """

    def deep_merge_2dicts(dic1: MutableMapping, dic2: Mapping) -> MutableMapping:
        """
        辞書1に辞書2をディープマージ
        Args:
            dic1(MutableMapping): 辞書1
            dic2:(Mapping): 辞書2
        Returns:
            辞書2がマージされた辞書1(MutableMapping)
        """
        for key in dic2.keys():
            if isinstance(dic2[key], dict) and key in dic1 and isinstance(dic1[key], dict):
                deep_merge_2dicts(dic1[key], dic2[key])
            else:
                dic1[key] = dic2[key]
        return dic1

    merged_dict: MutableMapping = dict()
    return reduce(deep_merge_2dicts, dicts, merged_dict)


def dict_raiser(function_position: str, err: KeyError | str) -> NoReturn:
    """
    辞書のKeyErrorの共通例外
    Args:
        function_position(str): 関数の位置
        err(KeyError | str): 辞書が上げたKeyErrorか、任意のメッセージ
    Raises:
        DataReadError: 辞書にエントリがない
    """
    message: str = err.args if isinstance(err, KeyError) else err
    raise DataReadError(f"Setting dictionary does not have necessary key contents."
                        f" Maybe a key in setting file is invalid"
                        f" ({function_position}, message: {message})")


def find_if(
        sequence: Sequence[T],
        predicate: Callable[[T], bool]
) -> Optional[T]:
    """
    列の中で述語を満たす最初の要素。なければNone。
    Args:
        sequence(Sequence[T]): 列
        predicate(Callable[[T], bool]): 述語
    Returns:
        最初の要素 or None(Optional[T])
    """
    if len(sequence) == 0:
        return None
    return next(dropwhile(lambda element: not predicate(element), sequence))


def dictionary_entry(
        dictionary: Mapping,
        keys: Sequence
) -> Optional:
    """
    ネストされた辞書とキー列から、再帰的にキーをたどった値
    キーに対応する辞書や値がないならNone
    Args:
        dictionary(Mapping): 辞書
        keys(Sequence): キー列
    Returns:
        キーをたどった値 or None
    Raises:
        DataReadError: キー列が空
    """
    if not isinstance(dictionary, abc.Mapping):
        return None
    if len(keys) == 0:
        raise DataReadError(f"Please specify a dictionary and keys for the dictionary."
                            f" ({inspect.currentframe().f_code.co_name} in module {__name__})")
    if len(keys) == 1:
        return dictionary.get(keys[0], None)
    sub_dictionary: Optional[Mapping] = dictionary.get(keys[0], None)
    return dictionary_entry(sub_dictionary, keys[1:]) if sub_dictionary is not None else None


def split_sequence(sequence: Sequence, sublist_size: int) -> Generator[Sequence, Any, None]:
    """
    Sequenceをsub-sequencesに分割する
    Args:
        sequence(Sequence): リスト
        sublist_size(int): サブリストの要素数
    Returns:
        サブリストのGenerator
    """
    for index in range(0, len(sequence), sublist_size):
        yield sequence[index:index + sublist_size]


def split_sequence_eager(sequence: Sequence, sublist_size: int) -> Sequence[Sequence]:
    """
    Sequenceをsub-sequencesに分割し、内部は評価したリスト
    Args:
        sequence(Sequence): リスト
        sublist_size(int): サブリストの要素数
    Returns:
        サブリストのリスト(Sequence[Sequence])
    """
    return list(split_sequence(sequence, sublist_size))


def transpose(matrix: Iterable[Iterable[T]]) -> Iterable[Iterable[T]]:
    """
    行列の転置。ジェネレータなどの場合でも評価はされない。
    Args:
        matrix(Iterable[Iterable[T]]): 行列

    Returns:
        転置行列(Iterable[Iterable[T]])
    """
    return zip(*matrix)


def transpose_eager(matrix: Iterable[Iterable[T]]) -> Sequence[Sequence[T]]:
    """
    行列の転置。要素を評価してリストのリストにする。
    Args:
        matrix(Iterable[Iterable[T]]): 行列

    Returns:
        転置行列(Sequence[Sequence[T]])
    """
    return list(list(column for column in row) for row in transpose(matrix))


def array2d_sequence2_2d_sequence_array(
        array2d_sequence: Sequence[Sequence[Sequence[T]]]
) -> Sequence[Sequence[Sequence[T]]]:
    """
    すべて同じ形の2D配列の列から、対応する要素の列の2D配列
    e.g.
    [[[1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]],
    [[10, 11, 12],
     [13, 14, 15],
     [15, 16, 17]]]

->  [[[1, 10], [2, 11], [3, 12]],
     [[4, 13], [5, 14], [6, 15]],
     [[7, 15], [8, 16], [9, 17]]]

    Args:
        array2d_sequence(Sequence[Sequence[Sequence[T]]]): 2D配列の列
    Returns:
        要素の列の2D配列(Sequence[Sequence[Sequence[T]]])
    """
    return [transpose_eager(transposed_array) for transposed_array in transpose(array2d_sequence)]


def flatten(matrix: Iterable[Iterable[T]]) -> Iterable[T]:
    """
    行列の平坦化
    e.g. [[0,1,2,3],[4,5,6,7]] -> [0,1,2,3,4,5,6,7]
    ジェネレータなどの場合、評価はされない。
    Args:
        matrix(Iterable[Iterable[T]]): 行列

    Returns:
        平坦化されたIterable(Iterable[T])
    """
    return chain(*matrix)


def flatten_eager(matrix: Iterable[Iterable[T]]) -> Sequence[T]:
    """
    行列の平坦化
    e.g. [[0,1,2,3],[4,5,6,7]] -> [0,1,2,3,4,5,6,7]
    要素は評価する。
    Args:
        matrix(Iterable[Iterable[T]]): 行列

    Returns:
        要素を評価して平坦化されたSequence(Sequence[T])
    """
    return list(chain(*[list(column for column in row) for row in matrix]))


# def average_arrays(arrays: Sequence[np.ndarray], nan_value=disabled_value(np.float64)) -> np.ndarray:
#     """
#     同じshapeを持つnumpy.ndarrayの列の平均。
#     無効値を含む場合、
#     ある位置でどの配列でも無効値が入っているなら無効値、
#     そうではないが、ある位置でどれかの配列での値が無効値の場合は、無効値だけを無視した平均を取る。
#     e.g.
#     nan_value=-1のとき
#     a1=array([[ 0,  1],
#               [-1, -1]])
#
#     a2=array([[ 4, -1],
#               [ 6, -1]])
#
#     ->return  array([[ 2.,  1.],
#                      [ 6., -1]])
#     Args:
#         arrays(Sequence[np.ndarray]): データ配列
#         nan_value(np.float64): 無効値として設定する値
#     Returns:
#         平均した配列(np.ndarray)
#     """
#     if len(arrays) == 0:
#         raise UsageError(f"attempt to average empty array sequence."
#                          f" ({inspect.currentframe().f_code.co_name} in module {__name__})")
#     if len(arrays) == 1:
#         return np.copy(arrays[0])
#
#     shape: tuple[int, int] = arrays[0].shape
#     if not all([array.shape == shape for array in arrays]):
#         raise UsageError(f"attempt to average array sequence with different shapes."
#                          f" ({inspect.currentframe().f_code.co_name} in module {__name__})")
#
#     if all([np.all(array != nan_value) for array in arrays]):
#         return np.mean(arrays, axis=0)
#     nanned_arrays: np.ndarray = np.array([np.where(array == nan_value, np.nan, array) for array in arrays])
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         averaged_with_nan: np.ndarray = (
#                 np.nansum(nanned_arrays, axis=0) / np.count_nonzero(~np.isnan(nanned_arrays), axis=0))
#     np.nan_to_num(averaged_with_nan, nan=nan_value)


def identity() -> Callable[[T], T]:
    """
    Returns:
        恒等関数(Callable[[T], T])
    """
    return lambda x: x


def split_to_groups(iterator: Iterator[T], split_predicate: Callable[[T], bool]) -> Sequence[Sequence[T]]:
    """
    イテレータから読み出して、グループ列に分割
    Args:
        iterator(Iterator): イテレータ
        split_predicate(Callable[[T], bool]): グループの分割述語
    Returns:
        Iterator[Sequence[str]]: グループ列
    """
    groups: MutableSequence[MutableSequence[T]] = list()
    first_item: T = next(iterator)
    if split_predicate(first_item):
        groups.append([first_item])
    else:
        groups.append(list())
    for item in iterator:
        if split_predicate(item):
            groups.append([item])
            continue
        groups[-1].append(item)
    return groups
