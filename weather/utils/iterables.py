import re
import itertools as it

from typing import List, Iterable, Iterator


def sorted_alphanumeric(data: Iterable[str]) -> List[str]:
    """Sort iterable alphanumerically similar to the OS"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def lst_chunk(lst: List, n: int) -> List[List]:
    """Yield successive n-sized chunks from lst."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def pairwise(iterable: Iterable) -> Iterator:
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


def main():
    pairwise((0, 0.1, 1, 3, 5, 10, 30, 100, 300))


if __name__ == '__main__':
    main()
