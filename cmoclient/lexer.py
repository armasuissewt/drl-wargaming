# lexer.py
#
# Simple string lexer
#
# Adapted from: codebrainz [https://gist.github.com/codebrainz/ffbd2fde8d44b93c22f0]
#
# Author: Giacomo Del Rio

import re
from collections import namedtuple
from typing import List, Tuple, Iterator


class Tokenizer:
    Token = namedtuple('Token', 'name text')

    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        pat_list = []
        for tok, pat in self.tokens:
            pat_list.append(f'(?P<{tok}>{pat})')
        self.re = re.compile('|'.join(pat_list))

    def iter_tokens(self, in_str: str, ignore_ws=True) -> Iterator[Tuple[str, str]]:
        for match in self.re.finditer(in_str):
            if ignore_ws and match.lastgroup == 'WHITESPACE':
                continue
            yield Tokenizer.Token(match.lastgroup, match.group(0))
        yield Tokenizer.Token('EOF', None)

    def tokenize(self, in_str: str, ignore_ws=True) -> List[Tuple[str, str]]:
        return list(self.iter_tokens(in_str, ignore_ws))


# Test
if __name__ == "__main__":
    txt = "[1] = { guid = '38ad0146-f62d-4d72-a54e-134292155432', name = 'SAM Bty (Patriot [PAC-2 GEM+, PAC-3 " \
          "ERINT/MSE])' }, [2] = { guid = 'f8ux3a-0hmbjij2jeqbh', name = 'MIM-104F Patriot PAC-3 MSE #62' }"

    TOKENS = [
        ('NUMBER', r'\d+(?:\.\d+)?'),
        ('STRING', r"'[^']*'"),
        ('SYMBOL', r'[a-zA-Z0-9_]+'),
        ('LPAREN', r'{'),
        ('RPAREN', r'}'),
        ('LSQUARE', r'\['),
        ('RSQUARE', r'\]'),
        ('COMMA', r','),
        ('EQUAL', r'='),
        ('WHITESPACE', r'\s+'),
        ('ERROR', r'.'),
    ]

    for t in Tokenizer(TOKENS).iter_tokens(txt):
        print(t)
