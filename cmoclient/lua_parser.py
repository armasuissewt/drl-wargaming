# lua_parser.py
#
# Parser for LUA data snippets
#
# Author: Giacomo Del Rio

from typing import List, Dict, Tuple, Optional, Union

from cmoclient.lexer import Tokenizer


class LuaParser:
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

    def __init__(self, in_str: str):
        self.lexer = Tokenizer(LuaParser.TOKENS).iter_tokens(in_str)
        self.lookahead: Tokenizer.Token = self.lexer.__next__()

    def parse_array(self) -> List[Tuple[str, Dict]]:
        res = []
        self.__match("LPAREN")

        while self.lookahead.name != "RPAREN":
            self.__match("LSQUARE")
            self.__match("NUMBER")
            self.__match("RSQUARE")
            self.__match("EQUAL")
            res.append(self.parse_object())

            if self.lookahead.name == 'COMMA':
                self.__match("COMMA")

        self.__match("RPAREN")
        return res

    def parse_object(self) -> Tuple[Optional[str], Dict]:
        if self.lookahead.name == 'SYMBOL':
            obj_class = self.__match("SYMBOL")
        else:
            obj_class = None
        obj = self.parse_dict()
        return obj_class, obj

    def parse_dict(self) -> Dict:
        res = {}
        self.__match("LPAREN")

        while True:
            key = self.__match("SYMBOL")
            self.__match("EQUAL")
            if self.lookahead.name == 'LPAREN':
                inner_array = self.parse_array()
                res[key] = inner_array
            elif self.lookahead.name == 'NUMBER':
                val = self.__match("NUMBER")
                res[key] = self.__int_or_float(val)
            else:  # STRING
                val = self.__match("STRING")
                res[key] = val[1:-1]

            if self.lookahead.name == 'COMMA':
                self.__match("COMMA")

            if self.lookahead.name == 'RPAREN':
                break

        self.__match("RPAREN")
        return res

    def __match(self, token: str):
        if self.lookahead.name == token:
            res = self.lookahead.text
            if self.lookahead.name != 'EOF':
                self.lookahead: Tokenizer.Token = self.lexer.__next__()
            return res
        else:
            raise RuntimeError(f"Unexpected token {self.lookahead.text}. {token} expected.")

    @staticmethod
    def __int_or_float(v) -> Union[int, float]:
        try:
            return int(v)
        except ValueError as e:
            return float(v)


# Test
if __name__ == "__main__":
    txt = "{[1] = { guid = '38ad0146-f62d-4d72-a54e-134292155432', name = 'SAM Bty (Patriot [PAC-2 GEM+, PAC-3 " \
          "ERINT/MSE])' }, [2] = { guid = 'f8ux3a-0hmbjij2jeqbh', name = 'MIM-104F Patriot PAC-3 MSE #62' }}"

    lap = LuaParser(txt)
    arr = lap.parse_array()
    for i in arr:
        print(f"{i}")
