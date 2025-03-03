from lark import Lark, Tree
import logging
import re

#logging.basicConfig(level=logging.DEBUG)
#logging.getLogger("lark").setLevel(logging.DEBUG)
DEFAULT_GRAMMAR = """
    start: value

    value: object 
         | array 
         | string 
         | NUMBER 
         | "true" 
         | "false" 
         | "null"

    object: "{" [pair ("," pair)*] "}"
    pair: string ":" value

    array: "[" [value ("," value)*] "]"

    string: ESCAPED_STRING
    NUMBER: /-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?/

    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS
"""

def tree_to_dict(tree):
    """Converts a Lark parse tree into a dictionary format."""
    return {
        'name': tree.data,
        'children': [tree_to_dict(child) if isinstance(child, Tree) else {'name': str(child)}
                     for child in tree.children],
    }

def parse_input(text_input, grammar=DEFAULT_GRAMMAR):
    """
    Parses input text using the provided grammar.
    Returns the parsed tree in dictionary format.
    """
    try:
        print("init Lark")
        parser = Lark(format_to_lark_ebnf(grammar), start="start", debug=True)
        print("starting parser.parse")
        tree = parser.parse(text_input)
        print("Sucess")
        return {'status': 'success', 'output': text_input, 'parse_tree': tree_to_dict(tree)}
    except Exception as e:
        print("fail")
        print(str(e))
        return {'status': 'error', 'message': str(e)}

def format_to_lark_ebnf(grammar):
    cleaned_grammar = re.sub("::=",":",grammar)
    cleaned_grammar = re.sub("root","start",cleaned_grammar)
    print(cleaned_grammar)
    return cleaned_grammar

# Default grammar

NO_NESTING_GRAMMAR = """
    start: greeting
    greeting: "hello" | "hi" | "hey"
    
    %import common.WS
    %ignore WS
"""

test_samples1 = [
    "hello",
    "hi",
    "hey",
    "hello",
    "hi"
]

NO_NESTING_GRAMMAR_TESTCASE = [
    "hello",
    "hi",
    "hey",
    "hello",
    "hi"
]
NESTING_GRAMMAR = """
    start: expr
    expr: NUMBER
        | expr "+" expr -> add
        | "(" expr ")"
    NUMBER: /[0-9]+/
    
    %import common.WS
    %ignore WS
"""

NESTING_GRAMMAR_TESTCASE = [
    "42",
    "1 + 2",
    "(42)",
    "(1 + 2)",
    "1 + (2 + 3)"
]

QUANTIFIERS_GRAMMAR = """
    start: list
    list: "[" items? "]"
    items: item ("," item)*
    item: NUMBER
    NUMBER: /[0-9]+/
    
    %import common.WS
    %ignore WS
"""

QUANTIFIERS_GRAMMAR_TESTCASES = [
    "[]",
    "[1]",
    "[1, 2]",
    "[1, 2, 3]",
    "[1, 2, 3, 4]"
]