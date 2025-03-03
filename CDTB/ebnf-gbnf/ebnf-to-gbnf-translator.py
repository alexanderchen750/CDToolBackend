import re
import argparse
from typing import Dict, List, Tuple, Optional, Set
import sys
import os
from enum import Enum, auto

class GrammarFormat(Enum):
    """Enum for different grammar format variations."""
    LARK_EBNF = auto()  # Lark's EBNF variant
    STANDARD_EBNF = auto()  # Standard EBNF (ISO/IEC 14977)
    ANTLR = auto()  # ANTLR grammar format

class EBNFToGBNFTranslator:
    """
    Translator for converting EBNF (Extended Backus-Naur Form) grammar to GBNF (GGML BNF) format.
    
    EBNF is often used with parsers like Lark, while GBNF is used for constrained 
    decoding in language models via libraries like llama.cpp and xgrammar.
    """
    
    def __init__(self, format_type: GrammarFormat = GrammarFormat.LARK_EBNF):
        self.rules: Dict[str, str] = {}
        self.referenced_rules: Set[str] = set()
        self.format_type = format_type
        self.terminals: Set[str] = set()  # Collect terminal symbols
        self.root_rule = "root"  # Default root rule name
        
    def parse_ebnf(self, ebnf_text: str) -> None:
        """Parse EBNF grammar text into rules dictionary."""
        self.rules = {}
        self.referenced_rules = set()
        
        # Preprocessing: remove comments based on format
        if self.format_type in [GrammarFormat.LARK_EBNF, GrammarFormat.ANTLR]:
            # Remove single line comments
            ebnf_text = re.sub(r'//.*$', '', ebnf_text, flags=re.MULTILINE)
            # Remove multi-line comments
            ebnf_text = re.sub(r'/\*[\s\S]*?\*/', '', ebnf_text)
            # Remove Lark-style comments
            ebnf_text = re.sub(r'#.*$', '', ebnf_text, flags=re.MULTILINE)
        elif self.format_type == GrammarFormat.STANDARD_EBNF:
            # Standard EBNF uses (* comment *) format
            ebnf_text = re.sub(r'\(\*[\s\S]*?\*\)', '', ebnf_text)
        
        # Extract rules based on format
        if self.format_type == GrammarFormat.LARK_EBNF:
            # Lark uses both := and : for rule definitions
            rule_pattern = re.compile(r'(\w[\w\-_]*)\s*(?::=|:)\s*([^;]*?)(?:;|\n\s*(?=\w[\w\-_]*\s*(?::=|:)))', re.DOTALL)
        elif self.format_type == GrammarFormat.ANTLR:
            # ANTLR uses : for rule definitions and ; for termination
            rule_pattern = re.compile(r'(\w[\w\-_]*)\s*:\s*([^;]*);', re.DOTALL)
        else:  # Standard EBNF
            rule_pattern = re.compile(r'(\w[\w\-_]*)\s*=\s*([^;]*);', re.DOTALL)
        
        matches = rule_pattern.findall(ebnf_text)
        
        for name, definition in matches:
            name = name.strip()
            definition = definition.strip()
            
            # Convert rule name to lowercase with dashes for GBNF compatibility
            gbnf_name = self._normalize_rule_name(name)
            
            self.rules[gbnf_name] = definition
            
            # Find all referenced rules to check for undefined rules later
            self._find_referenced_rules(gbnf_name, definition)
    
    def _normalize_rule_name(self, name: str) -> str:
        """Convert rule names to GBNF compatible format (lowercase with dashes)."""
        # GBNF requires rule names to be lowercase with dashes
        normalized = name.lower()
        
        # Replace underscores with dashes
        normalized = normalized.replace('_', '-')
        
        # Handle camelCase or PascalCase by inserting dashes
        normalized = re.sub(r'([a-z])([A-Z])', r'\1-\2', normalized)
        
        return normalized
        
    def _find_referenced_rules(self, current_rule: str, definition: str) -> None:
        """
        Identify rule references within a definition.
        This requires parsing the definition to distinguish between terminals and rule references.
        """
        # First, extract all strings which are likely terminals
        string_literals = set()
        # Handle different string delimiters based on format
        if self.format_type == GrammarFormat.LARK_EBNF:
            # Lark supports both single and double quotes
            string_literals.update(re.findall(r'"([^"]*)"', definition))
            string_literals.update(re.findall(r"'([^']*)'", definition))
        else:
            # Standard EBNF typically uses double quotes
            string_literals.update(re.findall(r'"([^"]*)"', definition))
        
        # Add these to our terminals
        self.terminals.update(string_literals)
        
        # Now find all word-like tokens that might be rule references
        for word in re.findall(r'\b(\w[\w\-_]*)\b', definition):
            normalized_word = self._normalize_rule_name(word)
            if (normalized_word != current_rule and 
                normalized_word not in ['true', 'false', 'null'] and
                not re.match(r'^[0-9]+$', normalized_word)):  # Skip numbers
                self.referenced_rules.add(normalized_word)
    
    def convert_ebnf_rule_to_gbnf(self, rule_name: str, definition: str) -> str:
        """Convert a single EBNF rule definition to GBNF format."""
        # First, we need to tokenize the definition to properly handle nested structures
        tokens = self._tokenize_definition(definition)
        
        # Process tokens to convert to GBNF format
        gbnf_tokens = self._process_tokens(tokens)
        
        # Join processed tokens back into a string
        gbnf_definition = ''.join(gbnf_tokens)
        
        # Normalize whitespace
        gbnf_definition = re.sub(r'\s+', ' ', gbnf_definition).strip()
        
        return f"{rule_name} ::= {gbnf_definition}"
    
    def _tokenize_definition(self, definition: str) -> List[str]:
        """
        Tokenize an EBNF definition into a list of tokens.
        This ensures we handle nested structures correctly.
        """
        tokens = []
        i = 0
        length = len(definition)
        
        while i < length:
            char = definition[i]
            
            # Handle string literals
            if char in ['"', "'"]:
                quote = char
                start = i
                i += 1
                while i < length and definition[i] != quote:
                    if definition[i] == '\\' and i + 1 < length:
                        i += 2  # Skip escape sequence
                    else:
                        i += 1
                
                if i < length:  # Found closing quote
                    tokens.append(definition[start:i+1])
                    i += 1
                else:  # No closing quote
                    tokens.append(definition[start:])
                    i = length
            
            # Handle character classes
            elif char == '[':
                start = i
                i += 1
                bracket_depth = 1
                
                while i < length and bracket_depth > 0:
                    if definition[i] == '[':
                        bracket_depth += 1
                    elif definition[i] == ']':
                        bracket_depth -= 1
                    elif definition[i] == '\\' and i + 1 < length:
                        i += 1  # Skip escape
                    i += 1
                
                tokens.append(definition[start:i])
            
            # Handle parentheses for grouping
            elif char == '(':
                start = i
                i += 1
                paren_depth = 1
                
                while i < length and paren_depth > 0:
                    if definition[i] == '(':
                        paren_depth += 1
                    elif definition[i] == ')':
                        paren_depth -= 1
                    elif definition[i] == '\\' and i + 1 < length:
                        i += 1  # Skip escape
                    i += 1
                
                tokens.append(definition[start:i])
            
            # Handle repetition operators
            elif char in ['{', '?', '*', '+']:
                if char == '{':
                    start = i
                    i += 1
                    while i < length and definition[i] != '}':
                        i += 1
                    if i < length:
                        i += 1  # Include the closing brace
                    tokens.append(definition[start:i])
                else:
                    tokens.append(char)
                    i += 1
            
            # Handle alternatives
            elif char == '|':
                tokens.append(char)
                i += 1
            
            # Handle whitespace
            elif char.isspace():
                i += 1
                while i < length and definition[i].isspace():
                    i += 1
                tokens.append(' ')
            
            # Handle rule references and other symbols
            else:
                start = i
                while i < length and not (definition[i] in '"|\'()[]{}?*+' or definition[i].isspace()):
                    i += 1
                tokens.append(definition[start:i])
        
        return tokens
    
    def _process_tokens(self, tokens: List[str]) -> List[str]:
        """Process tokenized EBNF into GBNF-compatible tokens."""
        result = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Convert string literals
            if (token.startswith("'") and token.endswith("'")) or (token.startswith('"') and token.endswith('"')):
                # Extract the content and convert to double-quoted string for GBNF
                content = token[1:-1]
                # Escape any double quotes if we're converting from single quotes
                if token.startswith("'"):
                    content = content.replace('"', '\\"')
                result.append(f'"{content}"')
            
            # Handle character classes
            elif token.startswith('[') and token.endswith(']'):
                # GBNF character classes use the same format but may have different escape sequences
                result.append(self._convert_character_class(token))
            
            # Handle grouping
            elif token.startswith('(') and token.endswith(')'):
                # Process the content inside parentheses recursively
                inner_tokens = self._tokenize_definition(token[1:-1])
                processed_inner = self._process_tokens(inner_tokens)
                inner_str = ''.join(processed_inner)
                result.append(f'({inner_str})')
            
            # Handle repetition
            elif token in ['?', '*', '+']:
                result.append(token)
            
            # Handle exact repetition {n}
            elif token.startswith('{') and token.endswith('}'):
                inside = token[1:-1].strip()
                
                # Check if it's {n} or {n,m} format
                if ',' in inside:
                    # Range repetition {n,m}
                    result.append(token)
                elif inside.isdigit():
                    # Exact repetition {n}
                    result.append(token)
                else:
                    # EBNF {x} for 0+ becomes x* in GBNF
                    result.append('*')
            
            # Handle alternatives
            elif token == '|':
                result.append(' | ')
            
            # Handle whitespace
            elif token.isspace():
                result.append(' ')
            
            # Handle rule references and other symbols
            else:
                # Check if it's a rule reference (non-terminal)
                if token.strip():
                    # Convert to GBNF rule name format if needed
                    normalized = self._normalize_rule_name(token.strip())
                    result.append(normalized)
            
            i += 1
        
        return result
    
    def _convert_character_class(self, char_class: str) -> str:
        """Convert EBNF character class to GBNF format."""
        # GBNF character classes are similar: [a-z] means any character from a to z
        
        # Remove outer brackets to process the content
        content = char_class[1:-1]
        
        # Check if it's a negated class
        if content.startswith('^'):
            negated = True
            content = content[1:]
        else:
            negated = False
        
        # Process escape sequences if needed
        # For now, we'll pass through most escapes as-is since GBNF handles similar escape sequences
        
        # Reconstruct the character class
        if negated:
            return f'[^{content}]'
        else:
            return f'[{content}]'
    
    def convert_to_gbnf(self) -> str:
        """Convert the full EBNF grammar to GBNF format."""
        gbnf_rules = []
        
        # Check for undefined rules
        undefined = self.referenced_rules - set(self.rules.keys())
        if undefined:
            print(f"Warning: Referenced but undefined rules: {', '.join(undefined)}", file=sys.stderr)
        
        # Find the root rule - if 'root' exists, use it, otherwise use the first rule or 'start'
        root_rule = None
        
        # Check for 'root' rule first (GBNF convention)
        if 'root' in self.rules:
            root_rule = 'root'
        # Then check for 'start' rule (common EBNF convention)
        elif 'start' in self.rules:
            root_rule = 'start'
            # We'll need to add a 'root' rule that points to 'start'
            self.rules['root'] = 'start'
            print("Note: Using 'start' rule as the grammar entry point via 'root'", file=sys.stderr)
        # Otherwise use the first rule
        elif self.rules:
            first_rule = next(iter(self.rules))
            root_rule = first_rule
            # Add a 'root' rule that points to the first rule
            self.rules['root'] = first_rule
            print(f"Note: No 'root' or 'start' rule found. Using '{first_rule}' as the grammar entry point via 'root'", file=sys.stderr)
        else:
            print("Error: No rules found in the grammar", file=sys.stderr)
            return ""
        
        # Process root rule first if it exists directly
        if root_rule == 'root':
            gbnf_rules.append(self.convert_ebnf_rule_to_gbnf('root', self.rules['root']))
        else:
            # Create a root rule that points to the actual starting rule
            gbnf_rules.append(f"root ::= {root_rule}")
        
        # Process all other rules
        for name, definition in self.rules.items():
            if name != 'root':  # Skip 'root' as we've already processed it
                gbnf_rules.append(self.convert_ebnf_rule_to_gbnf(name, definition))
        
        # Join rules with double newlines for readability
        return '\n\n'.join(gbnf_rules)

    def process_ebnf_file(self, filename: str) -> str:
        """Process an EBNF grammar file and return the equivalent GBNF grammar."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                ebnf_text = f.read()
            
            self.parse_ebnf(ebnf_text)
            return self.convert_to_gbnf()
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}", file=sys.stderr)
            raise

    def save_gbnf_file(self, gbnf_text: str, output_filename: str) -> None:
        """Save the GBNF grammar to a file."""
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(gbnf_text)
            
            print(f"Converted grammar saved to {output_filename}")
        except Exception as e:
            print(f"Error saving to file {output_filename}: {str(e)}", file=sys.stderr)
            raise

def detect_grammar_format(filename: str) -> GrammarFormat:
    """Attempt to detect the grammar format from file content."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for Lark-style grammar indicators
        if re.search(r'\b\w+\s*:=', content) or re.search(r'%import', content):
            return GrammarFormat.LARK_EBNF
        # Check for ANTLR-style grammar indicators
        elif re.search(r'grammar\s+\w+;', content) or re.search(r'\b\w+\s*:', content):
            return GrammarFormat.ANTLR
        # Default to standard EBNF
        else:
            return GrammarFormat.STANDARD_EBNF
    except Exception:
        # Default to Lark EBNF if detection fails
        return GrammarFormat.LARK_EBNF

def main():
    parser = argparse.ArgumentParser(description='Convert EBNF grammar to GBNF format')
    parser.add_argument('input_file', help='Input EBNF grammar file')
    parser.add_argument('-o', '--output', help='Output GBNF grammar file')
    parser.add_argument('-f', '--format', choices=['lark', 'standard', 'antlr'], 
                        help='Input grammar format (lark=Lark EBNF, standard=ISO EBNF, antlr=ANTLR grammar)')
    
    args = parser.parse_args()
    
    # Determine grammar format
    if args.format:
        if args.format == 'lark':
            format_type = GrammarFormat.LARK_EBNF
        elif args.format == 'standard':
            format_type = GrammarFormat.STANDARD_EBNF
        elif args.format == 'antlr':
            format_type = GrammarFormat.ANTLR
    else:
        # Auto-detect format
        format_type = detect_grammar_format(args.input_file)
        print(f"Auto-detected grammar format: {format_type.name}")
    
    # Create translator with the appropriate format
    translator = EBNFToGBNFTranslator(format_type)
    
    try:
        # Process the input file
        gbnf_grammar = translator.process_ebnf_file(args.input_file)
        
        # Determine output filename
        output_file = args.output or os.path.splitext(args.input_file)[0] + '.gbnf'
        
        # Save the converted grammar
        translator.save_gbnf_file(gbnf_grammar, output_file)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()