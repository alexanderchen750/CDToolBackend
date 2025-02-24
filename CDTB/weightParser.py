from lark import Lark, Tree, Token, Transformer
from collections import defaultdict
import json
import re
from parser_utils import NO_NESTING_GRAMMAR, NO_NESTING_GRAMMAR_TESTCASE, NESTING_GRAMMAR, NESTING_GRAMMAR_TESTCASE, QUANTIFIERS_GRAMMAR, QUANTIFIERS_GRAMMAR_TESTCASES

def number_alternatives_grammar(grammar_str):
    """Convert a grammar to explicitly number alternatives, handling multi-line rules"""
    lines = grammar_str.split('\n')
    numbered_lines = []
    current_rule = None
    current_alternatives = []
    
    for line in lines:
        stripped_line = line.strip()
        
        # Skip empty lines and directives
        if not stripped_line or stripped_line.startswith('%'):
            # If we were processing a rule, finish it before adding the empty line
            if current_rule:
                # Number the collected alternatives
                numbered_alts = [
                    f"({alt}) -> {current_rule}_{i}" 
                    for i, alt in enumerate(current_alternatives)
                ]
                numbered_lines.append(f"{current_rule}: {' | '.join(numbered_alts)}")
                current_rule = None
                current_alternatives = []
            numbered_lines.append(line)
            continue
            
        # Check if this is the start of a new rule
        if ':' in line:
            # If we were processing a previous rule, finish it
            if current_rule:
                numbered_alts = [
                    f"({alt}) -> {current_rule}_{i}" 
                    for i, alt in enumerate(current_alternatives)
                ]
                numbered_lines.append(f"{current_rule}: {' | '.join(numbered_alts)}")
            
            # Start new rule
            name, productions = line.split(':', 1)
            name = name.strip()
            
            # Skip terminal rules
            if name.isupper():
                numbered_lines.append(line)
                current_rule = None
                current_alternatives = []
                continue
                
            current_rule = name
            current_alternatives = []
            
            # Add any alternatives from this line
            if productions.strip():
                for alt in productions.strip().split('|'):
                    # Remove existing names (anything after ->)
                    alt = alt.split('->')[0].strip()
                    if alt:  # Only add non-empty alternatives
                        current_alternatives.append(alt)
                    
        # If this is a continuation line of the current rule
        elif current_rule and stripped_line.startswith('|'):
            # Split and clean alternatives from this line
            for alt in stripped_line[1:].split('|'):  # Skip the first | character
                # Remove existing names (anything after ->)
                alt = alt.split('->')[0].strip()
                if alt:  # Only add non-empty alternatives
                    current_alternatives.append(alt)
    
    # Don't forget to process the last rule if there is one
    if current_rule:
        numbered_alts = [
            f"({alt}) -> {current_rule}_{i}" 
            for i, alt in enumerate(current_alternatives)
        ]
        numbered_lines.append(f"{current_rule}: {' | '.join(numbered_alts)}")
    
    return '\n'.join(numbered_lines)

class AlternativeTracker(Transformer):
    def __init__(self):
        super().__init__()
        # Track counts for each sample separately
        self.sample_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # Track overall counts
        self.total_counts = defaultdict(lambda: defaultdict(int))
        
    def __default__(self, data, children, meta):
        """Track alternatives for any rule"""
        rule_name = data
        
        # Check if this is a numbered alternative
        if '_' in rule_name:
            # Extract rule name and alternative number
            parent_rule, alt_num = rule_name.rsplit('_', 1)
            if alt_num.isdigit():
                alt_num = int(alt_num)
                # Update counts for current sample
                self.sample_counts[self.current_sample][parent_rule][alt_num] += 1
                # Update total counts
                self.total_counts[parent_rule][alt_num] += 1
        
        return children

    def get_probabilities(self):
        """Calculate probabilities per sample and overall"""
        sample_probabilities = {}
        total_probabilities = {}
        
        # Calculate probabilities for each sample
        for sample, rule_counts in self.sample_counts.items():
            sample_probabilities[sample] = {}
            for rule, alternatives in rule_counts.items():
                total = sum(alternatives.values())
                if total > 0:
                    prob_dict = {
                        f"alternative_{idx}": count/total 
                        for idx, count in sorted(alternatives.items())
                    }
                    sample_probabilities[sample][rule] = prob_dict
        
        # Calculate overall probabilities
        for rule, alternatives in self.total_counts.items():
            total = sum(alternatives.values())
            if total > 0:
                prob_dict = {
                    f"alternative_{idx}": count/total 
                    for idx, count in sorted(alternatives.items())
                }
                total_probabilities[rule] = prob_dict
        
        return {
            "per_sample": sample_probabilities,
            "overall": total_probabilities
        }

def analyze_grammar_probabilities(grammar_str, input_samples):
    """Analyze alternative probabilities in a grammar"""
    try:
        # Print original grammar for debugging
        print("Original Grammar:")
        print(grammar_str)
        
        # Number the alternatives in the grammar
        numbered_grammar = number_alternatives_grammar(grammar_str)
        
        # Print numbered grammar for debugging
        print("\nNumbered Grammar:")
        print(numbered_grammar)
        
        # Create parser with numbered grammar
        parser = Lark(numbered_grammar, parser='lalr', debug=True)
        
        # Initialize tracker
        tracker = AlternativeTracker()
        
        # Parse each input and track alternatives
        for i, sample in enumerate(input_samples):
            try:
                # Set current sample identifier
                tracker.current_sample = f"sample_{i}"
                tree = parser.parse(sample)
                tracker.transform(tree)
            except Exception as e:
                print(f"Failed to parse sample: {sample}")
                print(f"Error: {e}")
                continue
        
        return tracker.get_probabilities()
        
    except Exception as e:
        print(f"Error analyzing grammar: {e}")
        return {}

# Keep number_alternatives_grammar function as is...

if __name__ == "__main__":
    test_grammar = NESTING_GRAMMAR
    test_samples = NESTING_GRAMMAR_TESTCASE
    
    probabilities = analyze_grammar_probabilities(test_grammar, test_samples)
    print("\nProbabilities:")
    print(json.dumps(probabilities, indent=2))