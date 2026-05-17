import sys; sys.path.insert(0, 'D:\\distributed-llm\\src')
from distllm.core.grammar_decoder import GBNFParser, GBNFFSM, LiteralNode, AltNode, SeqNode

# 1. Parser test
grammar = 'root ::= "hello"'
parser = GBNFParser(grammar)
rules = parser.parse()
node = rules['root']
print(f'Parser: root = {type(node).__name__}')
assert isinstance(node, LiteralNode)
assert node.value == 'hello'

# 2. Alternation
grammar2 = 'root ::= "yes" | "no"'
parser2 = GBNFParser(grammar2)
rules2 = parser2.parse()
print(f'Parser alt: root = {type(rules2["root"]).__name__}')

# 3. Basic FSM - literal
fsm = GBNFFSM('root ::= "hello"')
allowed = fsm.get_allowed_bytes()
print(f'FSM literal first: {[chr(b) for b in sorted(allowed)]}')
assert 0x68 in allowed  # 'h'

# 4. Track literal transitions
fsm.transition(0x68)  # h
all2 = fsm.get_allowed_bytes()
print(f'After h: {[chr(b) for b in sorted(all2)]}')
assert 0x65 in all2  # 'e'
assert not fsm.is_accepting()

fsm.transition(0x65)  # e
fsm.transition(0x6C)  # l
fsm.transition(0x6C)  # l
fsm.transition(0x6F)  # o
print(f'After hello, accepting={fsm.is_accepting()}')
assert fsm.is_accepting()

# 5. Char class
fsm2 = GBNFFSM('root ::= [0-9]+')
all3 = fsm2.get_allowed_bytes()
assert 0x30 in all3
assert 0x39 in all3
print(f'CharClass: digits 0-9 allowed: {0x30 in all3 and 0x39 in all3}')

fsm2.transition(0x35)  # 5
all4 = fsm2.get_allowed_bytes()
assert 0x30 in all4
print(f'After 5, digits still allowed: {0x30 in all4}')

# 6. Alternation
fsm3 = GBNFFSM('root ::= "yes" | "no"')
all5 = fsm3.get_allowed_bytes()
assert 0x79 in all5  # y
assert 0x6E in all5  # n
print(f'Alternation: y={0x79 in all5} n={0x6E in all5}')

# 7. With constrained decoder
from distllm.core.constrained_decoder import SchemaConstrainedDecoder, ConstrainedConstraint, TokenIndex
class FakeTokenizer:
    vocab_size = 100
    eos_token_id = 1
    def get_vocab(self):
        return {chr(i): i for i in range(32, 96)} | {f't{i}': i for i in range(96, 100)}
tok = FakeTokenizer()
decoder = SchemaConstrainedDecoder(tok)
constraint = decoder.grammar('root ::= [a-zA-Z]+')
import torch
mask = constraint.get_logits_mask(100)
print(f'Grammar mask: {mask.sum().item()} allowed / 100 total')

print('\nALL GBNF TESTS PASSED')
