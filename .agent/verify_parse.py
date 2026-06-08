import ast, sys

files = [
    'D:/distributed-llm/src/distllm/api/server.py',
    'D:/distributed-llm/src/distllm/api/routes/chat.py',
    'D:/distributed-llm/src/distllm/core/cost_tracker.py',
    'D:/distributed-llm/src/distllm/constants.py',
]

for f in files:
    try:
        tree = ast.parse(open(f, encoding='utf-8').read())
        print(f'OK: {f}')
    except SyntaxError as e:
        print(f'FAIL: {f} - {e}')
        sys.exit(1)

print('ALL PARSE OK')
