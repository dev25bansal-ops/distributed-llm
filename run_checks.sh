#!/bin/sh
python -c "
import ast
files = [
    'src/distllm/cli/client.py',
    'src/distllm/core/auto_discovery.py',
    'src/distllm/core/model_sizing.py',
]
for f in files:
    try:
        with open(f) as fh:
            ast.parse(fh.read())
        print(f'OK: {f}')
    except SyntaxError as e:
        print(f'ERROR: {f}: {e}')
    except Exception as e:
        print(f'ERROR: {f}: {e}')
print('--- SYNTAX CHECK DONE ---')
"