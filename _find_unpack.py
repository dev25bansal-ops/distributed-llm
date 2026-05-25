import os

src = r'D:\distributed-llm\src\distllm'
targets = ['output, new_kv', 'output, new_past_kv', 'logits, new_kv']

for root, dirs, files in os.walk(src):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path) as fh:
                    for i, line in enumerate(fh, 1):
                        for t in targets:
                            if t in line:
                                print(f'{path}:{i}: {line.rstrip()}')
            except:
                pass
