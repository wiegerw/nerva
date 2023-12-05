#!/usr/bin/env python3
#
# This script generates LaTeX tables for the output of mlp_experiment.sh.

import re
from pathlib import Path

# epoch   1  lr: 0.01000000  loss: 0.20351325  train accuracy: 0.94016667  test accuracy: 0.93890000  time: 4.90890760s
def make_table(name: str, dataset: str, epoch: int):
    filename = f'{name}.log'
    text = Path(filename).read_text()
    lines = text.strip().split('\n')

    tool = 'Unknown'

    result = []
    for line in lines:
        if 'tool:' in line:
            line = line.replace('tool:', '')
            line = line.replace('=', '')
            tool = line.strip()
        elif re.search(rf'epoch\s+{epoch} ', line):
            line = line.replace(r'\\', '')
            line = re.sub(r'epoch\s+\d+', '', line)
            line = line.replace('lr:', '')
            line = line.replace('loss:', '')
            line = line.replace('train accuracy:', '')
            line = line.replace('test accuracy:', '')
            line = line.replace('time:', '')
            line = line.replace('s', '')
            words = line.strip().split()
            del words[0]
            for i in range(len(words)):
                words[i] = f'{float(words[i]):.3f}'
            result.append(' & '.join([tool, dataset] + words) + r' \\')
    result.insert(1, '\midrule')
    result.insert(0, '\midrule')
    print('\n'.join(result))

make_table('mnist', 'MNIST', 1)
make_table('cifar10', 'CIFAR-10', 1)
