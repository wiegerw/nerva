#!/usr/bin/env python3

import argparse
from pathlib import Path

if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("file", help="The input file", type=str)
    args = cmdline_parser.parse_args()

    feedforward = 0.0
    backpropagate = 0.0

    text = Path(args.file).read_text()
    for line in text.strip().split('\n'):
        if line.startswith('feedforward'):
            label, seconds = line.split()
            seconds = seconds.replace('s', '')
            feedforward += float(seconds)
        elif line.startswith('backpropagate'):
            label, seconds = line.split()
            seconds = seconds.replace('s', '')
            backpropagate += float(seconds)

    print(f'feedforward  : {feedforward:.8f}')
    print(f'backpropagate: {backpropagate:.8f}')
