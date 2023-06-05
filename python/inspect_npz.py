#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import torch
from snn.datasets import load_dict_from_npz
from snn import pp


def inspect_npz_file(filename: str):
    data = load_dict_from_npz(filename)
    for key, value in data.items():
        pp(key, value)
        print()


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('filename', metavar='FILE', type=str, help='an .npz file containing a dictionary')
    args = cmdline_parser.parse_args()
    torch.set_printoptions(precision=8, edgeitems=3, threshold=5, sci_mode=False, linewidth=160)
    inspect_npz_file(args.filename)


if __name__ == '__main__':
    main()
