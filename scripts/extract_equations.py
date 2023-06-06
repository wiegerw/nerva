#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


class Processor(object):
    def __init__(self):
        # the current item
        self.header = ''
        self.feedforward = ''
        self.backpropagation = ''

        # maps headers to (feedforward, backpropagation) pairs
        self.result = []

    def finish_current_item(self):
        if self.header:
            self.result.append((self.header, self.feedforward, self.backpropagation))
            self.header = ''

    def process_paragraph(self, text: str) -> None:
        lines = text.split('\n')
        header = lines[0].strip()

        if header.startswith('def') and 'layer' in header:
            m = re.search(r'def test_(\w+)\(self\)', header)
            if m:
                self.header = m.group(1)
                self.feedforward = ''
                self.backpropagation = ''
        elif '# feedforward' in header:
            lines = [line for line in lines if not 'substitute' in line and not 'feedforward' in line and not 'backpropagation' in line]
            lines = [line for line in lines if not re.match(r'\w = \w$', line.strip())]
            lines = [line.lstrip() for line in lines]
            self.feedforward = '\n'.join(lines)
        elif '# backpropagation' in header:
            lines = [line for line in lines if not 'substitute' in line and not 'feedforward' in line and not 'backpropagation' in line]
            lines = [line for line in lines if not re.match(r'\w = \w$', line.strip())]
            lines = [line.lstrip() for line in lines]
            self.backpropagation = '\n'.join(lines)
            self.finish_current_item()


def print_header(header):
    indent = ' '*10
    return f'#------------------------------------------#\n#{indent}{header}\n#------------------------------------------#\n'


def to_text(header, feedforward, backpropagation) -> str:
    return print_header(header) + '\n' + feedforward + '\n\n' + backpropagation


def to_latex(header, feedforward, backpropagation) -> str:
    start = r'''\textsc{implementation} \vspace{-0.1cm}
\begin{footnotesize}
\begin{verbatim}
'''

    end = r'''\end{verbatim}
\end{footnotesize}
'''
    return print_header(header) + '\n' + start + feedforward + '\n\n' + backpropagation + '\n' + end

# N.B. this transformation is not very robust
def to_cpp(header, feedforward, backpropagation) -> str:
    def f(line: str) -> str:
        line = line.replace('.T', '.transpose()')
        line = line.replace('ones', 'ones<eigen::matrix>')
        line = line.replace('identity', 'identity<eigen::matrix>')
        line = re.sub(r'\bsoftmax_colwise\(', 'stable_softmax_colwise()(', line)
        line = re.sub(r'\blog_softmax_colwise\(', 'stable_log_softmax_colwise()(', line)
        line = re.sub(r'\bsoftmax_rowwise\(', 'stable_softmax_rowwise()(', line)
        line = re.sub(r'\blog_softmax_rowwise\(', 'stable_log_softmax_rowwise()(', line)
        m = re.match(r'R = (.*)$', line)
        if m:
            line = f'R = ({m.group(1)}).eval()'
        return line + ';'

    feedforward = '\n'.join([f(line) for line in feedforward.strip().split('\n')])
    backpropagation = '\n'.join([f(line) for line in backpropagation.strip().split('\n')])
    return print_header(header) + '\n' + feedforward + '\n\n' + backpropagation + '\n'


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('--latex', help='generate LaTeX output', action='store_true')
    cmdline_parser.add_argument('--cpp', help='generate C++ output', action='store_true')
    args = cmdline_parser.parse_args()

    processor = Processor()
    for path in sorted(Path('../python/tests').glob('test_layer_*.py')):
        text = path.read_text()
        paragraphs = re.split(r'\n(\s*\n)+', text, flags=re.MULTILINE)[::2]
        for paragraph in paragraphs:
            processor.process_paragraph(paragraph)

    if args.latex:
        paragraphs = [to_latex(header, feedforward, backpropagation) for (header, feedforward, backpropagation) in processor.result]
    elif args.cpp:
        paragraphs = [to_cpp(header, feedforward, backpropagation) for (header, feedforward, backpropagation) in processor.result]
    else:
        paragraphs = [to_text(header, feedforward, backpropagation) for (header, feedforward, backpropagation) in processor.result]

    print('\n\n'.join(paragraphs))


if __name__ == '__main__':
    main()
