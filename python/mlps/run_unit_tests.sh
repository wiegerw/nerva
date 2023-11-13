#!/bin/bash

cd nerva-sympy/tests

for file in test*.py ; do
    echo "$file"
        ./"$file"
done
