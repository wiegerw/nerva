FROM ubuntu:latest

# 1. Install packages
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    pybind11-dev \
    python3-pip \
    vim-tiny \
    wget \
    zip

# 1. Install python frameworks
RUN pip3 install jax numpy sympy tensorflow torch scikit-learn

# 2. Download Nerva
RUN cd ~/ && wget https://www.win.tue.nl/~wieger/tmp/dist.zip

# 3. Unzip the nerva package
RUN cd ~/ && unzip dist.zip

# 4. Install the nerva packages
RUN cd ~/dist && sh install_packages.sh
