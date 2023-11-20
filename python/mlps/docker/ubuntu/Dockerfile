FROM ubuntu:latest

# 1. Install packages
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    pybind11-dev \
    python3-pip \
    wget \
    zip

# Download Nerva
RUN cd ~/ && wget https://www.win.tue.nl/~wieger/tmp/dist.zip

# 2. Download and unzip the nerva package (normally this would be a GitHub install)
RUN cd ~/ && unzip dist.zip

# 3. Install python frameworks
RUN pip3 install jax numpy sympy tensorflow torch

# 4. Install the nerva packages
RUN cd ~/dist && sh install_packages.sh
