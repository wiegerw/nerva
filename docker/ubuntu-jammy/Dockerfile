FROM ubuntu:jammy

# 1. Install packages
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  build-essential \
  libeigen3-dev \
  libmkl-dev \
  pybind11-dev \
  python3-pip \
  python3-pybind11 \
  python3-numpy \
  python3-torch \
  python3-torchvision \
  wget \
  zip

# 2. Download and unzip the nerva package (normally this would be a GitHub install)
RUN cd ~/ && wget https://www.win.tue.nl/~wieger/tmp/nerva-main.zip && unzip nerva-main.zip

# 3. Install the nerva package
RUN cd ~/nerva-main && pip3 install .

# 4. Test the nerva package (this should be done in an interactive session)
RUN cd ~/nerva-main/python && ./chessboard.py
