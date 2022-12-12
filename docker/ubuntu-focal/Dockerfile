FROM ubuntu:focal

# 1. Install packages
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  build-essential \
  wget \
  zip \
  libmkl-dev \
  pybind11-dev \
  python3-pip \
  python3-pybind11 \
  python3-numpy \
  python3-keras-applications \
  python3-keras-preprocessing

RUN pip3 install tensorflow

# A newer version of Eigen is required
RUN cd ~/ && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && tar xzvf eigen-3.4.0.tar.gz

# Note that "~/eigen-3.4.0" and "$HOME/eigen-3.4.0" doesn't work here
ENV EIGENROOT="/root/eigen-3.4.0"

# 2. Download and unzip the nerva package
RUN cd ~/ && wget https://www.win.tue.nl/~wieger/tmp/nerva-main.zip && unzip nerva-main.zip

# 3. Install the nerva package
RUN cd ~/nerva-main && pip3 install . --user

# 4. Test the installation
RUN cd ~/nerva-main/python && python3 cifar10.py
