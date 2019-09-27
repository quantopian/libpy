# DOCKER-VERSION 1.3.0
FROM ubuntu:16.04
MAINTAINER Quantopian Inc.

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y software-properties-common wget
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main" >> \
    /etc/apt/sources.list.d/llvm.list && apt-get update
RUN add-apt-repository ppa:deadsnakes/ppa
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y \
    clang-tools-6.0 \
    doxygen \
    g++-8 \
    gcc-8 \
    gfortran \
    git \
    libatlas-base-dev \
    libblas-dev \
    libffi-dev \
    libpcre2-dev \
    libsparsehash-dev \
    libssl-dev \
    make \
    openssh-client \
    python-dev \
    python-tox \
    python3.6-dev \
    python3.6-venv \
    tzdata \
    util-linux \
    valgrind \
    virtualenv

RUN mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /src/
COPY . ./libpy
COPY ./etc/Makefile.buildkite ./libpy/Makefile.local

# clean up by resetting DEBIAN_FRONTEND to its default value:
ENV DEBIAN_FRONTEND newt
