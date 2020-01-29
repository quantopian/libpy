# DOCKER-VERSION 1.3.0
FROM ubuntu:18.04
LABEL MAINTAINER Quantopian Inc.

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y software-properties-common wget
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main" >> \
    /etc/apt/sources.list.d/llvm.list && apt-get update
RUN add-apt-repository ppa:deadsnakes/ppa
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
# set to python3.5 for py3 build
ARG PYTHON_BINARY_NAME=python2.7
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
    python3.6-dev \
    python3.6-venv \
    tzdata \
    util-linux \
    valgrind \
    # contained in the same step to reduce size and improve speed; unquoted
    # because we do want word splitting here
    $(if [ "${PYTHON_BINARY_NAME}" = "python3.5" ]; then \
        echo "python3-pip python3.5-dev"; \
    elif [ "${PYTHON_BINARY_NAME}" = "python3.6" ]; then \
        echo "python-pip python3.6-dev"; \
    elif [ "${PYTHON_BINARY_NAME}" = "python2.7" ]; then \
        echo "python-pip python2.7-dev"; \
    fi) \
    # confirm that we have the right python
    && which "${PYTHON_BINARY_NAME}" \
    && "${PYTHON_BINARY_NAME}" --version \
    && "${PYTHON_BINARY_NAME}" -m pip --version \
    # clean up apt caches
    && rm -rf /var/lib/apt/lists/*


RUN mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /src/
COPY . ./libpy
COPY ./etc/Makefile.buildkite ./libpy/Makefile.local

# clean up by resetting DEBIAN_FRONTEND to its default value:
ENV DEBIAN_FRONTEND newt
