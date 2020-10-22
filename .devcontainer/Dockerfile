# [Choice] Debian / Ubuntu version: debian-10, debian-9
ARG VARIANT=buster
FROM mcr.microsoft.com/vscode/devcontainers/cpp:dev-${VARIANT}

# [Optional] Uncomment this section to install additional packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends libsparsehash-dev doxygen python3-dev python3-pip

ENV CC=gcc-9
ENV CXX=g++-9
ENV PYTHON=python3
