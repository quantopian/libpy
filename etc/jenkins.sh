#!/bin/bash
# Set exit on error to make sure that we notice build errors.
set -e

# Clean out build artifacts form the repo and submodules before calling
# docker build, which copies the source tree into the image.
make clean
docker build -t libpy/test-image .

# Create a directory to share as a volume, into which the tests will
# write the test-output
shared_volume=$(mktemp -d --suffix="-libpy")
docker run --cap-add SYS_PTRACE -v $shared_volume:/test-output libpy/test-image \
       bash -c "cd /src/libpy && . VENV_ACTIVATE_27 && OUTPUT_DIR=/test-output ./etc/runtests"
docker run --cap-add SYS_PTRACE -v $shared_volume:/test-output libpy/test-image \
       bash -c "cd /src/libpy && . VENV_ACTIVATE_36 && OUTPUT_DIR=/test-output ./etc/runtests"

# Copy libpy_report.xml out of shared volume.
cp $shared_volume/libpy_report.xml .
rm -rf $shared_volume
