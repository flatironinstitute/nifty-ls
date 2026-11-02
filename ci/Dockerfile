FROM docker.io/nvidia/cuda:12.9.1-devel-rockylinux9

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN dnf -y install \
    git \
    fftw-devel

# Rocky 9's python3 is 3.9, older than requires-python
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.13
