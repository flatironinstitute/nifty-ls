FROM nvidia/cuda:12.9.1-devel-rockylinux9

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN dnf -y install \
    python3 \
    python3-devel \
    git
