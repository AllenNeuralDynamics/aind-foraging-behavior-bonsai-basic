# hash:sha256:67dc5d5180c837a867505d0765059ece78ce6d7a6bc94109babf1d49e68f8914
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    matplotlib==3.7.1 \
    pandas==2.0.1 \
    plotly \
    pynwb==2.3.2 \
    seaborn==0.12.2 \
    xarray==2023.4.2 \
    zarr==2.14.2
