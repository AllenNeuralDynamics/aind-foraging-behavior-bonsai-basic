# hash:sha256:1e3693de94598c93b871591c195c3800206d17a70c8d072b153a41f3c549d8d3
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    matplotlib==3.7.1 \
    pandas==2.0.1 \
    plotly==5.15.0 \
    pynwb==2.3.2 \
    seaborn==0.12.2 \
    xarray==2023.4.2 \
    zarr==2.14.2 \
    ipykernel \
    s3fs \
    awscli \
    aind_codeocean_api==0.2.3\
    aind-dynamic-foraging-models==0.1.5

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server

COPY postInstall /
RUN /postInstall
