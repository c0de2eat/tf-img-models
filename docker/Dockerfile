FROM nvcr.io/nvidia/tensorflow:21.08-tf2-py3

RUN apt-get update && \
    apt-get install -y graphviz

RUN mkdir /tfim
ADD . /tfim
RUN cd /tfim && pip install -e .

RUN useradd -ms /bin/bash tfim-docker
USER tfim-docker

WORKDIR /tfim
