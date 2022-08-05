FROM rapidsai/rapidsai-core:22.06-cuda11.4-runtime-ubuntu18.04-py3.8
RUN apt update
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y vim && apt-get install -y git
RUN pip install --upgrade pip
RUN pip install matplotlib seaborn scikit-learn scipy pandas numpy jupyter torch
RUN apt-get install -y fonts-nanum
RUN rm -rf ~/.cache/matplotlib/*
WORKDIR /
RUN pip freeze > requirements.txt
ARG UNAME
ARG UID
ARG GID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME