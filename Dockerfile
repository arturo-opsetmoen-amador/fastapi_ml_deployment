FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo

RUN apt-get update && apt upgrade -y
RUN apt-get install software-properties-common build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev -y
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
RUN tar -xf Python-3.10.0.tgz
RUN cd Python-3.10.0/ && ./configure --enable-optimizations && make altinstall -j8
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 1
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt install python3.10
RUN which python
RUN ls /usr/bin/python*
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python 1
RUN ./usr/bin/python --version
RUN pip install --upgrade pip
RUN useradd -m -s /bin/bash arturo_docker
USER arturo_docker
ENV PATH "$PATH:/home/arturo_docker/.local/bin"
COPY ./requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

WORKDIR /home/arturo_docker/
USER root
RUN apt-get install curl unzip -y
USER arturo_docker
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/arturo_docker/awscliv2.zip"
RUN unzip awscliv2.zip
USER root
RUN  ./aws/install
RUN rm -rf aws awscliv2.zip
RUN curl https://cli-assets.heroku.com/install-ubuntu.sh | sh
USER arturo_docker
USER root
RUN apt-get install git -y
RUN rm /Python-3.10.0.tgz
RUN rm -rf /Python-3.10.0

USER arturo_docker
# ENV PYTHONPATH "${PYTHONPATH}:/clean_churn/tests:/clean_churn:"
# WORKDIR /clean_churn
WORKDIR /home/arturo_docker/
ENTRYPOINT ["bash"]
CMD ["run.sh"]
