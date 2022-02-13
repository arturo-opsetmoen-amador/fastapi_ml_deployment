FROM python:3.10.0

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo

RUN pip install --upgrade pip
RUN useradd -m -s /bin/bash arturo_docker
USER arturo_docker
ENV PATH "$PATH:/home/arturo_docker/.local/bin"
COPY ./requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

WORKDIR /home/arturo_docker/
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/arturo_docker/awscliv2.zip"
RUN unzip awscliv2.zip
USER root
RUN  ./aws/install
RUN rm -rf aws awscliv2.zip
USER arturo_docker

# ENV PYTHONPATH "${PYTHONPATH}:/clean_churn/tests:/clean_churn:"
# WORKDIR /clean_churn
WORKDIR /home/arturo_docker/
ENTRYPOINT ["bash"]
CMD ["run.sh"]
