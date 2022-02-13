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

ENV PYTHONPATH "${PYTHONPATH}:/clean_churn/tests:/clean_churn:"

WORKDIR /clean_churn
ENTRYPOINT ["bash"]
CMD ["run.sh"]
