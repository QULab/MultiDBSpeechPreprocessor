FROM mambaorg/micromamba:debian13-slim
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# set up micromamba environment and python modules
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml .
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt .
RUN micromamba install -y -n base -f env.yaml && \
    micromamba clean --all --yes && \
    pip install -r requirements.txt

USER root
RUN chmod 777 -R /home/$MAMBA_USER
WORKDIR /resonanz
