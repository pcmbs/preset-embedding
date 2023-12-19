# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime as dev 

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
# # avoids warnings when you go on to work with your container. 
ENV DEBIAN_FRONTEND=noninteractive
# disable the display in X11 to avoid MIT-SHM error (req. for dawdreamer)
ENV DISPLAY=none
# allow CUBLAS deterministic behavior
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# required for audio plugins 
RUN apt-get update && \
    apt-get -y --no-install-recommends install libgl1 libatomic1 libfreetype6 libasound2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Creates a non-root group and user with an explicit GID and UID and 
# change the required folders ownership to this user and give read/write access
RUN addgroup --gid 1000 nonroot && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" nonroot && \
    chown -R nonroot:nonroot /workspace && \
    chmod -R 0775 /workspace 

# Set the working directory
WORKDIR /workspace

# install third-party packages
COPY --chown=nonroot:nonroot environment.yml .
RUN conda env update -n base --file environment.yml

# copy source code
COPY --chown=nonroot:nonroot . .

# install local packages packages
RUN /opt/conda/bin/pip install -e .

# set the default user when running the image if last stage.
USER nonroot

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
ENTRYPOINT ["python"]
CMD ["src/train.py"]
