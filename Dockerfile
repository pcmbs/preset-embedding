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

# Arguments for username, UID and GID (to be passed on build)
ARG UNAME=nonroot
ARG UID=1000
ARG GID=1000
ARG ADDITIONAL_GROUPS=


# Creates a non-root group and user with an explicit GID and UID and 
# change the required folders ownership to this user and give read/write access
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN groupadd -g $GID $UNAME && \
    useradd -u $UID -g $GID -l -m -s /bin/bash -d /home/$UNAME -p '' $UNAME && \
    # Add user to additional groups if provided
    if [ -n "$ADDITIONAL_GROUPS" ]; then \
    for group in $(echo $ADDITIONAL_GROUPS | tr ',' ' '); do \
    groupadd -g $group group$group && usermod -a -G group$group $UNAME; \
    done; \
    fi && \
    chown -R $UNAME:$UNAME /workspace && \
    chmod -R 0775 /workspace 

# Set the working directory
WORKDIR /workspace

# install third-party packages
COPY --chown=$UNAME:$UNAME environment.yml .
RUN conda env update -n base --file environment.yml

# set the default user when running the image if last stage.
USER $UNAME

# copy source code
COPY --chown=$UNAME:$UNAME . .

# install local packages
RUN /opt/conda/bin/pip install -e .

# During debugging, this entry point will be overridden.
CMD ["bash"]
