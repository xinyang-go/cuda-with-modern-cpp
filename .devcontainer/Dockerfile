FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# install common develop tools
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y sudo wget curl git ninja-build \
    && rm -rf /var/lib/apt/lists/*
# setup timezone
ENV TZ=Asia/Shanghai
RUN apt-get update && apt-get install -y tzdata && rm -rf /var/lib/apt/lists/*
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# install latest cmake
RUN wget -qO- https://apt.kitware.com/kitware-archive.sh | bash \
    && apt-get install -y cmake \
    && rm -rf /var/lib/apt/lists/*
# create user
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && usermod --shell /bin/bash $USERNAME
USER $USERNAME
WORKDIR /home/$USERNAME