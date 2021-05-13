# FROM node:alpine
# RUN apk add --no-cache git
# RUN apk add --no-cache openssh
# WORKDIR /data
# RUN git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing.git /data/app
# WORKDIR /data/app
# EXPOSE 3000 

FROM ubuntu:18.04

ENV PYTHONUNBUFFERED TRUE

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    python3-dev \
    python3-distutils \
    python3-venv \
    openjdk-11-jre-headless \
    curl \
    # wget \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

RUN python3 -m venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

RUN mkdir /home \
        && cd /home \        
        && git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing.git

#Set working directory
WORKDIR /home
RUN cd /home/Self-Correction-Human-Parsing
RUN pip install gdown

# RUN git 
# RUN git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing.git
# WORKDIR /data
# RUN git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing.git /data/app
# WORKDIR /data/app
# EXPOSE 3000 
# RUN mkdir checkpoints

# RUN cd ./Self-Correction-Human-Parsing
# RUN pip install -r requirements.txt

# RUN pip install --no-cache-dir torch==1.7.0+cpu torchvision==0.8.1+cpu torchserve==0.2.0 -f https://download.pytorch.org/whl/torch_stable.html

# COPY /checkpoints/ /checkpoints/

# RUN useradd -m model-server

# COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
# COPY config.properties /home/model-server/config.properties
# COPY model-store /home/model-server/model-store

# RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
#     && mkdir -p /home/model-server/tmp \
#     && chown -R model-server /home/model-server \
#     && chown -R model-server /home/venv

# RUN chown -R model-server /home/model-server/model-store \
#     && cd /home/model-server/model-store

# EXPOSE 8080 8081 8082

# USER model-server
# WORKDIR /home/model-server
# ENV TEMP=/home/model-server/tmp
# ENTRYPOINT ["/usr/local/bin/entryFile.sh"]


# FROM ubuntu:18.04
# LABEL key = "Doku"
# #Install git
# RUN apt-get update \        
#     && apt-get install -y git
# RUN mkdir /home/sampleTest \
#         && cd /home/sampleTest \        
#         && git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing.git



# #Set working directory
# WORKDIR /home/
