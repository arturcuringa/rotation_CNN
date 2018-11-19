FROM nvidia/cuda:9.0-base-ubuntu16.04
# adapted from https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker
LABEL maintainer="many"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
       build-essential \
       cuda-command-line-tools-9-0 \
       cuda-cublas-9-0 \
       cuda-cufft-9-0 \
       cuda-curand-9-0 \
       cuda-cusolver-9-0 \
       cuda-cusparse-9-0 \
       curl \
       libcudnn7=7.0.5.15-1+cuda9.0 \
       libfreetype6-dev \
       libhdf5-serial-dev \
       libpng12-dev \
       libzmq3-dev \
       pkg-config \
       python3 \
       python3-dev \
       rsync \
       software-properties-common \
       libsm6 \
       libxext6 \
       libxrender-dev \
       libqt5x11extras5 \
       unzip \
       && \
   apt-get clean && \
   rm -rf /var/lib/apt/lists/*

# apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys E1DD270288B4E6030699E45FA1715D88E1DF1F24
# echo 'deb http://ppa.launchpad.net/git-core/ppa/ubuntu trusty main' > /etc/apt/sources.list.d/git.list
# apt-get update
# apt-get install git

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
   python3 get-pip.py && \
   rm get-pip.py

RUN pip --no-cache-dir install \
       Pillow \
       h5py \
       ipykernel \
       jupyter \
       matplotlib \
       numpy \
       pandas \
       scipy \
       sklearn \
       opencv-python \
       && \
   python3 -m ipykernel.kernelspec


RUN pip --no-cache-dir install \
   https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp35-cp35m-linux_x86_64.whl

RUN pip --no-cache-dir install \
   keras==2.2.2

RUN ln -s -f /usr/bin/python3 /usr/bin/python

ADD . /logical
WORKDIR "/logical"

#install deps
#RUN pip install -e /logical/venv/nnutil


CMD ["/bin/bash"]
