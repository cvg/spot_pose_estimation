FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential git cmake wget python-opencv

RUN alias pip="python3 -m pip"
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install loguru wis3d h5py

# install colmap
RUN apt-get update && apt-get install -y -qq \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-regex-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        cuda-cudart-dev-12-1 \
        cuda-curand-dev-10-2 \
        libflann-dev \
        libfreeimage3 \
        libfreeimage-dev \
        liblz4-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libceres-dev \
        libatlas3-base \
        libboost-test1.65.1 \
        libcgal13 \
        libcgal-qt5-13 \
        libgflags2.2 \
        libglew2.0 \
        libgoogle-glog0v5 \
        libqt5opengl5 \
        libamd2 \
        libbtf1 \
        libcamd2 \
        libccolamd2 \
        libcholmod3 \
        libcolamd2 \
        libcxsparse3 \
        libgraphblas1 \
        libklu1 \
        libldl2 \
        librbio2 \
        libspqr2 \
        libsuitesparseconfig5 \
        libumfpack5
COPY ./colmap /colmap
WORKDIR /colmap
RUN mkdir build && cd build && cmake .. -GNinja && ninja && ninja install
# RUN apt install -y colmap


COPY ./OnePose_Plus_Plus_Spot /OnePose_Plus_Plus_Spot
WORKDIR /OnePose_Plus_Plus_Spot
RUN python3 -m pip install -r /OnePose_Plus_Plus_Spot/requirements.txt
WORKDIR /OnePose_Plus_Plus_Spot/submodules/DeepLM
RUN mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON && make -j8
RUN CUDACXX=/usr/local/cuda-11.3/bin/nvcc sh example.sh
RUN cp /OnePose_Plus_Plus_Spot/backup/deeplm_init_backup.py /OnePose_Plus_Plus_Spot/submodules/DeepLM/__init__.py
RUN mkdir /OnePose_Plus_Plus_Spot/weight
WORKDIR /OnePose_Plus_Plus_Spot/weight
RUN wget https://zenodo.org/record/8086894/files/LoFTR_wsize9.ckpt?download=1 -O LoFTR_wsize9.ckpt
RUN wget https://zenodo.org/record/8086894/files/OnePosePlus_model.ckpt?download=1 -O OnePosePlus_model.ckpt
# RUN mkdir -p /OnePose_Plus_Plus_Spot/data/demo/sfm_model/outputs_softmax_loftr_loftr/
# WORKDIR /OnePose_Plus_Plus_Spot/data/demo/sfm_model/outputs_softmax_loftr_loftr/
# RUN wget https://zenodo.org/record/8086894/files/SpotRobot_sfm_model.tar?download=1 -O SpotRobot_sfm_model.tar
# RUN tar -xvf SpotRobot_sfm_model.tar
WORKDIR /OnePose_Plus_Plus_Spot
