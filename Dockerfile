FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

COPY ./OnePose_Plus_Plus_Spot /OnePose_Plus_Plus_Spot
WORKDIR /OnePose_Plus_Plus_Spot
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
WORKDIR /OnePose_Plus_Plus_Spot/submodules/DeepLM
RUN sh example.sh
RUN cp /OnePose_Plus_Plus_Spot/backup/deeplm_init_backup.py /OnePose_Plus_Plus_Spot/submodules/DeepLM/__init__.py
RUN mkdir /OnePose_Plus_Plus_Spot/weights
WORKDIR /OnePose_Plus_Plus_Spot/weights
RUN wget https://zenodo.org/record/8086894/files/LoFTR_wsize9.ckpt?download=1
RUN wget https://zenodo.org/record/8086894/files/OnePosePlus_model.ckpt?download=1
RUN mkdir -p /OnePose_Plus_Plus_Spot/data/demo/sfm_model/outputs_softmax_loftr_loftr/
WORKDIR /OnePose_Plus_Plus_Spot/data/demo/sfm_model/outputs_softmax_loftr_loftr/
RUN wget https://zenodo.org/record/8086894/files/SpotRobot_sfm_model.tar?download=1
RUN tar -xvf SpotRobot_sfm_model.tar
WORKDIR /OnePose_Plus_Plus_Spot
