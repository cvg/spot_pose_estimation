FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

COPY ./OnePose_Plus_Plus_Spot /OnePose_Plus_Plus_Spot
WORKDIR /OnePose_Plus_Plus_Spot
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
