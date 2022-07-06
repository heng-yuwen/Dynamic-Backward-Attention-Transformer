FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
USER root
WORKDIR /home/root

RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt update && apt-get update && \
    apt install -y git wget curl vim unzip zip libgl1-mesa-dev
RUN chmod 777 /home/root
SHELL ["/bin/bash", "-c"]

ENV PATH=/home/root/miniconda3/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
RUN bash ./Miniconda3-py38_4.11.0-Linux-x86_64.sh -b && rm ./Miniconda3-py38_4.11.0-Linux-x86_64.sh && \
    source $HOME/miniconda3/bin/activate && \
    conda init bash && \ 
    conda config --set auto_activate_base false && \
    . $HOME/.bashrc && \
    conda update conda && \
    conda create -n cv python=3.8 && \ 
    conda activate cv && \
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch && \
    pip install pytorch-lightning==1.2.3 &&\
    pip install segmentation-models-pytorch==0.2.0 &&\
    pip install mmcv==1.3.17 && \
    pip install timm==0.4.12 && \
    conda clean -ay

RUN source $HOME/miniconda3/bin/activate && \
    . $HOME/.bashrc && \
    conda activate cv && \
    pip install opencv-python opencv-python-headless randaugment ptflops && \
    conda install pandas && \
    conda clean -ay

RUN echo "conda activate cv" >> ~/.bashrc
 
