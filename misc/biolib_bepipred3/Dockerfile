FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

#Keys to make nvidias repos work 
RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

#install python and pip package manager
RUN apt update
RUN apt install --no-install-recommends python3 python3-pip git -y
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

#install all requiremnets
COPY requirements.txt . 
RUN pip3 install --no-cache-dir -r requirements.txt

#create alias for python executable to be python3 executable
RUN ln -s $(which python3) /usr/bin/python

#copy CLI script and model esm1b model into image
COPY bepipred3_CLI.py .
RUN mkdir /models/
ADD models /models/
COPY example_antigens.fasta .
