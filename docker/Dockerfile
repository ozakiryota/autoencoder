########## Pull ##########
FROM nvidia/cuda:11.6.2-base-ubuntu20.04
########## User ##########
ARG home_dir="/home/user"
COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh
RUN apt-get update && \
	apt-get install -y \
		gosu \
		sudo && \
	chmod +x /usr/local/bin/entrypoint.sh && \
	mkdir -p $home_dir
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
########## Non-interactive ##########
ENV DEBIAN_FRONTEND=noninteractive
########## Common Tools ##########
RUN apt-get update && \
    apt-get install -y \
	    vim \
    	wget \
    	unzip \
    	git \
		python3-tk
########## PyTorch ##########
RUN apt-get update && \
    apt-get install -y \
	    python3-pip && \
	pip3 install \
		torch==1.12.1+cu116 \
		torchvision==0.13.1+cu116 \
		torchaudio==0.12.1 \
		--extra-index-url https://download.pytorch.org/whl/cu116 \
		tensorboardX \
		tensorboard
########## Cache Busting ##########
ARG cache_bust=1
########## autoencoder ##########
# RUN cd $home_dir && \
# 	git clone https://github.com/ozakiryota/autoencoder.git && \
# 	pip3 install \
# 		matplotlib \
# 		tqdm
RUN pip3 install \
		matplotlib \
		tqdm
########## Initial Position ##########
WORKDIR $home_dir/autoencoder
CMD ["bash"]