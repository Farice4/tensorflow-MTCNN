FROM tensorflow/tensorflow:1.2.1

RUN apt-get -y update && apt-get install -y libsm6 libxext6 libxrender-dev git vim wget

RUN pip install numpy==1.13.3 scipy==0.19.1 opencv-python==4.1.1.26 Keras==2.3.0 tensorflow==1.2.1 tensorboard==1.8.0 easydict==1.9 Pillow==6.1.0 tqdm==4.36.1 Flask==1.1.1

ADD . /root/tensorflow-MTCNN

WORKDIR /root/tensorflow-MTCNN
