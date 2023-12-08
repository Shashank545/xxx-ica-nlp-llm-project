FROM python:3.9-bullseye

ARG ENVIRONMENT=local
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget
RUN apt install -y libgl1-mesa-glx
RUN wget "https://guillaumejaume.github.io/FUNSD/dataset.zip"
RUN unzip dataset.zip && mv dataset data && rm -rf dataset.zip
RUN git clone https://github.com/airsplay/py-bottom-up-attention.git

COPY requirements.txt .

RUN pip3 install -r requirements.txt

WORKDIR /py-bottom-up-attention

RUN pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# RUN pip3 install torch torchvision torchaudio

# RUN python setup.py build develop
RUN pip3 install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html

RUN echo 'YAY we have completed the setup'

WORKDIR /

RUN mkdir /features

RUN mkdir /models

RUN mkdir /features/visual_features

RUN mkdir /features/visual_features/detectron_train

RUN mkdir /features/visual_features/detectron_test

RUN wget "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl" -P /models

COPY image_preprocessing.py .

RUN python image_preprocessing.py

RUN echo 'Image Feature Vectors Generated and Stored successfully'

RUN pip3 install pandas scikit-learn transformers

RUN mkdir /features/textual_features

RUN mkdir /utils

COPY /utils/bert_downloader.py /utils

RUN pip3 install requests==2.27.1

RUN python /utils/bert_downloader.py

RUN echo 'BERT pre-trained models are downloaded successfully'

COPY text_preprocessing.py .

RUN python text_preprocessing.py

RUN echo 'Text Feature Vectors Generated and Stored successfully'

RUN pip3 install nltk 

RUN pip3 install -U spacy

RUN pip3 install python-certifi-win32

RUN python -m spacy download en_core_web_sm

COPY /utils/scenegraph_generator.py /utils

RUN python /utils/scenegraph_generator.py

RUN echo 'ALl Scene Graph generated successfully'