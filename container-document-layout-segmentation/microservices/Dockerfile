ARG REGISTRY=kitmicroservices.azurecr.io

# Stage #1 - Install Python depdencies
FROM ${REGISTRY}/python:3.9-buildpack as base
ARG PIP_EXTRA_INDEX_URL

COPY microservices/requirements.txt ./

RUN --mount=type=secret,id=build_secrets.sh  . import_secret_env && \
    pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    # All packages below are automatically installed as a sub-dependency from other packages in req.txt. \
    # So they are removed in this way.
    # Can work without Onnx - Also has vulnerability in the current dependency in req.txt that can't be solved atm.
    pip3 uninstall -y onnx && \
    # Can work without Zope
    pip3 uninstall -y zope-event && \
    pip3 uninstall -y zope-interface && \
    # PyMuPDF uses AGPL 3.0 license that is not allowed.
    pip3 uninstall -y PyMuPDF


# Stage #2 - Create runtime image
FROM ${REGISTRY}/python:3.9-base AS output
ARG VERSION
ENV VERSION=$VERSION

COPY --from=base /opt/. /opt
WORKDIR /opt/app

RUN apt-get update \
 && apt-get install -y sudo python3-opencv \
 && apt-get install -y poppler-utils

COPY microservices/scripts/entry.sh /entry.sh
COPY microservice.yaml /opt/app/
COPY document_layout_segmentation_service/ /opt/app/document_layout_segmentation_service
# Copy ML-models to the expected directories
COPY models/paddleocr /home/app/.paddleocr/
COPY models/paddleclas /home/app/.paddleclas/

RUN useradd -ms /bin/bash -u 1000 app
RUN chmod +x /entry.sh

USER app

ENTRYPOINT [ "/entry.sh" ]

FROM output AS dev_image

USER root
RUN pip3 install pydevd-pycharm
RUN pip3 install debugpy
RUN pip3 install watchdog
USER app
RUN mkdir -p /tmp/jobfs \
             /tmp/asyncjobqueue