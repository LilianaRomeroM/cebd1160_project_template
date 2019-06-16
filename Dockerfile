FROM ubuntu:latest

COPY requirements.txt ./

RUN apt update && \
    apt install -y python3 \
                   python3-pip && \
    pip3 install -r requirements.txt

COPY DIABETES_EXPERIMENT.py ./

CMD ["python3", "-u",  "DIABETES_EXPERIMENT.py"]
