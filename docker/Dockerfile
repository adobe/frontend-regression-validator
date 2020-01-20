FROM python:3.7-slim-buster

WORKDIR /fred

# Installing build dependencies
RUN apt-get update && apt-get install -y curl libnss3-dev libcups2-dev libasound2-dev libatk1.0-dev libatk-bridge2.0-dev libgtk-3-dev libpangocairo-1.0-0 python3-pip xorg && apt-get clean

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pyppeteer-install

COPY fred/ .

# Extract the model files
RUN cd inference && cat model_files.bz2.parta* > model_files.bz2 && tar xjf model_files.bz2 && rm -f model_files.bz2*

CMD python3 run.py
EXPOSE 5000
