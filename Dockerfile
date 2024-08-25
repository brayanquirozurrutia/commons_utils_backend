FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/tesseract-ocr/tessdata/raw/main/spa.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata/

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

EXPOSE 5000

CMD ["python", "app.py"]
