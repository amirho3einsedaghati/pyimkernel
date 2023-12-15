FROM python:3.9-slim
WORKDIR /app
COPY requires.txt .
# The dependencies of the opencv library
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0
RUN pip install --upgrade pip
RUN pip install -r requires.txt
COPY . .
EXPOSE 5400

LABEL version='1.0'
LABEL description='Applying some image kernel(s) on a grayscale or RGB|BGR color-scale image'
LABEL maintainer="Amir Hosein Sedaghati <<amirhosseinsedaghati42@gmail.com>>"

CMD ["python", "pyimkernel/main.py"]
