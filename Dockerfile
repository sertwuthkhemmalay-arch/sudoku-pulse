FROM python:3.14.6-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Dependencies required to build Tesseract 5.5.3
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    libtool \
    pkg-config \
    g++ \
    make \
    cmake \
    libleptonica-dev \
    libicu-dev \
    libpango1.0-dev \
    libcairo2-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    zlib1g-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build and install the same Tesseract version used locally
WORKDIR /tmp
RUN wget -O tesseract-5.5.3.tar.gz \
    https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.5.3.tar.gz \
    && tar -xzf tesseract-5.5.3.tar.gz \
    && cd tesseract-5.5.3 \
    && ./autogen.sh \
    && ./configure \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && tesseract --version

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads output debug_digits

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 180 app:app"]
