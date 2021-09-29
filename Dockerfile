# syntax=docker/dockerfile:1

FROM python:3.8

WORKDIR /usr/src/nalcos

COPY . .

RUN make install

CMD ["bash"]
