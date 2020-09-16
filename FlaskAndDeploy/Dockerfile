FROM python:3.8

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN apt-get update
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --use-feature=2020-resolver

COPY . /

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["app.py"]
