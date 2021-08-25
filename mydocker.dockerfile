FROM python:3.7.9
RUN python3 -m pip install --upgrade pip
RUN apt-get update
RUN python3 -m pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
WORKDIR /app
COPY . /app
EXPOSE 8066
CMD ["paddlex_restful","--start_restful","--port","8066","--workspace_dir","/app" ]




