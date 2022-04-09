FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./
RUN python3.8 -m pip install -r requirements.txt -t .

COPY models models
COPY app.py ./
COPY preprocessing.py ./

CMD ["app.lambda_handler"]
