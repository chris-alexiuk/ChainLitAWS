FROM python:3.9
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["chainlit", "run", "app.py", "--port", "7860"]