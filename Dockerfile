# docker build . -t figchat
FROM python:3.10-slim

RUN apt update
RUN apt install -y gcc pkg-config cmake libcairo2-dev libjpeg-dev libgif-dev

WORKDIR /usr/src/app
COPY . .
RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir gradio
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# CMD ["python", "figchat_gradio.py"]

# CMD ["python", "-m", "chainlit", "run", "app.py", "-h", "--port", "7860", "--host", "0.0.0.0", "--root-path", "/figchat"]
ENTRYPOINT ["python", "-m", "chainlit", "run", "app.py", "-h", "--port", "7860", "--host", "0.0.0.0", "--root-path", "/figchat"]
