FROM python:3.9-slim

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

RUN pip3 install --no-cache-dir --upgrade \
    pip \
    virtualenv

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git

USER appuser
WORKDIR /home/appuser

# This line clones a generic streamlit example. 
# You should replace this with copying your actual application code.
# For example, if your app is in a local 'my_streamlit_app' directory:
# COPY ./my_streamlit_app ./app
# And ensure your requirements.txt is in 'my_streamlit_app/requirements.txt'
RUN git clone https://github.com/aavalose/msds_chat_v2.git app
RUN mv app/data . # Move the data directory to /home/appuser/data

ENV VIRTUAL_ENV=/home/appuser/venv
RUN virtualenv ${VIRTUAL_ENV}

# This assumes your requirements.txt is inside the 'app' directory cloned above.
# If you copied your own app, ensure the path to requirements.txt is correct.
RUN . ${VIRTUAL_ENV}/bin/activate && pip install -r app/requirements.txt

EXPOSE 8501

COPY --chown=appuser:appuser run.sh /home/appuser/
RUN chmod +x /home/appuser/run.sh

ENTRYPOINT ["./run.sh"] 