# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10.12


EXPOSE 8080

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install -r requirements.txt

RUN pip install opentelemetry-sdk==1.24.0
RUN pip install opentelemetry-semantic-conventions==0.48b0
RUN pip install opentelemetry-api==1.24.0




# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD streamlit run --server.port 8080 --server.enableCORS false app.py