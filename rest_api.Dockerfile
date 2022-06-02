FROM python:3.9-slim as base

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1


FROM base AS python-deps

# Install pipenv and compilation dependencies
RUN pip install pipenv
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Install python dependencies in /.venv
COPY Pipfile .
COPY Pipfile.lock .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy


FROM base AS runtime

ARG PORT 8000

# Copy virtual env from python-deps stage
COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

# Create and switch to a new user
RUN groupadd -r pipelineG && useradd --create-home --no-log-init -g pipelineG pipeline

# Install application into container
COPY /rest-api /home/pipeline/code

# change the permission of code files and change working directory
RUN chown -R pipeline:pipelineG /home/pipeline/code
USER pipeline:pipelineG
WORKDIR /home/pipeline/code

EXPOSE ${PORT}


CMD ["sh", "-c", "uvicorn rest_api:app --host 0.0.0.0 --port $PORT"]
