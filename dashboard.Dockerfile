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

ARG PORT 8085

# Copy virtual env from python-deps stage
COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

# Create and switch to a new user
RUN groupadd -r dashboardG && useradd --create-home --no-log-init -g dashboardG dashboard

# Install application into container
COPY /dashboard /home/dashboard/code

# change the permission of code files and change working directory
RUN chown -R dashboard:dashboardG /home/dashboard/code
USER dashboard:dashboardG
WORKDIR /home/dashboard/code

EXPOSE ${PORT}


CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT dashboard:server"]
