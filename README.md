# QA Online (Incremental) Machine Learning Challenge


## Run the Solution
To run the solution, use the below command
```shell
docker-compose build # To create the images
docker-compose up    # To run the containers
```
Plus, you should clone the [main project](https://gitlab.com/qa-public/qa-streaming-pipeline-challenge) and run it by docker-compose as the solution docker-compose expected the network of this app to be available to attach to it.

Then follow the two mentioned URLs below to use the system endpoints and dashboard

For endpoints
> http://localhost:8000/docs

For dashboard
>http://localhost:8085/


## Description of Implementaion:
The REST API of the system is developed by using **FastAPI** and **SQLAlchemy**, and the dashboard is created by **Dash(Plotly)**. Also, **River** is used for Online ML learning of the challenge.

- After data is ingested from the REST API of the challenge, it will be fed to the classification and regression models. Then prediction, data, and metrics will be stored in the SQLite database, which models are also stored as pickle type. The mentioned process will be repeated every two seconds as an async task to API. Furthermore, the dashboard app will use the REST endpoints to get the latest record from the database with the help of a time trigger. Then, Plotly will plot the result for each model and show the prediction for the data.
- The second part of the challenge is solved and explained in Jupiter Notebook named `variable_analysis.ipynb`. The regression model is included in the challenge code even though it may not be the best model to use.


## Local Deployment
For local Deployment, the `requirements.txt` is included if, instead of `pipenv`, the `pip` is used. In the case of `pipenv`, just use the `pipenv shell`. Please be careful about the environmental variables and how to change or set them to system works. You can find Envs in the `.env` files.

