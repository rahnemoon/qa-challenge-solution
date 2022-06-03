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

- After data is ingested from the REST API of the challenge, it will be fed to the classification and regression models. Then prediction, data, and metrics will be stored in the SQLite database, which models are also stored as pickle type. The mentioned process will be repeated every two seconds as an async task to API. Furthermore, the dashboard app will use the REST endpoints to get the latest record from the database with the help of a time trigger. Then, Plotly will plot the result for each model and show the prediction for the data.Three main decisions in design are described more in detail in the following:
	- For the time being, the ingest part of the challenge is simple, which is a Get request to an only endpoint. A time-trigger-based async task from *FastAPI* will do the job of feeding the data to the ML pipeline and storing the result in a database. However, in a more complex scenario and a production environment, this task is better to be done by **workers** and **message brokers** like *Celery* and *Redis* or *RabbitMQ*.

	- Both endpoints of REST API for classification and regression data will return the latest record in the database, and the API will run in the background and refine the models. At any time opening the dashboard, we can get the latest result, not all the previous results, as we can't do this in a stream of data. Even if the data is stored, the retrieval will hurt the performance. Plus, multiple users can use the dashboard simultaneously as there is no stage held in the app. It has a time trigger to get the latest result, which can be configured to be set by the user if the user wants fewer updates.

	- The River is used as an ML library for training which more or less is a wrapper for both *creme* and *scikit-multiflow* packages, which in incremental learning are popular. This library will make partial training of models way easier. For the dashboard, the Dash from Plotly allows having dynamic and interactive plots in which code is written in Python, and then an application is made by Flask and React.


- The second part of the challenge is solved and explained in Jupiter Notebook named `variable_analysis.ipynb`. The regression model is included in the challenge code even though it may not be the best model to use.


## Local Deployment
For local Deployment, the `requirements.txt` is included if, instead of `pipenv`, the `pip` is used. In the case of `pipenv`, just use the `pipenv shell`. Please be careful about the environmental variables and how to change or set them to system works. You can find Envs in the `.env` files.

