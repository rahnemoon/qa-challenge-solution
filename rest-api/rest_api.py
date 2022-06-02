from typing import List, Optional

import time
import asyncio
import ingest_api_data
import pipeline_ml
import json
import copy
import os

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from fastapi import FastAPI
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from fastapi_utils.session import FastAPISessionMaker
from fastapi_utils.tasks import repeat_every

import models as models
from db import get_db, engine
from repository import ClassificationRepo, RegressionRepo


regre_model = pipeline_ml.RegressionModel()
class_model = pipeline_ml.ClassificationModel(
    is_mini_batch_mode=os.getenv("MINI_BATCH_MODE", "False") == "True"
)
ingest_api = ingest_api_data.IngestAPI()
app = FastAPI(
    title="REST back-end for dashboard",
    description="Ingest API data and train models and provide endpoints to see the training process",
    version="1.0.0",
)

models.Base.metadata.create_all(bind=engine)


def train_classification_model(data_batch: list) -> None:
    result = class_model.train(mini_batch=data_batch)
    local_session = sessionmaker(autocommit=True, autoflush=True, bind=engine)
    with local_session.begin() as db:
        ClassificationRepo.create(
            db,
            metric_report=class_model.serialize_report(),
            data_batch=result,
            model=copy.deepcopy(class_model),
        )


def train_regression_model(data_batch: list) -> None:
    result = regre_model.train(mini_batch=data_batch)
    local_session = sessionmaker(autocommit=True, autoflush=True, bind=engine)
    with local_session.begin() as db:
        RegressionRepo.create(
            db,
            metric_report=regre_model.serialize_report(),
            data_batch=result,
            model=copy.deepcopy(regre_model),
        )


@app.on_event("startup")
@repeat_every(seconds=2)
def mock_msg_queue() -> None:
    data_batch = ingest_api.get_data()
    train_classification_model(data_batch)
    train_regression_model(data_batch)


@app.exception_handler(Exception)
def validation_exception_handler(request, err):
    base_error_message = f"Failed to execute: {request.method}: {request.url}"
    return JSONResponse(
        status_code=400, content={"message": f"{base_error_message}. Detail: {err}"}
    )


@app.get("/api/v1/classification", tags=["Classification"])
def get_latest_items(db: Session = Depends(get_db)):
    db_item = ClassificationRepo.fetch_lastest(db)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item.jsonfy()


@app.get("/api/v1/regression", tags=["Regression"])
def get_latest_items(db: Session = Depends(get_db)):
    db_item = RegressionRepo.fetch_lastest(db)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item.jsonfy()


if __name__ == "__main__":
    uvicorn.run("rest_api:app", port=9000, reload=True, debug=True)
