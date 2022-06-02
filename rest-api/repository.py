from sqlalchemy.orm import Session

import models


class ClassificationRepo:
    def create(db: Session, metric_report, data_batch, model):
        db_item = models.Classification(
            metric_report=metric_report, data_batch=data_batch, model=model
        )
        db.add(db_item)
        return db_item

    def fetch_lastest(db: Session):
        return (
            db.query(models.Classification)
            .order_by(models.Classification.model_id.desc())
            .first()
        )


class RegressionRepo:
    def create(db: Session, metric_report, data_batch, model):
        db_item = models.Regression(
            metric_report=metric_report, data_batch=data_batch, model=model
        )
        db.add(db_item)
        return db_item

    def fetch_lastest(db: Session):
        return (
            db.query(models.Regression)
            .order_by(models.Regression.model_id.desc())
            .first()
        )
