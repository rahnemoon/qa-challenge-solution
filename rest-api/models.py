from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    Table,
    JSON,
    PickleType,
    DateTime,
)

from datetime import datetime
from db import Base


class Classification(Base):
    __tablename__ = "Classification_model"
    model_id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    metric_report = Column(JSON, nullable=False)
    data_batch = Column(JSON, nullable=False)
    model = Column(PickleType, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now())

    def jsonfy(self):
        return dict(
            model_id=self.model_id,
            metric_report=self.metric_report,
            data_batch=self.data_batch,
            created_at=self.created_at,
        )


class Regression(Base):
    __tablename__ = "regression_model"
    model_id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    metric_report = Column(JSON, nullable=False)
    data_batch = Column(JSON, nullable=False)
    model = Column(PickleType, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now())

    def jsonfy(self):
        return dict(
            model_id=self.model_id,
            metric_report=self.metric_report,
            data_batch=self.data_batch,
            created_at=self.created_at,
        )
