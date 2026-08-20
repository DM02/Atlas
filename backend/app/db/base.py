from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    # Every Mapped[datetime] column gets a timezone-aware Postgres timestamptz,
    # so app code can consistently use timezone-aware datetimes end to end.
    type_annotation_map = {datetime: DateTime(timezone=True)}
