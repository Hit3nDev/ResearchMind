from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text

from app.db.database import Base


class QueryRow(Base):
    __tablename__ = "queries"
    query_id = Column(String, primary_key=True)
    question = Column(Text, nullable=False)
    status = Column(String, default="processing")
    stage = Column(String, default="searching")
    papers_found = Column(Integer, default=0)
    papers_processed = Column(Integer, default=0)
    paper_ids = Column(JSON, default=list)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PaperRow(Base):
    __tablename__ = "papers"
    paper_id = Column(String, primary_key=True)
    query_id = Column(String, index=True, nullable=True)
    data = Column(JSON, nullable=False)          # serialized schemas.Paper (minus full_text)
    full_text = Column(Text, nullable=True)


class ReportRow(Base):
    __tablename__ = "reports"
    query_id = Column(String, primary_key=True)
    data = Column(JSON, nullable=False)          # serialized schemas.Report


class ProjectRow(Base):
    __tablename__ = "projects"
    project_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    query_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
