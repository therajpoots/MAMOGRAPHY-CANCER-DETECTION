import os
import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, Date, DateTime, ForeignKey, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from werkzeug.security import generate_password_hash

DATABASE_URL = "sqlite:///e:/CONFERENCE/mammography_platform.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    role = Column(String, default="Radiologist") # Radiologist, Clinician, Admin
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    mrn = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    birth_date = Column(Date, nullable=False)
    gender = Column(String, default="F")
    family_history = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    studies = relationship("Study", back_populates="patient", cascade="all, delete-orphan")

class Study(Base):
    __tablename__ = "studies"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    accession_number = Column(String, unique=True, index=True, nullable=False)
    study_date = Column(DateTime, nullable=False)
    density_category = Column(String, nullable=True) # A, B, C, D
    current_status = Column(String, default="Uploaded") # Uploaded, Processing, Needs Review, Approved
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    patient = relationship("Patient", back_populates="studies")
    images = relationship("Image", back_populates="study", cascade="all, delete-orphan")
    analysis_results = relationship("AnalysisResult", back_populates="study", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="study", cascade="all, delete-orphan")

class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(Integer, ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    modality = Column(String, default="MG")
    image_view = Column(String, nullable=False) # L-CC, R-CC, L-MLO, R-MLO
    width = Column(Integer, default=512)
    height = Column(Integer, default=512)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    study = relationship("Study", back_populates="images")

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(Integer, ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    segmentation_metrics = Column(JSON, nullable=False)
    classification_metrics = Column(JSON, nullable=False)
    birads_rating = Column(String, nullable=False)
    mask_path = Column(String, nullable=True)
    xai_overlay_path = Column(String, nullable=True)
    completed_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    study = relationship("Study", back_populates="analysis_results")

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(Integer, ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)
    author_name = Column(String, nullable=False)
    report_text = Column(Text, nullable=False)
    status = Column(String, default="Draft") # Draft, Signed
    approved_at = Column(DateTime, nullable=True)
    signature_checksum = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    study = relationship("Study", back_populates="reports")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, nullable=True)
    event_type = Column(String, nullable=False)
    details = Column(Text, nullable=False)
    ip_address = Column(String, default="127.0.0.1")
    occurred_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # 1. Create a default radiologist user
        if not db.query(User).filter(User.email == "rad@hospital.org").first():
            user = User(
                email="rad@hospital.org",
                password_hash=generate_password_hash("password123"),
                first_name="Eleanor",
                last_name="Vance",
                role="Radiologist"
            )
            db.add(user)
            db.commit()
            print("[DB] Created default radiologist account: rad@hospital.org")

        # Mock clinical data insertion was removed.
        pass
            
    finally:
        db.close()
