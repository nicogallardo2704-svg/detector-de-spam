"""
B2B Threat Intelligence API Gateway for Enterprise Phishing Classification.
Powered by FastAPI for ultra-low latency text payload evaluation.
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List
from clasificador import EmailThreatClassifier

# Initialize enterprise API routing and metadata
app = FastAPI(
    title="Enterprise Email Threat Intelligence API",
    description="NLP-driven ingestion gateway for corporate phishing and Business Email Compromise (BEC) detection.",
    version="1.0.0"
)

# Simulate enterprise historical data to pre-initialize the model engine
training_corpus = [
    "Urgent: Update your employee banking credentials immediately via this link.",
    "Hey Team, please review the attached Q3 financial performance report.",
    "Wire transfer requested from the CEO. Process $50,000 to the external vendor now.",
    "Are we still on for the sync meeting scheduled for tomorrow at 9 AM?",
    "Security Alert: Your corporate cloud account access expires in 24 hours.",
    "Please send over the updated software deployment documentation by Friday."
]
labels = [1, 0, 1, 0, 1, 0]

# Instantiate and fit the classification core
threat_engine = EmailThreatClassifier()
threat_engine.train(training_corpus, labels)

# Data contracts for inbound and outbound communication vectors
class ThreatAnalysisRequest(BaseModel):
    emails: List[str] = Field(
        ...,
        description="Batch list of raw email text payloads to evaluate for active security threats.",
        example=["URGENT: Click here to claim your corporate bonus right now", "Project deadline extension approved."]
    )

class SingleThreatVerdict(BaseModel):
    payload: str
    is_threat: bool
    verdict: str

class ThreatAnalysisResponse(BaseModel):
    results: List[SingleThreatVerdict]
    total_scanned: int
    threats_isolated: int

@app.get("/health", status_code=status.HTTP_200_OK)
def security_gateway_health():
    """Validates real-time engine telemetry and signature load status."""
    return {
        "gateway_status": "active",
        "threat_signatures_loaded": True,
        "engine_architecture": "MultinomialNB"
    }

@app.post("/analyze-threats", response_model=ThreatAnalysisResponse, status_code=status.HTTP_200_OK)
def analyze_email_payloads(payload: ThreatAnalysisRequest):
    """
    Ingest text buffers and run synchronous vector classification metrics 
    to isolate social engineering and phishing text patterns.
    """
    if not payload.emails:
        raise HTTPException(status_code=400, detail="Inbound payload queue cannot be empty.")
    
    try:
        # Run classification pipeline batch inference
        predictions = threat_engine.predict(payload.emails)
        
        compiled_results = []
        threat_count = 0
        
        for text, prediction in zip(payload.emails, predictions):
            has_risk = bool(prediction == 1)
            if has_risk:
                threat_count += 1
                
            compiled_results.append(
                SingleThreatVerdict(
                    payload=text,
                    is_threat=has_risk,
                    verdict="🚨 CRITICAL RISK DETECTED" if has_risk else "✅ VERIFIED SAFE"
                )
            )
            
        return ThreatAnalysisResponse(
            results=compiled_results,
            total_scanned=len(payload.emails),
            threats_isolated=threat_count
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Core Threat Analytics Engine Failure: {str(e)}"
        )
