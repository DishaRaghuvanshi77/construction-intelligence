from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2, numpy as np

app = FastAPI(title="Multimodal Construction Intelligence Platform")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Valuation(BaseModel):
    area: float
    rate: float = 5500
    age: int = 0
    quality: float = 1.0


class Plan(BaseModel):
    area: float
    floors: int = 1
    cost_per_sqft: float = 2000


class Safety(BaseModel):
    description: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/crack")
async def crack(file: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 80, 160)
    pct = float(edges.mean() / 255 * 100)
    severity = "Minor" if pct < 2 else "Moderate" if pct < 6 else "Severe"
    return {"edge_density_pct": round(pct, 2), "severity": severity}


@app.post("/valuation")
def valuation(v: Valuation):
    dep = min(0.4, v.age * 0.012)
    value = v.area * v.rate * (1 - dep) * v.quality
    return {"estimate": round(value), "low": round(value * .92), "high": round(value * 1.08)}


@app.post("/plan")
def plan(p: Plan):
    total = p.area * p.floors * p.cost_per_sqft
    months = 4 + p.floors * 2 + p.area * p.floors / 1500
    return {"total_cost": round(total), "months": round(months, 1)}


@app.post("/safety")
def safety(s: Safety):
    # Replace with an LLM / vision-model call (e.g. Claude API) for production
    rules = {"helmet": "Helmet missing", "harness": "Harness missing", "exposed wire": "Electrical hazard"}
    text = s.description.lower()
    return {"violations": [m for k, m in rules.items() if k in text]}
