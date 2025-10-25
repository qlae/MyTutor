import os, sys
# Ensure project root (one level up from tests/) is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

def test_subjects():
    client = app.test_client()
    r = client.get("/subjects")
    assert r.status_code == 200
    data = r.get_json()
    assert "subjects" in data
    assert isinstance(data["subjects"], list)
    assert set(["Math","Science","English"]).issuperset(data["subjects"])

def test_flow():
    client = app.test_client()
    r = client.post("/start", json={"subject":"Math"})
    assert r.status_code == 200
    diag = r.get_json()["diagnostic"]
    assert isinstance(diag, list) and len(diag) >= 1

    r2 = client.post("/submit_diagnostic", json={})
    assert r2.status_code == 200
    data = r2.get_json()
    assert "difficulty" in data and "questions" in data
    assert isinstance(data["questions"], list) and len(data["questions"]) >= 1

    # Pick the first option for each question to submit
    payload = {"answers": {f"q{i}": q["options"][0] for i, q in enumerate(data["questions"])}}
    r3 = client.post("/submit_quiz", json=payload)
    assert r3.status_code == 200
    res = r3.get_json()
    assert set(["score","total","accuracy","reward","next"]).issubset(res.keys())
