from starlette.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Laya" in response.text
    assert "text/html" in response.headers.get("content-type", "")


def test_api_examples_list():
    response = client.get("/api/examples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    ids = [item["id"] for item in data]
    assert "librivox_01_pitch_tier5" in ids


def test_api_example_detail():
    response = client.get("/api/example/librivox_01_pitch_tier5")
    assert response.status_code == 200
    data = response.json()
    assert "flaw_regions" in data
    assert "ground_truth" in data
    assert data["ground_truth"] is not None
    assert data["ground_truth"]["flaw_type"] == "pitch"
    assert data.get("transcript_source") == "provided"
    assert "transcript" in data


def test_api_ladders():
    response = client.get("/api/ladders")
    assert response.status_code == 200
    data = response.json()
    assert "pitch" in data
    assert "rate" in data
    assert len(data["pitch"]) > 0
    assert len(data["rate"]) > 0


def test_api_audio_file():
    response = client.get("/api/audio?path=dataset/sources/librivox_01.wav")
    assert response.status_code == 200
    assert response.headers.get("content-type") == "audio/wav"


def test_api_audio_missing():
    response = client.get("/api/audio?path=dataset/sources/nonexistent_file.wav")
    assert response.status_code == 404
