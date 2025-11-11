import sys
import os
import pytest
import pymongo
from dotenv import load_dotenv
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

VALID_TOKEN = os.getenv("VALID_TEST_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB")
LOG_COLLECTION_NAME = "PROD_intent_logs" 
@pytest.fixture(scope="module")
def db_connection():
    if not MONGO_URI or not MONGO_DB_NAME:
        pytest.skip("MONGO_URI/MONGO_DB não definidos. Pulando testes de integração.")
    
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
    except Exception as e:
        pytest.skip(f"Não foi possível conectar ao MongoDB Atlas: {e}")

    yield db 
    print(f"\n[Teardown] Limpando coleção {LOG_COLLECTION_NAME}...")
    db[LOG_COLLECTION_NAME].delete_many({})
    client.close()


def test_predict_integration_saves_to_db(db_connection, mocker):
      if not VALID_TOKEN:
        pytest.skip("Secret VALID_TEST_TOKEN is not set. Skipping integration test.")

        from app.app import app
        client = TestClient(app)
    
        logs_collection = db_connection[LOG_COLLECTION_NAME]

        logs_collection.delete_many({})
        assert logs_collection.count_documents({}) == 0
    
        input_text = "qual o status do meu pedido de integração?"

        response = client.post(
        "/predict",
        json={"text": input_text},
        headers={"Authorization": f"Bearer {VALID_TOKEN}"}
    )

        assert response.status_code == 200
    
        data = response.json()
        assert data["text"] == input_text
        assert "id" in data 

        assert logs_collection.count_documents({}) == 1
        saved_doc = logs_collection.find_one()
        assert saved_doc is not None
        assert saved_doc["text"] == input_text
        assert "predictions" in saved_doc