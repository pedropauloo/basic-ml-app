import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

try:
    from app.app import app
except Exception as e:
    print(f"Aviso ao importar app.app (pode ser esperado): {e}")
    pass

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_dev_mode_success(mocker):
    """Testa o /predict em modo 'dev', pulando auth e mockando o DB."""

    def mock_insert_one_side_effect(document_to_insert):
        document_to_insert["_id"] = "fake_mongo_id_12345"
        return MagicMock(inserted_id="fake_mongo_id_12345")

    mock_collection = MagicMock()
    mock_collection.insert_one.side_effect = mock_insert_one_side_effect
    mocker.patch('app.app.collection', mock_collection)

    mock_model = MagicMock()
    mock_model.predict.return_value = ("intent_falsa", {"intent_falsa": 1.0})
    mocker.patch.dict('app.app.MODELS', {'modelo_falso': mock_model})

    response = client.post(
        "/predict",
        json={"text": "Oi, tudo bem?"}
    )

    # Verifica os resultados
    assert response.status_code == 200
    data = response.json()
    assert data["owner"] == "dev_user"
    assert data["text"] == "Oi, tudo bem?"
    assert data["id"] == "fake_mongo_id_12345"
    mock_collection.insert_one.assert_called_once()


def test_predict_prod_mode_auth_failure(mocker):
    """Testa /predict em modo 'prod' sem token (deve falhar 401)."""

    mocker.patch('app.app.ENV', 'prod')

    response = client.post(
        "/predict",
        json={"text": "Oi, tudo bem?"}
    )

    assert response.status_code == 401
    assert "Missing Authorization header" in response.json().get("detail", "")


def test_predict_prod_mode_auth_invalid_token(mocker):
    """Testa /predict em 'prod' com token inválido (deve falhar 403)."""

    mocker.patch('app.app.ENV', 'prod')

    mock_tokens_collection = MagicMock()
    mock_tokens_collection.find_one.return_value = None
    mocker.patch('app.auth.get_mongo_collection', return_value=mock_tokens_collection)

    response = client.post(
        "/predict",
        json={"text": "Oi, tudo bem?"},
        headers={"Authorization": "Bearer token_falso_que_nao_existe"}
    )

    assert response.status_code == 403
    assert "Invalid or inactive token" in response.json().get("detail", "")