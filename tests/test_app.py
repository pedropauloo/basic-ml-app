"""
Testes Unitários para a Aplicação FastAPI (app/app.py).

Destaques da Cobertura de Testes:

1.  Mocking de Dependências:
    - Usa `pytest.fixture` e `monkeypatch` para automaticamente "mockar" (simular)
      e substituir todas as dependências externas:
        - `app.app.collection` (Banco de dados MongoDB)
        - `app.app.MODELS` (Modelos de ML)
        - `app.app.verify_token` (Função de autenticação)
    - Isso isola a lógica dos endpoints da aplicação para testes rápidos
      e previsíveis.

2.  Lógica de Ambiente (ENV):
    - Testa o comportamento dos endpoints (`/` e `/predict`) quando a
      variável global `ENV` é definida como "dev" vs. "prod".

3.  Fluxo de Autenticação (conditional_auth):
    - `test_predict_dev_mode`: Verifica se a autenticação é contornada
      corretamente.
    - `test_predict_prod_mode_auth_success`: Verifica o "caminho feliz" onde
      a autenticação tem sucesso e uma predição é retornada.
    - `test_predict_prod_mode_auth_fail`: Verifica o "caminho triste" onde
      a autenticação falha, um 401 é retornado, e *crucialmente*, o modelo
      de ML e o banco de dados *não* são chamados.

4.  Integração com Modelo Real (Teste de Integração):
    - Inclui um teste (`@pytest.mark.integration`) que carrega o modelo
      *real* `confusion.keras` para garantir que ele é compatível
      com o código da aplicação. Este teste é pulado por padrão
      em uma execução normal do `pytest` (execute com `pytest -m "integration"`).
"""

import os
import sys
import pytest
from unittest.mock import MagicMock
from dotenv import load_dotenv

# --- Configuração do Path ---
# Este é um passo crucial. Adicionamos o diretório raiz do projeto (um nível acima de 'tests')
# ao path do sistema. Isso permite que o Python encontre e importe módulos dos
# diretórios 'app' e 'intent_classifier'.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from fastapi import HTTPException

# Agora que o path está configurado, podemos importar nosso app e outros módulos.
from app.app import app
from intent_classifier import IntentClassifier

# --- Constantes de Arquivos de Teste ---
# Caminhos para os arquivos do modelo e config usados nos testes de integração
TEST_MODELS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "intent_classifier", "models"
))
CONFUSION_MODEL_PATH = os.path.join(TEST_MODELS_DIR, "confusion.keras")
CONFUSION_CONFIG_PATH = os.path.join(TEST_MODELS_DIR, "confusion_config.yml")

# --- Fixtures ---
# Fixtures são funções reutilizáveis de setup/teardown para testes.

@pytest.fixture(scope="function", autouse=True)
def mock_app_dependencies(monkeypatch):
    """
    Esta é uma fixture "autouse". Ela rodará automaticamente antes de CADA teste
    neste arquivo, garantindo um ambiente limpo e mockado para nossos testes unitários.
    
    Ela encontra e substitui todas as dependências externas em 'app/app.py' por
    objetos 'MagicMock' controláveis.
    """
    print("\n--- [Setup da Fixture] ---")
    
    # 1. Mock da Coleção do MongoDB
    # Substituímos o objeto 'collection' real em 'app.app' por um mock.
    print("  > Mockando: app.services.load_all_models (Modelos de ML)")
    mock_models = MagicMock()
    monkeypatch.setattr("app.services.load_all_models", mock_models)

    # 2. Mock do(s) Modelo(s) de ML
    # Criamos um 'IntentClassifier' mockado e configuramos seu método 'predict'
    # para retornar um valor conhecido e previsível.
    print("  > Mockando: app.app.MODELS (Modelo de ML)")
    mock_model = MagicMock(spec=IntentClassifier)
    mock_model.predict.return_value = ("mock_intent", {"mock_intent": 0.9, "other": 0.1})
    
    # Substituímos o dicionário MODELS inteiro em 'app.app'.
    # Isso evita o processo lento de carregar modelos reais do disco.
    monkeypatch.setattr("app.app.MODELS", {"mock-model": mock_model})

    # 3. Mock da Função de Autenticação
    # Substituímos a função 'verify_token' real por um mock que
    # simplesmente retorna um usuário falso.
    print("  > Mockando: app.app.verify_token (Auth)")
    mock_verify_token = MagicMock(return_value="mock_prod_user")
    monkeypatch.setattr("app.app.verify_token", mock_verify_token)
    
    # 4. Mock da Coleção do MongoDB (se necessário)
    mock_collection = MagicMock()
    mock_collection.insert_one = MagicMock()
    
    print("--- [Setup da Fixture Completo] ---")

    # 'yield' passa os mocks para qualquer teste que os solicite.
    # O código resumirá aqui *depois* que o teste terminar.
    yield mock_collection, mock_model, mock_verify_token
    
    print("--- [Teardown da Fixture] ---")
    # (Nenhum teardown explícito necessário, os mocks são resetados pelo pytest)

@pytest.fixture(scope="function")
def client():
    """
    Uma fixture que cria e fornece um 'TestClient' para nosso 'app'.
    Este cliente faz requisições em memória para nosso app, não chamadas HTTP reais.
    Ele é criado *depois* que a fixture 'mock_app_dependencies' rodou.
    """
    print("  > Criando TestClient para o app")
    with TestClient(app) as test_client:
        yield test_client

# --- Testes Unitários ---
# Estes testes verificam pequenas peças isoladas de lógica (endpoints)
# usando as dependências mockadas.

def test_get_root_dev_mode(client, monkeypatch):
    """
    Teste 1.1: Verifica o endpoint GET / quando ENV é 'dev'.
    """
    print("\n[Teste] Rodando: GET / (Modo Dev)")
    
    # Define a variável global ENV dentro de 'app.app' como 'dev'
    monkeypatch.setattr("app.app.ENV", "dev")
    
    # Faz a requisição
    response = client.get("/")
    
    # --- Verificações (Assertions) ---
    print("  > Verificando status code... 200")
    assert response.status_code == 200
    print("  > Verificando JSON da resposta...")
    assert response.json() == {"message": "Basic ML App is running in dev mode"}
    print("[Teste] Passou: GET / (Modo Dev)")

def test_get_root_prod_mode(client, monkeypatch):
    """
    Teste 1.2: Verifica o endpoint GET / quando ENV é 'prod'.
    """
    print("\n[Teste] Rodando: GET / (Modo Prod)")
    
    # Define a variável global ENV dentro de 'app.app' como 'prod'
    monkeypatch.setattr("app.app.ENV", "prod")
    
    response = client.get("/")
    
    # --- Verificações (Assertions) ---
    print("  > Verificando status code... 200")
    assert response.status_code == 200
    print("  > Verificando JSON da resposta...")
    assert response.json() == {"message": "Basic ML App is running in prod mode"}
    print("[Teste] Passou: GET / (Modo Prod)")

def test_predict_dev_mode(client, monkeypatch, mock_app_dependencies):
    """
    Teste 2.1: Verifica o endpoint POST /predict em modo 'dev'.
    Deve pular a autenticação e retornar uma predição.
    """
    print("\n[Teste] Rodando: POST /predict (Modo Dev)")
    
    # Define ENV como 'dev'
    monkeypatch.setattr("app.app.ENV", "dev")
    
    # Pega nossos mocks da fixture para verificá-los depois
    mock_collection, mock_model, mock_verify_token = mock_app_dependencies
    
    # Faz a requisição
    test_text = "hello dev mode"
    response = client.post("/predict", params={"text": test_text})
    
    # --- Verificações (Assertions) ---
    print("  > Verificando status code... 200")
    assert response.status_code == 200
    
    # Verifica os dados da resposta
    data = response.json()
    print(f"  > Verificando dados da resposta (owner, text, prediction)...")
    assert data["owner"] == "dev_user"  # Autenticação foi pulada
    assert data["text"] == test_text
    assert "mock-model" in data["predictions"]
    assert data["predictions"]["mock-model"]["top_intent"] == "mock_intent"

    # Verifica se nossos mocks foram chamados corretamente
    print("  > Verificando chamadas das funções mockadas...")
    mock_verify_token.assert_not_called()  # Auth *não* foi chamada
    mock_model.predict.assert_called_with(test_text) # Modelo foi chamado
    mock_collection.insert_one.assert_called_once()  # DB foi chamado
    
    print("[Teste] Passou: POST /predict (Modo Dev)")

def test_predict_prod_mode_auth_success(client, monkeypatch, mock_app_dependencies):
    """
    Teste 3.1: Verifica o endpoint POST /predict em modo 'prod'
    com autenticação bem-sucedida.
    """
    print("\n[Teste] Rodando: POST /predict (Modo Prod - Auth Sucesso)")
    
    monkeypatch.setattr("app.app.ENV", "prod")
    
    mock_collection, mock_model, mock_verify_token = mock_app_dependencies
    
    test_text = "hello prod mode"
    response = client.post(
        "/predict",
        params={"text": test_text},
        headers={"Authorization": "Bearer valid_token"}
    )
    
    # --- Verificações (Assertions) ---
    print("  > Verificando status code... 200")
    assert response.status_code == 200
    
    # Verifica dados da resposta
    data = response.json()
    print(f"  > Verificando dados da resposta (owner)...")
    assert data["owner"] == "mock_prod_user"  # Auth teve sucesso

    # Verifica chamadas dos mocks
    print("  > Verificando chamadas das funções mockadas...")
    mock_verify_token.assert_called_once()  # Auth *foi* chamada
    mock_model.predict.assert_called_with(test_text)
    mock_collection.insert_one.assert_called_once()

    print("[Teste] Passou: POST /predict (Modo Prod - Auth Sucesso)")

def test_predict_prod_mode_auth_fail(client, monkeypatch, mock_app_dependencies):
    """
    Teste 3.2: Verifica o endpoint POST /predict em modo 'prod'
    quando a autenticação falha.
    """
    print("\n[Teste] Rodando: POST /predict (Modo Prod - Auth Falha)")
    
    monkeypatch.setattr("app.app.ENV", "prod")
    
    mock_collection, mock_model, mock_verify_token = mock_app_dependencies
    
    # Configura o mock de auth para *lançar um erro* ao ser chamado
    mock_verify_token.side_effect = HTTPException(status_code=401, detail="Invalid Token")
    
    response = client.post(
        "/predict",
        params={"text": "wont work"},
        headers={"Authorization": "Bearer invalid_token"}
    )
    
    # --- Verificações (Assertions) ---
    print("  > Verificando status code... 401")
    assert response.status_code == 401
    print("  > Verificando detalhe do erro...")
    assert "Invalid Token" in response.json()["detail"]

    # CRÍTICO: Verifica se as funções "downstream" *não* foram chamadas
    print("  > Verificando se funções mockadas *não* foram chamadas...")
    mock_model.predict.assert_not_called()
    mock_collection.insert_one.assert_not_called()
    
    print("[Teste] Passou: POST /predict (Modo Prod - Auth Falha)")

def test_predict_no_models_loaded(client, monkeypatch, mock_app_dependencies):
    """
    Teste 4.1: Verifica o comportamento do app se o dict MODELS estiver vazio.
    Deve ainda logar a requisição no DB, mas não retornar predições.
    """
    print("\n[Teste] Rodando: POST /predict (Nenhum Modelo Carregado)")
    
    monkeypatch.setattr("app.app.ENV", "dev") # Usa modo dev para pular auth
    
    # Sobrescreve o mock da fixture 'autouse' para MODELS com um dict vazio
    print("  > Sobrescrevendo mock MODELS para ser {}")
    monkeypatch.setattr("app.app.MODELS", {})
    
    mock_collection, _, _ = mock_app_dependencies
    
    response = client.post("/predict", params={"text": "no models"})
    
    # --- Verificações (Assertions) ---
    print("  > Verificando status code... 200")
    assert response.status_code == 200
    
    # Verifica que as predições estão vazias
    data = response.json()
    print("  > Verificando se as predições estão vazias...")
    assert data["predictions"] == {}
    
    # Verifica se o log no DB *ainda* foi chamado
    print("  > Verificando se o banco de dados ainda foi chamado...")
    mock_collection.insert_one.assert_called_once()
    
    print("[Teste] Passou: POST /predict (Nenhum Modelo Carregado)")

def test_model_config_category_mismatch(monkeypatch):
    """
    Teste 5.1: Verifica o comportamento quando o modelo tem um número diferente
    de categorias de saída do que está especificado em config.codes.
    
    Este é um cenário crítico que indica um modelo inválido ou config incorreto.
    O IntentClassifier deve detectar essa incompatibilidade durante a inicialização
    e fornecer uma mensagem de erro clara sugerindo verificar o arquivo de config
    e treinar um novo modelo.
    """
    print("\n[Teste] Rodando: IntentClassifier (Mismatch Model-Config Categories)")
    
    # Cria um mock de modelo Keras com output size diferente do config
    from unittest.mock import Mock, MagicMock
    import tensorflow as tf
    
    # Simula um modelo que foi treinado com 5 categorias
    mock_model = MagicMock()
    mock_model.output_shape = (None, 5)  # Modelo tem 5 classes de saída
    
    # Cria um mock de config com apenas 3 categorias (mismatch!)
    mock_config_obj = MagicMock()
    mock_config_obj.codes = ["intent1", "intent2", "intent3"]  # 3 categorias
    mock_config_obj.stop_words_file = None
    mock_config_obj.dataset_name = "test"
    mock_config_obj.architecture = "v0.1.5"
    mock_config_obj.task = "predict"
    mock_config_obj.min_words = 1
    mock_config_obj.embedding_model = 'https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/multilingual/2'
    mock_config_obj.sent_hl_units = 32
    mock_config_obj.sent_dropout = 0.1
    mock_config_obj.l1_reg = 0.01
    mock_config_obj.l2_reg = 0.01
    mock_config_obj.epochs = 500
    mock_config_obj.callback_patience = 20
    mock_config_obj.learning_rate = 5e-3
    mock_config_obj.validation_split = 0.2
    
    # Mocka tf.keras.models.load_model para retornar nosso modelo mockado
    def mock_load_model(path):
        return mock_model
    
    monkeypatch.setattr("tensorflow.keras.models.load_model", mock_load_model)
    
    # Mocka fetch_model_from_wandb para retornar um path fake
    def mock_fetch_model(url):
        return "/fake/model/path.keras"
    
    monkeypatch.setattr("intent_classifier.fetch_model_from_wandb", mock_fetch_model)
    
    # Mocka wandb para evitar inicialização real
    monkeypatch.setattr("intent_classifier.wandb.login", MagicMock())
    monkeypatch.setattr("intent_classifier.wandb.init", MagicMock())
    
    # Tenta inicializar IntentClassifier com mismatch
    # Isso deve lançar ValueError durante a validação
    print("  > Tentando inicializar IntentClassifier com modelo-config mismatch...")
    
    with pytest.raises(ValueError) as exc_info:
        # Cria um Config object com 3 categorias
        from intent_classifier import Config
        config = Config(
            dataset_name="test",
            codes=["intent1", "intent2", "intent3"],  # 3 categorias
            stop_words_file=None
        )
        
        # Tenta carregar um modelo (que tem 5 categorias)
        classifier = IntentClassifier(
            config=config,
            load_model="/fake/model/path.keras"  # Modelo mockado tem 5 categorias
        )
    
    # Verifica que o erro foi lançado com a mensagem correta
    error_message = str(exc_info.value)
    print(f"  > Erro capturado: {error_message[:150]}...")
    
    # Verifica que a mensagem de erro contém informações importantes
    assert "Model-config mismatch detected" in error_message
    assert "5 categories" in error_message or "5" in error_message
    assert "3 categories" in error_message or "3" in error_message
    assert "config file" in error_message.lower() or "config" in error_message.lower()
    assert "train" in error_message.lower()
    assert "intent_classifier.py train" in error_message or "train" in error_message
    
    print("  > Mensagem de erro contém todas as informações necessárias:")
    print("    - Detecta o mismatch entre modelo e config")
    print("    - Indica o número de categorias em cada um")
    print("    - Sugere verificar o config file")
    print("    - Fornece instruções para treinar um novo modelo")
    
    print("[Teste] Passou: IntentClassifier (Mismatch Model-Config Categories)")

# --- Testes de Integração ---
# Estes testes são marcados com '@pytest.mark.integration'
# Eles podem ser pulados rodando: pytest -m "not integration"
# Eles testam os componentes *reais*, removendo um ou mais mocks.

@pytest.mark.integration
def test_integration_real_model_predict(client, monkeypatch, mock_app_dependencies):
    """
    Teste de Integração 1: Testa o app com o modelo REAL 'confusion'.
    - *Mantemos* o mock para o banco de dados (collection).
    - *Mantemos* o mock para auth (definindo ENV=dev).
    - *Removemos* o mock do dict MODELS e carregamos o real.
    """
    print("\n[Teste de Integração] Rodando: POST /predict (Usando modelo REAL)")

    load_dotenv() # Garante que o .env local seja lido (para seu PC)
    if not os.getenv("WANDB_API_KEY"):
        print("  > PULANDO: WANDB_API_KEY não configurada no ambiente.")
        pytest.skip("PULANDO teste de integração: WANDB_API_KEY não configurada.")
    
    monkeypatch.setattr("app.app.ENV", "dev") # Mantém auth mockada
    
    # 1. Encontra o arquivo do modelo real
    if os.getenv("WANDB_MODEL_URL"):
        real_model_path = os.getenv("WANDB_MODEL_URL")
    else:
        real_model_path = CONFUSION_MODEL_PATH
    
    print(f"  > Carregando modelo real de: {real_model_path}")
    if not os.path.exists(real_model_path):
        print("  > PULANDO: Modelo não encontrado.")
        pytest.skip("Pulando teste do modelo: modelo não encontrado")
        
    # 2. Carrega o modelo real
    try:
        real_model = IntentClassifier(config=CONFUSION_CONFIG_PATH, 
                                      load_model=real_model_path)
    except Exception as e:
        print(f"  > FALHOU: Não foi possível carregar o modelo real. Erro: {e}")
        pytest.fail(f"Falha ao carregar modelo real de {real_model_path}: {e}")
        
    # 3. Desfaz o mock da variável global 'MODELS' substituindo-o pelo modelo real
    print("  > Injetando modelo real em 'app.app.MODELS'")
    monkeypatch.setattr("app.app.MODELS", {"confusion": real_model})
    
    # 4. Faz a requisição com um texto conhecido de "confusion"
    test_text = "wait what?"
    response = client.post("/predict", params={"text": test_text})
    
    # --- Verificações (Assertions) ---
    print("  > Verificando status code... 200")
    assert response.status_code == 200
    
    data = response.json()
    
    # Verifica se o modelo real produziu o output esperado
    print(f"  > Verificando se predição do modelo real é 'confusion'...")
    assert "confusion" in data["predictions"]
    prediction = data["predictions"]["confusion"]["top_intent"]
    print(f"  > Modelo real predisse: '{prediction}'")
    assert prediction == "confusion"
    
    # Verifica se o DB mockado ainda foi chamado
    mock_collection, _, _ = mock_app_dependencies
    mock_collection.insert_one.assert_called_once()
    
    print("[Teste de Integração] Passou: (Usando modelo REAL)")