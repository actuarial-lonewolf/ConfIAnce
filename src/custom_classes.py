# Libraries
import os
import sys
import logging
import requests
import psycopg2
import psycopg2.pool
from dotenv import load_dotenv
from typing import Dict, List, Optional
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

# Add the parent directory to the path so we can import modules without path errors
sys.path.append("..")

# Load environment variables
assert load_dotenv()

# Set the logging level for Azure SDK to WARNING to avoid printing INFO
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)


class OpenAI:
    def __init__(self, 
                 kv_client: SecretClient,
                    model: Optional[str] = "gpt-4o" ,
                    emb_model: Optional[str] = "text-embedding-ada-002",
                 ):
        
        # Service
        self.incubator_endpoint = kv_client.get_secret("EYQ-INCUBATOR-ENDPOINT").value
        self.incubator_key = kv_client.get_secret("EYQ-INCUBATOR-KEY").value
        self.api_version = kv_client.get_secret("EYQ-API-VERSION").value
        
        # Model
        if model not in ["o1-mini", "o1-preview", "gpt-4o-mini", "gpt-4o","gpt-4-turbo-ga","gpt-4-turbo","gpt-35-turbo-16k","gpt-35-turbo"]:
            raise ValueError(
                "Model not supported."
            )
        else:
            self.model = model
        
        # Embeddings Models
        if emb_model not in ["text-embedding-3-large","text-embedding-ada-002"]:
            raise ValueError(
                "Embedding model not supported."
            )
        else:
            self.emb_model = emb_model
    

    def generic_completion(
        self, prompt: Optional[str] = None, messages: Optional[Dict[str, List]] = None
    ):
        self.url = (
            self.incubator_endpoint
            + "/openai/deployments/"
            + self.model
            + "/chat/completions"
        )
                
        if not messages:
            messages = {
                "messages": [
                    {
                        "role": "system",
                        "content": prompt,
                    }
                ],
            }
        response = requests.post(
            self.url,
            json=messages,
            headers={"api-key": self.incubator_key},
            params={"api-version": self.api_version},
        )
        status_code = response.status_code
        response = response.json()

        if status_code == 200:
            try:
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                print("Error: ", e)
                return (
                    "An error occurred during completion. Please try again. The response was: "
                    + str(response)
                )

        print("Error: ", status_code)
        print("Response: ", response["error"])
        return "An error occurred during completion. Please try again."
    
    def get_embedding(
        self, texts: Optional[List[str]] = None
    ):
        self.url = (
            self.incubator_endpoint
            + "/openai/deployments/"
            + self.emb_model
            + "/embeddings"
        )
        
        response = requests.post(
            self.url,
            json={
                    "input": texts,
                    "model": self.emb_model,
                },
            headers={"api-key": self.incubator_key},
            params={"api-version": self.api_version},
        )
        status_code = response.status_code
        response = response.json()

        if status_code == 200:
            try:
                return response["data"]
            except Exception as e:
                print("Error: ", e)
                return (
                    "An error occurred during completion. Please try again. The response was: "
                    + str(response)
                )

        print("Error: ", status_code)
        print("Response: ", response["error"])
        return "An error occurred during completion. Please try again."
    

class BlobClient:

    def __init__(self, kv_client: SecretClient):
        self.blob_service_client = BlobServiceClient.from_connection_string(
            kv_client.get_secret("BLOBSTORAGECONNECTIONSTRING").value
        )
        self.container_name = kv_client.get_secret("BLOB-CONTAINER-NAME").value


class PostgreSQLDatabase:
    def __init__(self, kv_client: SecretClient):
        self.kv_client = kv_client
        self._pool = self._init_pool()

    def _init_pool(self):
        try:
            return psycopg2.pool.SimpleConnectionPool(
                minconn=2,
                maxconn=10,
                user=self.kv_client.get_secret("DATABASE-USER").value,
                password=self.kv_client.get_secret("DATABASE-PASSWORD").value,
                host=self.kv_client.get_secret("DATABASE-SERVER").value,
                port=self.kv_client.get_secret("DATABASE-SERVER-PORT").value,
                database=self.kv_client.get_secret("DATABASE-NAME").value,
            )
        except Exception as e:
            print(f"Error initializing connection pool: {e}")
            return None

    def getconn(self):
        if self._pool:
            return self._pool.getconn()
        else:
            raise ConnectionError("Connection pool is not initialized.")

    def putconn(self, conn):
        if self._pool and conn:
            self._pool.putconn(conn)

    def execute_query(self, query, params=None):
        conn = self.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                if cur.description:
                    result = cur.fetchall()
                else:
                    conn.commit()
                    result = None
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
        finally:
            self.putconn(conn)

if __name__ == "__main__":
    # SPN with KeyVault Access
    AZURE_TENANT_ID = os.environ["AZURE_TENANT_ID"]
    AZURE_CLIENT_ID = os.environ["AZURE_CLIENT_ID"]
    AZURE_CLIENT_SECRET = os.environ["AZURE_CLIENT_SECRET"]
    AZURE_KEYVAULT_NAME = os.environ["AZURE_KEYVAULT_NAME"]

    # Create Key Vault client
    kv_client = SecretClient(
        vault_url=f"https://{AZURE_KEYVAULT_NAME}.vault.azure.net/",
        credential=ClientSecretCredential(
            AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
        ),
    )

    llm_conn = OpenAI(kv_client=kv_client)
    blob_conn = BlobClient(kv_client=kv_client)
    postgres_conn = PostgreSQLDatabase(kv_client=kv_client)