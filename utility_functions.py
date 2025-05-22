import re
from pathlib import Path
import re 
import sys
from dotenv import load_dotenv
import pandas as pd
import os

from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

# Add the parent directory to the path so we can import modules without path errors
sys.path.append("..")

# Load environment variables
assert load_dotenv()

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

from src.custom_classes import OpenAI
llm_conn = OpenAI(kv_client=kv_client)



ROOT_DIR = Path.cwd()

# Get the root directory (assuming this script is in src/)
project_root = Path(ROOT_DIR).resolve()#.parents[0] # a adapter pour le notebook


data_path = project_root / 'data'
data_path


sheet = "example"

data = pd.DataFrame()
# print(data_path)
for file in os.listdir(data_path):
    
    # print(file)
    if file.endswith(".xlsx") and re.match(r"^\d", file):
         current_df = pd.read_excel(data_path / file, sheet_name=sheet)
         current_df["file"] = file
         data = pd.concat(
            [
                data,
                current_df
            ]
        )


train = data[data["TYPE"]=="TRAIN"]
   
# filter data where "Email_scam" is notnull and keep only email column
# Extract emails from the dataset where Email_scam is not null
scam_emails = train["Email_scam"].dropna().tolist()
unique_scam_emails = list(set(scam_emails))

scam_phone = train["Phone"].dropna().tolist()
unique_scam_phone = list(set(scam_emails))

#website semble contenir pas mal de choses... skip pour l'instant


potential_scam_pseudo = data["Pseudo"].dropna().tolist()
unique_scam_pseudo = list(set(potential_scam_pseudo))





def extract_phone_number(text):
    """
    Extracts the first phone number found in the input string.
    Supports various common phone number formats.
    Returns the phone number as a string, or None if not found.
    """
    phone_pattern = re.compile(
        r'(\+?\d{1,3}[\s\-\.]?)?(\(?\d{3}\)?[\s\-\.]?)?\d{3}[\s\-\.]?\d{4}'
    )
    match = phone_pattern.search(text)
    if match:
        return match.group()
    return None

def extract_email_address(text):
    """
    Extracts the first email address found in the input string.
    Returns the email address as a string, or None if not found.
    """
    email_pattern = re.compile(
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    )
    match = email_pattern.search(text)
    if match:
        return match.group()
    return None


def extract_url_with_llm(text, llm_conn) -> str:
    """
    Uses the LLM (via generic_completion) to extract the first URL from the input text.
    Returns the URL as a string, or None if not found.
    """
    prompt = (
        "Extract the first URL from the following text. "
        "If there is no URL, respond with 'None'.\n\n"
        f"Text: {text}"
    )
    response = llm_conn.generic_completion(prompt=prompt)
    url = response.strip()
    if url.lower() == "none":
        return None
    return url

def extract_url(text, llm_conn):
    """
    Extracts the first URL found in the input string.
    Returns the URL as a string, or None if not found.
    """
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    match = url_pattern.search(text)
    if match:
        return match.group()
    else:
        # Fallback: Use an LLM to extract the URL if regex fails
        return extract_url_with_llm(text, llm_conn)

def extract_pseudo(text):
    """
    Extracts the first pseudo (username) found in the input string.
    Returns the pseudo as a string, or None if not found.
    """
    prompt = (
        "Extract the first pseudo (username) from the following text."
        "If there is no pseudo, respond with 'None'.\n\n"
        f"Examples of pseudo are: {unique_scam_pseudo[:10]}.\n\n"
        f"Text: {text}"
    )
    print(unique_scam_pseudo[:10])
    response = llm_conn.generic_completion(prompt=prompt)
    username = response.strip()
    if username.lower() == "none":
        return None
    return username


def identify_scam_email(text):
    """
    Identifies if the input text contains a scam email.
    Returns True if a scam email is found, otherwise False.
    """
    found_email = extract_email_address(text)
    if found_email:
        for scam_email in unique_scam_emails:
            if scam_email.lower() in found_email.lower():
                return True, scam_email
    return False,None

def identify_scam_phone(text):
    """
    Identifies if the input text contains a scam phone number.
    Returns True if a scam phone number is found, otherwise False.
    """
    found_phone = extract_phone_number(text)
    if found_phone:
        for scam_phone in unique_scam_phone:
            if scam_phone in found_phone:
                return True, scam_phone
    return False, None

def identify_scam_pseudo(text):
    """
    Identifies if the input text contains a scam pseudo (username).
    Returns True if a scam pseudo is found, otherwise False.
    """
    found_pseudo = extract_pseudo(text)
    if found_pseudo:
        for scam_pseudo in unique_scam_pseudo:
            if scam_pseudo.lower() == found_pseudo.lower():
                return True
    return False

if __name__ == "__main__":
    # Example usage
    text = "Contact me at someone@outlook.ca"
    print("Extracted Email:", extract_email_address(text))
    print("Extracted Phone:", extract_phone_number(text))
    print("Scam email?", identify_scam_email(text))

    # Example usage
    text = "Contact me at r_wilson@primeoakmont.com"
    print("Extracted Email:", extract_email_address(text))
    print("Extracted Phone:", extract_phone_number(text))
    print("Scam email?", identify_scam_email(text))

    # Example usage
    text = "Contact me at 1-800-555-0199"
    print("Extracted Email:", extract_email_address(text))
    print("Extracted Phone:", extract_phone_number(text))
    print("Scam email?", identify_scam_email(text))
    print("Scam phone?", identify_scam_phone(text))

    # Example usage
    text = "My name is Simon. Am I a scammer?"
    print("Extracted username:", extract_pseudo(text))
    print("Scam username?", identify_scam_pseudo(text))