# Given the client ID and tenant ID for an app registered in Azure,
# provide an Azure AD access token and a refresh token.

# If the caller is not already signed in to Azure, the caller's
# web browser will prompt the caller to sign in first.

# pip install msal
from msal import PublicClientApplication
import sys
import json
import re
from pathlib import Path

# You can hard-code the registered app's client ID and tenant ID here,
# or you can provide them as command-line arguments to this script.
client_id = '55a73d26-841d-4980-9157-d1a8152be7bb'
tenant_id = 'e009942d-3130-4196-8091-8c4d4c8e44a1'
home_path = Path.home()
# Do not modify this variable. It represents the programmatic ID for
# Azure Databricks along with the default scope of '/.default'.
scopes = ['2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default']

# Check for too few or too many command-line arguments.
if (len(sys.argv) > 1) and (len(sys.argv) != 3):
    print("Usage: get-tokens.py <client ID> <tenant ID>")
    exit(1)

# If the registered app's client ID and tenant ID are provided as
# command-line variables, set them here.
if len(sys.argv) > 1:
    client_id = sys.argv[1]
    tenant_id = sys.argv[2]

app = PublicClientApplication(
    client_id=client_id,
    authority="https://login.microsoftonline.com/" + tenant_id
)

acquire_tokens_result = app.acquire_token_interactive(
    scopes=scopes
)

if 'error' in acquire_tokens_result:
    print("Error: " + acquire_tokens_result['error'])
    print("Description: " + acquire_tokens_result['error_description'])
else:
    print("Successful!!")

    d = {
        "access_token": acquire_tokens_result['access_token'],
        "refresh_token": acquire_tokens_result['refresh_token']

    }

    with open(home_path.joinpath(".tokens.json"), "w") as j:
        json.dump(d, j, ensure_ascii=False, indent=2)
    # Update token in "~/.databrickscfg"
    with open(home_path.joinpath(".databrickscfg")) as r:
        cg = r.read()
    cg = re.sub("(?<=token = )(.*)(?=\n)", acquire_tokens_result['access_token'], cg)
    with open(home_path.joinpath(".databrickscfg"), "w") as wr:
        wr.write(cg)
    print("Updated databrickscfg file!!")
