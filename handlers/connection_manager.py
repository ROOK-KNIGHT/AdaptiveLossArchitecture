#connection_manager.py

import base64
import requests
import webbrowser
import json
import urllib.parse
import os
from datetime import timedelta, datetime
import time

# Load API credentials from external file
def load_api_keys():
    """Load API keys from the external KEYS.json file"""
    keys_path = "/Users/isaac/Desktop/Projects/CS_KEYS/KEYS.json"
    try:
        with open(keys_path, 'r') as f:
            keys = json.load(f)
        return keys["APP_KEY"], keys["APP_SECRET"]
    except Exception as e:
        print(f"Error loading API keys from {keys_path}: {e}")
        raise

# Load keys
APP_KEY, APP_SECRET = load_api_keys()

# Configuration
REDIRECT_URI = "https://127.0.0.1"
AUTH_URL = f"https://api.schwabapi.com/v1/oauth/authorize?response_type=code&client_id={APP_KEY}&redirect_uri={REDIRECT_URI}&scope=readonly"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

# Path to save tokens - Update this path for your system
TOKEN_FILE = "cs_tokens.json"

def save_tokens(tokens):
    # Calculate and save the expiration time as a string
    expires_at = datetime.now() + timedelta(seconds=int(tokens['expires_in']))
    tokens['expires_at'] = expires_at.isoformat()  # Store it as an ISO string
    
    with open(TOKEN_FILE, 'w') as f:
        json.dump(tokens, f)


def load_tokens():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            return json.load(f)
    return None

def get_authorization_code():
    print("Manual authentication required. Go to the following URL to authenticate:")
    print(AUTH_URL)
    webbrowser.open(AUTH_URL)
    
    returned_url = input("Paste the full returned URL here as soon as you get it: ")
    
    # Extract the authorization code from the returned URL
    parsed_url = urllib.parse.urlparse(returned_url)
    code = urllib.parse.parse_qs(parsed_url.query).get('code', [None])[0]
    
    if not code:
        raise ValueError("Failed to extract code from the returned URL")
    
    return code


def get_tokens(code):
    print("Exchanging authorization code for tokens...")
    credentials = f"{APP_KEY}:{APP_SECRET}"
    base64_credentials = base64.b64encode(credentials.encode()).decode("utf-8")

    headers = {
        "Authorization": f"Basic {base64_credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI
    }
    
    token_response = requests.post(TOKEN_URL, headers=headers, data=payload)
    if token_response.status_code == 200:
        tokens = token_response.json()
        save_tokens(tokens)
        return tokens
    else:
        print("Failed to get tokens")
        print("Status Code:", token_response.status_code)
        print("Response:", token_response.text)
        return None

def refresh_tokens(refresh_token):
    print("Refreshing access token...")
    credentials = f"{APP_KEY}:{APP_SECRET}"
    base64_credentials = base64.b64encode(credentials.encode()).decode("utf-8")

    headers = {
        "Authorization": f"Basic {base64_credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }

    refresh_response = requests.post(TOKEN_URL, headers=headers, data=payload)
    
    if refresh_response.status_code == 200:
        new_tokens = refresh_response.json()
        save_tokens(new_tokens)
        return new_tokens
    else:
        print("Failed to refresh tokens")
        print("Status Code:", refresh_response.status_code)
        print("Response:", refresh_response.text)
        return None


# Modify ensure_valid_tokens to check expiration time
def ensure_valid_tokens():
    tokens = load_tokens()
    if tokens:
        expires_at = tokens.get('expires_at')
        
        # Check if 'expires_at' exists and is a valid string
        if expires_at:
            try:
                expires_at = datetime.fromisoformat(expires_at)
            except ValueError:
                print("Invalid 'expires_at' format in tokens. Re-authentication required.")
                tokens = None  # Force re-authentication
        else:
            print("'expires_at' missing from tokens. Re-authentication required.")
            tokens = None  # Force re-authentication

        if tokens:
            refresh_token = tokens.get("refresh_token")
            # Check if access token is expired or about to expire (within a buffer, e.g., 2 minutes)
            if datetime.now() >= expires_at - timedelta(minutes=2):
                print("Access token is about to expire or has expired, attempting to refresh...")
                new_tokens = refresh_tokens(refresh_token)
                if new_tokens:
                    return new_tokens  # Token successfully refreshed
                else:
                    print("Failed to refresh tokens. Please re-authenticate.")
            else:
                return tokens  # Access token is still valid

    # If no tokens or refreshing failed, require manual re-authentication
    print("Manual re-authentication required.")
    code = get_authorization_code()
    return get_tokens(code)


def get_account_numbers(access_token):
    url = "https://api.schwabapi.com/trader/v1/accounts/accountNumbers"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    retries = 5
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)  # 10 seconds timeout
            response.raise_for_status()  # Raise error for bad status codes
            return response.json()
        except requests.exceptions.ReadTimeout:
            print(f"Request timed out on attempt {attempt + 1}/{retries}. Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}, attempt {attempt + 1}/{retries}")
        time.sleep(2 ** attempt)  # Exponential backoff

    raise Exception(f"Failed to fetch account numbers after {retries} attempts")

def get_account_details(access_token, account_number, field):
    #print(f"DEBUG : Fetching account details for account {account_number}...")
    url = f"https://api.schwabapi.com/trader/v1/accounts/{account_number}?fields={field}"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        print(f"Rate limit exceeded. Retry after {retry_after} seconds.")
        return None
    else:
        print(f"Failed to get account details\nStatus Code: {response.status_code}\nResponse: {response.text}")
        return None

def get_positions(access_token, account_number):
    """Get current positions for the specified account"""
    url = f"https://api.schwabapi.com/trader/v1/accounts/{account_number}/positions"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                positions = response.json()
                # Format the positions data
                formatted_positions = []
                for pos in positions.get('positions', []):
                    formatted_positions.append({
                        'symbol': pos.get('symbol'),
                        'quantity': pos.get('quantity'),
                        'cost_basis': pos.get('costBasis'),
                        'market_value': pos.get('marketValue'),
                        'unrealized_pl': pos.get('unrealizedPL'),
                        'unrealized_pl_percent': pos.get('unrealizedPLPercent')
                    })
                return formatted_positions
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limit exceeded. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                print(f"Failed to get positions. Status Code: {response.status_code}")
                print(f"Response: {response.text}")
                time.sleep(2 ** attempt)  # Exponential backoff
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None

def get_all_positions():
    """Get positions for all accounts"""
    try:
        # Ensure we have valid tokens
        tokens = ensure_valid_tokens()
        if not tokens:
            print("Failed to get valid tokens")
            return None
            
        access_token = tokens['access_token']
        
        # Get accounts with positions
        url = "https://api.schwabapi.com/trader/v1/accounts?fields=positions"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to get accounts. Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
        accounts = response.json()
        all_positions = {}
        
        for account in accounts:
            securities_account = account.get('securitiesAccount', {})
            account_number = securities_account.get('accountNumber')
            positions = securities_account.get('positions', [])
            
            if account_number and positions:
                formatted_positions = []
                for pos in positions:
                    instrument = pos.get('instrument', {})
                    formatted_positions.append({
                        'symbol': instrument.get('symbol'),
                        'quantity': pos.get('longQuantity', 0) - pos.get('shortQuantity', 0),
                        'cost_basis': pos.get('averagePrice', 0) * pos.get('longQuantity', 0),
                        'market_value': pos.get('marketValue', 0),
                        'unrealized_pl': pos.get('longOpenProfitLoss', 0) + pos.get('shortOpenProfitLoss', 0),
                        'unrealized_pl_percent': (pos.get('currentDayProfitLossPercentage', 0))
                    })
                if formatted_positions:
                    all_positions[account_number] = formatted_positions
                    
        return all_positions
    except Exception as e:
        print(f"Error getting all positions: {str(e)}")
        return None
