import hmac # compare_digest = safe way to compare secrets
from fastmcp import FastMCP
from fastmcp.server.auth import require_scopes
from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair
from fastmcp.server.dependencies import get_access_token
from starlette.requests import Request
from starlette.responses import JSONResponse


mcp = FastMCP("mymcpserver")

HOST = "127.0.0.1"
PORT = 8004

ISSUER = f"http://{HOST}:{PORT}" # who created the token
AUDIENCE = "secure-mcp-server" # "aud": which server the token is for
TOKEN_LIFETIME_SECONDS = 3

key_pair = RSAKeyPair.generate()# public/private keys, access token

RESTRICTED_CLIENTS = {
    "reader-app":{
        "secret":"reader-secret",
        "allowed_scopes": ["app:read"],
    },
    "admin-app":{
        "secret":"admin-secret",
        "allowed_scopes":["app:read","db:reset"],
    },
}

mcp = FastMCP(
    "production-style Secure Server",
    auth=JWTVerifier(
        public_key=key_pair.public_key,
        issuer=ISSUER,
        audience=AUDIENCE,
    ),
    
)

@mcp.tool(auth=require_scopes("app:read"))
def greet(name:str)->str:
    """Return a freindly greeting message on the given name"""
    return f'hey hi {name} welcome to you to the vrushabhs mcp server'

@mcp.tool(auth=require_scopes("app:read"))
def whami()->str:
    """tell the person who they are and what can they do"""
    token = get_access_token()
    return f'hey you are the {token.client_id} and you the scope of {token.scopes}'

@mcp.tool(auth=require_scopes("db:reset"))
def dbreset()->str:
    """this is the high risk action names db reset need the db:reset scope for this"""
    print("[server] reset command ran!")
    return "DB reset is done succsesfully"

@mcp.custom_route("/token",methods=["POST"])
async def issue_token(request:Request)->JSONResponse:
    form = await request.form()

    #1. Check
    if form.get("grant_type") != "client_credentials":
        return error_response("unsupported_grant_type", 400)

    #2. Check the client
    client_id = str(form.get("client_id", ""))
    client_secret = str(form.get("client_secret", ""))
    client = REGISTERED_CLIENTS.get(client_id)

    if not is_valid_client(client, client_secret):
        return error_response("invalid_client", 401)

    #3. Check the scope
    requested_scopes = (str(form.get("scope", "")).split()
                        or client["allowed_scopes"])
    for scope in requested_scopes:
        if scope not in client["allowed_scopes"]:
            return error_response("invalid_scope", 400)

    access_token = key_pair.create_token(
        subject=client_id, # "sub": who owns the token
        issuer=ISSUER, # this server is the issuer here
        audience=AUDIENCE, # "aud": this MCP server
        scopes=requested_scopes, # what the token allows
        expires_in_seconds=TOKEN_LIFETIME_SECONDS, # "exp"
        )

    return JSONResponse({
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": TOKEN_LIFETIME_SECONDS,
        "scope": " ".join(requested_scopes)
        })



