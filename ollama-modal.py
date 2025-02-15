import modal
import os
import subprocess
import time
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import ollama
import asyncio
from typing import AsyncGenerator, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = os.environ.get("MODEL", "mistral-small")  # Updated to correct model name
MAX_RETRIES = 3
RETRY_DELAY = 2

app = modal.App("ollama-api")

def setup_ollama():
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Setting up Ollama (attempt {attempt + 1}/{MAX_RETRIES})")
            
            # Initialize Ollama service
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "enable", "ollama"], check=True)
            subprocess.run(["systemctl", "start", "ollama"], check=True)
            time.sleep(15)  # Wait longer for service to start
            
            # Pull model
            subprocess.run(["ollama", "pull", MODEL], check=True)
            
            # Verify Ollama is responding
            ollama.list()
            logger.info("Ollama is running and responding")
            return
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "systemctl")
    .pip_install("ollama", "fastapi", "uvicorn", "httpx")
    .run_commands(
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama",
        "usermod -a -G ollama $(whoami)",
    )
    .add_local_file("ollama.service", "/etc/systemd/system/ollama.service", copy=True)
)

web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_chat_chunk(content: str = "", finish_reason: str = None) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-" + os.urandom(12).hex(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason
            }
        ]
    }

async def stream_chat_response(messages: list) -> AsyncGenerator[str, None]:
    try:
        # Start response
        yield f"data: {json.dumps(create_chat_chunk())}\n\n"
        
        # Get response from Ollama
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            stream=True
        )
        
        for chunk in response:
            content = chunk["message"]["content"]
            if content:
                yield f"data: {json.dumps(create_chat_chunk(content=content))}\n\n"
        
        # Send finish chunk
        yield f"data: {json.dumps(create_chat_chunk(finish_reason='stop'))}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in stream generation: {str(e)}")
        yield f"data: {json.dumps(create_chat_chunk(content=f'Error: {str(e)}', finish_reason='error'))}\n\n"
        yield "data: [DONE]\n\n"

@web_app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    try:
        # Ensure Ollama service is running
        setup_ollama()
        
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        
        if stream:
            return StreamingResponse(
                stream_chat_response(messages),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            try:
                response = ollama.chat(
                    model=MODEL,
                    messages=messages,
                    stream=False
                )
                
                completion_response = {
                    "id": "chatcmpl-" + os.urandom(12).hex(),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response["message"]["content"]
                            },
                            "finish_reason": "stop"
                        }
                    ]
                }
                
                return completion_response
            except Exception as e:
                logger.error(f"Error in non-streaming response: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )
            
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.function(
    image=image,
    gpu="h100",
    container_idle_timeout=300,
    timeout=300
)
@modal.asgi_app()
def fastapi_app():
    # Initialize Ollama when the container starts
    setup_ollama()
    return web_app
