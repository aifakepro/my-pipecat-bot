import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

app = FastAPI()

# Serve client.html at root
@app.get("/")
async def root():
    return FileResponse("client.html")

@app.post("/connect")
async def connect():
    """Create Daily room and start bot"""
    try:
        # Get Daily API key
        daily_api_key = os.getenv("DAILY_API_KEY")
        if not daily_api_key:
            raise HTTPException(status_code=500, detail="DAILY_API_KEY not set")
        
        # Create Daily room
        response = requests.post(
            "https://api.daily.co/v1/rooms",
            headers={
                "Authorization": f"Bearer {daily_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "properties": {
                    "exp": int(asyncio.get_event_loop().time()) + 300,  # 5 minutes
                    "enable_chat": True,
                    "enable_screenshare": False
                }
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to create room: {response.text}")
        
        room_data = response.json()
        room_url = room_data["url"]
        
        # Create token for user
        token_response = requests.post(
            "https://api.daily.co/v1/meeting-tokens",
            headers={
                "Authorization": f"Bearer {daily_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "properties": {
                    "room_name": room_data["name"],
                    "is_owner": True
                }
            }
        )
        
        if token_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to create token: {token_response.text}")
        
        token_data = token_response.json()
        
        # Start bot in background
        from bot_gemini import bot
        from pipecat.runner.types import RunnerArguments
        
        bot_token_response = requests.post(
            "https://api.daily.co/v1/meeting-tokens",
            headers={
                "Authorization": f"Bearer {daily_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "properties": {
                    "room_name": room_data["name"],
                    "is_owner": False
                }
            }
        )
        
        bot_token = bot_token_response.json()["token"]
        
        # Start bot in background task
        asyncio.create_task(bot(RunnerArguments(
            room_url=room_url,
            token=bot_token
        )))
        
        return JSONResponse({
            "room_url": room_url,
            "token": token_data["token"]
        })
        
    except Exception as e:
        print(f"Error in /connect: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
