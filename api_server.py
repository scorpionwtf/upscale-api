#api_server.py
import os
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
# from pymongo import MongoClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

from transformations.upscale import upscale

import uvicorn

from environs import Env
import logging
import base64
from PIL import Image
from io import BytesIO

env = Env()
env.read_env() # Load environment variables from a .env file

print(1)
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app and rate limiter
app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


class TransformRequest(BaseModel):
    userId: str = Field(..., min_length=1, max_length=50, pattern=r'^[a-zA-Z0-9_-]+$')
    transformationType: str = Field(..., min_length=1, max_length=50, pattern=r'^[a-zA-Z_]+$')
    upscale_factor: str = Field(..., min_length=1)
    imageData: str = Field(..., min_length=1)

def verify_api_key(x_api_key: str = Header(...)):

    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/transform")
@limiter.limit("5000/minute")  # Apply rate limiting: 5000 requests per minute per IP address
async def transform(request: Request, transform_request: TransformRequest, x_api_key: str = Depends(verify_api_key)):
    # try:
    #     user_id = ObjectId(transform_request.userId)
    #     logging.debug(f"Converted userId to ObjectId: {user_id}")
    #     user = users_collection.find_one({"_id": user_id})
    # except Exception as e:
    #     logging.error(f"Error converting userId: {e}")
    #     raise HTTPException(status_code=400, detail="Invalid userId format")

    # if not user:
    #     logging.debug("User not found")
    #     raise HTTPException(status_code=404, detail="User not found")
    
    # if user['credits'] <= 0:
    #     raise HTTPException(status_code=402, detail="Insufficient credits")

    # Decode the base64 image data
    image_data = base64.b64decode(transform_request.imageData)
    

    

    # Perform the requested transformation
    try:
        if transform_request.transformationType == "upscale":
            result_image = upscale(image_data, transform_request.upscale_factor)
            print(transform_request.upscale_factor)
        else:
            raise HTTPException(status_code=400, detail="Invalid transformation type")
    
    except Exception as e:
        logging.error(f"Error transforming image")
        raise HTTPException(status_code=400, detail="Transformation failed")

    
    # Encode the result image to base64
    buffered = BytesIO()
    result_image.save(buffered, format="PNG")
    result_base64 = base64.b64encode(buffered.getvalue()).decode("ascii")
    
    # Deduct one credit from the user's account
    # users_collection.update_one({"_id": user_id}, {"$inc": {"credits": -1}})
    
    return JSONResponse(content={"result": result_base64})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
