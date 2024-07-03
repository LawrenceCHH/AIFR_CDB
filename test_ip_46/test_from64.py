from fastapi import FastAPI, Request, Query
from pydantic import BaseModel
import json
import pandas as pd
# Cross origin and allow origins
from fastapi.middleware.cors import CORSMiddleware

# Lifespan
from contextlib import asynccontextmanager
# Log
import logging

# 404 redirect to frontend template
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import (
    Deque, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union, Any
)
from math import ceil






app = FastAPI()


origins = [
    '*',
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 假設的 get_users 函數，這裡只是返回一個範例數據
def get_users():
    return [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"}
    ]

# 假設的 paginate 函數，這裡根據 params 做簡單的分頁處理
def paginate(data, params: Dict[str, Any]):
    size = params["size"]
    page = params["page"]
    start = (page - 1) * size
    end = start + size
    return data[start:end]

# 定義接受的參數模型
class PaginationParams(BaseModel):
    page: Optional[int] = 1
    page_size: Optional[int] = 2

# # 分頁獲取用戶
# @app.get("/")
# async def index_page():
#     return {'ky':123}
# # 分頁獲取用戶
@app.get("/users/", response_model=Dict[str, Any])
async def paginate_users(page: int, size: int):
    users = get_users()
    paginated_users = paginate(users, {'page': page, 'size': size})
    return {
        "data": paginated_users,
        "page": page,
        "page_size": size,
        "total": len(users)
    }



from fastapi.staticfiles import StaticFiles

# frontend_template_dir = '/home/lawrencechh/AIFR_CDB/test'
frontend_template_dir = '/home/lawrencechh/AIFR_CDB/test/t'


app.mount('/', StaticFiles(directory=frontend_template_dir, html=True))
domain_setting = {'host': '140.114.80.195', 'port': 6728}
domain = f"http://{domain_setting['host']}:{domain_setting['port']}" + '/'
import uvicorn
if __name__ == '__main__':
    # uvicorn.run('cdb_api:app', host="127.0.0.1", port=6128)
    # uvicorn.run('cdb_api:app', host="140.114.80.195", port=6128)
    print(domain_setting['host'])
    # Formal server
    uvicorn.run('test_from64:app', host=domain_setting['host'], port=domain_setting['port'], forwarded_allow_ips='*')
    # uvicorn.run('cdb_api_new:app', host=domain_setting['host'], port=domain_setting['port'], forwarded_allow_ips='*')

# # Commands
# ngrok tunnel --label edge=edghts_2b8EWy9H5bevmDCX2UwiHmpksel http://localhost:8000
# CHH python cdb_api.py
# Server
# pm2 start cdb_api.py --name cdb
# pm2 start /home/lawrencechh/AIFR_CDB/test/test_from64.py --name test

    