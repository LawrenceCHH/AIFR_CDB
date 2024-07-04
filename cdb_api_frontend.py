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
    Deque, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union
)
from math import ceil





app = FastAPI()


# origins = [
#     domain,
#     domain.split('/')[0] + ':8000',
#     domain + ':8000',
#     "http://localhost",
#     "http://localhost:8000",
#     '*'
# ]
origins = [
    # '*',
    '127.0.0.1',
    '140.114.80.195',
    'https://jerry914.github.io',
    'https://jerry914.github.io/ai-annotated-judgment-database',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi.staticfiles import StaticFiles

frontend_template_dir = '/home/lawrencechh/AIFR_CDB/frontend_deployment/20240704_dist'


app.mount('/', StaticFiles(directory=frontend_template_dir, html=True), name='ai-annotated-judgment-database')

@app.exception_handler(404)
async def redirect_all_requests_to_frontend(request: Request, exc: HTTPException):

    request_url = str(request.url)
    splitted_url = request_url.split('/')[3]
    splitted_url = 'search-result?' if splitted_url.startswith('search-result?') else splitted_url
    vue_router_paths = ['about', 'search-result?', 'members']
    path_validated = splitted_url in vue_router_paths
    if path_validated:
        return HTMLResponse(open(frontend_template_dir+"/index.html").read())
    else:
        return JSONResponse({"detail":"Not Found"})

import uvicorn
domain_setting = {'host': '127.0.0.1', 'port': 6128}
domain = f"http://{domain_setting['host']}:{domain_setting['port']}" + '/'

if __name__ == '__main__':
    # uvicorn.run('cdb_api:app', host="127.0.0.1", port=6128)
    # uvicorn.run('cdb_api:app', host="140.114.80.195", port=6128)
    print(domain_setting['host'])
    # Formal server
    uvicorn.run('cdb_api_frontend:app', host=domain_setting['host'], port=domain_setting['port'], forwarded_allow_ips='*')
    # uvicorn.run('cdb_api_new:app', host=domain_setting['host'], port=domain_setting['port'], forwarded_allow_ips='*')

# # Commands
# ngrok tunnel --label edge=edghts_2b8EWy9H5bevmDCX2UwiHmpksel http://localhost:8000
# CHH python cdb_api.py
# Server
# pm2 start /home/lawrencechh/AIFR_CDB/cdb_api_frontend.py --name cdb