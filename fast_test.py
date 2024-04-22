from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles


@asynccontextmanager
async def lifespan(app: FastAPI):
    print('test')
    yield

app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import re

@app.exception_handler(404)
async def redirect_all_requests_to_frontend(request: Request, exc: HTTPException):
 
    request_url = str(request.url)
    vue_router_paths = ['about', 'search-result?', 'members']
    path_validated = request_url.split('/')[3] in vue_router_paths
    if path_validated:
        return HTMLResponse(open("/workspace/Projects/AIFR_CDB/frontend_deployment/20240416_dist/index.html").read())
    else:
        return JSONResponse({"detail":"Not Found"})
app.mount('/', StaticFiles(directory='/workspace/Projects/AIFR_CDB/frontend_deployment/20240416_dist', html=True))

import uvicorn
if __name__ == '__main__':
    # uvicorn.run('cdb_api:app', host=domain_setting['host'], port=domain_setting['port'], forwarded_allow_ips='*')
    uvicorn.run('fast_test:app', host='127.0.0.1', port=8000, forwarded_allow_ips='*', reload=True)