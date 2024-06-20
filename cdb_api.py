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


on_server = True


def get_merged_df(main_basic_df, category_df, based_column='UID'):
    # category_df['order_index'] = range(len(category_df))
    # category_df['jud_url'] = [f'https://judgment.judicial.gov.tw/FJUD/data.aspx?ty=JD&id={JID}' for JID in category_df['JID']]
    # category_df['same_distance_num'] = ""
    # category_df['show_unique_result'] = ""
    target_basic_df_columns = main_basic_df.columns.tolist()
    del target_basic_df_columns[target_basic_df_columns.index('JID')]
    merged_df = category_df.merge(main_basic_df[target_basic_df_columns], on=based_column, how='left')
    return merged_df
def get_sorted_res(res_df, search_method):
    if search_method=='keyword' and len(res_df)>0:
        res_df['jud_date'] = res_df['jud_date'].astype(str)

        res_df.sort_values(by='jud_date', inplace=True)

    return res_df
def get_formatted_res(res_df, query_dict, result_dict):
    if query_dict=={}:
        query_type = ""
        query_type_ch = ""
        query_text = ""
    else:
        query_type, query_text = list(query_dict.items())[0]
        query_type_ch = preloaded_data['db'][query_type][2]

    res_data_length = len(res_df)
    if res_data_length>0:
        jud_num = len(list(set(res_df['UID'].tolist())))
        res_data = res_df.to_json(orient='records', force_ascii=False)
        res_data = json.loads(res_data)
    else:
        jud_num = 0
        res_data = []
    

    res_json = {
        'query_info': {
            'query_type': query_type,
            'query_type_ch': query_type_ch,
            'query_text': query_text,
        }, 
        'condition_info':{
            'available': result_dict['available'],
            'unavailable': result_dict['unavailable'],
        },
        'summary': [
            {
                'name': "裁判書",
                'unit': "篇",
                'count': jud_num
            },
            {
                'name': "涵攝",
                'unit': "筆",
                'count': 0
            },
            {
                'name': "見解",
                'unit': "筆",
                'count': 0
            },
            {
                'name': "心證",
                'unit': "筆",
                'count': 0
            }
        ],
        'data': res_data
    }
    if query_dict!={}:
        for index, summary_type in enumerate(res_json['summary']):
            if summary_type["name"]==query_type_ch:
                res_json['summary'][index]['count'] = res_data_length
    elif res_data_length>0 and query_dict=={}:
        for query_type in preloaded_data['db'].keys():
            tmp_category_df = res_df[res_df[query_type]!="無資料"]
            query_type_ch = preloaded_data['db'][query_type][2]

            for index, summary_type in enumerate(res_json['summary']):
                if summary_type["name"]==query_type_ch:
                    # res_json['summary'][index]['count'] = len(tmp_category_df)
                    res_json['summary'][index]['count'] = sum(map(len, tmp_category_df[query_type]))
                    

    # db = {'fee': [fee_df, fee_flat, '心證'], 'sub': [sub_df, sub_flat, '涵攝'], 'opinion': [opinion_df, opinion_flat, '見解']}
    return res_json
def get_keywords_df(query_text, target_df, target_column = 'sentence'):
    if query_text=="":
        return target_df
    tmp_df = target_df
    tmp_df = tmp_df.fillna("無資料")
    for keyword in query_text.split(' '):
        if keyword!="":
            tmp_df = tmp_df[tmp_df[target_column].str.contains(keyword)]
    tmp_df = tmp_df.drop_duplicates()
    return tmp_df

def get_condition_filtered_result(query_text, target_df, target_column):
    if query_text=="":
        return target_df
    if target_column=="jud_date":
        # Split the user input into start_date and end_date
        start_date, end_date = query_text.split('-')
        # Filter the DataFrame based on the date range
        target_df['jud_date'] = target_df['jud_date'].astype(str)
        output_df = target_df[(target_df['jud_date'] >= str(start_date)) & (target_df['jud_date'] <= str(end_date))]
    elif target_column=="court_type":
        target_courts = query_text.strip().split(' ')
        output_df = target_df[target_df[target_column].isin(target_courts)]
    else:
        output_df = get_keywords_df(query_text, target_df, target_column)

    return output_df
def get_condition_filtered_dict(condition_dict, target_df):
    # available and unavailable list [['target_column1', 'query_text2], ...]
    result_dict = {'result_df': None, 'available': [], 'unavailable': []}
    result_df = target_df.copy()
    last_result_df = result_df.copy()
    for target_column, query_text in condition_dict.items():
        if query_text=="":
            continue
        if target_column=="jud_date":
            # Split the user input into start_date and end_date
            start_date, end_date = query_text.split('-')
            # Filter the DataFrame based on the date range
            result_df['jud_date'] = result_df['jud_date'].astype(str)
            result_df = result_df[(result_df['jud_date'] >= str(start_date)) & (result_df['jud_date'] <= str(end_date))]
        elif target_column=="court_type":
            target_courts = query_text.strip().split(' ')
            result_df = result_df[result_df[target_column].isin(target_courts)]
        else:
            result_df = get_keywords_df(query_text, result_df, target_column)
        if len(result_df)==0:
            result_dict['unavailable'].append([target_column, query_text])
            # Use last available conditions
            # result_df = last_result_df
            # Break if condition not met
            break
        else:
            result_dict['available'].append([target_column, query_text])
            last_result_df = result_df.copy()

    result_dict['result_df'] = result_df
    return result_dict


preloaded_data = {}
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger = logging.getLogger("uvicorn.access")
    if on_server:
        handler = logging.handlers.RotatingFileHandler("verdictdb_api.log",mode="a",maxBytes = 100*1024*1024, backupCount = 10)
    else:
        handler = logging.handlers.RotatingFileHandler("verdictdb_api_chh.log",mode="a",maxBytes = 100*1024*1024, backupCount = 10)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    if on_server:
        main_basic_df = pd.read_csv('~/workspace/data/CDB-2017-2021/6. Merged df/20240617_2017_2021_juds_basic_info.csv', lineterminator='\n')
        opinion_df = pd.read_csv('~/workspace/data/CDB-2017-2021/6. Merged df/20240617_2017_2021判決書_無簡_sentence_opinion.csv', lineterminator='\n')
        sub_df = pd.read_csv('~/workspace/data/CDB-2017-2021/6. Merged df/20240617_2017_2021判決書_無簡_paragraph_sub.csv', lineterminator='\n')
        fee_df = pd.read_csv('~/workspace/data/CDB-2017-2021/6. Merged df/20240617_2017_2021判決書_無簡_paragraph_fee.csv', lineterminator='\n')
    else:
        main_basic_df = pd.read_csv('/workspace/data/CDB-2017-2021/6. Merged df/20240617_2017_2021_juds_basic_info.csv', lineterminator='\n')
        opinion_df = pd.read_csv('/workspace/data/CDB-2017-2021/6. Merged df/20240617_2017_2021判決書_無簡_sentence_opinion.csv', lineterminator='\n')
        sub_df = pd.read_csv('/workspace/data/CDB-2017-2021/6. Merged df/20240617_2017_2021判決書_無簡_paragraph_sub.csv', lineterminator='\n')
        fee_df = pd.read_csv('/workspace/data/CDB-2017-2021/6. Merged df/20240617_2017_2021判決書_無簡_paragraph_fee.csv', lineterminator='\n')

    db = {'fee': [fee_df, None, '心證'], 'sub': [sub_df, None, '涵攝'], 'opinion': [opinion_df, None, '見解']}

    # Remove unnecessary column
    main_basic_df.drop(columns=['JFULL'], inplace=True)
    preloaded_data['main_basic_df'] = main_basic_df
    preloaded_data['opinion_df'] = opinion_df
    preloaded_data['sub_df'] = sub_df
    preloaded_data['fee_df'] = fee_df
    preloaded_data['db'] = db
    yield
    # Clean up the models and release the resources
    preloaded_data.clear()


if on_server:

    domain_setting = {'host': '140.114.80.195', 'port': 6128}
    domain = f"http://{domain_setting['host']}:{domain_setting['port']}" + '/'
else:
    domain_setting = {'host': '127.0.0.1', 'port': 8000}
    # domain_setting = {'host': '127.0.0.1', 'port': 4000}
    # domain = f"http://{domain_setting['host']}:{domain_setting['port']}" + '/'
    domain = 'https://namely-fast-ocelot.ngrok-free.app' + '/'

if on_server:
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
else:
    app = FastAPI(lifespan=lifespan)

# origins = [
#     domain,
#     domain.split('/')[0] + ':8000',
#     domain + ':8000',
#     "http://localhost",
#     "http://localhost:8000",
#     '*'
# ]
if on_server:
    origins = [
        # '*',
        '140.114.80.195',
        'https://jerry914.github.io',
        'https://jerry914.github.io/ai-annotated-judgment-database',
    ]
else:
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

# Get method
@app.get("/api/search/all")
async def search_all(
    search_method: str,
    court_type: str | None = None, 
    jud_date: str | None = None, 
    case_num: str | None = None, 
    case_type: str | None = None, 
    basic_info: str | None = None, 
    syllabus: str | None = None, 
    opinion: str | None = None, 
    fee: str | None = None, 
    sub: str | None = None, 
    # jud_full: str | None = None 
    ):
    # jud_full
    jud = {'search_method':search_method, 'court_type':court_type, 'jud_date':jud_date, 'case_num': case_num, 'case_type': case_type, 'basic_info':basic_info, 'syllabus':syllabus, 'opinion':opinion, 'fee':fee, 'sub':sub}
    
    # Frontend request only necessary input, other unused variable use null
    query_dict = {key: jud[key] for key in jud.keys() & {'fee', 'sub', 'opinion'} if jud[key]!=None}
    condition_dict = {key: jud[key] for key in jud.keys() & {'court_type', 'jud_date', 'case_num', 'case_type', 'basic_info', 'syllabus'} if jud[key]!=None}
    search_method = jud['search_method']

    # Search juds with only conditions, category is not searched
    if query_dict=={}:
        result_dict = get_condition_filtered_dict(condition_dict, preloaded_data['main_basic_df'])
        res_df = result_dict['result_df']
        # Get every fee, sub, or opinion of each UID
        for query_type in preloaded_data['db'].keys():
            query_df = preloaded_data['db'][query_type][0]
            grouped_df = query_df.groupby('UID')['sentence'].agg(list).reset_index()
            res_df = res_df.merge(grouped_df[['UID', 'sentence']], on='UID', how='left')
            res_df['sentence'].fillna("無資料", inplace=True)
            res_df.rename(columns={'sentence': query_type}, inplace=True)
        res_df = res_df[~((res_df['fee']=='無資料') & (res_df['sub']=='無資料') & (res_df['opinion']=='無資料'))][:100]

    # Category is searched
    else:
        query_type, query_text = list(query_dict.items())[0]
        print(query_type, query_text, search_method)

        if search_method=='keyword':
            res_df = get_keywords_df(query_text, preloaded_data['db'][query_type][0])

        # Check if df has data
        if len(res_df)>0:
            merged_df = get_merged_df(preloaded_data['main_basic_df'], res_df, based_column='UID')
            result_dict = get_condition_filtered_dict(condition_dict, merged_df)
            res_df = result_dict['result_df'][:100]

            # # Append additional data to res_df
            # res_df['jud_date'] = res_df['jud_date'].astype(str)
            # res_df['jud_url'] = [f'https://judgment.judicial.gov.tw/FJUD/data.aspx?ty=JD&id={JID}' for JID in res_df['JID']]
            result_dict['available'].append([query_type, query_text])
        else:
            result_dict = {'result_df': res_df, 'available': [], 'unavailable': [[query_type, query_text]]}

        # The resulting sentences are in "sentence", rename them to "fee", "opinion" or "sub" 
        res_df.rename(columns={'sentence': query_type}, inplace=True)

    res_df = get_sorted_res(res_df, search_method)
    res_json = get_formatted_res(res_df, query_dict, result_dict)

    return res_json



def list_paginated_dict(data, page, size, request_params):
    request_url_prefix = domain + 'api/search' + "?" + "".join([f'{key}={request_params[key]}&' for key in request_params.keys() if request_params[key]!=None])
    total = len(data)
    if size == 0:
        total_pages = 0
    elif total is not None:
        total_pages = ceil(total / size)
    else:
        total_pages = None
    # if page > total_pages or page < 1:
    #     page = 1
    if page>0:
        start = (page - 1) * size
        end = start + size
        output_data = data[start:end]
    else:
        output_data = []
    # next = f"?page={page + 1}&size={size}" if (page + 1) <= total_pages else "null"
    next = request_url_prefix + f"page={page + 1}&size={size}" if page * size < total and page > -1 else "null"
    previous = request_url_prefix + f"page={page - 1}&size={size}" if (page - 1) >= 1 and (page - 1) * size <= total else "null"
    meta = {'page': page, 
            'size': size, 
            'next_page_url': next, 
            'previous_page_url': previous, 
            'total_pages': total_pages, 
            'total': total
            }
    return {'paginated_data': output_data, 'meta': meta}

class QueryInfo(BaseModel):
    query_type: str
    query_type_ch: str
    query_text: str

class ConditionInfo(BaseModel):
    available: List[List[str]]
    unavailable: List

class Summary(BaseModel):
    name: str
    unit: str
    count: int

class CaseData(BaseModel):
    # EID: Optional[int] | None = None
    JID: str
    opinion: Union[str, list] | None = None
    sub: Union[str, list] | None = None
    fee: Union[str, list] | None = None
    type: str | None = None
    UID: int
    case_num: str
    basic_info: str
    case_type: str
    court_type: str
    jud_date: str
    jud_url: str
    syllabus: str

class MetaData(BaseModel):
    page: int
    size: int
    next_page_url: str | None = None
    previous_page_url: Optional[str] | None = None
    total_pages: int
    total: int

class ResponseData(BaseModel):
    query_info: QueryInfo
    condition_info: ConditionInfo
    summary: List[Summary]
    data: List[CaseData] | None = None
    meta: MetaData

class ResponseModel(BaseModel):
    opinion: ResponseData | None = None
    sub: ResponseData | None = None
    fee: ResponseData | None = None
    jud: ResponseData | None = None

@app.get("/api/search", response_model=ResponseModel)
# 輸入所需的參數
async def search(
    search_method: str = Query("keyword"),
    page: int = Query(1),
    size: int = Query(2),
    court_type: str | None = None, 
    jud_date: str | None = None, 
    case_num: str | None = None,
    case_type: str | None = None,
    basic_info: str | None = None, 
    syllabus: str | None = None, 
    prediction: str | None = None, 
    # opinion: str | None = None, 
    # fee: str | None = None, 
    # sub: str | None = None, 
    ):
    # res_json = await search_all(search_method, court_type, jud_date, case_num, case_type, basic_info, syllabus, opinion, fee, sub)
    request_params = {'search_method':search_method, 'court_type':court_type, 'jud_date':jud_date, 'case_num': case_num, 'case_type': case_type, 'basic_info':basic_info, 'syllabus':syllabus, 'prediction':prediction}
    
    if prediction:
        output_dict = {'opinion': None, 'fee': None, 'sub': None}
        for i, prediction_category in enumerate(['opinion', 'fee', 'sub']):
            reset_prediction = [None, None, None]
            reset_prediction[i] = prediction
            opinion, fee, sub = reset_prediction
            res_json = await search_all(search_method, court_type, jud_date, case_num, case_type, basic_info, syllabus, opinion, fee, sub)
            
            paginated_dict = list_paginated_dict(res_json['data'], page, size, request_params)
            res_json['meta'] = paginated_dict['meta']
            res_json['data'] = paginated_dict['paginated_data']
            output_dict[prediction_category] = res_json
     
            
    else:
        reset_prediction = [None, None, None]
        opinion, fee, sub = reset_prediction
        res_json = await search_all(search_method, court_type, jud_date, case_num, case_type, basic_info, syllabus, opinion, fee, sub)
        paginated_dict = list_paginated_dict(res_json['data'], page, size, request_params)
        res_json['meta'] = paginated_dict['meta']
        res_json['data'] = paginated_dict['paginated_data']
        output_dict = {'jud': res_json}
    return output_dict


# Accept UID(int) or JID(str)
@app.get("/api/article/")
async def get_jud_from_id(id):
    main_basic_df = preloaded_data['main_basic_df']
    opinion_df = preloaded_data['opinion_df']
    sub_df = preloaded_data['sub_df']
    fee_df = preloaded_data['fee_df']
    target_column = 'UID'
    try:
        id = int(id)
    except:
        target_column = 'JID'
          
    tmp_df = main_basic_df.copy()
    tmp_df = tmp_df.fillna('無資料')
    tmp_df['jud_date'] = tmp_df['jud_date'].astype(str)
    jud = tmp_df[tmp_df[target_column]==id].iloc[0].to_dict()
    jud['opinion'] = opinion_df[opinion_df[target_column]==id]['sentence'].tolist()
    jud['sub'] = sub_df[sub_df[target_column]==id]['sentence'].tolist()
    jud['fee'] = fee_df[fee_df[target_column]==id]['sentence'].tolist()
    # return JUD_item(**jud)
    return jud



# @app.get('/testip')
# def index(real_ip: str = Header(None, alias='X-Real-IP')):
#     print(real_ip)
#     return real_ip

from fastapi.staticfiles import StaticFiles
if on_server:
    frontend_template_dir = '/home/lawrencechh/AIFR_CDB/frontend_deployment/20240620_dist'
else:
    frontend_template_dir = '/workspace/Projects/AIFR_CDB/frontend_deployment/20240620_dist'
    # # Redirect
    # from fastapi.responses import RedirectResponse
    # @app.get("/ai-annotated-judgment-database/")
    # async def redirect():
    #     response = RedirectResponse(url='/')
    #     return response

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
if __name__ == '__main__':
    # uvicorn.run('cdb_api:app', host="127.0.0.1", port=6128)
    # uvicorn.run('cdb_api:app', host="140.114.80.195", port=6128)
    print(domain_setting['host'])
    # Formal server
    uvicorn.run('cdb_api:app', host=domain_setting['host'], port=domain_setting['port'], forwarded_allow_ips='*')
    # uvicorn.run('cdb_api_new:app', host=domain_setting['host'], port=domain_setting['port'], forwarded_allow_ips='*')

# # Commands
# ngrok tunnel --label edge=edghts_2b8EWy9H5bevmDCX2UwiHmpksel http://localhost:8000
# CHH python cdb_api.py
# Server
# pm2 start cdb_api.py --name cdb

    