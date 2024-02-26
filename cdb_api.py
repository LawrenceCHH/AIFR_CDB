from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

import os

import json
import pandas as pd
from tqdm import tqdm
import gc
import numpy as np


# Pagination
from typing import List
from fastapi_pagination import add_pagination, paginate
from fastapi_pagination.links import Page

# Custom Page
from typing import Any, Generic, Optional, Sequence, TypeVar

from fastapi import Query
from pydantic import BaseModel
from typing_extensions import Self

from fastapi_pagination.bases import AbstractPage, AbstractParams, RawParams

# Cross origin and allow origins
from fastapi.middleware.cors import CORSMiddleware

# Lifespan
from contextlib import asynccontextmanager
# Log
import logging

import ssl


def load_ori_glm2(llm_path="/workspace/LLM/chatglm2-6b"):
    config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True, output_hidden_states=True, output_attentions = True)
    model = AutoModel.from_pretrained(llm_path, config=config, trust_remote_code=True).quantize(4).half().cuda()
    model = model.eval()
    return model

def load_glm_checkpoint(checkpoint_path, llm_path):

    # 载入Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    # config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True, pre_seq_len=1024)

    # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True, pre_seq_len=1024, output_hidden_states=True, output_attentions = True)

    # # model = AutoModel.from_pretrained("/content/drive/MyDrive/share_p/20230416_chatglm6b_model", config=config, trust_remote_code=True)
    model = AutoModel.from_pretrained(llm_path, config=config, trust_remote_code=True)
    print("Parameter Merging!")
    prefix_state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))

    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    print("Model Quantizationing!")
    model = model.quantize(4)
    model = model.half().cuda()
    model.transformer.prefix_encoder.float()
    model = model.eval()
    print("Model Loaded!")
    return model

# def get_opinion_embedding(query_text, tokenizer, model):
#     query_text = query_text[:1500]
#     embedding = get_mean_pooling_embedding(query_text, KeywordSearchConfig.tokenizer, KeywordSearchConfig.merged_model)
#     return embedding

def get_mean_pooling_embedding(input_text, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, return_attention_mask=True)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # hidden state shape (batch_size, sequence_length, hidden_size)
    # (input_tokens_length, 1, 4096)
    last_hidden_state = outputs[2][-1]
    input_tokens_length = last_hidden_state.shape[0]
    # (1, 4096)
    embedding = torch.sum(last_hidden_state, 0)
    embedding = embedding[0] / input_tokens_length
    embedding = embedding.to("cpu").numpy().astype(np.float32)
    
    return embedding

def get_faiss_dataframe(query_embedding, query_index_flat, query_df, k=2048, mode='cpu', res_data_length=10000):
    # 1, 4096
    query_embedding = np.array([query_embedding])
    # print(query_embedding.shape)

    # if mode=='gpu':
    #     D, I = KeywordSearchConfig.gpu_index_flat.search(query_embedding, k) 
    #     result_indexes = I[0]
    #     result_distances = D[0]
    # elif mode=='cpu':
    # tmp, D, I = query_index_flat.range_search(query_embedding, 2000) 
    # output = sorted(zip(D, I))[:res_data_length]
    # output = np.array(output)
    # result_indexes = output[:, 1].tolist()
    # result_distances = output[:, 0].tolist()

    D, I = query_index_flat.search(query_embedding, 2000) 
    result_indexes = I[0]
    result_distances = D[0]

    # print('result_indexes: ', len(result_indexes))

    faiss_result = [result_indexes, result_distances]
    faiss_df = query_df.iloc[faiss_result[0]].copy()
    faiss_df['distance'] = faiss_result[1]
    faiss_df['order_index'] = range(0, len(faiss_df))
    faiss_df['show_unique_result'] = False
    # faiss_df['jud_url'] = [f'https://judgment.judicial.gov.tw/FJUD/data.aspx?ty=JD&id={JID}' for JID in faiss_df['JID']]
    return faiss_df

def get_filtered_res_data(faiss_df, condition="", filter_mode=0):
    # 篩選出符合faiss_result的結果
    # faiss_df = df.iloc[faiss_result[0]].copy()
    # faiss_df['distance'] = faiss_result[1]
    # faiss_df['order_index'] = range(0, len(faiss_df))
    # faiss_df['show_unique_result'] = False
    # faiss_df['jud_url'] = [f'https://judgment.judicial.gov.tw/FJUD/data.aspx?ty=JD&id={JID}' for JID in faiss_df['JID']]

    # 依欄位分組
    grouped_df = faiss_df.groupby("distance")
    # 取得分組後特定欄位的數目
    same_opinion_count = grouped_df['sentence'].count().reset_index(name='opinion_count')
    found_unique_data_num = 500
    grouped_keys = list(grouped_df.groups.keys())[:found_unique_data_num]

    # 依據distance篩選出前found_unique_data_num個unique的資料
    limited_df = faiss_df[faiss_df['distance'].isin(grouped_keys)].copy()

    # 計算同一個distance情況下共有幾個見解
    same_opinion_count = same_opinion_count[:found_unique_data_num]
    # 設定用於mapping的dict，key為distance，value為對應的opinion_count
    same_opinion_dict = same_opinion_count.set_index(['distance']).to_dict()['opinion_count']
    # print(same_opinion_dict)

    #依據不同的distance指派特定的opinion_count
    limited_df['same_distance_num'] = limited_df['distance'].map(same_opinion_dict)

    # 找出各組第一筆資料，設定其show_unique_result為True，讓前端根據此數值過濾掉其他多餘的資料，若需要全部資料呈現也可以保留
    limited_df['show_unique_result'].mask(limited_df['order_index'].isin(grouped_df.first()['order_index'].tolist()), True, inplace=True)


    # # 1 court
    # if filter_mode==1:
    #     limited_df = limited_df[limited_df['court']==condition]

    # res_data = limited_df.to_dict(orient='records')

    return limited_df
def get_merged_df(main_basic_df, category_df, based_column='UID'):
    # category_df['order_index'] = range(len(category_df))
    # category_df['jud_url'] = [f'https://judgment.judicial.gov.tw/FJUD/data.aspx?ty=JD&id={JID}' for JID in category_df['JID']]
    # category_df['same_distance_num'] = ""
    # category_df['show_unique_result'] = ""
    target_basic_df_columns = main_basic_df.columns.tolist()
    del target_basic_df_columns[target_basic_df_columns.index('JID')]
    merged_df = category_df.merge(main_basic_df[target_basic_df_columns], on=based_column, how='left')
    return merged_df
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
                    res_json['summary'][index]['count'] = len(tmp_category_df)


    # db = {'fee': [fee_df, fee_flat, '心證'], 'sub': [sub_df, sub_flat, '涵攝'], 'opinion': [opinion_df, opinion_flat, '見解']}
# 
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
            break
            # Use last available conditions
            # result_df = last_result_df
        else:
            result_dict['available'].append([target_column, query_text])
            last_result_df = result_df.copy()

    # Drop unwanted columns 
    if len(result_df) > 0:
        print(result_df.iloc[0])
        result_df = result_df.drop('jud_full', axis=1)

    result_dict['result_df'] = result_df
    return result_dict

llm_path = r"/workspace/LLM/chatglm2-6b"

preloaded_data = {

}
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Use vector or not
    load_LLM = False
    on_server = True

    # Load data
    logger = logging.getLogger("uvicorn.access")
    handler = logging.handlers.RotatingFileHandler("api.log",mode="a",maxBytes = 100*1024, backupCount = 3)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    if on_server:
        main_basic_df = pd.read_csv('~/workspace/111資料/db_loaded/20240120_main_basic.csv')
        opinion_df = pd.read_csv('~/workspace/111資料/db_loaded/20240120_category_opinion.csv')
        sub_df = pd.read_csv('~/workspace/111資料/db_loaded/20240225_category_sub.csv')
        fee_df = pd.read_csv('~/workspace/111資料/db_loaded/20240225_category_fee.csv')
    else:
        main_basic_df = pd.read_csv('/workspace/111資料/db_loaded/20240120_main_basic.csv')
        opinion_df = pd.read_csv('/workspace/111資料/db_loaded/20240120_category_opinion.csv')
        sub_df = pd.read_csv('/workspace/111資料/db_loaded/20240225_category_sub.csv')
        fee_df = pd.read_csv('/workspace/111資料/db_loaded/20240225_category_fee.csv')

    db = {'fee': [fee_df, None, '心證'], 'sub': [sub_df, None, '涵攝'], 'opinion': [opinion_df, None, '見解']}
    if load_LLM:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        import faiss

        ori_glm2_model = load_ori_glm2(llm_path)
        tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        if on_server:
            opinion_flat = faiss.read_index('~/workspace/111資料/db_loaded/0114_op_sentence_district_TARGET_embedding.bin')
            fee_flat = faiss.read_index('~/workspace/111資料/db_loaded/20240225_embedding_fee.bin')
            sub_flat = faiss.read_index('~/workspace/111資料/db_loaded/20240225_embedding_sub.bin')
        else:
            opinion_flat = faiss.read_index('/workspace/111資料/db_loaded/0114_op_sentence_district_TARGET_embedding.bin')
            fee_flat = faiss.read_index('/workspace/111資料/db_loaded/20240225_embedding_fee.bin')
            sub_flat = faiss.read_index('/workspace/111資料/db_loaded/20240225_embedding_sub.bin')
        preloaded_data['ori_glm2_model'] = ori_glm2_model
        preloaded_data['tokenizer'] = tokenizer
        preloaded_data['opinion_flat'] = opinion_flat
        preloaded_data['fee_flat'] = fee_flat
        preloaded_data['sub_flat'] = sub_flat
        db = {'fee': [fee_df, fee_flat, '心證'], 'sub': [sub_df, sub_flat, '涵攝'], 'opinion': [opinion_df, opinion_flat, '見解']}

    preloaded_data['main_basic_df'] = main_basic_df
    preloaded_data['opinion_df'] = opinion_df
    preloaded_data['sub_df'] = sub_df
    preloaded_data['fee_df'] = fee_df
    preloaded_data['db'] = db
    yield
    # Clean up the models and release the resources
    preloaded_data.clear()

domain = 'https://5737-140-114-83-23.ngrok-free.app' + '/'
# domain = '127.0.0.1:8000' + '/'
# domain = 'http://140.114.80.195:6127' + '/'

app = FastAPI(lifespan=lifespan)
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('/home/lawrencechh/AIFR_CDB/cert.pem', keyfile='/home/lawrencechh/AIFR_CDB/key.pem')
# app = FastAPI()



# origins = [
#     domain,
#     domain.split('/')[0] + ':8000',
#     domain + ':8000',
#     "http://localhost",
#     "http://localhost:8000",
#     '*'
# ]
origins = [
    '*'
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# class JUD(BaseModel):
#     court_type: str | None = None
#     jud_date: str | None = None
#     # case_kind and basic_info are merged to one paragraph
#     basic_info: str | None = None
    
#     syllabus: str | None = None
#     opinion: str | None = None
#     fee: str | None = None
#     sub: str | None = None
#     jud_full: str | None = None
#     # keyword or vector
#     search_method: str 
# @app.post("/api/search/all")
# async def search_all(jud: JUD):
   
#     # Frontend request only necessary input, other unused variable use null
#     query_dict = jud.model_dump(include=['fee', 'sub', 'opinion'], exclude_none=True, exclude_unset=True)
#     condition_dict = jud.model_dump(include=['court_type', 'jud_date', 'basic_info', 'syllabus', 'jud_full'], exclude_none=True, exclude_unset=True)
#     search_method = jud.search_method
#     query_type, query_text = list(query_dict.items())[0]
#     print(query_type, query_text, search_method)

#     if search_method=='keyword':
#         res_df = get_keywords_df(query_text, db[query_type][0])
#     elif search_method=='vector':
#         query_text = query_text[:1500]
#         query_embedding = get_mean_pooling_embedding(query_text, tokenizer, ori_glm2_model)
#         faiss_df = get_faiss_dataframe(query_embedding=query_embedding, query_index_flat=db[query_type][1], query_df=db[query_type][0])
#         res_df = get_filtered_res_data(faiss_df, condition="", filter_mode=0)

#     merged_df = get_merged_df(main_basic_df, res_df, based_column='UID')
#     for key, value in condition_dict.items():
#         print(key, value)
#         merged_df = get_condition_filtered_dict(value, merged_df, key)
#     res_json = get_formatted_res(res_df=merged_df, query_text=query_text, query_type=query_type)
#     return res_json
#     # print(list(query_dict.items()))
#     # print(list(condition_dict.items()))
#     # print(list(jud.model_dump().items()))

# Get method
@app.get("/api/search/all")
async def search_all(
    search_method: str,
    court_type: str | None = None, 
    jud_date: str | None = None, 
    basic_info: str | None = None, 
    syllabus: str | None = None, 
    opinion: str | None = None, 
    fee: str | None = None, 
    sub: str | None = None, 
    jud_full: str | None = None 
    ):
    # jud_full
    jud = {'search_method':search_method, 'court_type':court_type, 'jud_date':jud_date, 'basic_info':basic_info, 'syllabus':syllabus, 'opinion':opinion, 'fee':fee, 'sub':sub, 'jud_full': jud_full}
    # jud = {'search_method':search_method, 'court_type':court_type, 'jud_date':jud_date, 'basic_info':basic_info, 'syllabus':syllabus, 'opinion':opinion, 'fee':fee, 'sub':sub}
    # Frontend request only necessary input, other unused variable use null
    query_dict = {key: jud[key] for key in jud.keys() & {'fee', 'sub', 'opinion'} if jud[key]!=None}
    condition_dict = {key: jud[key] for key in jud.keys() & {'court_type', 'jud_date', 'basic_info', 'syllabus', 'jud_full'} if jud[key]!=None}
    search_method = jud['search_method']

    # Search juds with only conditions
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
        # res_df = res_df[~res_df['fee']=='無資料' and res_df['sub']=='無資料' and res_df['opinion']=='無資料']
        res_df = res_df[~((res_df['fee']=='無資料') & (res_df['sub']=='無資料') & (res_df['opinion']=='無資料'))]
   
    # Vector search
    else:
        query_type, query_text = list(query_dict.items())[0]
        print(query_type, query_text, search_method)

        if search_method=='keyword':
            res_df = get_keywords_df(query_text, preloaded_data['db'][query_type][0])
        elif search_method=='vector':
            query_embedding = get_mean_pooling_embedding(query_text[:1500], preloaded_data['tokenizer'], preloaded_data['ori_glm2_model'])
            faiss_df = get_faiss_dataframe(query_embedding=query_embedding, query_index_flat=preloaded_data['db'][query_type][1], query_df=preloaded_data['db'][query_type][0])
            res_df = get_filtered_res_data(faiss_df, condition="", filter_mode=0)

        if len(res_df)>0:
            merged_df = get_merged_df(preloaded_data['main_basic_df'], res_df, based_column='UID')
            # for key, value in condition_dict.items():
            #     print(key, value)
            #     merged_df = get_condition_filtered_result(value, merged_df, key)
            result_dict = get_condition_filtered_dict(condition_dict, merged_df)
            res_df = result_dict['result_df']
            res_df['jud_date'] = res_df['jud_date'].astype(str)
            res_df['jud_url'] = [f'https://judgment.judicial.gov.tw/FJUD/data.aspx?ty=JD&id={JID}' for JID in res_df['JID']]
            result_dict['available'].append([query_type, query_text])
        else:
            result_dict = {'result_df': res_df, 'available': [], 'unavailable': [[query_type, query_text]]}
        # The resulting sentences are in "sentence" or "fee", "opinion" or "sub" 
        res_df.rename(columns={'sentence': query_type}, inplace=True)

    res_json = get_formatted_res(res_df, query_dict, result_dict)

    return res_json

from typing import Any, Generic, Optional, Sequence, TypeVar

from fastapi import Query
from pydantic import BaseModel
from typing_extensions import Self

from fastapi_pagination.bases import AbstractPage, AbstractParams, RawParams
from math import ceil

class JSONAPIParams(BaseModel, AbstractParams):
    page: int = Query(1, ge=1, description="Page number")
    size: int = Query(10, ge=1, le=100, description="Page size")
    def to_raw_params(self) -> RawParams:
        return RawParams(limit=self.size, offset=self.size*(self.page-1))


# class JSONAPIPageInfoMeta(BaseModel):
#     total: int

# class JSONAPIPageMeta(BaseModel):
#     page: JSONAPIPageInfoMeta

T = TypeVar("T")


class JSONAPIPage(AbstractPage[T], Generic[T]):
    meta: dict
    query_info: dict
    condition_info: dict
    summary: list
    data: Sequence[T]
    

    __params_type__ = JSONAPIParams

    @classmethod
    def create(
        cls,
        data: Sequence[T],
        params: AbstractParams,
        *,
        total: Optional[int] = None,
        query_info: dict,
        condition_info: dict,
        summary: dict,
        request_url_prefix: str,
        **kwargs: Any,
    ) -> Self:
        assert isinstance(params, JSONAPIParams)
        assert total is not None
        size = params.size if params.size is not None else total
        page = params.page if params.page is not None else 1

        if size == 0:
            total_pages = 0
        elif total is not None:
            total_pages = ceil(total / size)
        else:
            total_pages = None
            
        # next = f"?page={page + 1}&size={size}" if (page + 1) <= total_pages else "null"
        next = request_url_prefix + f"page={page + 1}&size={size}" if page * size < total else "null"
        previous = request_url_prefix + f"page={page - 1}&size={size}" if (page - 1) >= 1 else "null"
        # next = request_url_prefix + f"page={page + 1}&size={size}" if page * size < total else request_url_prefix + f"page={page}&size={size}"
        # previous = request_url_prefix + f"page={page - 1}&size={size}" if (page - 1) >= 1 else request_url_prefix + f"page={page}&size={size}"
        # first={"page": 1},
        # last={"page": ceil(total / size) if total > 0 and size > 0 else 1},
 

        return cls(
            # meta={'page': page, 'total':total, 'page':page, 'size':size, 'next' : next, 'previous' : previous, 'total_pages' : total_pages},
            # meta={'page': page, 'total':total, 'page':page, 'size':size, 'next_page_url' : next, 'previous_page_url' : previous, 'total_pages' : total_pages},
            meta={
                'page': page, 
                'size':size, 
                'next_page_url' : next, 
                'previous_page_url' : previous, 
                'total_pages' : total_pages,
                'total':total, 
            },

            query_info=query_info,
            condition_info=condition_info,
            summary=summary,
            data=data,
            **kwargs,
        )
# Get method
class JUD_item(BaseModel):
    
    EID: int | None = None
    UID: int 
    JID: str 
    court_type: str 
    jud_date: str 
    # case_kind and basic_info are merged to one paragraph
    basic_info: str | None = None
    syllabus: str | None = None
    sentence: str | None = None
    fee: List | str | None = None
    opinion: List | str | None = None
    sub: List | str | None = None
    # jud_full: str 
    jud_url: str 
    type: str | None = None
    distance: float | None = None
    order_index: int | None = None
    show_unique_result: bool | None = None
    same_distance_num: int | None = None


@app.get("/api/search")
async def search(
    search_method: str,
    page: str,
    size: str,
    court_type: str | None = None, 
    jud_date: str | None = None, 
    basic_info: str | None = None, 
    syllabus: str | None = None, 
    opinion: str | None = None, 
    fee: str | None = None, 
    sub: str | None = None, 
    # jud_full: str | None = None, 
    )-> JSONAPIPage[JUD_item]:
    # jud_full
    # res_json = await search_all(search_method, court_type, jud_date, basic_info, syllabus, opinion, fee, sub, jud_full)
    # request_params = {'search_method':search_method, 'court_type':court_type, 'jud_date':jud_date, 'basic_info':basic_info, 'syllabus':syllabus, 'opinion':opinion, 'fee':fee, 'sub':sub, 'jud_full': jud_full}
    res_json = await search_all(search_method, court_type, jud_date, basic_info, syllabus, opinion, fee, sub)
    request_params = {'search_method':search_method, 'court_type':court_type, 'jud_date':jud_date, 'basic_info':basic_info, 'syllabus':syllabus, 'opinion':opinion, 'fee':fee, 'sub':sub}
    request_url_prefix = domain + 'api/search/' + "?" + "".join([f'{key}={request_params[key]}&' for key in request_params.keys() if request_params[key]!=None])
    paged_res_json = paginate(res_json["data"], additional_data={'query_info': res_json['query_info'], 'condition_info': res_json['condition_info'], 'summary': res_json['summary'], 'request_url_prefix': request_url_prefix})

    return paged_res_json

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
add_pagination(app)
