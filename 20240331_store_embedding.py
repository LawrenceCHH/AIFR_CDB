import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import json
import pandas as pd
from tqdm import tqdm
import gc
import numpy as np


def load_ori_glm1(llm_path="/workspace/LLM/chatglm-6b"):
    # config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, pre_seq_len=1024, output_hidden_states=True, output_attentions = True)
    # config = AutoConfig.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True, output_hidden_states=True, output_attentions = True)
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", config=config, trust_remote_code=True).half().cuda()
    config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True, output_hidden_states=True, output_attentions = True)
    # model = AutoModel.from_pretrained(llm_path, config=config, trust_remote_code=True).half().cuda()
    model = AutoModel.from_pretrained(llm_path, config=config, trust_remote_code=True).quantize(4).half().cuda()
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    return model
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

def get_mean_pooling_embedding(input_text, tokenizer, model):
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, return_attention_mask=True, truncation=True, max_length=2048)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    # print(len(inputs['input_ids'][0]))

    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()
        outputs = model(**inputs)
    # hidden state shape (batch_size, sequence_length, hidden_size)
    # (input_tokens_length, 1, 4096)
    last_hidden_state = outputs[2][-1]
    input_tokens_length = last_hidden_state.shape[0]
    # (1, 4096)
    embedding = torch.sum(last_hidden_state, 0)
    embedding = embedding[0] / input_tokens_length
    torch.cuda.empty_cache()
    gc.collect()
    return embedding


checkpoint_paths = [
    r'/workspace/data/CDB_models/20240304 110juds models/20230620_724td_opinion_-6b-pt-token-1024-3e-3_0818/checkpoint-1200', 
    # r'/workspace/111資料/20230819_1070td_ft_-6b-pt-token-1024-3e-3_0818_2/checkpoint-1200', 
    # r'/workspace/111資料/if_sub_train_09262209_-6b-pt-token1024-2e-2/checkpoint-1500'
                    ]

target_df_paths = [
    [r'/workspace/data/CDB_20240304_110juds/20240320_final_merged/20240322_110_category_opinion.csv', 0],
    # [r'/workspace/111資料/111判決書提出標註目標/0114_ft_paragraph_district_TARGET.csv', 0],
    # [r'/workspace/111資料/sub/0114_sub_paragraph_district_TARGET.csv', 0]
                   ]


llm_path = r'/workspace/LLM/chatglm2-6b'
for df_index in range(len(target_df_paths)):
    merged_model = load_glm_checkpoint(checkpoint_paths[df_index], llm_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)

    path = target_df_paths[df_index][0]
    output_csv_path = path.split('.csv')
    output_csv_path = output_csv_path[0] + '_embedding.csv'
    print('output_csv_path: ', output_csv_path)
    df = pd.read_csv(path, encoding='utf-8-sig')
    output_df = pd.DataFrame(columns=list(df.columns)+['embedding'])
    if os.path.isfile(output_csv_path) == False:

        print("output_csv created.")
        output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        embedding_df_length = target_df_paths[df_index][1]
    else:
        print("output_csv existed, append data to existed file.")
        embedding_df = pd.read_csv(output_csv_path, encoding='utf-8-sig')
        embedding_df_length = len(embedding_df)
        df = df[embedding_df_length:].copy()

    stop_num = 0
    for i in tqdm(range(len(df))):
        tmp_data_dict = dict(df.iloc[i])
        input_text = tmp_data_dict['sentence']
        embedding = get_mean_pooling_embedding(input_text, tokenizer, merged_model)
        embedding = str(embedding.to("cpu").numpy().astype(np.float32).tolist())        
        tmp_data_dict['embedding'] = embedding
        
        output_df.loc[0] = tmp_data_dict
        output_df.to_csv(output_csv_path, mode="a", index=False, header=False, encoding='utf-8-sig')

        gc.collect()
        embedding = None
        torch.cuda.empty_cache()
        gc.collect()
        

    