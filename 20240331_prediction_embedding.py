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
    r'/cluster/home/lawrencechh.cs/RAT/20240304/20230620_724td_opinion_-6b-pt-token-1024-3e-3_0818/checkpoint-1200', 
    # r'/workspace/111資料/20230819_1070td_ft_-6b-pt-token-1024-3e-3_0818_2/checkpoint-1200', 
    # r'/workspace/111資料/if_sub_train_09262209_-6b-pt-token1024-2e-2/checkpoint-1500'
                    ]

target_df_paths = [
    [r'/cluster/home/lawrencechh.cs/data/2017-2021/2017_2021判決書_無簡_sentence.csv', 0],
    # [r'/workspace/111資料/111判決書提出標註目標/0114_ft_paragraph_district_TARGET.csv', 0],
    # [r'/workspace/111資料/sub/0114_sub_paragraph_district_TARGET.csv', 0]
                   ]
output_csv_postfix = '_20240331_predicted_opinions.csv'
count_for_output = 10
# Add new column to output_csv, ex: prediction, embedding
added_csv_columns = ['prediction']
llm_path = r'THUDM/chatglm2-6b'


for target_df_index in range(len(target_df_paths)):
    merged_model = load_glm_checkpoint(checkpoint_paths[target_df_index], llm_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)

    path = target_df_paths[target_df_index][0]
    output_csv_path = path.split('.csv')
    output_csv_path = output_csv_path[0] + output_csv_postfix
    print('Load source csv...')
    df = pd.read_csv(path, encoding='utf-8-sig')
    tmp_df = pd.DataFrame(columns=list(df.columns)+added_csv_columns)
    print('output_csv_path: ', output_csv_path)
    if os.path.isfile(output_csv_path):

        output_df = pd.read_csv(output_csv_path, encoding="utf-8-sig", lineterminator='\n')
        output_df_length = len(output_df)
        print("output_csv existed, append data to existed file. \n output_csv length: {}".format(output_df_length))
        del output_df
        df = df[output_df_length:].copy()
    else:
        print("output_csv created.")
        tmp_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        output_df_length = target_df_paths[target_df_index][1]

    current_count = 0
    # for i in tqdm(range(len(df))):
    for df_index, row in tqdm(df.iterrows(), total=df.shape[0]):
        
        # tmp_data_dict = dict(df.iloc[i])
        # input_text = tmp_data_dict['sentence']
        input_text = row['sentence']

        # # Store embedding
        # embedding = get_mean_pooling_embedding(input_text, tokenizer, merged_model)
        # embedding = str(embedding.to("cpu").numpy().astype(np.float32).tolist())        
        # row['embedding'] = embedding
        
        # # Clear embedding
        # gc.collect()
        # embedding = None
        # torch.cuda.empty_cache()
        # gc.collect()


        # Store prediction
        if type(input_text)==str:
            input_text = input_text[0:1024]
        else:
            input_text = 'no data'
        
        prediction, history = merged_model.chat(tokenizer, input_text, history=[])
        row['prediction'] = prediction

        tmp_df.loc[len(tmp_df)] = row
        # tmp_df.to_csv(output_csv_path, mode="a", index=False, header=False, encoding='utf-8-sig', lineterminator='\n')

        current_count += 1
        if current_count>=count_for_output:
            tmp_df.to_csv(output_csv_path, mode="a", index=False, header=False, encoding="utf-8-sig", lineterminator='\n')
            tmp_df = pd.DataFrame(columns=list(df.columns)+added_csv_columns)
            current_count = 0

# Start from server
# pm2 start 20240331_prediction_embedding.py --name prediction --interpreter=/cluster/home/lawrencechh.cs/miniconda3/envs/chatglm2_env/bin/python   
        

    