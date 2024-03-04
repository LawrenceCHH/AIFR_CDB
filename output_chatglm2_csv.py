import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import json
import pandas as pd
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]='0'#調整cuda號碼


def load_glm_checkpoint(checkpoint_path):
    # checkpoint_path = r"F:\Lawrence\ChatGLM-6B\ptuning\output\20230614-hackson-2500-6b-pt-1024-2e-2\checkpoint-300"
    llm_path = "F:\\ChatGLM2-6B-main\\LLM"
    # 载入Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    # config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True, pre_seq_len=1024)
    
    # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    config = AutoConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, pre_seq_len=1024)

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

def read_json(json_path):
    with open(json_path, "r", encoding="utf-8-sig") as json_file:
        # json_list = json_file.readlines()
        json_list = [json.loads(line) for line in json_file]
        keys = [key for key in json_list[0].keys()]
        print(f"json length:{len(json_list)}\njson keys:{keys}")
    return json_list

start_point=0
end_point=0
prediction_csv_path = r"F:\110判決書_無簡\110juds_nosimple_paragraph.csv" #改
# output_df = pd.DataFrame(columns=csv_columns)
# for index, line in enumerate(test_list):
#     output_df.loc[len(output_df)] = [test_list[index]["content"], test_list[index]["summary"]]
# output_df.to_csv(prediction_csv_path, index=False, encoding="utf-8-sig")
checkpoint_path = r"F:\ChatGLM2-6B-main\ptuning\output\0303_PARAGRAPH_sub\\checkpoint-{}"  #改
checkpoint_nums = [str(num) for num in range(500,550,50)]    #留意
#改!!!!!!!!!否則覆蓋!!!!!!!!!
#改!!!!!!!!!否則覆蓋!!!!!!!!!
#改!!!!!!!!!否則覆蓋!!!!!!!!!
output_csv_path = r"F:\110判決書_無簡\0304_110判決書_無簡_paragraph_sub.csv"   #改
#改!!!!!!!!!否則覆蓋!!!!!!!!!
#改!!!!!!!!!否則覆蓋!!!!!!!!!
#改!!!!!!!!!否則覆蓋!!!!!!!!!
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
count_for_output = 50

# -----------------------------

prediction_df = pd.read_csv(prediction_csv_path)[start_point:]#[start_point:]#[1615614:1822495]  #改數量/區間0   [0:1000] 之類的 到1822494
# Create initial csv
source_csv_columns = list(prediction_df.keys()) + checkpoint_nums
pd.DataFrame(columns=source_csv_columns).to_csv(output_csv_path, index=False, encoding="utf-8-sig")

# 20230621 modified - periodically save
from transformers import AutoTokenizer
#import pandas as pd
from tqdm import tqdm

# 調整參數區
# # Write initial contents from testing.json 
# csv_columns = ["sentence", "label"]


current_count = 0
tmp_df = pd.DataFrame(columns=source_csv_columns)
for str_num in checkpoint_nums:
    
    print(f"Checkpoint-{str_num} loading...")
    merged_model = load_glm_checkpoint(checkpoint_path.format(str_num))
    
    print(f"Checkpoint-{str_num} Predicting...")
    count=0
    for data_index in tqdm(range(len(prediction_df))):
        count+=1
        input_text = str(prediction_df.iloc[data_index]["sentence"])#原sentence for 主文庫改input_text=input_text[0:1000]   #改
        input_text=input_text[0:1024].replace(r"\n","").replace(r"\r","").replace(r"\u3000","")
        

        prediction, history = merged_model.chat(tokenizer, input_text, history=[])
        # prediction_df.loc[data_index, str_num] = prediction        
        tmp_df.loc[len(tmp_df)] = prediction_df.iloc[data_index]
        tmp_df.loc[len(tmp_df)-1, str_num] = prediction
        
        current_count += 1
        if current_count>=count_for_output:
            #tmp_df.to_csv(output_csv_path, mode="a", index=False, header=False, encoding="utf-8-sig")
            tmp_df.to_csv(output_csv_path, mode="a", index=False, header=False, encoding="utf-8-sig")
            tmp_df = pd.DataFrame(columns=source_csv_columns)
            current_count = 0
    #tmp_df.to_csv(output_csv_path, mode="a", index=False, header=False, encoding="utf-8-sig")
    tmp_df.to_csv(output_csv_path, mode="a", index=False, header=False, encoding="utf-8-sig")
    tmp_df = pd.DataFrame(columns=source_csv_columns)
    current_count = 0
    

    # prediction_df.to_csv(r"F:\Lawrence\ChatGLM-6B\ptuning\datasets\\槍砲案一人一罪pre.csv", index=False, encoding="utf-8-sig")
    print(f"Checkpoint-{str_num} Saved!\n------------\n")