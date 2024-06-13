from fastapi import FastAPI, HTTPException, Query
from typing import Optional, Dict, Any
from pydantic import BaseModel

app = FastAPI()

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

# 分頁獲取用戶
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

# 如果你想要運行此應用程序，你需要使用以下命令：
# uvicorn your_script_name:app --reload
