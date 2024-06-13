from typing import Any, Generic, Optional, Sequence, TypeVar

from fastapi import Query
from pydantic import BaseModel
from typing_extensions import Self

from fastapi_pagination.bases import AbstractPage, AbstractParams, RawParams
from typing import List, Dict

from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

from fastapi_pagination import Page, add_pagination, paginate

# class JSONAPIParams(BaseModel, AbstractParams):
#     page: int = Query(1, ge=1, description="Page number")
#     size: int = Query(10, ge=1, le=100, description="Page size")
#     def to_raw_params(self) -> RawParams:
#         return RawParams(limit=self.size, offset=self.size*(self.page-1))


# class JSONAPIPageInfoMeta(BaseModel):
#     total: int


# class JSONAPIPageMeta(BaseModel):
#     page: JSONAPIPageInfoMeta


# T = TypeVar("T")


# class JSONAPIPage(AbstractPage[T], Generic[T]):
#     data: Sequence[T]
#     meta: JSONAPIPageMeta

#     __params_type__ = JSONAPIParams

#     @classmethod
#     def create(
#         cls,
#         items: Sequence[T],
#         params: AbstractParams,
#         *,
#         total: Optional[int] = None,
#         **kwargs: Any,
#     ) -> Self:
#         assert isinstance(params, JSONAPIParams)
#         assert total is not None

#         return cls(
#             data=items,
#             meta={"page": {"total": total}},
#             **kwargs,
#         )



# app = FastAPI()


# class UserOut(BaseModel):
#     name: str
#     email: EmailStr


# users: List[UserOut] = [
#     UserOut(name="John Doe", email="john-doe@example.com"),
# ]


# @app.get("/users")
# def get_users(data, page, size) -> JSONAPIPage[UserOut]:
#     json_params = JSONAPIParams(size=size, page=page)
#     print(123)
#     print(paginate(data, params=json_params))
#     return paginate(data, params=json_params)

# @app.get("/paginate")
# async def paginate_user(page, size):
#     return await get_users(users, page, size)
# add_pagination(app)

from fastapi import FastAPI, Query, Depends



class JSONAPIParams(BaseModel, AbstractParams):
    page: int = Query(1, ge=1, description="Page number")
    size: int = Query(10, ge=1, le=100, description="Page size")
    def to_raw_params(self) -> RawParams:
        return RawParams(limit=self.size, offset=self.size*(self.page-1))


class JSONAPIPageInfoMeta(BaseModel):
    total: int


class JSONAPIPageMeta(BaseModel):
    page: JSONAPIPageInfoMeta


T = TypeVar("T")


class JSONAPIPage(AbstractPage[T], Generic[T]):
    data: Sequence[T]
    meta: dict

    __params_type__ = JSONAPIParams

    @classmethod
    def create(
        cls,
        items: Sequence[T],
        params: AbstractParams,
        *,
        total: Optional[int] = None,
        **kwargs: Any,
    ) -> Self:
        assert isinstance(params, JSONAPIParams)
        assert total is not None

        return cls(
            data=items,
            meta={"page": {"total": total}},
            **kwargs,
        )

app = FastAPI()

class UserOut(BaseModel):
    name: str
    email: EmailStr

users: List[UserOut] = [UserOut(name="John Doe", email="john-doe@example.com")]


@app.get("/users", response_model=JSONAPIPage[UserOut])
async def get_users(data, params: JSONAPIParams = Depends()) -> JSONAPIPage[UserOut]:

    return paginate(data, params=params)

# @app.get("/paginate", response_model=JSONAPIPage[UserOut])
@app.get("/paginate")
# async def paginate_user(page: int, size: int) -> JSONAPIPage[UserOut]:
async def paginate_user(page: int, size: int):
    params = JSONAPIParams(page=page, size=size)
    result_dict = {'1':"", '2':"", '3':""}
    for i in range(3):
        res = await get_users(users, params)
        res_dict = dict(res)
        result_dict[str(i)] = res_dict
        print(res_dict)
    return result_dict

add_pagination(app)


