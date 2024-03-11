from fastapi.responses import FileResponse

@app.get("/app")
def read_index():
    return FileResponse("./app.html")