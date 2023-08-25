from aiohttp import web
import json
from llama_cpp import Llama


llm = Llama(model_path="model/llama-2-7b.ggmlv3.q2_K.bin")
routes = web.RouteTableDef()


@routes.get("/")
async def healthcheck(request):
    return web.Response()


@routes.post("/predict")
async def predict(request):
    try:
        req = await request.json()
        instruction = req.get("instruction", "What is the capital of Spain?")
        response = llm(instruction, max_tokens=32, stop=["Q:", "\n"], echo=True)
        resp = {"result": response}
        return web.Response(text=json.dumps(resp))
    except Exception as e:
        return web.Response(text=str(e), status=500)


def create_app():
    app = web.Application()
    app.add_routes(routes)
    return app


async def create_gunicorn_app():
    return create_app()


if __name__ == "__main__":
    web.run_app(create_app(), port=8080)
