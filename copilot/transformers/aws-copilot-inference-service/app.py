from aiohttp import web
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import json

checkpoint = "./model/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,device_map='auto',torch_dtype=torch.float32)
pipe = pipeline('text2text-generation',model=base_model, tokenizer=tokenizer, max_length=512, do_sample=True,
                temperature=0.3, top_p=0.95)

routes = web.RouteTableDef()


@routes.get('/')
async def healthcheck(request):
    return web.Response()


@routes.post('/predict')
async def predict(request):
    try:
        req = await request.json()
        instruction = req.get('instruction', 'What is the capital of Spain?')
        generated_text = pipe(instruction)
        response = ''
        for text in generated_text:
            response += text['generated_text']
        resp = {
            'result': response
        }
        return web.Response(text=json.dumps(resp))
    except Exception as e:
        return web.Response(text=str(e), status=500)


def create_app():
    app = web.Application()
    app.add_routes(routes)
    return app


async def create_gunicorn_app():
    return create_app()


if __name__ == '__main__':
    web.run_app(create_app(), port=8080)
