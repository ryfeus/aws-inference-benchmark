import aiohttp
from aiohttp import web
import onnxruntime as ort
import numpy as np
import io
import asyncio
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

sess = ort.InferenceSession('efficientnet-lite4-11.onnx')
with open("labels_map.txt") as f:
    label_map = json.load(f)

routes = web.RouteTableDef()
executor = ThreadPoolExecutor()


@routes.get('/')
async def hello(request):
    return web.Response()


def image_preprocess(image):
    image = image.resize((224, 224), resample=Image.BICUBIC)
    image = np.array(image).astype('float32')
    image = image[:, :, :3]
    image = np.expand_dims(image, axis=0)
    image -= [127.0, 127.0, 127.0]
    image /= [128.0, 128.0, 128.0]
    return image


# Set up the HTTP server
@routes.post('/predict')
async def predict(request):
    try:
        image_data = await request.content.read()
        image = Image.open(io.BytesIO(image_data))

        image = image_preprocess(image)

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(executor, sess.run, [output_name], {input_name: image})

        result = reversed(results[0][0].argsort()[-5:])
        preds = []
        for r in result:
            preds.append({"label": label_map[str(r)], "score": str(results[0][0][r])})

        resp = {
            'preds': preds
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
