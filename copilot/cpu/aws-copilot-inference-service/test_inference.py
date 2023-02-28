from aiohttp import web
import json
from app import create_app
import pytest


@pytest.fixture
def cli(loop, aiohttp_client):
    app = create_app()
    return loop.run_until_complete(aiohttp_client(app))


class TestObjectDetection:
    @pytest.mark.asyncio
    async def test_healthcheck(self, cli):
        resp = await cli.get('/')
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_inference(self, cli):
        with open('panda.png', 'rb') as f:
            image = f.read()

        resp = await cli.post('/predict', data={'image': image})

        assert resp.status == 200

        resp_json = json.loads(await resp.text())
        assert resp_json['preds'][0]['label'] == 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca'
