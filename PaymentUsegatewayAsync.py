import aiohttp
from loguru import logger
import uuid
import requests


class PaymentsUsegateway:
    base_path = 'https://api.usegateway.net/v1/payments/'

    def __init__(self, api_key, timeout=150):
        self.api_key = api_key
        self.timeout = timeout

    async def create(self, payload):
        async with aiohttp.ClientSession() as session:
            logger.debug(f"SEND gateway request create: {payload['metadata']['uuid4']}")
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'x-api-key': self.api_key
            }
            try:
                async with session.post(self.base_path, json=payload, headers=headers, timeout=self.timeout) as resp:
                    response = await resp.json()
                    logger.debug(f"RECEIVE gateway request create: {payload['metadata']['uuid4']}")
                    return response
            except Exception as e:
                logger.error(f"gateway request create {payload['metadata']['uuid4']} exception: {e}")
                return None

    async def find_one(self, payment_id):
        async with aiohttp.ClientSession() as session:
            #logger.debug(f'SEND gateway request find payment: {payment_id}')
            headers = {
                'Accept': 'application/json',
                'x-api-key': self.api_key
            }
            try:
                async with session.get(self.base_path+payment_id+'/', headers=headers, timeout=self.timeout) as resp:
                    response = await resp.json()
                    #logger.debug(f'RECEIVE gateway request find payment: {payment_id}')
                    return response
            except Exception as e:
                logger.error(f'gateway request find payment exception: {e}')
                return None

