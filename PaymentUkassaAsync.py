import aiohttp
from loguru import logger
import uuid


class PaymentsUkassa:
    base_path = 'https://api.yookassa.ru/v3/payments/'

    def __init__(self, shopID, secretKey, timeout=150):
        self.shopID = shopID
        self.secretKey = secretKey
        self.timeout = timeout

    async def find_one(self, payment_id):
        async with aiohttp.ClientSession(auth=aiohttp.BasicAuth(login=self.shopID, password=self.secretKey)) as session:
            #logger.debug(f'SEND yookassa request find payment: {payment_id}')
            try:
                async with session.get(self.base_path+payment_id, timeout=self.timeout) as resp:
                    response = await resp.json()
                    #logger.debug(f'RECEIVE yookassa request find payment: {payment_id}')
                    return response
            except Exception as e:
                logger.error(f'yookassa request find payment exception: {e}')
                return None

    async def create(self, payload):
        async with aiohttp.ClientSession(auth=aiohttp.BasicAuth(login=self.shopID, password=self.secretKey)) as session:
            idempotency_key = str(uuid.uuid4())
            logger.debug(f'SEND yookassa request create: {idempotency_key}')
            headers = {
                'Idempotence-Key': idempotency_key
            }
            try:
                async with session.post(self.base_path, json=payload, headers=headers, timeout=self.timeout) as resp:
                    response = await resp.json()
                    logger.debug(f'RECEIVE yookassa request create: {idempotency_key}')
                    return response
            except Exception as e:
                logger.error(f'yookassa request create exception: {e}')
                return None

