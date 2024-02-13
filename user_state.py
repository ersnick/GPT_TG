import asyncio
from aredis import StrictRedis


class UserState:
    def __init__(self, redis,
                 user_telegram_id: int):
        self.user_telegram_id = user_telegram_id
        self.redis = redis

    async def set_state(self, key, value):
        key = f'{self.user_telegram_id}_{key}'
        await self.redis.set(key, value)

    async def get_state(self, key):
        key = f'{self.user_telegram_id}_{key}'
        res = None
        if await self.redis.exists(key) is True:
            res = await self.redis.get(key)
        return res
