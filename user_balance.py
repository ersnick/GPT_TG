from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select
import sqlalchemy

from orm_models import CommonBalance
import datetime


class Balance:
    def __init__(self, async_engine: sqlalchemy.ext.asyncio.engine.AsyncEngine,
                 user_telegram_id: int, limit, func_check_subs):
        self.user_telegram_id = user_telegram_id
        self.async_engine = async_engine
        self.limit = limit
        self.func_check_subs = func_check_subs

    async def get_balance(self):
        async with AsyncSession(self.async_engine) as async_session:
            # выбираем из таблицы соответствующего пользователя
            stmt = select(CommonBalance).where(CommonBalance.telegram_id == self.user_telegram_id)
            items = await async_session.execute(stmt)
            item = items.scalars().first()

            cur_dt = datetime.datetime.utcnow().date()
            # если такого пользователя еще нет, значит надо создать
            if item is None:
                item = CommonBalance()
                item.balance = await self.func_check_subs(self.user_telegram_id, self.limit)
                item.save_dt = cur_dt
                item.telegram_id = self.user_telegram_id
                user_balance = item.balance
                async_session.add(item)
                await async_session.commit()
                return user_balance
            # если есть такой пользователь
            else:
                user_balance = item.balance

                # если даты разные, значит новые сутки пошли и можно начислить баланс
                if cur_dt != item.save_dt:
                    item.save_dt = cur_dt
                    # проверка на подписку
                    limit = await self.func_check_subs(self.user_telegram_id, self.limit)
                    # максимальное ограничение в 100к токенов
                    if (item.balance + limit) > 100000:
                        limit = 100000 - item.balance
                    if limit < 0:
                        limit = 0
                    item.balance += limit
                    user_balance = item.balance

                    async_session.add(item)
                    await async_session.commit()
                    return user_balance
                else:
                    # если баланс достаточен для текущего запроса, то продолжаем
                    return user_balance

    async def update_balance(self, decrement_tokens):
        async with AsyncSession(self.async_engine) as async_session:
            # выбираем из таблицы соответствующего пользователя
            stmt = select(CommonBalance).where(CommonBalance.telegram_id == self.user_telegram_id)
            items = await async_session.execute(stmt)
            item = items.scalars().first()

            item.balance -= decrement_tokens

            async_session.add(item)
            await async_session.commit()
