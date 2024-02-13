from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select
import sqlalchemy

from orm_models import User, Message, StartHistory
import datetime

class History:
    def __init__(self, async_engine: sqlalchemy.ext.asyncio.engine.AsyncEngine,
                 user_telegram_id: int, username: str,
                 bot_id: int):
        self.user_telegram_id = user_telegram_id
        self.async_engine = async_engine
        self.username = username
        self.bot_id = bot_id

    async def get_history(self):
        async with AsyncSession(self.async_engine) as async_session:
            stmt = select(User.id, Message.id, Message.datetime) \
                .join(StartHistory, Message.id == StartHistory.msg_id) \
                .join(User, StartHistory.user_id == User.id) \
                .where(User.telegram_id == self.user_telegram_id) \
                .order_by(Message.datetime.desc()).limit(1).subquery()
            stmt2 = select(Message) \
                .join(stmt, (Message.user_id_src == stmt.c.id) | (Message.user_id_dest == stmt.c.id)) \
                .where(stmt.c.datetime <= Message.datetime)
            items = await async_session.execute(stmt2)
            items = items.scalars().all()

            # возвращаем сообщения не считая стартового
            return items[1:]

    async def add_msg(self, text: str, timestamp: datetime.datetime, is_human: bool, is_sum=False):
        user_id = 0
        msg_id = 0
        async with AsyncSession(self.async_engine) as async_session:
            stmt = select(User).where(User.telegram_id == self.user_telegram_id)
            item = await async_session.execute(stmt)
            user = item.scalars().first()
            if user is None:
                user = User()
                user.username = self.username
                user.telegram_id = self.user_telegram_id
                async_session.add(user)
                await async_session.commit()

                item = await async_session.execute(select(User).where(User.telegram_id == self.user_telegram_id))
                user = item.scalars().first()

            user_id = user.id

            msg = Message()
            msg.datetime = timestamp
            msg.text = text
            msg.is_sum = is_sum
            if is_sum:
                msg.user_id_src = user.id
                msg.user_id_dest = self.bot_id
            else:
                if is_human:
                    msg.user_id_src = user.id
                    msg.user_id_dest = self.bot_id
                else:
                    msg.user_id_src = self.bot_id
                    msg.user_id_dest = user.id

            async_session.add(msg)
            await async_session.commit()

            item = await async_session.execute(select(Message).where(Message.datetime == timestamp))
            msg = item.scalars().first()

            msg_id = msg.id
        return msg_id, user_id

    async def clear_history(self):
        # добавляем стартовое сообщение и юзера, если такого еще нет
        msg_id, user_id = await self.add_msg('Начало', datetime.datetime.now(), True)
        async with AsyncSession(self.async_engine) as async_session:
            stmt = select(StartHistory).where(StartHistory.user_id == user_id)
            item = await async_session.execute(stmt)
            start_hist = item.scalars().first()
            if start_hist is None:
                start_hist = StartHistory()
                start_hist.user_id = user_id

            start_hist.msg_id = msg_id
            async_session.add(start_hist)
            await async_session.commit()
