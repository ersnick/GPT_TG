FROM python:3.9-buster

ENV TZ=Asia/Yekaterinburg
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV DEBIAN_FRONTEND noninteractive

RUN pip install sqlalchemy[asyncio]
RUN pip install asyncpg
RUN pip install requests
RUN pip install python-decouple
RUN pip install loguru==0.6.0
RUN pip install transformers==4.25.1
RUN pip install aiogram==2.24
RUN pip install aiohttp==3.8.3
RUN pip install pydantic
RUN pip install aredis
RUN pip install tiktoken
RUN pip install tenacity

WORKDIR /home
RUN mkdir /home/static_texts
COPY main.py /home
COPY openai.py /home
COPY history.py /home
COPY keyboard.py /home
COPY orm_models.py /home
COPY .env /home
COPY user_balance.py /home
COPY user_state.py /home
COPY PaymentUkassaAsync.py /home
COPY PaymentUsegatewayAsync.py /home

ENTRYPOINT python main.py
