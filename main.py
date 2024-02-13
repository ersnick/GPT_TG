from tenacity import retry, stop_after_attempt
import copy
import datetime

import asyncio
import uuid

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types.message import ContentType
from aiogram.utils.deep_linking import get_start_link, decode_payload

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import select

from orm_models import User, CommonBalance, CommonPayment

from loguru import logger
from decouple import config

from keyboard import BotKeyboard, InlineBotKeyboard

from history import History
from openai import Tokenizer

import aiohttp
import json
from aiogram.types import ChatActions
from user_balance import Balance
from user_state import UserState

from aredis import StrictRedis

import re

from PaymentUkassaAsync import PaymentsUkassa
from PaymentUsegatewayAsync import PaymentsUsegateway

def retry_callback(retry_state):
    return {'success': False, 'id':'timeout', 'error': 'exception in parsing api'}

@retry(retry_error_callback=retry_callback, stop=stop_after_attempt(2))
async def send_api(dp, id, username, prompt, temp, top_p, max_tokens):
    dp.cur_openai_api_text = (dp.cur_openai_api_text + 1) % len(dp.openai_services_text)
    async with aiohttp.ClientSession() as session:
        payload = {
            "prompt": prompt,
            "temp": temp,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "presence_penalty": float(config('PRESENCE_PENALTY')),
            "frequency_penalty": float(config('FREQUENCY_PENALTY')),
            "best_of": int(config('BEST_OF')),
        }
        try:
            cur_service = dp.openai_services_text[dp.cur_openai_api_text]
            logger.debug(f"request is send to openai_api {cur_service['host']}:{cur_service['port']} user id: {id} prompt: {prompt}")
            async with session.post(f"http://{cur_service['host']}:{cur_service['port']}/completions", json=payload, timeout=250) as resp:
                response = await resp.json()
                logger.debug(f'response from openai api is received: {response}')
                if 'choices' in response:
                    response['success'] = True
                    response["prompt"] = response["choices"][0]['text'].replace('<|endofstatement|>', '').split('Human:')[0]
                else:
                    response['success'] = False
                return response
        except Exception as e:
            logger.error(f'exception in parsing openai api: {e}')
            raise Exception
        
@retry(retry_error_callback=retry_callback, stop=stop_after_attempt(2))
async def send_api_chat(dp, id, username, messages, temperature, top_p, max_tokens):
    dp.cur_openai_api_text = (dp.cur_openai_api_text + 1) % len(dp.openai_services_text)
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "presence_penalty": int(config('PRESENCE_PENALTY')),
            "frequency_penalty": int(config('FREQUENCY_PENALTY')),
        }
        try:
            cur_service = dp.openai_services_text[dp.cur_openai_api_text]
            logger.debug(f"request is send to openai_api {cur_service['host']}:{cur_service['port']} user id: {id} messages: {messages}")
            async with session.post(f"http://{cur_service['host']}:{cur_service['port']}/chat/completions", json=payload, timeout=250) as resp:
                response = await resp.json()
                logger.debug(f'response from openai api is received: {response}')
                if 'choices' in response:
                    response['success'] = True
                    response["prompt"] = response["choices"][0]['message']['content']
                else:
                    response['success'] = False
                return response
        except Exception as e:
            logger.error(f'exception in parsing openai api: {e}')
            raise Exception
        
@retry(retry_error_callback=retry_callback, stop=stop_after_attempt(2))
async def send_api_image(dp, id, prompt, size='1024x1024', n=1):
    dp.cur_openai_api_image = (dp.cur_openai_api_image + 1) % len(dp.openai_services_image)
    async with aiohttp.ClientSession() as session:
        payload = {
            "prompt": prompt,
            "image_size": size,
            "n": n
        }
        try:
            cur_service = dp.openai_services_image[dp.cur_openai_api_image]
            logger.debug(f"request is send to openai_api {cur_service['host']}:{cur_service['port']} user id: {id} prompt: {prompt}")
            async with session.post(f"http://{cur_service['host']}:{cur_service['port']}/get_image", json=payload, timeout=250) as resp:
                response = await resp.json()
                logger.debug(f'response from openai api is received: {response}')
                return response
        except Exception as e:
            logger.error(f'exception in parsing openai api: {e}')
            raise Exception

async def send_api_internet(dp, id, prompt, model, market, use_chain):
    dp.cur_openai_api_internet = (dp.cur_openai_api_internet + 1)%len(dp.openai_services_internet)
    async with aiohttp.ClientSession() as session:
        payload = {
            "query": prompt,
            "model": model,
            "use_chain": use_chain,
            "market": market,
            'id': str(id)
        }
        try:
            cur_service = dp.openai_services_internet[dp.cur_openai_api_internet]
            logger.debug(f"request is send to openai_api internet {cur_service['host']}:{cur_service['port']} user id: {id} query: {prompt}")
            async with session.post(f"http://{cur_service['host']}:{cur_service['port']}/search_internet", json=payload, timeout=600) as resp:
                response = await resp.json()
                logger.debug(f'response from openai api internet is received: {response}')
                return response
        except Exception as e:
            logger.error(f'exception in parsing openai api internet: {e}')
            return {'success': False, 'id':'timeout', 'error': f'exception in parsing openai api: {e}'}
        
@retry(retry_error_callback=retry_callback, stop=stop_after_attempt(2))
async def send_api_search_country(dp, prompt):
    dp.cur_openai_api_internet = (dp.cur_openai_api_internet + 1)%len(dp.openai_services_internet)
    async with aiohttp.ClientSession() as session:
        payload = {
            "query": prompt
        }
        try:
            cur_service = dp.openai_services_internet[dp.cur_openai_api_internet]
            logger.debug(f"request is send to openai_api internet {cur_service['host']}:{cur_service['port']} query: {prompt}")
            async with session.post(f"http://{cur_service['host']}:{cur_service['port']}/search_country", json=payload, timeout=250) as resp:
                response = await resp.json()
                logger.debug(f'response from openai api internet is received: {response}')
                return response
        except Exception as e:
            logger.error(f'exception in parsing openai api internet: {e}')
            raise Exception

async def message_answer(msg, *args, **kwargs):
    # слишком длинный ответ надо как то разбивать
    answer = copy.copy(args[0])
    l = answer.split('```')
    if len(l)%2 != 0 and len(l)>1:
        answer = ''
        rplc = True
        for item in l:
            if rplc:
                answer += item.replace('\\','\\\\').replace('_','\_').replace('*','\*').replace('[','\[').replace(']','\]').replace('(','\(').replace(')','\)').replace('~','\~').replace('>','\>').replace('<','\<').replace('&','\&').replace('#','\#').replace('+','\+').replace('-','\-').replace('=','\=').replace('|','\|').replace('{','\{').replace('}','\}').replace('.','\.').replace('!','\!')
            else:
                answer += '```'+item+'```'
            rplc = not rplc
    else:
        answer = answer.replace('\\','\\\\').replace('_','\_').replace('*','\*').replace('[','\[').replace(']','\]').replace('(','\(').replace(')','\)').replace('~','\~').replace('>','\>').replace('<','\<').replace('&','\&').replace('#','\#').replace('+','\+').replace('-','\-').replace('=','\=').replace('|','\|').replace('{','\{').replace('}','\}').replace('.','\.').replace('!','\!')
  
    while(len(answer) > 0):
        try:
            await msg.answer(answer[:4095], *args[1:], **kwargs, parse_mode='MarkdownV2')
        except:
            await msg.answer(answer[:4095], *args[1:], **kwargs)
        answer = answer[4095:]
    logger.info(f'BOT to user: {msg.from_user.id} {args[0]}')


async def bot_send(*args, **kwargs):
    logger.info(f'BOT to user: {args[0]}: {args[1]}')
    await bot.send_message(*args, **kwargs)

async def loop_periodic(dp):
    while True:
        try:
            # обновление статусов, что бот сейчас печатает
            cur_dict = copy.deepcopy(dp.user_in_request)
            tasks = []
            res={}
            for key, value in cur_dict.items():
                task = asyncio.ensure_future(bot.send_chat_action(value[0], value[1]))
                tasks.append(task)
                res[key] = task
            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                logger.error(f'exception in loop_periodic: {e}')

            await asyncio.sleep(3)
            for key, task in res.items():
                if task.exception():
                    if key in dp.user_in_request:
                        del dp.user_in_request[key]
        except Exception as e:
            logger.error(f'excepion in loop periodic: {e}')


def read_static_files(dp):
    try:
        with open("config/conversation_starter_pretext.txt", "r") as f:
            pretext = f.read()
        dp.pretext = pretext

        with open("config/conversation_starter_pretext_programmer.txt", "r") as f:
            pretext_programmer = f.read()
        dp.pretext_programmer = pretext_programmer

        with open('config/config.json', 'r', encoding='utf-8') as f:
            json_str = f.read()
        ava_langs = json.loads(json_str)

        with open('config/openai_services.json', 'r', encoding='utf-8') as f:
            json_str = f.read()
        dp.openai_services_text = json.loads(json_str)['services_text']
        dp.openai_services_image = json.loads(json_str)['services_image']
        dp.openai_services_internet = json.loads(json_str)['services_internet']

        langs_buttons_dict = {
            'ru': 'Русский',
            'en': 'English',
            'fr': 'Français',
            'spa': 'Español'
        }
        dp.ava_langs = {item[0]: langs_buttons_dict[item[0]] for item in ava_langs.items() if item[1]}
        dp.inv_but_langs = {item[1]: item[0] for item in dp.ava_langs.items()}

        dp.inv_but_model = {'GPT_3.5':'gpt', 'GPT_3.5_Turbo':'turbo'}

        for lang in dp.ava_langs.keys():
            with open(f'static_texts/language_headers_{lang}.json', 'r', encoding='utf-8') as f:
                json_str = f.read()
            dp.static_headers[lang] = json.loads(json_str)

    except Exception as e:
        logger.error(f'exception in read_static_files: {e}')

PAYMENTS = config("PAYMENTS").lower() in ('true', '1', 't')
PAYMENTS_CRYPTO = config("PAYMENTS_CRYPTO").lower() in ('true', '1', 't')
PAYMENTS_STRIPE = config("PAYMENTS_STRIPE").lower() in ('true', '1', 't')
REDIRECT = config("REDIRECT").lower() in ('true', '1', 't')

model_dict = {"turbo": "gpt-3.5-turbo", "gpt": "text-davinci-003"}

bot = Bot(config('BOT_TOKEN'))
dp = Dispatcher(bot)
dp.user_in_request = dict()
bot_id=int(config('BOT_POSTGRES_ID'))
if PAYMENTS:
    dp.ukassa = PaymentsUkassa(config('SHOP_ID'), config('SHOP_KEY'))
if PAYMENTS_CRYPTO:
    dp.usegateway = PaymentsUsegateway(config('GATEWAY_API_KEY'))
dp.cur_openai_api_text = 0
dp.cur_openai_api_image = 0
dp.cur_openai_api_internet = 0

async_engine = create_async_engine(config('BD_PATH'), pool_pre_ping=True, pool_size=30, max_overflow=30)
async_engine_common_balance = create_async_engine(config('BD_PATH_COMMON_BALANCE'), pool_pre_ping=True, pool_size=30, max_overflow=30)


dp.redis = StrictRedis(config('REDIS_HOST'), int(config('REDIS_PORT')), int(config('REDIS_DB')), config('REDIS_PASSWORD'))

# костыльно записываем какой именно это бот english или russian
if PAYMENTS_STRIPE:
    dp.cur_name_bot = 'eng'
else:
    dp.cur_name_bot = 'rus'

# Wait for Redis to be ready
async def aredis_wait():
    ping = False
    while ping == False:
        try:
          ping = await dp.redis.ping()
        except:
          pass
        logger.info(str('Redis Alive:'+str(ping)))
        await asyncio.sleep(1)

asyncio.get_event_loop().run_until_complete(aredis_wait())

dp.reply_keyboard = dict()
dp.inline_keyboard = dict()
dp.static_headers = dict()
read_static_files(dp)


for cur_lang in dp.ava_langs.keys():
    dp.reply_keyboard[cur_lang] = BotKeyboard(dp.static_headers[cur_lang], list(dp.ava_langs.values()), list(dp.inv_but_model.keys()))

    dp.inline_keyboard[cur_lang] = InlineBotKeyboard(dp.static_headers[cur_lang])

tokenizer = Tokenizer()


async def is_user_admin(telegram_id):
    async with AsyncSession(async_engine) as async_session:
        res = False
        # выбираем из таблицы соответствующего пользователя
        stmt = select(User).where(User.telegram_id == telegram_id)
        items = await async_session.execute(stmt)
        item = items.scalars().first()
        if item is not None:
            res = item.is_admin == True
        return res


async def get_id_users():
    async with AsyncSession(async_engine) as async_session:
        stmt = select(User).order_by(User.id)
        items = await async_session.execute(stmt)
        items = items.scalars().all()
        return [(item.telegram_id,item.username) for item in items[1:]]


async def get_id_from_telegram_id_payments(telegram_id: int):
    async with AsyncSession(async_engine_common_balance) as async_session:
        res = 0
        # выбираем из таблицы соответствующего пользователя
        stmt = select(CommonBalance).where(CommonBalance.telegram_id == telegram_id)
        items = await async_session.execute(stmt)
        item = items.scalars().first()
        if item is not None:
            res = item.id
        return res


async def increment_balance(user_id: int, tokens: int):
    async with AsyncSession(async_engine_common_balance) as async_session:
        stmt = select(CommonBalance).where(CommonBalance.id == user_id)
        item = await async_session.execute(stmt)
        dest_user = item.scalars().first()
        if dest_user is None:
            return False, f'User is not exist id: {user_id}'
        dest_user.balance += tokens
        async_session.add(dest_user)
        await async_session.commit()
        return True, 'Success'


async def increment_balance_pay(redis, user_id: int, trans_id: str, amount_currency: float, currency: str,
                                method: str = 'SBP'):
    # получаем цену
    if method == 'SBP':
        price = await redis.get('price')
        if price is None:
            price = 0.5
            await redis.set('price', price)
        else:
            price = float(price.decode('utf-8'))
    if method == 'CRYPTO' or method == 'STRIPE':
        price = await redis.get('price_usd')
        if price is None:
            price = 0.0066666666
            await redis.set('price_usd', price)
        else:
            price = float(price.decode('utf-8'))

    tokens = int(amount_currency * 1000 / price)

    ress, res_str = await increment_balance(user_id, tokens)
    if ress:
        async with AsyncSession(async_engine_common_balance) as async_session:
            payment = CommonPayment()
            payment.user_id = user_id
            payment.tokens = tokens
            payment.currency = currency
            payment.dt = datetime.datetime.now()
            payment.amount = amount_currency
            payment.transaction_id = trans_id
            payment.method = method
            payment.bot = dp.cur_name_bot
            async_session.add(payment)
            await async_session.commit()
            return True, 'Success'
    else:
        return ress, res_str


@dp.message_handler(commands=['send_all'])
async def cmd_send_all(message: types.Message):
    is_admin = await is_user_admin(message.from_user.id)
    if not is_admin:
        return
    msg = message.text.replace('/send_all ', '')
    parts = msg.split('|')
    tg_id = int(parts[0])
    msg = parts[1]
    users = await get_id_users()
    start_send = tg_id == 1
    for id, user in users:
        if tg_id == id:
            start_send = True
        if start_send is False:
            continue
        try:
            lang_mode = await user_lang.get(id)
            await bot.send_message(id, msg)
            logger.info(f'ADMIN send: {user} {id}')
        except Exception as e:
            logger.error(f'exception in aending all: {e}')
        await asyncio.sleep(0.5)
    logger.info(f'SEND ALL Developer: {message.from_user.username}: {message.text}')


@dp.message_handler(commands=['send_spec'])
async def cmd_send_spec(message: types.Message):
    is_admin = await is_user_admin(message.from_user.id)
    if not is_admin:
        return
    msg = message.text.replace('/send_spec ', '')
    parts = msg.split('|')
    tg_id = parts[0]
    msg = parts[1]
    lang_mode = await user_lang.get(tg_id)
    await bot.send_message(tg_id, msg)
    logger.info(f'SEND SPEC Developer: {message.from_user.username}: {message.text}')


class UserStorage:
    def __init__(self, key:str, default:str):
        self.key = key
        self.default = default

    async def get(self, id):
        user_mode = UserState(dp.redis, id)
        mode = await user_mode.get_state(self.key)
        if mode is None:
            await user_mode.set_state(self.key, self.default)
            mode = self.default
        else:
            mode = mode.decode('utf-8')
        return mode

    async def set(self, id, value):
        user_mode = UserState(dp.redis, id)
        await user_mode.set_state(self.key, value)


user_lang = UserStorage('language', 'en')
user_mode = UserStorage('mode', 'writer')
user_menu_state = UserStorage('menu', 'work')
user_paint_mode = UserStorage('paint', '1')
user_model = UserStorage('model', 'turbo')

user_internet_ref_mode = UserStorage('internet_ref_mode', 'only_ref')
user_internet_resp_mode = UserStorage('internet_resp_mode', 'no_chain')
user_internet_market_mode = UserStorage('internet_market_mode', 'en-US')


@dp.callback_query_handler(text='cash_in_button')
async def process_callback_cash_in_button(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)

    lang_mode = await user_lang.get(callback_query.from_user.id)
    if PAYMENTS_STRIPE:
        await bot_send(callback_query.from_user.id, dp.static_headers[lang_mode]['cash_in_select_pay_method_en'], reply_markup=dp.inline_keyboard[lang_mode].select_pay_method)
    else:
        await bot_send(callback_query.from_user.id, dp.static_headers[lang_mode]['cash_in_select_pay_method'], reply_markup=dp.inline_keyboard[lang_mode].select_pay_method)



async def create_payment_ukassa(value, description, email):
    payme = await dp.ukassa.create({
        "amount": {
            "value": value,
            "currency": "RUB"
        },
        "payment_method_data": {
            "type": "sbp"
        },
        "confirmation": {
            "type": "redirect",
            "return_url": "https://t.me/gpt3_unlim_chatbot"
        },
        "receipt": {
            "customer": {
                "email": email
            },
            "items": [{
                "description": description,
                "amount": {
                    "value": value,
                    "currency": "RUB"
                },
                "vat_code": 1,
                "quantity": 1
            }]
        },
        "capture": True,
        "description": description
    })
    if payme is None:
        logger.error(f'SBP payment creating is None')
    return payme


async def check_payment_ukassa(payment_id):
    while(1):
        payme = await dp.ukassa.find_one(payment_id)
        if payme is None:
            return False, None
        if payme['status'] != 'pending':
            break
        await asyncio.sleep(3)

    return payme['status'] == 'succeeded', payme


async def checkout_ukassa(trans_id, amount, currency, tg_id):
    try:
        lang_mode = await user_lang.get(tg_id)
        id = await get_id_from_telegram_id_payments(tg_id)
        res, msg = await increment_balance_pay(dp.redis, id, trans_id, amount, currency, 'SBP')
        if res is not True:
            await bot_send(tg_id,
                                   dp.static_headers[lang_mode]['cash_in_error'] + '\n' + msg)
            logger.error(f'SBP user is not exist in balance: id: {id} transaction_id: {trans_id}')
        else:
            await bot_send(tg_id,
                           dp.static_headers[lang_mode]['cash_in_success'].format(int(amount), currency))
            logger.info(
                f'SBP user_id: {id} telegram_id: {tg_id} paid: {amount} currency {currency} transaction_id: {trans_id}')
    except Exception as e:
        if res is not True:
            await bot.send_message(tg_id,
                                   dp.static_headers[lang_mode]['cash_in_error'] + '\n' + e)
            logger.error(f'SBP exception in payment user_id: {id} transaction_id: {trans_id}')


async def pay_ukassa(amount, user_id: int, mail):
    lang_mode = await user_lang.get(user_id)

    pay = await create_payment_ukassa(amount, f'Оплата заказа генерации контента на сумму: {amount} руб.', mail)
    await bot.send_message(user_id,
                           dp.static_headers[lang_mode]['cash_in_description'] + '\n'
                           + pay['confirmation']['confirmation_url'])

    res, pay_res = await check_payment_ukassa(pay['id'])
    if res:
        await checkout_ukassa(pay_res['id'], float(pay_res['amount']['value']),
                              pay_res['amount']['currency'], user_id)
    else:
        logger.info(f"SBP payment from tg_id: {user_id} pay_id: {pay['id']} amount: {float(pay['amount']['value'])} {pay['amount']['currency']} is not successed")





async def create_payment_usegateway(value, description):
    id = str(uuid.uuid4())
    payme = await dp.usegateway.create({
      "name": "Content generation",
      "description": description,
      "pricing_type": "fixed_price",
      "local_price": {
        "amount": value,
        "currency": "USD"
      },
      "metadata": {
        "uuid4": id
      },
      "redirect_url": "https://t.me/gpt3_unlim_chatbot",
      "cancel_url": "https://t.me/gpt3_unlim_chatbot"
    })
    if payme is None:
        logger.error(f'CRYPTO payment creating is None')
    return payme


async def check_payment_usegateway(payment_id):
    while(1):
        payme = await dp.usegateway.find_one(payment_id)
        if payme is None:
            return False, None
        for item in payme['timeline']:
            if item['status'] == 'Completed':
                return True, payme
            if item['status'] == 'Closed' or item['status'] == 'Unresolved (Underpaid)' or item['status'] == 'Expired':
                return False, payme
        await asyncio.sleep(3)


async def checkout_usegateway(trans_id, amount, currency, tg_id):
    try:
        lang_mode = await user_lang.get(tg_id)
        id = await get_id_from_telegram_id_payments(tg_id)
        res, msg = await increment_balance_pay(dp.redis, id, trans_id, amount, currency, 'CRYPTO')
        if res is not True:
            await bot_send(tg_id,
                                   dp.static_headers[lang_mode]['cash_in_error'] + '\n' + msg)
            logger.error(f'CRYPTO user is not exist in balance: id: {id} transaction_id: {trans_id}')
        else:
            await bot_send(tg_id,
                           dp.static_headers[lang_mode]['cash_in_success'].format(int(amount), currency))
            logger.info(
                f'CRYPTO user_id: {id} telegram_id: {tg_id} paid: {amount} currency {currency} transaction_id: {trans_id}')
    except Exception as e:
        if res is not True:
            await bot.send_message(tg_id,
                                   dp.static_headers[lang_mode]['cash_in_error'] + '\n' + e)
            logger.error(f'CRYPTO exception in payment user_id: {id} transaction_id: {trans_id}')


async def pay_usegateway(amount, user_id: int):
    lang_mode = await user_lang.get(user_id)

    pay = await create_payment_usegateway(amount, f'Payment for the content generation order in the amount of: {amount} $.')
    await bot.send_message(user_id,
                           dp.static_headers[lang_mode]['cash_in_description'] + '\n'
                           + pay['hosted_url'])

    res, pay_res = await check_payment_usegateway(pay['id'])
    if res:
        await checkout_usegateway(pay_res['id'], float(pay_res['transactions'][0]['value']['local']['amount']),
                              pay_res['transactions'][0]['value']['local']['currency'], user_id)
    else:
        logger.info(f"CRYPTO payment from tg_id: {user_id} pay_id: {pay['id']} amount: {float(pay['amount']['value'])} {pay['amount']['currency']} is not successed")

async def pay_stripe(amount, user_id: int, chat_id: int):
    lang_mode = await user_lang.get(user_id)

    PRICE = types.LabeledPrice(label="Content generation service", amount=amount*100)

    await bot.send_invoice(chat_id,
                           title=dp.static_headers[lang_mode]['tg_invoice_title'],
                           description=dp.static_headers[lang_mode]['tg_invoice_description'],
                           provider_token=config('PAYMENT_TOKEN'),
                           currency="usd",
                           is_flexible=False,
                           prices=[PRICE],
                           payload="stripe-invoice-payload")
    
    logger.info(f"STRIPE invoice for tg_id: {user_id} chat_id: {chat_id} amount: {amount} usd has been sent")

@dp.pre_checkout_query_handler(lambda query: True)
async def pre_checkout_query(pre_checkout_q: types.PreCheckoutQuery):
    await bot.answer_pre_checkout_query(pre_checkout_q.id, ok=True)
    logger.info(f"pre_checkout_query for pre_checkout_q.id: {pre_checkout_q.id} has been executed")

@dp.message_handler(content_types=ContentType.SUCCESSFUL_PAYMENT)
async def successful_payment(message: types.Message):
    logger.info(f"SUCCESSFUL PAYMENT {message.successful_payment.telegram_payment_charge_id}: {message.successful_payment.total_amount // 100} {message.successful_payment.currency}")
    await bot.send_message(message.chat.id,
                           f"{message.successful_payment.total_amount // 100} {message.successful_payment.currency} received")

    try:
        lang_mode = await user_lang.get(message.from_user.id)
        id = await get_id_from_telegram_id_payments(message.from_user.id)
        trans_id = message.successful_payment.provider_payment_charge_id
        amount = message.successful_payment.total_amount // 100	
        tg_id = message.from_user.id
        currency = 'USD'
        res, msg = await increment_balance_pay(dp.redis, id, trans_id, amount, currency, 'STRIPE')
        if res is not True:
            await bot_send(tg_id,
                                   dp.static_headers[lang_mode]['cash_in_error'] + '\n' + msg)
            logger.error(f'STRIPE user is not exist in balance: id: {id} transaction_id: {trans_id}')
        else:
            await bot_send(tg_id,
                           dp.static_headers[lang_mode]['cash_in_success'].format(int(amount), currency))
            logger.info(
                f'STRIPE user_id: {id} telegram_id: {tg_id} paid: {amount} currency {currency} transaction_id: {trans_id}')
    except Exception as e:
        if res is not True:
            await bot.send_message(tg_id,
                                   dp.static_headers[lang_mode]['cash_in_error'] + '\n' + e)
            logger.error(f'STRIPE exception in payment user_id: {id} transaction_id: {trans_id}')
    
    
    
async def save_pay_amount(user_id, amount):
    await dp.redis.set(str(user_id) + '_pay_amount', amount)
async def load_pay_amount(user_id):
    res = await dp.redis.get(str(user_id) + '_pay_amount')
    if res is None:
        return None
    val = res.decode('utf-8')
    return float(val)
async def save_mail(user_id, mail):
    await dp.redis.set(str(user_id)+"_mail", mail)
async def load_mail(user_id):
    res = await dp.redis.get(str(user_id) + '_mail')
    val = res.decode('utf-8')
    return val

@dp.callback_query_handler(text='no_mail_button')
async def process_callback_no_mail_button(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await user_menu_state.set(callback_query.from_user.id, 'work')
    amount = await load_pay_amount(callback_query.from_user.id)
    lang_mode = await user_lang.get(callback_query.from_user.id)
    if amount is None:
        await bot.send_message(callback_query.from_user.id,
                               dp.static_headers[lang_mode]['cash_in_welcome'],
                               reply_markup=dp.inline_keyboard[lang_mode].amount_sbp_panel)
        return

    await pay_ukassa(amount, callback_query.from_user.id, config('PAY_MAIL'))


async def save_payment_continue(callback_query: types.CallbackQuery, amount:int):
    await bot.answer_callback_query(callback_query.id)
    await save_pay_amount(callback_query.from_user.id, amount)
    lang_mode = await user_lang.get(callback_query.from_user.id)
    await user_menu_state.set(callback_query.from_user.id, 'payment')
    await bot_send(callback_query.from_user.id, dp.static_headers[lang_mode]['cash_in_mail_welcome'], reply_markup=dp.inline_keyboard[lang_mode].no_mail_panel)



@dp.callback_query_handler(text='sbp_button')
async def process_callback_sbp_button(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    lang_mode = await user_lang.get(callback_query.from_user.id)
    await bot.send_message(callback_query.from_user.id,
                           dp.static_headers[lang_mode]['cash_in_welcome'],
                           reply_markup=dp.inline_keyboard[lang_mode].amount_sbp_panel)


@dp.callback_query_handler(text='cash_in_50_button')
async def process_callback_cash_in_50_button(callback_query: types.CallbackQuery):
    await save_payment_continue(callback_query, 50)
@dp.callback_query_handler(text='cash_in_100_button')
async def process_callback_cash_in_100_button(callback_query: types.CallbackQuery):
    await save_payment_continue(callback_query, 100)
@dp.callback_query_handler(text='cash_in_200_button')
async def process_callback_cash_in_200_button(callback_query: types.CallbackQuery):
    await save_payment_continue(callback_query, 200)
@dp.callback_query_handler(text='cash_in_500_button')
async def process_callback_cash_in_500_button(callback_query: types.CallbackQuery):
    await save_payment_continue(callback_query, 500)
@dp.callback_query_handler(text='cash_in_1000_button')
async def process_callback_cash_in_1000_button(callback_query: types.CallbackQuery):
    await save_payment_continue(callback_query, 1000)
@dp.callback_query_handler(text='cash_in_2000_button')
async def process_callback_cash_in_2000_button(callback_query: types.CallbackQuery):
    await save_payment_continue(callback_query, 2000)
@dp.callback_query_handler(text='cash_in_5000_button')
async def process_callback_cash_in_5000_button(callback_query: types.CallbackQuery):
    await save_payment_continue(callback_query, 5000)




@dp.callback_query_handler(text='crypto_button')
async def process_callback_crypto_button(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    lang_mode = await user_lang.get(callback_query.from_user.id)
    await bot.send_message(callback_query.from_user.id,
                           dp.static_headers[lang_mode]['cash_in_welcome'],
                           reply_markup=dp.inline_keyboard[lang_mode].amount_crypto_panel)


async def payment_usegateway(callback_query: types.CallbackQuery, amount:int):
    await bot.answer_callback_query(callback_query.id)
    lang_mode = await user_lang.get(callback_query.from_user.id)
    if amount is None:
        await bot.send_message(callback_query.from_user.id,
                               dp.static_headers[lang_mode]['cash_in_welcome'],
                               reply_markup=dp.inline_keyboard[lang_mode].amount_crypto_panel)
        return

    await pay_usegateway(amount, callback_query.from_user.id)

@dp.callback_query_handler(text='cash_in_3usd_button')
async def process_callback_cash_in_3usd_button(callback_query: types.CallbackQuery):
    await payment_usegateway(callback_query, 3)
@dp.callback_query_handler(text='cash_in_5usd_button')
async def process_callback_cash_in_5usd_button(callback_query: types.CallbackQuery):
    await payment_usegateway(callback_query, 5)
@dp.callback_query_handler(text='cash_in_10usd_button')
async def process_callback_cash_in_10usd_button(callback_query: types.CallbackQuery):
    await payment_usegateway(callback_query, 10)
@dp.callback_query_handler(text='cash_in_20usd_button')
async def process_callback_cash_in_20usd_button(callback_query: types.CallbackQuery):
    await payment_usegateway(callback_query, 20)
@dp.callback_query_handler(text='cash_in_50usd_button')
async def process_callback_cash_in_50usd_button(callback_query: types.CallbackQuery):
    await payment_usegateway(callback_query, 50)
@dp.callback_query_handler(text='cash_in_100usd_button')
async def process_callback_cash_in_100usd_button(callback_query: types.CallbackQuery):
    await payment_usegateway(callback_query, 100)
@dp.callback_query_handler(text='cash_in_500usd_button')
async def process_callback_cash_in_500usd_button(callback_query: types.CallbackQuery):
    await payment_usegateway(callback_query, 500)


@dp.callback_query_handler(text='stripe_button')
async def process_callback_stripe_button(callback_query: types.CallbackQuery):
    if PAYMENTS:
        await bot.answer_callback_query(callback_query.id)
        lang_mode = await user_lang.get(callback_query.from_user.id)
        await bot.send_message(callback_query.from_user.id,
                            dp.static_headers[lang_mode]['redirect']
                            )
    else:
        await bot.answer_callback_query(callback_query.id)
        lang_mode = await user_lang.get(callback_query.from_user.id)
        await bot.send_message(callback_query.from_user.id,
                            dp.static_headers[lang_mode]['cash_in_welcome'],
                            reply_markup=dp.inline_keyboard[lang_mode].amount_stripe_panel)

async def payment_stripe(callback_query: types.CallbackQuery, amount:int):
    await bot.answer_callback_query(callback_query.id)
    lang_mode = await user_lang.get(callback_query.from_user.id)
    if amount is None:
        await bot.send_message(callback_query.from_user.id,
                               dp.static_headers[lang_mode]['cash_in_welcome'],
                               reply_markup=dp.inline_keyboard[lang_mode].amount_stripe_panel)
        return

    await pay_stripe(amount, callback_query.from_user.id, callback_query.message.chat.id)

@dp.callback_query_handler(text='cash_in_5usd_stripe_button')
async def process_cash_in_5usd_stripe_button(callback_query: types.CallbackQuery):
    await payment_stripe(callback_query, 5)
@dp.callback_query_handler(text='cash_in_10usd_stripe_button')
async def process_cash_in_10usd_stripe_button(callback_query: types.CallbackQuery):
    await payment_stripe(callback_query, 10)
@dp.callback_query_handler(text='cash_in_15usd_stripe_button')
async def process_cash_in_15usd_stripe_button(callback_query: types.CallbackQuery):
    await payment_stripe(callback_query, 15)
@dp.callback_query_handler(text='cash_in_25usd_stripe_button')
async def process_cash_in_25usd_stripe_button(callback_query: types.CallbackQuery):
    await payment_stripe(callback_query, 25)
@dp.callback_query_handler(text='cash_in_50usd_stripe_button')
async def process_cash_in_50usd_stripe_button(callback_query: types.CallbackQuery):
    await payment_stripe(callback_query, 50)
@dp.callback_query_handler(text='cash_in_100usd_stripe_button')
async def process_cash_in_100usd_stripe_button(callback_query: types.CallbackQuery):
    await payment_stripe(callback_query, 100)
@dp.callback_query_handler(text='cash_in_500usd_stripe_button')
async def process_cash_in_500usd_stripe_button(callback_query: types.CallbackQuery):
    await payment_stripe(callback_query, 500)

@dp.callback_query_handler(text='ref_button')
async def process_ref_button(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    lang_mode = await user_lang.get(callback_query.from_user.id)
    link = await get_start_link(str(callback_query.from_user.id), encode=True)
    await bot_send(callback_query.from_user.id, dp.static_headers[lang_mode]['ref_link_msg'].format(link))

async def check_subscription(telegram_id:int, limit:int):
    try:
        res = await bot.get_chat_member(config('CHANNEL_SUBSCRIPTION'), telegram_id)
        logger.debug(f"user: {telegram_id} subscription on channel {config('CHANNEL_SUBSCRIPTION')} is: {res.status}")
        if res.status == 'member':
            return limit
        else:
            return limit//2
    except Exception as e:
        logger.error(f'exception in subscription check: {e}')
        return limit



@dp.message_handler(commands=['help'])
async def cmd_help(message: types.Message):
    logger.info(f'HUMAN: {message.from_user.username}: {message.text}')
    lang_mode = await user_lang.get(message.from_user.id)
    if PAYMENTS_STRIPE:
        await message_answer(message, dp.static_headers[lang_mode]['help_en'])
    else:
        await message_answer(message, dp.static_headers[lang_mode]['help'])


@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    logger.info(f'HUMAN: {message.from_user.username}: {message.text}')
    lang_mode = await user_lang.get(message.from_user.id)

    args = message.get_args()
    if len(args):
        reference = int(decode_payload(args))
        id_in_our_db = await get_id_from_telegram_id_payments(message.from_user.id)
        # если юзера нет в нашей БД, значит он новый - можно начислить баланса рефералу
        if id_in_our_db == 0:
            user_id = await get_id_from_telegram_id_payments(reference)
            await increment_balance(user_id, int(config('REFERRAL_TOKENS')))
            await message_answer(message, dp.static_headers[lang_mode]['success_ref_msg'])
        else:
            await message_answer(message, dp.static_headers[lang_mode]['unsuccess_ref_msg'])



    # стираем историю, если есть
    hist = History(async_engine, user_telegram_id=message.from_user.id,
                   username=message.from_user.username, bot_id=bot_id)
    await hist.clear_history()

    mode = await user_mode.get(message.from_user.id)
    cur_reply_keyboard = dp.reply_keyboard[lang_mode].start
    if mode == 'painter':
        cur_reply_keyboard = dp.reply_keyboard[lang_mode].painter_panel
    await message_answer(message, dp.static_headers[lang_mode]['welcome'], reply_markup=cur_reply_keyboard)



@dp.message_handler(content_types=['text'])
async def msg_handler(message: types.Message):
    if message.from_user.is_bot:
        logger.info(f'MSG FROM GROUP: username: {message.from_user.username} '
                    f'id: {message.from_user.id} '
                    f'chat name: {message.chat.username}'
                    f'chat id: {message.chat.id}')
        return

    hist = History(async_engine, user_telegram_id=message.from_user.id,
                   username=message.from_user.username, bot_id=bot_id)
    balance = Balance(async_engine_common_balance, message.from_user.id, int(config('MAX_DAY_USER_TOKENS')), check_subscription)

    logger.info(f'HUMAN: {message.from_user.username}: {message.text}')

    lang_mode = await user_lang.get(message.from_user.id)
    mode = await user_mode.get(message.from_user.id)  
    model_mode = await user_model.get(message.from_user.id)
    # обработка меню
    menu_state = await user_menu_state.get(message.from_user.id)

    cur_reply_keyboard = dp.reply_keyboard[lang_mode].start
    if mode == 'painter' or (message.text == dp.static_headers[lang_mode]['painter_button'] and menu_state == 'mode'):
        copy_keyboard = copy.deepcopy(dp.reply_keyboard[lang_mode])
        cur_reply_keyboard = copy_keyboard.painter_panel
        n_pict = await user_paint_mode.get(message.from_user.id)
        n_pict = int(n_pict)
        if n_pict == 1:
            copy_keyboard.one_pict_but.text =  '*'+copy_keyboard.one_pict_but.text+'*'
            copy_keyboard.three_pict_but.text = copy_keyboard.three_pict_but.text.replace('*','')
            copy_keyboard.five_pict_but.text = copy_keyboard.five_pict_but.text.replace('*','')
        if n_pict == 3:
            copy_keyboard.three_pict_but.text =  '*'+copy_keyboard.three_pict_but.text+'*'
            copy_keyboard.one_pict_but.text = copy_keyboard.one_pict_but.text.replace('*','')
            copy_keyboard.five_pict_but.text = copy_keyboard.five_pict_but.text.replace('*','')
        if n_pict == 5:
            copy_keyboard.five_pict_but.text =  '*'+copy_keyboard.five_pict_but.text+'*'
            copy_keyboard.three_pict_but.text = copy_keyboard.three_pict_but.text.replace('*','')
            copy_keyboard.one_pict_but.text = copy_keyboard.one_pict_but.text.replace('*','')
        message.text = message.text.replace('*','')
    elif mode == 'internet':
        cur_reply_keyboard = dp.reply_keyboard[lang_mode].internet_panel

    # обработка кнопок
    if message.text == dp.static_headers[lang_mode]['restart_button']:
        await hist.clear_history()
        await message_answer(message, dp.static_headers[lang_mode]['welcome'], reply_markup=cur_reply_keyboard)
        return
    if message.text == dp.static_headers[lang_mode]['language_button']:
        await message_answer(message, dp.static_headers[lang_mode]['language_msg'], reply_markup=dp.reply_keyboard[lang_mode].language_panel)
        await user_menu_state.set(message.from_user.id, 'lang')
        return
    if message.text == dp.static_headers[lang_mode]['select_mode_button']:
        cur_user_keyboard = copy.deepcopy(dp.reply_keyboard[lang_mode])
        if mode == 'painter':
            cur_user_keyboard.painter_but.text = '*'+cur_user_keyboard.painter_but.text+'*'
            cur_user_keyboard.writer_but.text = cur_user_keyboard.writer_but.text.replace('*','')
            cur_user_keyboard.coder_but.text = cur_user_keyboard.coder_but.text.replace('*', '')
            cur_user_keyboard.internet_but.text = cur_user_keyboard.internet_but.text.replace('*', '')
        if mode == 'writer':
            cur_user_keyboard.writer_but.text = '*'+cur_user_keyboard.writer_but.text+'*'
            cur_user_keyboard.painter_but.text = cur_user_keyboard.painter_but.text.replace('*','')
            cur_user_keyboard.coder_but.text = cur_user_keyboard.coder_but.text.replace('*', '')
            cur_user_keyboard.internet_but.text = cur_user_keyboard.internet_but.text.replace('*','')
        if mode == 'programmer':
            cur_user_keyboard.coder_but.text = '*'+cur_user_keyboard.coder_but.text+'*'
            cur_user_keyboard.painter_but.text = cur_user_keyboard.painter_but.text.replace('*','')
            cur_user_keyboard.writer_but.text = cur_user_keyboard.writer_but.text.replace('*', '')
            cur_user_keyboard.internet_but.text = cur_user_keyboard.internet_but.text.replace('*','')
        if mode == 'internet':
            cur_user_keyboard.internet_but.text = '*'+cur_user_keyboard.internet_but.text+'*'
            cur_user_keyboard.painter_but.text = cur_user_keyboard.painter_but.text.replace('*','')
            cur_user_keyboard.writer_but.text = cur_user_keyboard.writer_but.text.replace('*', '')
            cur_user_keyboard.coder_but.text = cur_user_keyboard.coder_but.text.replace('*','')

        await message_answer(message, dp.static_headers[lang_mode]['mode_msg'], reply_markup=cur_user_keyboard.select_mode_panel)
        await user_menu_state.set(message.from_user.id, 'mode')
        return
    if message.text == dp.static_headers[lang_mode]['model_button']:
        cur_user_keyboard = copy.deepcopy(dp.reply_keyboard[lang_mode])
        if model_mode == 'gpt':
            cur_user_keyboard.models_buts[0].text = '*'+cur_user_keyboard.models_buts[0].text+'*'
            cur_user_keyboard.models_buts[1].text = cur_user_keyboard.models_buts[1].text.replace('*','')
        if model_mode == 'turbo':
            cur_user_keyboard.models_buts[1].text = '*'+cur_user_keyboard.models_buts[1].text+'*'
            cur_user_keyboard.models_buts[0].text = cur_user_keyboard.models_buts[0].text.replace('*','')

        await message_answer(message, dp.static_headers[lang_mode]['model_msg'], reply_markup=cur_user_keyboard.model_panel)
        await user_menu_state.set(message.from_user.id, 'model')
        return
    if message.text == dp.static_headers[lang_mode]['balance_button']:
        if PAYMENTS_STRIPE:
            msg = dp.static_headers[lang_mode]['balance_msg_en']
        else:
            msg = dp.static_headers[lang_mode]['balance_msg']
        tokens = await balance.get_balance()
        if PAYMENTS or PAYMENTS_CRYPTO or PAYMENTS_STRIPE:
            await message_answer(message, msg.format(tokens), reply_markup=dp.inline_keyboard[lang_mode].cash_in_panel)
        else:
            await message_answer(message, msg.format(tokens), reply_markup=cur_reply_keyboard)
        return
    if message.text == dp.static_headers[lang_mode]['help_button']:
        if PAYMENTS_STRIPE:
            await message_answer(message, dp.static_headers[lang_mode]['help_en'], reply_markup=cur_reply_keyboard)
        else:
            await message_answer(message, dp.static_headers[lang_mode]['help'], reply_markup=cur_reply_keyboard)
        return
    if message.text == dp.static_headers[lang_mode]['1picture']:
        if mode == 'painter':
            if copy_keyboard.one_pict_but.text[0] != '*':
                copy_keyboard.one_pict_but.text = '*' + copy_keyboard.one_pict_but.text + '*'
            copy_keyboard.three_pict_but.text = copy_keyboard.three_pict_but.text.replace('*','')
            copy_keyboard.five_pict_but.text = copy_keyboard.five_pict_but.text.replace('*','')
            await message_answer(message, dp.static_headers[lang_mode]['n_picture_description'].format(1),
                                 reply_markup=cur_reply_keyboard)
            await user_paint_mode.set(message.from_user.id, '1')
        return
    if message.text == dp.static_headers[lang_mode]['3picture']:
        if mode == 'painter':
            if copy_keyboard.three_pict_but.text[0] != '*':
                copy_keyboard.three_pict_but.text = '*' + copy_keyboard.three_pict_but.text + '*'
            copy_keyboard.one_pict_but.text = copy_keyboard.one_pict_but.text.replace('*','')
            copy_keyboard.five_pict_but.text = copy_keyboard.five_pict_but.text.replace('*','')
            await message_answer(message, dp.static_headers[lang_mode]['n_picture_description'].format(3),
                                 reply_markup=cur_reply_keyboard)
            await user_paint_mode.set(message.from_user.id, '3')
        return
    if message.text == dp.static_headers[lang_mode]['5picture']:
        if mode == 'painter':
            if copy_keyboard.five_pict_but.text[0] != '*':
                copy_keyboard.five_pict_but.text = '*' + copy_keyboard.five_pict_but.text + '*'
            copy_keyboard.three_pict_but.text = copy_keyboard.three_pict_but.text.replace('*','')
            copy_keyboard.one_pict_but.text = copy_keyboard.one_pict_but.text.replace('*','')
            await message_answer(message, dp.static_headers[lang_mode]['n_picture_description'].format(5),
                                 reply_markup=cur_reply_keyboard)
            await user_paint_mode.set(message.from_user.id, '5')
        return
    if message.text == dp.static_headers[lang_mode]['refs_quotes_button']:
        if mode == 'internet':
            internet_ref_mode = await user_internet_ref_mode.get(message.from_user.id)
            cur_reply_keyboard = copy.deepcopy(dp.reply_keyboard[lang_mode])

            if internet_ref_mode == 'no_ref':
                await user_internet_ref_mode.set(message.from_user.id, 'only_ref')
                internet_ref_mode = 'only_ref'
            # подсветить выбранный вариант
            if internet_ref_mode == 'only_ref':
                cur_reply_keyboard.only_ref_but.text = '*'+cur_reply_keyboard.only_ref_but.text+'*'
            if internet_ref_mode == 'ref_quotes':
                cur_reply_keyboard.ref_quotes_but.text = '*'+cur_reply_keyboard.ref_quotes_but.text+'*'

            await user_menu_state.set(message.from_user.id, 'internet_ref_mode_select')
            await message_answer(message, dp.static_headers[lang_mode]['internet_ref_mode_select'], reply_markup=cur_reply_keyboard.ref_mode_select)
        return
    if message.text == dp.static_headers[lang_mode]['select_mode_chain_button']:
        if mode == 'internet':
            use_chain = await user_internet_resp_mode.get(message.from_user.id)
            # подсветить выбранный вариант
            cur_reply_keyboard = copy.deepcopy(dp.reply_keyboard[lang_mode])
            if use_chain == 'use_chain':
                cur_reply_keyboard.use_chain.text = '*'+cur_reply_keyboard.use_chain.text+'*'
            if use_chain == 'no_chain':
                cur_reply_keyboard.no_chain.text = '*'+cur_reply_keyboard.no_chain.text+'*'

            await user_menu_state.set(message.from_user.id, 'internet_use_chain_mode_select')
            await message_answer(message, dp.static_headers[lang_mode]['internet_use_chain_mode_select'], reply_markup=cur_reply_keyboard.use_chain_select)
        return
    if message.text == dp.static_headers[lang_mode]['market_button']:
        if mode == 'internet':
            cur_reply_keyboard = dp.reply_keyboard[lang_mode].empty
            await user_menu_state.set(message.from_user.id, 'internet_market_mode_select')
            await message_answer(message, dp.static_headers[lang_mode]['internet_market_select'], reply_markup=cur_reply_keyboard)
        return


    if menu_state == 'lang':
        if message.text in dp.inv_but_langs:
            cur_lang_msg = dp.inv_but_langs[message.text]
            await user_lang.set(message.from_user.id, cur_lang_msg)
            await user_menu_state.set(message.from_user.id, 'work')
            cur_reply_keyboard = dp.reply_keyboard[cur_lang_msg].start
            if mode == 'painter':
                cur_reply_keyboard = dp.reply_keyboard[cur_lang_msg].painter_panel
            if mode == 'internet':
                cur_reply_keyboard = dp.reply_keyboard[cur_lang_msg].internet_panel
            await message_answer(message, message.text, reply_markup=cur_reply_keyboard)
            return
        else:
            await message_answer(message, dp.static_headers[lang_mode]['language_msg'],
                                 reply_markup=dp.reply_keyboard[lang_mode].language_panel)
            return
    if menu_state == 'model':
        message.text = message.text.replace('*', '')
        if message.text in dp.inv_but_model:
            cur_model_msg = dp.inv_but_model[message.text]
            await user_model.set(message.from_user.id, cur_model_msg)
            model_mode = await user_model.get(message.from_user.id)
            await user_menu_state.set(message.from_user.id, 'work')
            #cur_reply_keyboard = dp.reply_keyboard[lang_mode].start
            if mode == 'painter':
                cur_reply_keyboard = dp.reply_keyboard[lang_mode].painter_panel
            await message_answer(message, message.text, reply_markup=cur_reply_keyboard)
            return
        else:
            await message_answer(message, dp.static_headers[lang_mode]['model_msg'],
                                reply_markup=dp.reply_keyboard[lang_mode].model_panel)
            return
    if menu_state == 'mode':
        message.text = message.text.replace('*','')
        if message.text == dp.static_headers[lang_mode]['coder_button']:
            mode = await user_mode.get(message.from_user.id)
            await user_menu_state.set(message.from_user.id, 'work')
            if mode == 'painter' or mode == 'internet':
                await hist.clear_history()
            cur_reply_keyboard = dp.reply_keyboard[lang_mode].start
            await message_answer(message, dp.static_headers[lang_mode]['switch_programmer'],
                                 reply_markup=cur_reply_keyboard)
            await user_mode.set(message.from_user.id, 'programmer')
            return
        if message.text == dp.static_headers[lang_mode]['writer_button']:
            mode = await user_mode.get(message.from_user.id)
            await user_menu_state.set(message.from_user.id, 'work')
            if mode == 'painter' or mode == 'internet':
                await hist.clear_history()
            cur_reply_keyboard = dp.reply_keyboard[lang_mode].start
            await message_answer(message, dp.static_headers[lang_mode]['switch_writer'],
                                 reply_markup=cur_reply_keyboard)
            await user_mode.set(message.from_user.id, 'writer')
            return
        if message.text == dp.static_headers[lang_mode]['painter_button']:
            await user_menu_state.set(message.from_user.id, 'work')
            await message_answer(message, dp.static_headers[lang_mode]['switch_painter'],
                                 reply_markup=cur_reply_keyboard)
            await user_mode.set(message.from_user.id, 'painter')
            return
        if message.text == dp.static_headers[lang_mode]['internet_button']:
            await user_mode.set(message.from_user.id, 'internet')
            await user_menu_state.set(message.from_user.id, 'work')
            cur_reply_keyboard = dp.reply_keyboard[lang_mode].internet_panel
            await message_answer(message, dp.static_headers[lang_mode]['internet_welcome'], reply_markup=cur_reply_keyboard)
            return
        return
    if menu_state == 'internet_ref_mode_select':
        message.text = message.text.replace('*', '')
        is_success_select = False
        # if message.text == dp.static_headers[lang_mode]['none_ref_button']:
        #     await user_internet_ref_mode.set(message.from_user.id, 'no_ref')
        #     is_success_select = True
        if message.text == dp.static_headers[lang_mode]['only_ref_button']:
            await user_internet_ref_mode.set(message.from_user.id, 'only_ref')
            is_success_select = True
        if message.text == dp.static_headers[lang_mode]['ref_quotes_button']:
            await user_internet_ref_mode.set(message.from_user.id, 'ref_quotes')
            is_success_select = True
        if is_success_select:
            cur_reply_keyboard = dp.reply_keyboard[lang_mode].internet_panel
            await message_answer(message, dp.static_headers[lang_mode]['internet_welcome'], reply_markup=cur_reply_keyboard)
            await user_menu_state.set(message.from_user.id, 'work')
        return
    if menu_state == 'internet_use_chain_mode_select':
        message.text = message.text.replace('*', '')
        is_success_select = False
        if message.text == dp.static_headers[lang_mode]['no_chain_button']:
            await user_internet_resp_mode.set(message.from_user.id, 'no_chain')
            is_success_select = True
        if message.text == dp.static_headers[lang_mode]['use_chain_button']:
            await user_internet_resp_mode.set(message.from_user.id, 'use_chain')
            is_success_select = True
        if is_success_select:
            cur_reply_keyboard = dp.reply_keyboard[lang_mode].internet_panel
            await message_answer(message, dp.static_headers[lang_mode]['internet_welcome'], reply_markup=cur_reply_keyboard)
            await user_menu_state.set(message.from_user.id, 'work')
        return
    if menu_state == 'internet_market_mode_select':
        country_res = await send_api_search_country(dp, message.text)
        if country_res['success']:
            await user_mode.set(message.from_user.id, 'internet')
            await user_menu_state.set(message.from_user.id, 'work')
            await user_internet_market_mode.set(message.from_user.id, country_res['text'])
            await message_answer(message, dp.static_headers[lang_mode]['internet_welcome'],
                                 reply_markup=cur_reply_keyboard)
        else:
            cur_reply_keyboard = dp.reply_keyboard[lang_mode].empty
            await message_answer(message, dp.static_headers[lang_mode]['internet_market_select'],
                                 reply_markup=cur_reply_keyboard)

        return
    if menu_state == 'payment':
        try:
            grps = re.search(r"(\S*@[\S,^.]*\.\S*)", message.text).groups()
            if len(grps) > 0:
                mail = grps[0]
                await save_mail(message.from_user.id, mail)
                await user_menu_state.set(message.from_user.id, 'work')
                amount = await load_pay_amount(message.from_user.id)
                if amount is None:
                    await bot.send_message(message.from_user.id,
                                           dp.static_headers[lang_mode]['cash_in_welcome'],
                                           reply_markup=dp.inline_keyboard[lang_mode].amount_sbp_panel)
                    return
                await pay_ukassa(amount, message.from_user.id, mail)
        except Exception as e:
                await message_answer(message, dp.static_headers[lang_mode]['mail_error'], reply_markup=dp.inline_keyboard[lang_mode].no_mail_panel)
        return

    # проверка на то что запрос пользователь уже отправил, если нет, идем дальше, если да, отвечаем, что подожди формирую мысль
    if message.from_user.id in dp.user_in_request:
        await message_answer(message, dp.static_headers[lang_mode]['wait_answer'],
                             reply_markup=cur_reply_keyboard)
        return

    #запрашиваем текущий режим для пользователя
    mode = await user_mode.get(message.from_user.id)
    # проверка баланса
    cur_balance = await balance.get_balance()

    if mode == 'painter':
        if len(message.text) > 1000:
            await message_answer(message, dp.static_headers[lang_mode]["error_image_limit"],
                                 reply_markup=cur_reply_keyboard)
            return
        n_pict = await user_paint_mode.get(message.from_user.id)
        n_pict = int(n_pict)
        # проверка баланса
        res_balance = cur_balance >= 2000*n_pict
        if not res_balance:
            logger.info(
                f'user: {message.from_user.username} for image gen balance is finished, need: {2000*n_pict}, balance: {cur_balance}')
            if PAYMENTS or PAYMENTS_CRYPTO:
                await message_answer(message, dp.static_headers[lang_mode]['balance_is_over'],
                                     reply_markup=dp.inline_keyboard[lang_mode].cash_in_panel)
            elif PAYMENTS_STRIPE:
                await message_answer(message, dp.static_headers[lang_mode]['balance_is_over_en'],
                                     reply_markup=dp.inline_keyboard[lang_mode].cash_in_panel)

            else:
                await message_answer(message, dp.static_headers[lang_mode]['balance_is_over'],
                                     reply_markup=cur_reply_keyboard)
            return

        dp.user_in_request[message.from_user.id] = message.from_user.id, ChatActions.UPLOAD_PHOTO

        res = await send_api_image(dp, message.from_user.id, message.text, n=n_pict)
        try:
            if not res['success']:
                logger.debug(
                    f"response from openai api image is error: {res['error']} username: {message.from_user.username}")
                if res['error'] == 'openai server_error, safety system not allowed':
                    await message_answer(message, dp.static_headers[lang_mode]["error_image_safety"],
                                         reply_markup=cur_reply_keyboard)
                else:
                    await message_answer(message, dp.static_headers[lang_mode]['error_from_api'],
                                         reply_markup=cur_reply_keyboard)

            else:
                for url in res['url']:
                    await bot.send_photo(message.chat.id, photo=url, reply_markup=cur_reply_keyboard)
                    # обновляем баланс с учетом токенов запроса
                    await balance.update_balance(2000)

                    # только после удачного ответа, добавляем вопрос в историю
                    timestamp = datetime.datetime.now()
                    await hist.add_msg('gen image: '+message.text, timestamp, True)

                    # debug info
                    logger.debug(f'PROMPT HUMAN GEN IMAGE: {message.from_user.username}')

                    # debug info
                    logger.debug(
                        f"PROMPT BOT GEN IMAGE: {message.from_user.username}\n{url}")

                    # добавляем в историю ответ бота
                    timestamp = datetime.datetime.now()
                    await hist.add_msg('gen image: '+url, timestamp, False)
        except Exception as e:
            logger.error(f'exception in image gen: {e}')
        if message.from_user.id in dp.user_in_request:
            del dp.user_in_request[message.from_user.id]
        return


    if mode == 'internet':

        res_balance = cur_balance >= 5000
        if not res_balance:
            logger.info(
                f'user: {message.from_user.username} for internet query balance is finished, balance: {cur_balance}')
            if PAYMENTS or PAYMENTS_CRYPTO or PAYMENTS_STRIPE:
                await message_answer(message, dp.static_headers[lang_mode]['internet_balance_is_over'],
                                     reply_markup=dp.inline_keyboard[lang_mode].cash_in_panel)
            else:
                await message_answer(message, dp.static_headers[lang_mode]['internet_balance_is_over'],
                                     reply_markup=cur_reply_keyboard)
            return

        internet_ref_mode = await user_internet_ref_mode.get(message.from_user.id)
        if internet_ref_mode =='no_ref':
            await user_internet_ref_mode.set(message.from_user.id, 'only_ref')

        use_chain = await user_internet_resp_mode.get(message.from_user.id)
        market = await user_internet_market_mode.get(message.from_user.id)
        model = await user_model.get(message.from_user.id)
        if model == 'turbo':
            model = 'gpt-3.5-turbo'
        else:
            model = 'text-davinci-003'

        use_chain = use_chain == 'use_chain'

        dp.user_in_request[message.from_user.id] = message.from_user.id, ChatActions.TYPING
        res = await send_api_internet(dp, message.from_user.id, message.text, model, market, use_chain)
        if message.from_user.id in dp.user_in_request:
            del dp.user_in_request[message.from_user.id]

        if not res['success']:

            logger.error(
                f"response from internet api is error: {res['error']} username: {message.from_user.username} query: {message.from_user.id}")
            await message_answer(message, dp.static_headers[lang_mode]['error_from_api'],
                                 reply_markup=cur_reply_keyboard)
            return

        answer = res['text']
        await message_answer(message, answer, reply_markup=cur_reply_keyboard)


        if internet_ref_mode == 'ref_quotes':
            for ref in res['nodes']:
                refs = 'URL: ' + ref['url'] + '\n' + ref['text']
                await message_answer(message, refs, reply_markup=cur_reply_keyboard)
        elif internet_ref_mode == 'only_ref':
            refs = ''
            refs_set = set(rs['url'] for rs in res['nodes'])
            for ref in refs_set:
                refs += 'URL: ' + ref + '\n' + '\n'
            await message_answer(message, refs, reply_markup=cur_reply_keyboard)

        usage_total = res['token_usage']['embeddings_tokens']//50 + res['token_usage']['completions_tokens'] + \
                      res['token_usage']['chat_completions_tokens']//10
        await balance.update_balance(int(usage_total*1.2))

        return
    
    
    cur_pretext = dp.pretext
    if mode == 'programmer':
        cur_pretext = dp.pretext_programmer

    pretext = cur_pretext.replace('CURRENT_TIME', datetime.datetime.utcnow().strftime('%Y.%m.%d %H:%M:%S %A'))
    
    len_tokens_message = tokenizer.count_tokens(pretext+message.text, model_dict[model_mode])
    max_tokens = int(config('MAX_TOKENS')) - len_tokens_message
    if max_tokens < 500:
        logger.debug(
            f"user overloaded request: username: {message.from_user.username} len: {len_tokens_message} prompt: {message.text}")
        await message_answer(message, dp.static_headers[lang_mode]['max_tokens_in_request'], reply_markup=cur_reply_keyboard)
        return


    await bot.send_chat_action(message.chat.id, ChatActions.TYPING)
    dp.user_in_request[message.from_user.id] = message.from_user.id, ChatActions.TYPING
    items = await hist.get_history()

    msgs = []
    for item in items:
         # summarizing msg
        if item.is_sum:
            msgs.append(item.text + '\nThen conversation continues as follows:\n')
        elif item.user_id_dest == bot_id:
            msgs.append('Human: ' + item.text + '<|endofstatement|>\n')
        else:
            msgs.append('GPTie: ' + item.text + '<|endofstatement|>\n\n')

    text_hist = pretext+'\n' + ''.join(msgs)

    async def summary(msgs, max_tokens):
        summary_request_text = []
        summary_request_text.append(
            "The following is a conversation instruction set and a conversation"
            " between two people, a Human, and GPTie. Firstly, determine the Human's name from the conversation history, then summarize the conversation in the same language. Summarize the conversation in a detailed fashion. If Human mentioned their name, be sure to mention it in the summary. Pay close attention to things the Human has told you, such as personal details.\n"
        )
        summary_request_text.append(''.join(msgs))
        summary_request_text = "".join(summary_request_text)+"Summarize the context and conversation together in fine details step by step"
        len_tokens = tokenizer.count_tokens(summary_request_text, "gpt-3.5-turbo")
        res_balance = cur_balance >= len_tokens
        if not res_balance:
            if message.from_user.id in dp.user_in_request:
                del dp.user_in_request[message.from_user.id]
            logger.info(
                f'user: {message.from_user.username} balance is over, need: {len_tokens}, balance: {cur_balance}')
            if PAYMENTS_STRIPE:
                await message_answer(message, dp.static_headers[lang_mode]['balance_is_over_en'],
                                    reply_markup=cur_reply_keyboard)
            else:
                await message_answer(message, dp.static_headers[lang_mode]['balance_is_over'],
                                    reply_markup=cur_reply_keyboard)
            return None, 0
        
        logger.debug(f'SUMMARY HUMAN: {message.from_user.username}: len: {len_tokens}\n{summary_request_text}')
        res = await send_api_chat(dp, message.from_user.id, message.from_user.username, [{"role": "user", "content": summary_request_text}], 0.5, 1, max_tokens)
        if not res['success']:
            if message.from_user.id in dp.user_in_request:
                del dp.user_in_request[message.from_user.id]
            logger.error(f"response from openai api is error: {res['error']} username: {message.from_user.username} len: {len_tokens} prompt: {summary_request_text}")
            await message_answer(message, dp.static_headers[lang_mode]['error_from_api'], reply_markup=cur_reply_keyboard)
            return None, 0

        summarized_text = res['prompt']
        usage = res['usage']
        usage_total = usage['total_tokens']

        logger.debug(f"SUMMARY BOT: {message.from_user.username}: len_prompt: {usage['prompt_tokens']} len_completion: {usage['completion_tokens']} len_tot: {usage['total_tokens']}"
                     f"\n{summarized_text}")
        
        return summarized_text, usage_total
    
    summary_for_turbo_model=''
    #проверка на длину токенов
    len_history = tokenizer.count_tokens(text_hist, model_dict[model_mode])
    if tokenizer.count_tokens(text_hist, "gpt-3.5-turbo") > int(config('MAX_TOKENS'))-600:
        logger.error('SUMMARY chunks')
        if len(msgs)==2:
            msgs_tail = msgs
            msgs_head = []
        else:
            msgs_tail = msgs[:-2].copy()
            msgs_head = msgs[-2:].copy()
        
        tokens_tail = tokenizer.count_tokens(''.join(msgs_tail), "gpt-3.5-turbo")
        tokens_head = tokenizer.count_tokens(''.join(msgs_head), "gpt-3.5-turbo")
        
        if tokens_tail > tokens_head:
            chunk = msgs_tail
            tail = True
        else:
            chunk = msgs_head
            tail = False

        if tail:
            t = int(config('MAX_TOKENS'))-tokens_tail-120
            if t<=0:
                await message_answer(message, dp.static_headers[lang_mode]['max_tokens_in_request'], reply_markup=cur_reply_keyboard)
                if message.from_user.id in dp.user_in_request:
                    del dp.user_in_request[message.from_user.id]
                return
            
            summarized_text, usage_total = await summary(chunk, min(t,1200))
            if summarized_text is None:
                return
            msgs = [summarized_text+'\n']
            for m in msgs_head:
                msgs.append(m)
        else:
            t = int(config('MAX_TOKENS'))-tokens_head-120
            if t<=0:
                await message_answer(message, dp.static_headers[lang_mode]['max_tokens_in_request'], reply_markup=cur_reply_keyboard)
                if message.from_user.id in dp.user_in_request:
                    del dp.user_in_request[message.from_user.id]
                return
            
            summarized_text, usage_total = await summary(chunk, min(t,1200))
            if summarized_text is None:
                return
            msgs = []
            for m in msgs_tail:
                msgs.append(m)
            msgs.append('\nConversation continues as follows:\n'+summarized_text)

        text_hist = pretext+'\n'+ ''.join(msgs)+'\n'
        len_history = tokenizer.count_tokens(text_hist, model_dict[model_mode])
        await balance.update_balance(usage_total)
        summary_for_turbo_model = ''.join(msgs).replace('<|endofstatement|>', '')

    if (len_history > int(config('SUMMARIZING_THRESHOLD'))) or ((len_history+ len_tokens_message) > (int(config('MAX_TOKENS'))-600)):
        logger.error(f"SUMMARY: {len_history+ len_tokens_message}")
        #summarizing
        t = int(config('MAX_TOKENS'))-len_history
        if t<=0:
                await message_answer(message, dp.static_headers[lang_mode]['max_tokens_in_request'], reply_markup=cur_reply_keyboard)
                if message.from_user.id in dp.user_in_request:
                    del dp.user_in_request[message.from_user.id]
                return
        
        summarized_text, usage_total = await summary(msgs, min(t, 1350))
        if summarized_text is None:
            return
        await balance.update_balance(usage_total)
    
        #составить запрос заново
        new_conversation_history = ''
        new_conversation_history += "This conversation has some context from earlier, which has been summarized as follows:\n"
        new_conversation_history += summarized_text
        summary_for_turbo_model = new_conversation_history
        # обнулить историю и записать новое сообщение с пометкой, что это резюме
        await hist.clear_history()
        timestamp = datetime.datetime.now()
        await hist.add_msg(new_conversation_history, timestamp, True, True)

        new_conversation_history += "\nContinue the conversation, paying very close attention to things Human told you, such as their name, and personal details.\n"

        pretext = cur_pretext.replace('CURRENT_TIME', datetime.datetime.utcnow().strftime('%Y.%m.%d %H:%M:%S %A'))
        text_hist = pretext +'\n' + new_conversation_history+'\n'
        len_history = tokenizer.count_tokens(text_hist, model_dict[model_mode])

    # проверка баланса
    cur_balance = await balance.get_balance()
    res_balance = cur_balance >= (len_history+len_tokens_message)
    if not res_balance:
        logger.info(
            f'user: {message.from_user.username} balance is finished, need: {len_history+len_tokens_message}, balance: {cur_balance}')
        if PAYMENTS or PAYMENTS_CRYPTO:
            await message_answer(message, dp.static_headers[lang_mode]['balance_is_over'], reply_markup=dp.inline_keyboard[lang_mode].cash_in_panel)
        elif PAYMENTS_STRIPE:
            await message_answer(message, dp.static_headers[lang_mode]['balance_is_over_en'], reply_markup=dp.inline_keyboard[lang_mode].cash_in_panel)
        else:
            await message_answer(message, dp.static_headers[lang_mode]['balance_is_over'], reply_markup=cur_reply_keyboard)

        if message.from_user.id in dp.user_in_request:
            del dp.user_in_request[message.from_user.id]
        return

    # проверка, что запрос не превышает максимальное кол-во токенов за запрос
    max_tokens = int(config('MAX_TOKENS')) - (len_history+len_tokens_message+300) 
    if max_tokens < 0:
        logger.debug(
            f"user overloaded request: username: {message.from_user.username} len: {len_history+len_tokens_message} prompt: {text_hist}+'\nHuman: '+{message.text}")
        await message_answer(message, dp.static_headers[lang_mode]['max_tokens_in_request'], reply_markup=cur_reply_keyboard)
        if message.from_user.id in dp.user_in_request:
            del dp.user_in_request[message.from_user.id]
        return

    temp = float(config('TEMPERATURE'))
    top_p = float(config('TOP_P'))

    if mode == 'programmer':
        temp = 0.0
        top_p = 1.0



    if model_mode=='gpt':
        text_hist += 'Human: ' + message.text + '<|endofstatement|>\n' + 'GPTie: '
        logger.info(f'user: {message.from_user.username}, model: GPT')
        res = await send_api(dp, message.from_user.id, message.from_user.username, text_hist, temp, top_p, max_tokens)
        if message.from_user.id in dp.user_in_request:
            del dp.user_in_request[message.from_user.id]
        if not res['success']:
            logger.error(
                f"response from openai api is error: {res['error']} username: {message.from_user.username} len: {len_history+len_tokens_message} prompt: {text_hist}")
            await message_answer(message, dp.static_headers[lang_mode]['error_from_api'], reply_markup=cur_reply_keyboard)
            return
        usage = res['usage']
        usage_total = usage['total_tokens']
    else:
        logger.info(f'user: {message.from_user.username}, model: TURBO')
        msgs = []
        pretext = 'You are GPTie. You are talking to a Human.\n '+pretext.split('for GPTie')[1].split('Current time is')[0]
        msgs.append({"role": "system", "content": pretext+f" Today is {datetime.datetime.utcnow().strftime('%Y.%m.%d %H:%M:%S %A')} UTC"})
        if len(summary_for_turbo_model)>2:
            msgs.append({"role": "user", "content": summary_for_turbo_model})
            msgs.append({"role": "assistant", "content": "I'm GPTie. I'm ready to answer your questions."})
            msgs.append({"role": "user", "content": message.text})
        else:
            for item in items:
                if item.is_sum:
                    msgs.append({"role": "user", "content": item.text})
                elif item.user_id_dest == bot_id:
                    msgs.append({"role": "user", "content": item.text})
                else:
                    msgs.append({"role": "assistant", "content": item.text})
            msgs.append({"role": "user", "content": message.text})

        res = await send_api_chat(dp, message.from_user.id, message.from_user.username, msgs, temp, top_p, max_tokens)
        if message.from_user.id in dp.user_in_request:
            del dp.user_in_request[message.from_user.id]
        if not res['success']:
            logger.error(
                f"response from openai api chat completions: username: {message.from_user.username} len: {len_history+len_tokens_message} prompt: {msgs}")
            await message_answer(message, dp.static_headers[lang_mode]['error_from_api'], reply_markup=cur_reply_keyboard)
            return
        usage = res['usage']
        usage_total = usage['total_tokens']//10
        text_hist = msgs[-1]

    answer = res['prompt']
    
    # обновляем баланс с учетом токенов запроса и токенов ответа
    await balance.update_balance(usage_total)

    # только после удачного ответа, добавляем вопрос в историю
    timestamp = datetime.datetime.now()
    await hist.add_msg(message.text, timestamp, True)

    # debug info
    logger.debug(f'PROMPT HUMAN: {message.from_user.username}: len: {len_history+len_tokens_message}\n{text_hist}')

    # debug info
    logger.debug(
        f"PROMPT BOT: {message.from_user.username}: len_prompt: {usage['prompt_tokens']} len_completion: {usage['completion_tokens']} len_tot: {usage['total_tokens']}"
        f"\n{answer}")

    #добавляем в историю ответ бота
    timestamp = datetime.datetime.now()
    await hist.add_msg(answer, timestamp, False)

    await message_answer(message, answer, reply_markup=cur_reply_keyboard)


if __name__ == '__main__':
    logger.info('start')
    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(loop_periodic(dp)), loop.create_task(dp.start_polling())]
    wait_tasks = asyncio.wait(tasks)
    loop.run_until_complete(wait_tasks)
    loop.close()



