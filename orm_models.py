from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, String, text, Boolean, Date, Float
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class User(Base):
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True, unique=True, server_default=text("nextval('user_id_seq'::regclass)"))
    telegram_id = Column(BigInteger)
    username = Column(String)
    is_admin = Column(Boolean)


class Message(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True, unique=True, server_default=text("nextval('messages_id_seq'::regclass)"))
    datetime = Column(DateTime)
    user_id_src = Column(ForeignKey('user.id'))
    text = Column(String)
    user_id_dest = Column(Integer)
    is_sum = Column(Boolean)

    user = relationship('User')


class StartHistory(Base):
    __tablename__ = 'start_history'

    user_id = Column(ForeignKey('user.id'))
    msg_id = Column(ForeignKey('messages.id'))
    id = Column(Integer, primary_key=True, unique=True, server_default=text("nextval('start_history_id_seq'::regclass)"))

    msg = relationship('Message')
    user = relationship('User')


class UserBalance(Base):
    __tablename__ = 'user_balance'

    id = Column(Integer, primary_key=True, unique=True,
                server_default=text("nextval('user_balance_id_seq'::regclass)"))
    id_user = Column(ForeignKey('user.id'))
    balance = Column(Integer)
    save_dt = Column(Date)

    user = relationship('User')


class Payment(Base):
    __tablename__ = 'payments'

    id = Column(Integer, primary_key=True, unique=True, server_default=text("nextval('payments_id_seq'::regclass)"))
    user_id = Column(ForeignKey('user.id'))
    tokens = Column(Integer)
    dt = Column(DateTime)
    currency = Column(String)
    amount = Column(Float)
    transaction_id = Column(String)
    method = Column(String)

    user = relationship('User')

###########################################


class CommonBalance(Base):
    __tablename__ = 'common_balance'

    id = Column(Integer, primary_key=True, unique=True, server_default=text("nextval('common_balance_id_seq'::regclass)"))
    telegram_id = Column(BigInteger, nullable=False, unique=True)
    balance = Column(Integer, server_default=text("0"))
    save_dt = Column(Date)


class CommonPayment(Base):
    __tablename__ = 'common_payments'

    id = Column(Integer, primary_key=True, unique=True, server_default=text("nextval('common_payments_id_seq'::regclass)"))
    user_id = Column(ForeignKey('common_balance.id'))
    tokens = Column(Integer)
    dt = Column(DateTime)
    currency = Column(String)
    amount = Column(Float)
    transaction_id = Column(String)
    method = Column(String)
    bot = Column(String)

    user = relationship('CommonBalance')