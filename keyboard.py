from aiogram import Bot, types
from decouple import config

class BotKeyboard:
    def __init__(self, dp_lang, list_of_langs, list_of_models):

        self.empty = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        self.start = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)

        self.restart_but = types.KeyboardButton(dp_lang['restart_button'])
        self.start.row(self.restart_but)

        self.select_mode_but = types.KeyboardButton(dp_lang['select_mode_button'])
        self.model_but = types.KeyboardButton(dp_lang['model_button'])
        self.start.row(self.select_mode_but, self.model_but)

        self.help_but = types.KeyboardButton(dp_lang['help_button'])
        self.balance_but = types.KeyboardButton(dp_lang['balance_button'])
        self.language_but = types.KeyboardButton(dp_lang['language_button'])
        self.start.row(self.help_but, self.balance_but, self.language_but)
        ###
        self.select_mode_panel = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)

        self.coder_but = types.KeyboardButton(dp_lang['coder_button'])
        self.writer_but = types.KeyboardButton(dp_lang['writer_button'])
        self.select_mode_panel.row(self.coder_but, self.writer_but)
        self.painter_but = types.KeyboardButton(dp_lang['painter_button'])
        self.select_mode_panel.row(self.painter_but)
        self.internet_but = types.KeyboardButton(dp_lang['internet_button'])
        self.select_mode_panel.row(self.internet_but)
        ###
        self.painter_panel = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)

        self.painter_panel.row(self.select_mode_but)

        self.one_pict_but = types.KeyboardButton(dp_lang['1picture'])
        self.three_pict_but = types.KeyboardButton(dp_lang['3picture'])
        self.five_pict_but = types.KeyboardButton(dp_lang['5picture'])
        self.painter_panel.row(self.one_pict_but, self.three_pict_but, self.five_pict_but)
        self.painter_panel.row(self.help_but, self.balance_but, self.language_but)
        ###
        self.language_panel = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        self.model_panel = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)

        self.langs_buts = []
        for lang in list_of_langs:
            self.langs_buts.append(types.KeyboardButton(lang))
        self.language_panel.row(*self.langs_buts)

        self.models_buts = []
        for model in list_of_models:
            self.models_buts.append(types.KeyboardButton(model))
        self.model_panel.row(*self.models_buts)
        ###
        self.internet_panel = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        self.refs_quotes_but = types.KeyboardButton(dp_lang['refs_quotes_button'])
        self.use_chain_but = types.KeyboardButton(dp_lang['select_mode_chain_button'])
        self.market_but = types.KeyboardButton(dp_lang['market_button'])
        self.internet_panel.row(self.refs_quotes_but, self.use_chain_but, self.market_but)
        self.internet_panel.row(self.select_mode_but, self.model_but)
        self.internet_panel.row(self.help_but, self.balance_but, self.language_but)

        ###
        self.ref_mode_select = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        #self.none_ref_but = types.KeyboardButton(dp_lang['none_ref_button'])
        self.only_ref_but = types.KeyboardButton(dp_lang['only_ref_button'])
        self.ref_quotes_but = types.KeyboardButton(dp_lang['ref_quotes_button'])
        self.ref_mode_select.row(self.only_ref_but, self.ref_quotes_but)

        self.use_chain_select = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        self.no_chain = types.KeyboardButton(dp_lang['no_chain_button'])
        self.use_chain = types.KeyboardButton(dp_lang['use_chain_button'])
        self.use_chain_select.row(self.no_chain, self.use_chain)


class InlineBotKeyboard:
    def __init__(self, dp_lang):

        PAYMENTS = config("PAYMENTS").lower() in ('true', '1', 't')
        PAYMENTS_CRYPTO = config("PAYMENTS_CRYPTO").lower() in ('true', '1', 't')
        PAYMENTS_STRIPE = config("PAYMENTS_STRIPE").lower() in ('true', '1', 't')
        REDIRECT = config("REDIRECT").lower() in ('true', '1', 't')

        #############
        self.cash_in_but = types.InlineKeyboardButton(dp_lang['cash_in_button'], callback_data='cash_in_button')
        self.ref_but = types.InlineKeyboardButton(dp_lang['ref_button'], callback_data='ref_button')
        self.cash_in_panel = types.InlineKeyboardMarkup().add(self.cash_in_but, self.ref_but)

        pay_methods=[]
        if PAYMENTS:
            self.sbp_but = types.InlineKeyboardButton(dp_lang['sbp_button'], callback_data='sbp_button')
            pay_methods.append(self.sbp_but)
        if PAYMENTS_CRYPTO:
            self.crypto_but = types.InlineKeyboardButton(dp_lang['crypto_button'], callback_data='crypto_button')
            pay_methods.append(self.crypto_but)

        if PAYMENTS_STRIPE or REDIRECT:
            self.stripe_but = types.InlineKeyboardButton(dp_lang['stripe_button'], callback_data='stripe_button')
            pay_methods.append(self.stripe_but)

        if len(pay_methods) != 0:
            self.select_pay_method = types.InlineKeyboardMarkup().row(*pay_methods)


        self.no_mail_but = types.InlineKeyboardButton(dp_lang['no_mail_button'], callback_data='no_mail_button')
        self.no_mail_panel = types.InlineKeyboardMarkup().add(self.no_mail_but)

        if PAYMENTS:
            self.cash_in_50_but = types.InlineKeyboardButton(dp_lang['cash_in_50_button'],
                                                         callback_data='cash_in_50_button')
            self.cash_in_100_but = types.InlineKeyboardButton(dp_lang['cash_in_100_button'],
                                                          callback_data='cash_in_100_button')
            self.cash_in_200_but = types.InlineKeyboardButton(dp_lang['cash_in_200_button'],
                                                          callback_data='cash_in_200_button')
            self.cash_in_500_but = types.InlineKeyboardButton(dp_lang['cash_in_500_button'],
                                                          callback_data='cash_in_500_button')
            self.cash_in_1000_but = types.InlineKeyboardButton(dp_lang['cash_in_1000_button'],
                                                          callback_data='cash_in_1000_button')
            self.cash_in_2000_but = types.InlineKeyboardButton(dp_lang['cash_in_2000_button'],
                                                          callback_data='cash_in_2000_button')
            self.cash_in_5000_but = types.InlineKeyboardButton(dp_lang['cash_in_5000_button'],
                                                          callback_data='cash_in_5000_button')

            self.amount_sbp_panel = types.InlineKeyboardMarkup()
            self.amount_sbp_panel.row(self.cash_in_5000_but)
            self.amount_sbp_panel.row(self.cash_in_2000_but, self.cash_in_1000_but)
            self.amount_sbp_panel.row(self.cash_in_500_but, self.cash_in_200_but, self.cash_in_100_but, self.cash_in_50_but)

        if PAYMENTS_CRYPTO:
            self.cash_in_3usd_but = types.InlineKeyboardButton(dp_lang['cash_in_3usd_button'],
                                                             callback_data='cash_in_3usd_button')
            self.cash_in_5usd_but = types.InlineKeyboardButton(dp_lang['cash_in_5usd_button'],
                                                              callback_data='cash_in_5usd_button')
            self.cash_in_10usd_but = types.InlineKeyboardButton(dp_lang['cash_in_10usd_button'],
                                                              callback_data='cash_in_10usd_button')
            self.cash_in_20usd_but = types.InlineKeyboardButton(dp_lang['cash_in_20usd_button'],
                                                              callback_data='cash_in_20usd_button')
            self.cash_in_50usd_but = types.InlineKeyboardButton(dp_lang['cash_in_50usd_button'],
                                                               callback_data='cash_in_50usd_button')
            self.cash_in_100usd_but = types.InlineKeyboardButton(dp_lang['cash_in_100usd_button'],
                                                               callback_data='cash_in_100usd_button')
            self.cash_in_500usd_but = types.InlineKeyboardButton(dp_lang['cash_in_500usd_button'],
                                                               callback_data='cash_in_500usd_button')

            self.amount_crypto_panel = types.InlineKeyboardMarkup()
            self.amount_crypto_panel.row(self.cash_in_500usd_but)
            self.amount_crypto_panel.row(self.cash_in_100usd_but, self.cash_in_50usd_but)
            self.amount_crypto_panel.row(self.cash_in_20usd_but, self.cash_in_10usd_but, self.cash_in_5usd_but,
                                         self.cash_in_3usd_but)

        if PAYMENTS_STRIPE:
            self.cash_in_5usd_stripe_but = types.InlineKeyboardButton(dp_lang['cash_in_5usd_stripe_button'],
                                                             callback_data='cash_in_5usd_stripe_button')
            self.cash_in_10usd_stripe_but = types.InlineKeyboardButton(dp_lang['cash_in_10usd_stripe_button'],
                                                              callback_data='cash_in_10usd_stripe_button')
            self.cash_in_15usd_stripe_but = types.InlineKeyboardButton(dp_lang['cash_in_15usd_stripe_button'],
                                                              callback_data='cash_in_15usd_stripe_button')
            self.cash_in_25usd_stripe_but = types.InlineKeyboardButton(dp_lang['cash_in_25usd_stripe_button'],
                                                              callback_data='cash_in_25usd_stripe_button')
            self.cash_in_50usd_stripe_but = types.InlineKeyboardButton(dp_lang['cash_in_50usd_stripe_button'],
                                                               callback_data='cash_in_50usd_stripe_button')
            self.cash_in_100usd_stripe_but = types.InlineKeyboardButton(dp_lang['cash_in_100usd_stripe_button'],
                                                               callback_data='cash_in_100usd_stripe_button')
            self.cash_in_500usd_stripe_but = types.InlineKeyboardButton(dp_lang['cash_in_500usd_stripe_button'],
                                                               callback_data='cash_in_500usd_stripe_button')
            self.amount_stripe_panel = types.InlineKeyboardMarkup()

            self.amount_stripe_panel.row(self.cash_in_500usd_stripe_but)
            self.amount_stripe_panel.row(self.cash_in_100usd_stripe_but, self.cash_in_50usd_stripe_but)
            self.amount_stripe_panel.row(self.cash_in_25usd_stripe_but, self.cash_in_15usd_stripe_but, self.cash_in_10usd_stripe_but,
                                         self.cash_in_5usd_stripe_but)
            

