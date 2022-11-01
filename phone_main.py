import time
import glob
from datetime import datetime, timedelta, timezone
import threading
import random
import matplotlib.pyplot as plt
from tinkoff.invest.services import Services
from tinkoff.invest import (
    OrderDirection,
    OrderType,
    CandleInterval,
    CandleInstrument,
    LastPriceInstrument,
    Client,
    MarketDataRequest,
    MarketDataResponse,
    SubscribeCandlesRequest,
    SubscriptionAction,
    SubscriptionInterval,
    SubscribeLastPriceRequest,
    RequestError
)


#  print(help('modules'))


class Tools:
    """Класс инструментов обработки"""

    @staticmethod
    def compound(units, nano):
        return float(
            str(units) + "." + (str(nano) if len(str(nano)) == 9 else "0" + str(nano))
        ) if units >= 0 and nano >= 0 else -float(
            str(abs(units)) + "." + (str(abs(nano)) if len(str(abs(nano))) == 9 else "0" + str(abs(nano)))
        )

    @staticmethod
    def adding_missing_elements(old_list: list, new_list: list):
        for i in new_list:
            if i not in old_list:
                old_list.append(i)
        return old_list

    @staticmethod
    def start_timer(stop_timer: str):
        day, hour, minute = [int(i) for i in stop_timer.split(":")]
        print("\nОжидание...")
        timer = True
        while timer:
            datetime_now = datetime.now()
            if datetime_now.day == day:
                if datetime_now.hour == hour:
                    if datetime_now.minute == minute:
                        print("\nTimer stopped")
                        timer = False
                    else:
                        time.sleep(1)
                else:
                    time.sleep(59)
            else:
                time.sleep(59)

    @staticmethod
    def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end=""):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end, flush=True)
        if iteration == total:
            print()

    @staticmethod
    def plotting(
            x: list,
            y: list,
            x_label: str = 'Numb',
            y_label: str = 'Price',
            plot_color: str = "#00ff91",
            style: str = 'dark_background',
            show: bool = True,
            save: bool = False,
    ):
        plt.style.use(style)
        plt.plot(x, y, color=plot_color)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if save:
            plt.savefig("plot.png")
        if show:
            plt.show()


class SmoothingMethods:
    """Класс методов сглаживания числовых последовательностей"""

    def __init__(self, values: list, period: int):
        self.values = values
        self.len_values = len(self.values)
        self.period = period

    def ema_smoothing(self, w=1.0, out=0):
        ema_result, first_ema = [], (self.values[0] + self.values[0]) / 2
        alpha = w / (self.period + (w - 1)) if self.period > 0 else w / (self.len_values + (w - 1))
        for value in self.values[:self.period if self.period > 0 else self.len_values]:
            first_ema = alpha * value + (1 - alpha) * first_ema
        ema_result.append(first_ema)
        if 0 < self.period < self.len_values:
            for obj_numb in range(1, self.len_values - (self.period - 1)):
                ema_list = []
                for obj in self.values[obj_numb:obj_numb + self.period]:
                    ema_list.append(alpha * obj + (1 - alpha) * ema_result[-1])
                ema_result.append(ema_list[-1])
        return ema_result if out == 0 else ema_result[-1]

    def ma_smoothing(self, out=0):
        if out == 1:
            return sum(self.values[-self.period:]) / len(self.values[-self.period:])
        ma_result_list = []
        if 0 < self.period < self.len_values:
            for obj_numb in range(self.len_values - (self.period - 1)):
                ma_result_list.append(sum(self.values[obj_numb:obj_numb + self.period]) /
                                      len(self.values[obj_numb:obj_numb + self.period]))
        else:
            ma_result_list.append(sum(self.values) / self.len_values)
        return ma_result_list

    @staticmethod
    def ema_smoothing_plus_one(ema_last_value: float, new_value: float, ema_last_period: int, w=1.0):
        alpha = w / (ema_last_period + (w - 1))
        return alpha * new_value + (1 - alpha) * ema_last_value

    def __str__(self):
        return str(self.__dict__)


class TOKENS:
    """Класс определения токена"""
    with open("token.txt") as file:
        Tinkoff_Token = file.read()


class SettingsHandler:
    """Класс обработки настроек"""

    def __init__(self, settings: tuple):
        interval_argument = {"1m": CandleInterval.CANDLE_INTERVAL_1_MIN, "5m": CandleInterval.CANDLE_INTERVAL_5_MIN}
        subscription_interval_argument = {"1m": SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE,
                                          "5m": SubscriptionInterval.SUBSCRIPTION_INTERVAL_FIVE_MINUTES}
        self.days = int(settings[0])
        self.post_order_quantity = int(settings[1])
        self.trading_direction_mode = settings[2].replace(" ", "")
        self.strategy_name = settings[3].replace(" ", "")
        self.candle_type = int(settings[4]) if int(settings[4]) <= 2 else 1
        self.candle_interval = interval_argument[settings[5]]
        self.str_candle_type = CandleType.ClassicCandles \
            if settings[4] in (CandleType.ClassicCandles, str(CandleType.ClassicCandlesValue)) \
            else CandleType.HeikinAshiCandles
        self.str_candle_interval = settings[5].replace(" ", "")
        self.subscription_candle_interval = subscription_interval_argument[self.str_candle_interval]
        self.adx_length = int(settings[6])
        self.chandelier_exit_length = int(settings[7])
        self.chandelier_exit_factor = float(settings[8])
        self.ema_length = int(settings[9])
        self.stop_loss = [float(sp) for sp in settings[10].split(",")]
        self.stop_loss_level_update = float(settings[11])
        self.stop_loss_quantity = int(settings[12])
        self.take_profit = [float(sp) for sp in settings[13].split(",")]
        self.take_profit_quantity = int(settings[14])
        self.real_order = bool(int(settings[15]))
        self.first_signal_processing = bool(int(settings[16]))
        self.logging = bool(int(settings[17]))
        self.print_style = int(settings[18]) if int(settings[18]) <= 1 else 0
        self.strategy_analytics = bool(int(settings[19]))
        self.session_id = random.randrange(10000, 99999)

    def change_settings(self):
        def print_setting_arguments():
            print()
            item_numb = 0
            for item in self.__dict__:
                print(f"{item_numb}. {item} = {self.__dict__[item]}")
                item_numb += 1

        print_setting_arguments()
        finish_input = 0
        while finish_input not in ("", "\n", "-"):
            search_argument = input("\n'*' вывести параметры еще раз\n"
                                    "Введите номер или имя параметра которое нужно изменить:")
            if search_argument == "*":
                print_setting_arguments()
                continue
            try:
                key_list = [i for i in self.__dict__]
                setting_value = input(f"\n{key_list[int(search_argument)]} = ")
                if type(self.__dict__[key_list[int(search_argument)]]) == int:
                    self.__dict__[key_list[int(search_argument)]] = int(setting_value)
                elif type(self.__dict__[key_list[int(search_argument)]]) == float:
                    self.__dict__[key_list[int(search_argument)]] = float(setting_value)
                elif type(self.__dict__[key_list[int(search_argument)]]) == str:
                    self.__dict__[key_list[int(search_argument)]] = str(setting_value)
                elif type(self.__dict__[key_list[int(search_argument)]]) == bool:
                    self.__dict__[key_list[int(search_argument)]] = bool(int(setting_value))
                elif type(self.__dict__[key_list[int(search_argument)]]) == list \
                        and type(self.__dict__[key_list[int(search_argument)]][0]) == float:
                    self.__dict__[key_list[int(search_argument)]] = [float(sp) for sp in setting_value.split(",")]
                else:
                    print(f"\nТип данных '{type(self.__dict__[key_list[int(search_argument)]])}' "
                          f"не подходит для изменения")
            except (ValueError, TypeError, IndexError):
                if self.__dict__.get(search_argument):
                    setting_value = input(f"\n{search_argument} = ")
                    if type(self.__dict__[search_argument]) == int:
                        self.__dict__[search_argument] = int(setting_value)
                    elif type(self.__dict__[search_argument]) == float:
                        self.__dict__[search_argument] = float(setting_value)
                    elif type(self.__dict__[search_argument]) == str:
                        self.__dict__[search_argument] = str(setting_value)
                    elif type(self.__dict__[search_argument]) == bool:
                        self.__dict__[search_argument] = bool(int(setting_value))
                    elif type(self.__dict__[search_argument]) == list \
                            and type(self.__dict__[search_argument][0]) == float:
                        self.__dict__[search_argument] = [float(sp) for sp in setting_value.split(",")]
                    else:
                        print(f"\nТип данных '{type(self.__dict__[search_argument])}' не подходит для изменения")
                    print(self.adx_length, type(self.adx_length))
                else:
                    print(f"\nКлюч {search_argument} не найден!")
            finish_input = input("\nНажмите Enter для выхода из режима изменения настроек "
                                 "(любую кнопку чтобы продолжить):")

    def __str__(self):
        return str(self.__dict__)


class Settings:
    """Класс запроса на получение настроек"""

    def __init__(self, default_path: str = r"test_setting.txt"):
        self.default_path = default_path

    def setting_request(self):
        setting_list = glob.glob("*_setting.txt")
        print("Выбор профиля настроек\nВаши профили:\n")
        numb_setting, path = 0, None
        for setting in setting_list:
            print(f"{numb_setting}. {setting}")
            numb_setting += 1
        try:
            inp = input("\n'*' запустить программу в режиме анализа стратегии\n"
                        "'-' запустить программу построение графика\n"
                        "Введите номер профиля настроек который хотите выбрать:")
            if inp == "*":
                return "strategy_analysis_mode"
            elif inp == "-":
                return "plotting_mode"
            path = setting_list[int(inp)]
        except (IndexError, ValueError):
            print("\nОшибка ввода...\n")
            self.setting_request()
        try:
            with open(path, encoding='utf-8') as filehandle:
                return SettingsHandler(
                    tuple([rl.rstrip().split('=')[1].replace(" ", "") for rl in filehandle.readlines()])
                )
        except FileNotFoundError:
            with open(self.default_path, encoding='utf-8') as filehandle:
                return SettingsHandler(
                    tuple([rl.rstrip().split('=')[1].replace(" ", "") for rl in filehandle.readlines()])
                )


class CandleArguments:
    """Класс определения аргументов свечи"""

    def __init__(self, open_, high_, low_, close_, volume_, time_: datetime, type_=None, interval_=None):
        self.open = open_
        self.high = high_
        self.low = low_
        self.close = close_
        self.volume = volume_
        self.time = time_
        self.type = type_ if type_ else "None"
        self.interval = interval_ if interval_ else "None"

    def __str__(self):
        return str(self.__dict__)


class Indicators:
    """Класс индикаторов"""

    class AdxArguments:
        """Класс определения аргументов adx индикатора"""

        def __init__(self, adx: list, di_plus_list: list, di_minus_list: list, last_atr: float,
                     last_dm_plus: float, last_dm_minus: float):
            self.adx_list = adx
            self.di_plus_and_minus_list = (di_plus_list, di_minus_list)
            self.last_adx = adx[-1]
            self.last_di_plus = di_plus_list[-1]
            self.last_di_minus = di_minus_list[-1]
            self.previous_adx = adx[-2]
            self.previous_di_plus = di_plus_list[-2]
            self.previous_di_minus = di_minus_list[-2]
            self.last_atr = last_atr
            self.last_dm_plus = last_dm_plus
            self.last_dm_minus = last_dm_minus

        def get_only_di_plus_list(self):
            return self.di_plus_and_minus_list[0]

        def get_only_di_minus_list(self):
            return self.di_plus_and_minus_list[1]

        def __str__(self):
            return self.__dict__

    class ChandelierExitArguments:
        """Класс определения аргументов chandelier exit индикатора"""

        def __init__(self, chandelier_exit_long_list: list, chandelier_exit_short_list: list, last_atr: float):
            self.CE_long_and_short_list = (chandelier_exit_long_list, chandelier_exit_short_list)
            self.last_CE_long_and_short = (chandelier_exit_long_list[-1], chandelier_exit_short_list[-1])
            self.previous_CE_long_and_short = (chandelier_exit_long_list[-2], chandelier_exit_short_list[-2])
            self.last_CE_long = chandelier_exit_long_list[-1]
            self.last_CE_short = chandelier_exit_short_list[-1]
            self.previous_CE_long = chandelier_exit_long_list[-2]
            self.previous_CE_short = chandelier_exit_short_list[-2]
            self.last_atr = last_atr

        def get_only_ce_long_list(self):
            return self.CE_long_and_short_list[0]

        def get_only_ce_short_list(self):
            return self.CE_long_and_short_list[1]

        def __str__(self):
            return self.__dict__

    def __init__(self, candles: list[CandleArguments]):
        self.candle_info = CandlesInfo(candles)
        self.candles = candles
        self.len_candles = len(self.candles)
        self.return_adx: Indicators.AdxArguments = None
        self.return_chandelier_exit: Indicators.ChandelierExitArguments = None
        self.return_ema = None
        self.return_ma = None

    def adx(self, period: int, w: float = 1.0):
        tr_list, dm_plus_list, dm_minus_list, di_plus_list, di_minus_list, dx_list = [], [], [], [], [], []
        for numb in range(0, self.len_candles):
            prev_candle = self.candles[numb - 1 if numb > 0 else 0]
            tr_list.append(
                max(self.candles[numb].high - self.candles[numb].low,
                    abs(self.candles[numb].high - prev_candle.close),
                    abs(self.candles[numb].low - prev_candle.close))
            )
            dm_plus_list.append(
                self.candles[numb].high - prev_candle.high
                if (self.candles[numb].high - prev_candle.high) >
                   (prev_candle.low - self.candles[numb].low) and self.candles[numb].high - prev_candle.high > 0
                else 0
            )
            dm_minus_list.append(
                prev_candle.low - self.candles[numb].low
                if (prev_candle.low - self.candles[numb].low) >
                   (self.candles[numb].high - prev_candle.high) and prev_candle.low - self.candles[numb].low > 0
                else 0
            )
        atr = SmoothingMethods(tr_list, period).ema_smoothing(w=w)
        smoothed_dm_plus = SmoothingMethods(dm_plus_list, period).ema_smoothing(w=w)
        smoothed_dm_minus = SmoothingMethods(dm_minus_list, period).ema_smoothing(w=w)
        for numb in range(0, len(atr)):
            try:
                di_plus, di_minus = smoothed_dm_plus[numb] / atr[numb] * 100, smoothed_dm_minus[numb] / atr[numb] * 100
            except ZeroDivisionError:
                di_plus, di_minus = 0, 0
            di_plus_list.append(di_plus)
            di_minus_list.append(di_minus)
            dx_list.append(abs(di_plus - di_minus) / abs(di_plus + di_minus) * 100 if di_plus + di_minus > 0 else 0)
        self.return_adx = Indicators.AdxArguments(
            SmoothingMethods(dx_list, period).ema_smoothing(w=w),
            di_plus_list, di_minus_list, atr[-1], smoothed_dm_plus[-1], smoothed_dm_minus[-1])
        return self.return_adx

    def chandelier_exit(self, period: int, factor: float, w: float = 1.0):
        max_high_list, min_low_list = [], []
        chandelier_exit_long, chandelier_exit_short = [], []
        tr_list = []
        for ic in range(0, self.len_candles):
            prev_candle_numb = ic - 1 if ic > 0 else 0
            tr_list.append(
                max(self.candles[ic].high - self.candles[ic].low,
                    abs(self.candles[ic].high - self.candles[prev_candle_numb].close),
                    abs(self.candles[ic].low - self.candles[prev_candle_numb].close))
            )
            if ic < len(self.candles) - (period - 1):
                high_n_, low_n_ = [], []
                for ic_ in self.candles[ic:ic + period]:
                    high_n_.append(ic_.high)
                    low_n_.append(ic_.low)
                max_high_list.append(max(high_n_))
                min_low_list.append(min(low_n_))
        atr = SmoothingMethods(tr_list, period).ema_smoothing(w=w)
        chandelier_exit_long.append(max_high_list[0] - atr[0] * factor)
        chandelier_exit_short.append(min_low_list[0] + atr[0] * factor)
        for ic in range(1, len(atr)):
            chandelier_exit_long.append(
                max(max_high_list[ic] - atr[ic] * factor, chandelier_exit_long[-1])
                if self.candles[period - 1 + (ic - 1)].close > chandelier_exit_long[-1]
                else max_high_list[ic] - atr[ic] * factor
            )
            chandelier_exit_short.append(
                min(min_low_list[ic] + atr[ic] * factor, chandelier_exit_short[-1])
                if self.candles[period - 1 + (ic - 1)].close < chandelier_exit_short[-1]
                else min_low_list[ic] + atr[ic] * factor
            )
        self.return_chandelier_exit = Indicators.ChandelierExitArguments(
            chandelier_exit_long, chandelier_exit_short, atr[-1])
        return self.return_chandelier_exit

    def adx_plus_one(self, period: int, last_adx: AdxArguments, w: float = 1.0):
        last_adx_list = last_adx.adx_list
        last_di_plus_list, last_di_minus_list = last_adx.di_plus_and_minus_list
        prev_candle = self.candles[-2]
        tr = max(
            self.candles[-1].high - self.candles[-1].low,
            abs(self.candles[-1].high - prev_candle.close),
            abs(self.candles[-1].low - prev_candle.close))
        dm_plus = self.candles[-1].high - prev_candle.high if \
            (self.candles[-1].high - prev_candle.high) > \
            (prev_candle.low - self.candles[-1].low) and self.candles[-1].high - prev_candle.high > 0 else 0
        dm_minus = prev_candle.low - self.candles[-1].low if \
            (prev_candle.low - self.candles[-1].low) > \
            (self.candles[-1].high - prev_candle.high) and prev_candle.low - self.candles[-1].low > 0 else 0
        atr = SmoothingMethods.ema_smoothing_plus_one(last_adx.last_atr, tr, period, w=w)
        smoothed_dm_plus = SmoothingMethods.ema_smoothing_plus_one(last_adx.last_dm_plus, dm_plus, period, w=w)
        smoothed_dm_minus = SmoothingMethods.ema_smoothing_plus_one(last_adx.last_dm_minus, dm_minus, period, w=w)
        di_plus, di_minus = smoothed_dm_plus / atr * 100, smoothed_dm_minus / atr * 100
        dx = abs(di_plus - di_minus) / abs(di_plus + di_minus) * 100 if di_plus + di_minus > 0 else 0
        last_di_plus_list.append(di_plus)
        last_di_minus_list.append(di_minus)
        last_adx_list.append(SmoothingMethods.ema_smoothing_plus_one(last_adx_list[-1], dx, period, w=w))
        self.return_adx = Indicators.AdxArguments(last_adx_list, last_di_plus_list, last_di_minus_list, atr,
                                                  smoothed_dm_plus, smoothed_dm_minus)
        return self.return_adx

    def chandelier_exit_plus_one(
            self, period: int, factor: float, last_chandelier_exit: ChandelierExitArguments, w: float = 1.0
    ):
        tr = max(
            self.candles[-1].high - self.candles[-1].low,
            abs(self.candles[-1].high - self.candles[-2].close),
            abs(self.candles[-1].low - self.candles[-2].close))
        max_high_n = max(self.candle_info.get_only_high_candles()[-period:])
        min_low_n = min(self.candle_info.get_only_low_candles()[-period:])
        atr = SmoothingMethods.ema_smoothing_plus_one(last_chandelier_exit.last_atr, tr, period, w=w)
        last_chandelier_exit_long_list, last_chandelier_exit_short_list = last_chandelier_exit.CE_long_and_short_list
        chandelier_exit_long = max(max_high_n - atr * factor, last_chandelier_exit_long_list[-1]) if \
            self.candles[-1].close > last_chandelier_exit_long_list[-1] else max_high_n - atr * factor
        chandelier_exit_short = min(min_low_n + atr * factor, last_chandelier_exit_short_list[-1]) if \
            self.candles[-1].close < last_chandelier_exit_short_list[-1] else min_low_n + atr * factor
        last_chandelier_exit_long_list.append(chandelier_exit_long)
        last_chandelier_exit_short_list.append(chandelier_exit_short)
        self.return_chandelier_exit = Indicators.ChandelierExitArguments(
            last_chandelier_exit_long_list, last_chandelier_exit_short_list, atr)
        return self.return_chandelier_exit

    def ema(self, period: int, type_: str = "close", w=1.0, out=0):
        if type_ == "open":
            values = self.candle_info.get_only_open_candles()
        elif type_ == "high":
            values = self.candle_info.get_only_high_candles()
        elif type_ == "low":
            values = self.candle_info.get_only_low_candles()
        elif type_ == "close":
            values = self.candle_info.get_only_close_candles()
        elif type_ == "4ohlc":
            values = self.candle_info.get_only_4ohlc_candles()
        else:
            values = self.candle_info.get_only_close_candles()
        self.return_ema = SmoothingMethods(values, period).ema_smoothing(w, out)
        return self.return_ema

    def ma(self, period: int, type_: str = "close", out=0):
        if type_ == "open":
            values = self.candle_info.get_only_open_candles()
        elif type_ == "high":
            values = self.candle_info.get_only_high_candles()
        elif type_ == "low":
            values = self.candle_info.get_only_low_candles()
        elif type_ == "close":
            values = self.candle_info.get_only_close_candles()
        elif type_ == "4ohlc":
            values = self.candle_info.get_only_4ohlc_candles()
        else:
            values = self.candle_info.get_only_close_candles()
        self.return_ma = SmoothingMethods(values, period).ma_smoothing(out)
        return self.return_ma

    def ema_plus_one(self, previous_ema_period: int, previous_ema_list: list, type_: str = "close", w=1.0, out=0):
        if type_ == "open":
            value = self.candles[-1].open
        elif type_ == "high":
            value = self.candles[-1].high
        elif type_ == "low":
            value = self.candles[-1].low
        elif type_ == "close":
            value = self.candles[-1].close
        elif type_ == "4ohlc":
            value = self.candle_info.last_4ohlc_candle
        else:
            value = self.candles[-1].close
        previous_ema_list.append(
            SmoothingMethods.ema_smoothing_plus_one(previous_ema_list[-1], value, previous_ema_period, w=w))
        self.return_ema = previous_ema_list if out == 0 else previous_ema_list[-1]
        return self.return_ema

    def __str__(self):
        return str(self.__dict__)


class InstrumentArguments:
    """Класс определения аргументов инструмента торговли"""

    def __init__(
            self,
            name_=None,
            ticker_=None,
            figi_=None,
            uid_=None,
            type_=None,
            currency_=None,
            trading_status_=None,
            short_enabled_flag_=None,
            exchange_=None,
            real_exchange_=None,
            sector_=None,
            lot_=None,
            div_yield_flag_=None,
            country_of_risk_=None,
            country_of_risk_name_=None,
            api_trade_available_flag_=None,
            first_1min_candle_date_=None,
            first_1day_candle_date_=None
    ):
        self.name = name_ if name_ else "None"
        self.ticker = ticker_ if ticker_ else "None"
        self.figi = figi_ if figi_ else "None"
        self.uid = uid_ if uid_ else "None"
        self.type = type_ if type_ else "None"
        self.currency = currency_ if currency_ else "None"
        self.trading_status = trading_status_ if trading_status_ else "None"
        self.short_enabled_flag = short_enabled_flag_ if short_enabled_flag_ else "None"
        self.exchange = exchange_ if exchange_ else "None"
        self.real_exchange = real_exchange_ if real_exchange_ else "None"
        self.sector = sector_ if sector_ else "None"
        self.lot = lot_ if lot_ else "None"
        self.div_yield_flag = div_yield_flag_ if div_yield_flag_ else "None"
        self.country_of_risk = country_of_risk_ if country_of_risk_ else "None"
        self.country_of_risk_name = country_of_risk_name_ if country_of_risk_name_ else "None"
        self.api_trade_available_flag = api_trade_available_flag_ if api_trade_available_flag_ else "None"
        self.first_1min_candle_date = first_1min_candle_date_ if first_1min_candle_date_ else "None"
        self.first_1day_candle_date = first_1day_candle_date_ if first_1day_candle_date_ else "None"

    def get_last_price(self, client_: Services, figi: str = None):
        return float(
            [Tools.compound(lp.price.units, lp.price.nano)
             for lp in client_.market_data.get_last_prices(figi=[figi if figi else self.figi]).last_prices][-1]
        )

    def __str__(self):
        return str(self.__dict__)


class CandleType:
    """Класс хранение типов свечи"""
    ClassicCandles = "cl"
    HeikinAshiCandles = "ha"
    ClassicCandlesValue = 1
    HeikinAshiCandlesValue = 2


class CandleHandler:
    """Класс обработки свечи"""

    def __init__(self, candle, candle_interval: str = None):
        self.candle = CandleArguments(
            open_=Tools.compound(candle.open.units, candle.open.nano),
            high_=Tools.compound(candle.high.units, candle.high.nano),
            low_=Tools.compound(candle.low.units, candle.low.nano),
            close_=Tools.compound(candle.close.units, candle.close.nano),
            volume_=Tools.compound(candle.close.units, candle.close.nano),
            time_=candle.time,
            type_=CandleType.ClassicCandles,
        )
        self.candle_type = CandleType.ClassicCandles
        self.candle_interval = candle_interval if candle_interval else "None"

    def ha_candle_handler(self, previous_candle: CandleArguments):
        self.candle_type = CandleType.HeikinAshiCandles
        self.candle = CandleArguments(
            open_=(previous_candle.open + previous_candle.close) / 2,
            high_=max(
                (previous_candle.open + previous_candle.close) / 2,
                self.candle.high,
                (self.candle.open + self.candle.high + self.candle.low + self.candle.close) / 4
            ),
            low_=min(
                (previous_candle.open + previous_candle.close) / 2,
                self.candle.low,
                (self.candle.open + self.candle.high + self.candle.low + self.candle.close) / 4
            ),
            close_=(self.candle.open + self.candle.high + self.candle.low + self.candle.close) / 4,
            volume_=self.candle.volume,
            time_=self.candle.time,
            type_=CandleType.HeikinAshiCandles,
            interval_=self.candle.interval,
        ) if previous_candle else CandleArguments(
            open_=(self.candle.open + self.candle.close) / 2,
            high_=self.candle.high,
            low_=self.candle.low,
            close_=(self.candle.open + self.candle.high + self.candle.low + self.candle.close) / 4,
            volume_=self.candle.volume,
            time_=self.candle.time,
            type_=CandleType.HeikinAshiCandles,
            interval_=self.candle.interval,
        )
        return self.candle

    def __str__(self):
        return str(self.__dict__)


class CandlesInfo:
    """Класс получения информации о свечах"""

    def __init__(self, candles: list[CandleArguments]):
        self.candles = candles
        self.candles_quantity = len(candles)
        self.candle_interval = candles[-1].interval
        self.last_open_candle = candles[-1].open
        self.last_high_candle = candles[-1].high
        self.last_low_candle = candles[-1].low
        self.last_close_candle = candles[-1].close
        self.last_difference_high_low = round(self.last_high_candle - self.last_low_candle, 2)
        self.last_volume_candle = candles[-1].volume
        self.last_time_candle = candles[-1].time
        self.last_close_candle_positivity_negativity = "positive" if self.last_close_candle >= candles[-2].close \
            else "negative"
        self.last_4ohlc_candle = round((self.last_open_candle + self.last_high_candle +
                                        self.last_low_candle + self.last_close_candle) / 4, 2)
        self.previous_open_candle = candles[-2].open
        self.previous_high_candle = candles[-2].high
        self.previous_low_candle = candles[-2].low
        self.previous_close_candle = candles[-2].close
        self.previous_4ohlc_candle = round((self.previous_open_candle + self.previous_high_candle +
                                            self.previous_low_candle + self.previous_close_candle) / 4, 2)

    def get_only_weekdays_candles(self):
        return [i for i in self.candles if datetime.isoweekday(i.time) not in (6, 7)]

    def get_only_open_candles(self):
        return [i.open for i in self.candles]

    def get_only_high_candles(self):
        return [i.high for i in self.candles]

    def get_only_low_candles(self):
        return [i.low for i in self.candles]

    def get_only_close_candles(self):
        return [i.close for i in self.candles]

    def get_only_4ohlc_candles(self):
        return [(i.open + i.high + i.low + i.close) / 4 for i in self.candles]

    def get_only_time_candles(self):
        return [i.time for i in self.candles]

    def get_only_volume_candles(self):
        return [i.volume for i in self.candles]

    def get_only_close_positivity(self):
        return ["positive" if self.candles[i if i > 0 else i - 1].close >= self.candles[i].close else "negative"
                for i in range(len(self.candles))]

    def get_ma_close_positivity_negativity(self, period: int, out: int = 1):
        return ",".join([("positive" if i >= 0.5 else "negative") for i in SmoothingMethods(
            [int(self.candles[i].close >= self.candles[i - 1 if i > 0 else i].close) for i in range(len(self.candles))],
            period).ma_smoothing()][-1 if out == 1 else None:None])

    def get_ema_close_positivity_negativity(self, period: int, out: int = 1):
        return ",".join([("positive" if i >= 0.5 else "negative") for i in SmoothingMethods(
            [int(self.candles[i].close >= self.candles[i - 1 if i > 0 else i].close) for i in range(len(self.candles))],
            period).ema_smoothing()][-1 if out == 1 else None:None])

    def get_volume_ma(self, period: int, out: int = 1):
        return SmoothingMethods([i.volume for i in self.candles], period).ma_smoothing(out)

    def get_volume_ema(self, period: int, out: int = 1):
        return SmoothingMethods([i.volume for i in self.candles], period).ema_smoothing(out=out)

    def get_ma(self, period: int, out: int = 1):
        return SmoothingMethods([i.close for i in self.candles], period).ma_smoothing(out)

    def get_ema(self, period: int, out: int = 1):
        return SmoothingMethods([i.close for i in self.candles], period).ema_smoothing(out=out)

    def get_adx(self, period: int, w: float = 1.0, out: int = 1):
        return Indicators(self.candles).adx(period, w).adx_list if out == 0 \
            else round(Indicators(self.candles).adx(period, w).last_adx, 2)

    def get_di_trend(self, period: int, w: float = 1.0):
        di_plus_, di_minus_ = Indicators(self.candles).adx(period, w).di_plus_and_minus_list
        return "positive" if di_plus_[-1] >= di_minus_[-1] else "negative"

    def __str__(self):
        return str(self.__dict__)


class FindInstrument:
    """Класс поиска инструмента торговли"""

    def __init__(self, figi: str, instruments: list):
        self.figi = figi
        self.instrument = InstrumentArguments()
        for instrument in instruments:
            if instrument.figi == figi:
                self.instrument = instrument
                break

    @staticmethod
    def find_instrument(client_: Services, out: tuple = ("figi",), default_name: str = "AAPL"):
        search_results = []
        while len(search_results) < 1:
            find_instruments_input = input("\n'Enter' инструмент по умолчанию\n""'-' вызов последнего ввода поиска\n"
                                           "'*' вызов избранных инструментов\n\n""Поиск инструмента:")
            if find_instruments_input == "-":
                try:
                    with open("last_find_instrument.txt", "r") as file:
                        find_instruments_input = file.read().replace(" ", "").replace("\n", "")
                        print(f"\nРезультат последнего ввода поиска: {find_instruments_input}\n")
                except FileNotFoundError:
                    with open("last_find_instrument.txt", "w") as file:
                        file.write(default_name)
                    with open("last_find_instrument.txt", "r") as file:
                        find_instruments_input = file.read().replace(" ", "")
            if find_instruments_input == "*":
                favorite_instruments = client_.instruments.get_favorites().favorite_instruments
                for i, instrument in enumerate(favorite_instruments):
                    print(f"{i}. Ticker: {instrument.ticker.upper()}, "
                          f"Type: {instrument.instrument_type.upper()}, Flag: {instrument.api_trade_available_flag}")
                continue
            try:
                find_instruments = client_.instruments.find_instrument(query=find_instruments_input)
            except RequestError:
                find_instruments = client_.instruments.find_instrument(query=default_name)
            for instrument in find_instruments.instruments:
                if instrument.instrument_type in ("share", "currencies") and instrument.api_trade_available_flag:
                    print(f"{len(search_results)}.  {instrument.name}: {instrument.ticker}")
                    instrument_parameters = {'ticker': instrument.ticker, 'figi': instrument.figi,
                                             'name': instrument.name, 'uid': instrument.uid}
                    search_results.append(tuple([instrument_parameters[out_] for out_ in out]))
            inp_ = (input("\n [ - ] вернуться к поиску\nВыберите подходящий инструмент по номеру\nВвод:"
                          if len(search_results) > 1
                          else "\n [ - ] вернуться к поиску\nНажмите Enter для подтверждения результата поиска:")
                    if len(search_results) >= 1 else print("\nНичего не найдено!\nПопробуйте повторить попытку\n"))
            try:
                with open("last_find_instrument.txt", "w") as file:
                    file.write(find_instruments_input + "\n")
                return search_results[(int(inp_)) if len(search_results) > 1 or inp_ not in ["", "0"] else -1]
            except (ValueError, TypeError, IndexError):
                search_results.clear()


class GetHistoricalCandles:
    """Класс запросов на получения исторических свечей"""

    def __init__(self, client_: Services, figi: str, days: int):
        self.candle_interval_options = ["Интервал не определён", "1 минута", "5 минут", "15 минут", "1 час", "1 день"]
        self.Client = client_
        self.figi = figi
        self.days = days

    def get_classic_last_historical_candle(
            self, candle_interval: CandleInterval, from_minutes: int = 6, out: str = "last"
    ):
        candles = [CandleHandler(
            candle, self.candle_interval_options[candle_interval.value]
        ).candle for candle in self.Client.get_all_candles(
            figi=self.figi,
            from_=datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(minutes=from_minutes),
            interval=candle_interval
        )]
        return None if len(candles) < 1 else candles[-1] if out == "last" else candles

    def get_ha_last_historical_candle(self, candle_interval: CandleInterval, from_minutes: int = 6, out: str = "last"):
        historical_candles = []
        for candle in self.Client.get_all_candles(
                figi=self.figi,
                from_=datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(minutes=from_minutes),
                interval=candle_interval
        ):
            historical_candles.append(CandleHandler(
                candle, self.candle_interval_options[candle_interval.value]).ha_candle_handler(
                historical_candles[-1] if len(historical_candles) > 0 else None))
        return None if len(historical_candles) < 1 else historical_candles[-1] if out == "last" else historical_candles

    def get_classic_historical_candles(self, candle_interval: CandleInterval):
        return [CandleHandler(
            candle, self.candle_interval_options[candle_interval.value]
        ).candle for candle in self.Client.get_all_candles(
            figi=self.figi,
            from_=datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=self.days),
            interval=candle_interval
        )]

    def get_ha_historical_candles(self, candle_interval: CandleInterval):
        historical_candles = []
        for candle in self.Client.get_all_candles(
                figi=self.figi,
                from_=datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=self.days),
                interval=candle_interval
        ):
            historical_candles.append(CandleHandler(
                candle, self.candle_interval_options[candle_interval.value]).ha_candle_handler(
                historical_candles[-1] if len(historical_candles) > 0 else None))
        return historical_candles

    def get_classic_and_ha_historical_candles(self, candle_interval: CandleInterval):
        candle_request = self.Client.get_all_candles(
            figi=self.figi, from_=datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=self.days),
            interval=candle_interval)
        ha_historical_candles, cl_historical_candles = [], []
        for candle in candle_request:
            cl_historical_candles.append(CandleHandler(
                candle, self.candle_interval_options[candle_interval.value]).candle)
            ha_historical_candles.append(CandleHandler(
                candle, self.candle_interval_options[candle_interval.value]).ha_candle_handler(
                ha_historical_candles[-1] if len(ha_historical_candles) > 0 else None))
        return cl_historical_candles, ha_historical_candles

    def get_classic_1day_historical_candles(self):
        return self.get_classic_historical_candles(CandleInterval.CANDLE_INTERVAL_DAY)

    def get_ha_1day_historical_candles(self):
        return self.get_ha_historical_candles(CandleInterval.CANDLE_INTERVAL_DAY)

    def get_classic_1min_historical_candles(self):
        return self.get_classic_historical_candles(CandleInterval.CANDLE_INTERVAL_1_MIN)

    def get_ha_1min_historical_candles(self):
        return self.get_ha_historical_candles(CandleInterval.CANDLE_INTERVAL_1_MIN)

    def get_classic_5min_historical_candles(self):
        return self.get_classic_historical_candles(CandleInterval.CANDLE_INTERVAL_5_MIN)

    def get_ha_5min_historical_candles(self):
        return self.get_ha_historical_candles(CandleInterval.CANDLE_INTERVAL_5_MIN)


class PositionHandler:
    """Класс обработки информации о позициях"""

    def __init__(self, position, instrument_info: InstrumentArguments):
        self.name = instrument_info.name
        self.ticker = instrument_info.ticker
        self.currency = instrument_info.currency
        self.short_enabled_flag = instrument_info.short_enabled_flag
        self.trading_status = instrument_info.trading_status
        self.exchange = instrument_info.exchange
        self.real_exchange = instrument_info.real_exchange
        self.sector = instrument_info.sector
        self.lot = instrument_info.lot
        self.div_yield_flag = instrument_info.div_yield_flag
        self.country_of_risk = instrument_info.country_of_risk
        self.country_of_risk_name = instrument_info.country_of_risk_name
        self.api_trade_available_flag = instrument_info.api_trade_available_flag
        self.quantity = Tools.compound(position.quantity.units, position.quantity.nano)
        self.average_position_price = Tools.compound(position.average_position_price.units,
                                                     position.average_position_price.nano)
        self.average_position_price_fifo = Tools.compound(position.average_position_price_fifo.units,
                                                          position.average_position_price_fifo.nano)
        self.instrument_type = position.instrument_type.upper()
        self.current_price = Tools.compound(position.current_price.units, position.current_price.nano)
        self.expected_yield = Tools.compound(position.expected_yield.units, position.expected_yield.nano)
        self.total_percentage = round(self.expected_yield / self.average_position_price_fifo * 100, 2)
        self.total_price_quantity = round(self.expected_yield * self.quantity, 2)
        self.blocked = position.blocked

    def __str__(self):
        return str(self.__dict__)


class OperationHandler:
    """Класс обработки информации об операциях"""

    def __init__(self, operation, instrument_info: InstrumentArguments):
        self.name = instrument_info.name
        self.figi = operation.figi
        self.ticker = instrument_info.ticker
        self.currency = instrument_info.currency
        self.short_enabled_flag = instrument_info.short_enabled_flag
        self.trading_status = instrument_info.trading_status
        self.exchange = instrument_info.exchange
        self.real_exchange = instrument_info.real_exchange
        self.sector = instrument_info.sector
        self.lot = instrument_info.lot
        self.div_yield_flag = instrument_info.div_yield_flag
        self.country_of_risk = instrument_info.country_of_risk
        self.country_of_risk_name = instrument_info.country_of_risk_name
        self.api_trade_available_flag = instrument_info.api_trade_available_flag
        self.quantity = operation.quantity
        self.payment = Tools.compound(operation.payment.units,
                                      operation.payment.nano)
        self.price = Tools.compound(operation.price.units,
                                    operation.price.nano)
        self.instrument_type = operation.instrument_type.upper()
        self.trad_type = operation.type
        self.date = (operation.date.year, operation.date.month, operation.date.day, operation.date.hour,)

    def __str__(self):
        return str(self.__dict__)


class ClientInfo:
    """Класс получения информации об аккаунте"""

    def __init__(
            self,
            client_: Services,
            account_numb: int = 0,
            days: int = 90,
            methods: tuple = ('shares', 'currencies', 'etfs')
    ):
        self.Client = client_
        account = client_.users.get_accounts().accounts[account_numb]
        account_info = client_.users.get_info()
        margin_attributes = client_.users.get_margin_attributes(account_id=account.id)

        self.account_id = account.id
        self.account_name = account.name
        self.account_tariff = account_info.tariff.upper()
        self.account_qual_status = account_info.qual_status
        self.account_prem_status = account_info.prem_status
        self.account_access_level = "Полный доступ к счёту" if account.access_level.value == 1 else False
        self.account_status = "Открытый и активный счёт" if account.status.value == 2 else False
        self.account_type = "Брокерский счёт Тинькофф" if account.type.value == 1 else False
        self.account_liquid_portfolio = Tools.compound(
            margin_attributes.liquid_portfolio.units, margin_attributes.liquid_portfolio.nano
        ) if margin_attributes.liquid_portfolio else "None"
        self.account_amount_of_missing_funds = Tools.compound(
            margin_attributes.amount_of_missing_funds.units, margin_attributes.amount_of_missing_funds.nano
        ) if margin_attributes.amount_of_missing_funds else "None"
        self.margin_currency = margin_attributes.liquid_portfolio.currency.upper()
        self.account_opened_date = (account.opened_date.year, account.opened_date.month, account.opened_date.day)
        self.account_closed_date = False \
            if account.closed_date.year < account.opened_date.year \
            else (account.closed_date.year, account.closed_date.month, account.closed_date.day)
        self.account_money = {i.currency.upper(): Tools.compound(i.units, i.nano) for i in
                              client_.operations.get_positions(account_id=account.id).money}

        trading_status_argument = [
            "Торговый статус не определён", "Недоступен для торгов", "Период открытия торгов", "Период закрытия торгов",
            "Перерыв в торговле", "Нормальная торговля", "Аукцион закрытия", "Аукцион крупных пакетов",
            "Дискретный аукцион", "Аукцион открытия", "Период торгов по цене аукциона закрытия", "Сессия назначена",
            "Сессия закрыта", "Сессия открыта", "Доступна торговля в режиме внутренней ликвидности брокера",
            "Перерыв торговли в режиме внутренней ликвидности брокера",
            "Недоступна торговля в режиме внутренней ликвидности брокера"
        ]
        real_exchange_arguments = [
            "Тип не определён", "Московская биржа", "Санкт-Петербургская биржа", "Внебиржевой инструмент"
        ]
        instruments_list = []
        for method in methods:
            for instrument in getattr(client_.instruments, method)().instruments:
                instruments_list.append(
                    InstrumentArguments(
                        name_=instrument.name,
                        ticker_=instrument.ticker,
                        figi_=instrument.figi,
                        uid_=instrument.uid,
                        type_=method.upper(),
                        currency_=instrument.currency.upper(),
                        trading_status_=trading_status_argument[instrument.trading_status.value],
                        short_enabled_flag_=instrument.short_enabled_flag,
                        exchange_=instrument.exchange,
                        real_exchange_=real_exchange_arguments[instrument.real_exchange.value],
                        sector_="None" if method in ['currencies', 'bonds', 'futures'] else instrument.sector.upper(),
                        lot_=instrument.lot,
                        div_yield_flag_=instrument.div_yield_flag if method in ['shares'] else "None",
                        country_of_risk_=instrument.country_of_risk,
                        country_of_risk_name_=instrument.country_of_risk_name,
                        api_trade_available_flag_=instrument.api_trade_available_flag,
                        first_1min_candle_date_=(
                            instrument.first_1min_candle_date.year,
                            instrument.first_1min_candle_date.month,
                            instrument.first_1min_candle_date.day
                        ),
                        first_1day_candle_date_=(
                            instrument.first_1day_candle_date.year,
                            instrument.first_1day_candle_date.month,
                            instrument.first_1day_candle_date.day
                        )
                    )
                )
        self.instruments = instruments_list
        operations_info_list, refill_list, withdrawal_of_funds_list, service_fee_list = [], [], [], []
        commission_ = {"days": days, "USD": 0, "RUB": 0, "other": 0}
        for operation in client_.operations.get_operations(
                account_id=self.account_id, from_=datetime.today() - timedelta(days=days)
        ).operations:
            if operation.type == 'Удержание комиссии за операцию':
                if operation.currency == 'usd':
                    commission_["USD"] += Tools.compound(operation.payment.units, operation.payment.nano)
                elif operation.currency == 'rub':
                    commission_["RUB"] += Tools.compound(operation.payment.units, operation.payment.nano)
                else:
                    commission_["other"] += Tools.compound(operation.payment.units, operation.payment.nano)
            elif operation.type == "Пополнение брокерского счёта":
                refill_list.append({
                    "currency": operation.currency.upper(),
                    "quantity": Tools.compound(operation.payment.units, operation.payment.nano),
                    "data": (
                        operation.date.year, operation.date.month, operation.date.day,
                        operation.date.hour, operation.date.minute
                    )
                })
            elif operation.type == "Вывод денежных средств":
                withdrawal_of_funds_list.append({
                    "currency": operation.currency.upper(),
                    "quantity": Tools.compound(operation.payment.units, operation.payment.nano),
                    "data": (
                        operation.date.year, operation.date.month, operation.date.day,
                        operation.date.hour, operation.date.minute
                    )
                })
            elif operation.type == 'Удержание комиссии за обслуживание брокерского счёта':
                service_fee_list.append({
                    "currency": operation.payment.currency.upper(),
                    "quantity": Tools.compound(operation.payment.units, operation.payment.nano),
                    "data": (
                        operation.date.year, operation.date.month, operation.date.day,
                        operation.date.hour, operation.date.minute
                    )
                })
            elif operation.type in ("Покупка ценных бумаг", "Продажа ценных бумаг") and operation.figi:
                if len(operations_info_list) > 0 and operation.figi == operations_info_list[-1].figi:
                    operations_info_list.append(operations_info_list[-1])
                else:
                    operations_info_list.append(
                        OperationHandler(operation, FindInstrument(operation.figi, self.instruments).instrument)
                    )
            else:
                pass

        commission_["USD"], commission_["RUB"], commission_["other"] = \
            round(commission_["USD"], 2), round(commission_["RUB"], 2), round(commission_["other"], 2)

        self.get_account_positions = [
            PositionHandler(
                position, FindInstrument(position.figi, self.instruments).instrument
            ) for position in client_.operations.get_portfolio(account_id=self.account_id).positions
        ]
        self.get_account_operations = operations_info_list
        self.get_account_commission = commission_
        self.get_account_refill = refill_list
        self.get_account_withdrawal_of_funds = withdrawal_of_funds_list
        self.get_account_service_fee = service_fee_list

    def get_current_positions(self):
        return [
            PositionHandler(
                position, FindInstrument(position.figi, self.instruments).instrument
            ) for position in self.Client.operations.get_portfolio(account_id=self.account_id).positions
        ]

    def get_current_operations(self, days: int):
        operations_info_list = []
        from_ = datetime.today() - timedelta(days=days)
        for operation in self.Client.operations.get_operations(account_id=self.account_id, from_=from_).operations:
            if operation.type in ("Покупка ценных бумаг", "Продажа ценных бумаг"):
                if len(operations_info_list) > 0 and operation.figi == operations_info_list[-1].figi:
                    operations_info_list.append(operations_info_list[-1])
                else:
                    operations_info_list.append(
                        OperationHandler(operation, FindInstrument(operation.figi, self.instruments).instrument))
        return operations_info_list

    def get_current_commission(self, days: int):
        commission = {"days": days, "USD": 0, "RUB": 0, "other": 0}
        from_ = datetime.today() - timedelta(days=days)
        for operation in self.Client.operations.get_operations(account_id=self.account_id, from_=from_).operations:
            if operation.type == 'Удержание комиссии за операцию':
                if operation.currency == 'usd':
                    commission["USD"] += Tools.compound(operation.payment.units, operation.payment.nano)
                elif operation.currency == 'rub':
                    commission["RUB"] += Tools.compound(operation.payment.units, operation.payment.nano)
                else:
                    commission["other"] += Tools.compound(operation.payment.units, operation.payment.nano)
        commission["USD"], commission["RUB"], commission["other"] = \
            round(commission["USD"], 2), round(commission["RUB"], 2), round(commission["other"], 2)
        return commission

    def __str__(self):
        return str(self.__dict__)


class TradingDirectionArguments:
    """Класс определения аргументов о направление торговой операции"""

    def __init__(
            self,
            buy_price: float,
            stock_quantity: int,
            deals_quantity,
            transaction_time: datetime
    ):
        self.buy_price = buy_price
        self.stock_quantity = stock_quantity
        self.deals_quantity = deals_quantity
        self.transaction_time = transaction_time

    def __str__(self):
        return str(self.__dict__)


class TradingInfo:
    """Класс получения информации об информации торговли"""

    def __init__(
            self,
            settings: SettingsHandler,
            trading_direction_arguments: TradingDirectionArguments,
            last_price: float,
            signal_indicator: int,
            instrument_arguments: InstrumentArguments,
            indicators: Indicators,
            datetime_now: datetime,
            start_datatime: datetime,
            start_while_time: float,
            percentage_after_order: float,
            amount_after_order: float,
            take_profit_deals_quantity: int,
            stop_loss_deals_quantity: int,
            take_profit_numb: int,
            stop_loss_numb: int,
    ):
        self.settings = settings
        self.instrument_arguments = instrument_arguments
        self.last_price = last_price
        self.currency = self.instrument_arguments.currency
        self.indicators = indicators
        self.start_datatime = start_datatime
        self.start_while_time = start_while_time
        self.transaction_time = trading_direction_arguments.transaction_time
        self.working_hours_sec = round(abs(start_datatime - datetime_now).total_seconds(), 2)
        self.working_hours_min = round(self.working_hours_sec / 60, 2)
        self.working_hours_hour = round(self.working_hours_min / 60, 2)
        self.time_since_last_trade_sec = round(abs(self.transaction_time - datetime.now()).total_seconds(), 2)
        self.time_since_last_trade_min = round(self.time_since_last_trade_sec / 60, 2)
        self.start_datatime_str = f"{start_datatime.day}|{start_datatime.hour}:{start_datatime.minute}"
        self.signal_indicator = signal_indicator
        self.deal_direction_str = 'Long' if self.signal_indicator >= 0 else 'Short'
        self.trading_direction_mode = self.settings.trading_direction_mode
        self.buy_price = trading_direction_arguments.buy_price
        self.percentage_after_order = percentage_after_order
        self.amount_after_order = amount_after_order
        self.stock_quantity = trading_direction_arguments.stock_quantity
        self.price_lots_stock = round(self.stock_quantity * self.last_price, 2)
        self.deals_quantity = trading_direction_arguments.deals_quantity
        self.update_number = abs(self.signal_indicator)
        self.take_profit_trigger = self.settings.take_profit[take_profit_numb]
        self.stop_loss_trigger = self.settings.stop_loss[stop_loss_numb]
        self.take_profit_deals_quantity = take_profit_deals_quantity
        self.stop_loss_deals_quantity = stop_loss_deals_quantity
        self.cycle_speed = round(abs(time.time() - start_while_time), 4)
        self.last_chandelier_exit_long = round(self.indicators.return_chandelier_exit.last_CE_long, 2) if \
            self.indicators.return_chandelier_exit else "None"
        self.last_chandelier_exit_short = round(self.indicators.return_chandelier_exit.last_CE_short, 2) if \
            self.indicators.return_chandelier_exit else "None"
        self.last_adx = round(self.indicators.return_adx.last_adx, 2) if self.indicators.return_adx else "None"
        self.last_ema = round(self.indicators.return_ema[-1], 2) if self.indicators.return_ema else "None"

    def __str__(self):
        return str(self.__dict__)


class Strategies:
    """Класс стратегии обработки сигналов"""

    def __init__(self, signal_indicator: int, indicators: Indicators):
        self.SI = signal_indicator
        self.indicators = indicators

    def chandelier_exit_strategy(self):
        last_close = self.indicators.candle_info.last_close_candle
        prev_close = self.indicators.candle_info.previous_close_candle
        ce = self.indicators.return_chandelier_exit
        if ce:
            if last_close > ce.last_CE_short and prev_close < ce.previous_CE_short:
                self.SI = self.SI + 1 if self.SI > -1 else 1
            elif last_close < ce.last_CE_long and prev_close > ce.previous_CE_long:
                self.SI = self.SI - 1 if self.SI < 1 else -1
            else:
                self.SI = self.SI + 1 if self.SI > 0 else self.SI - 1
        else:
            print("Индикатор 'chandelier_exit' не определен")
            return ValueError
        return self.SI

    @staticmethod
    def strategy_pass():
        pass

    def __str__(self):
        return str(self.__dict__)


class StrategiesAnalytics:
    class ResultStrategiesAnalytics:
        class ListCellArg:
            def __init__(self, number: int, signal: int, value: float):
                self.number = number
                self.signal = signal
                self.value = value

            def __str__(self):
                return str(self.__dict__)

        class DealDataAnalysis:
            def __init__(
                    self,
                    deals_quantity: int,
                    profit_list: list[float],
                    profit_percentage_list: list[float],
                    numb_of_range_values_list: list[int],
                    max_up_percentage_list: list[float],
                    max_down_percentage_list: list[float],
                    profit_percentage_given_average_list: list[float],
                    average_percentage_list: list[float],
                    percentage_of_tr_list: list[float]
            ):
                self.average_profit_per_trade = round(sum(profit_list) / len(profit_list), 2)
                self.percent_of_profit_trades = round(len([i for i in profit_list if i > 0]) / deals_quantity * 100, 2)
                self.percent_of_profit_trades_given_the_range = round(
                    len([i for i in profit_percentage_given_average_list if i > 0]) / deals_quantity * 100, 2)
                self.average_profit_per_trade_percentage = round(
                    sum(profit_percentage_list) / len(profit_percentage_list), 2)
                self.average_numb_of_values_in_range = round(
                    sum(numb_of_range_values_list) / len(numb_of_range_values_list))
                self.average_max_up_percentage = round(sum(max_up_percentage_list) / len(max_up_percentage_list), 2)
                self.average_max_down_percentage = round(sum(max_down_percentage_list) / len(max_down_percentage_list),
                                                         2)
                maximum_height = round(max(max_up_percentage_list), 2)
                maximum_drawdown = round(min(max_down_percentage_list), 2)
                self.maximum_height = maximum_height if maximum_height != 0.0 else round(min(max_up_percentage_list), 2)
                self.maximum_drawdown = maximum_drawdown if maximum_drawdown != 0.0 \
                    else round(max(max_down_percentage_list), 2)
                self.average_percent_variability = round(sum(average_percentage_list) / len(average_percentage_list), 2)
                self.atr_percentage = round(sum(percentage_of_tr_list) / len(percentage_of_tr_list), 2)
                self.risk_index = round(
                    100 - ((self.percent_of_profit_trades * 2) + self.percent_of_profit_trades_given_the_range) / 3, 2)
                self.normalized_stop_loss_percent_level = round(
                    ((self.average_max_down_percentage * 4) + self.maximum_drawdown) / 5, 2)
                self.normalized_take_profit_percent_level = round(
                    ((self.average_max_up_percentage * 4) + self.maximum_height) / 5, 2)

            def __str__(self):
                return str(self.__dict__)

        def __init__(self, signal_list: list[int], value_list: list[float]):
            iter_numb, single_list, long_deals, short_deals = 0, [], [], []
            for signal, value in zip(signal_list, value_list):
                single_list.append(StrategiesAnalytics.ResultStrategiesAnalytics.ListCellArg(iter_numb, signal, value))
                if signal == 1:
                    long_deals.append(single_list[-1])
                elif signal == -1:
                    short_deals.append(single_list[-1])
                iter_numb += 1
            self.long_deals_quantity = len(long_deals)
            self.short_deals_quantity = len(short_deals)
            self.amount_of_deals = len(long_deals) + len(short_deals)
            pair_for_long = (long_deals, short_deals) if long_deals[0].number <= short_deals[0].number \
                else (long_deals, short_deals[1:])
            pair_for_short = (short_deals, long_deals) if short_deals[0].number <= long_deals[0].number \
                else (short_deals, long_deals[1:])
            long_profit_list, short_profit_list = [], []
            long_profit_percentage_list, short_profit_percentage_list = [], []
            long_profit_percentage_given_average_list, short_profit_percentage_given_average_list = [], []
            long_numb_of_range_values_list, short_numb_of_range_values_list = [], []
            long_max_up_percentage_list, long_max_down_percentage_list = [], []
            short_max_up_percentage_list, short_max_down_percentage_list = [], []
            long_average_percentage_list, short_average_percentage_list = [], []
            long_percentage_of_tr_list, short_percentage_of_tr_list = [], []
            for long_deal, short_deal in zip(*pair_for_long):
                long_slice = single_list[long_deal.number:short_deal.number + 1]
                long_profit_list.append(long_slice[-1].value - long_slice[0].value)
                long_profit_percentage_list.append(long_profit_list[-1] / long_slice[0].value * 100)
                long_numb_of_range_values_list.append(len(long_slice))
                long_slice_value = [i.value for i in long_slice]
                max_range_value, min_range_value = max(long_slice_value), min(long_slice_value)
                long_max_up_percentage_list.append(100 - (long_slice[0].value / max_range_value * 100))
                long_max_down_percentage_list.append(100 - (long_slice[0].value / min_range_value * 100))
                average_values_range = sum(long_slice_value) / long_numb_of_range_values_list[-1]
                long_average_percentage_list.append(100 - (long_slice[0].value / average_values_range * 100))
                long_profit_percentage_given_average_list.append(
                    (long_average_percentage_list[-1] + (long_profit_percentage_list[-1] * 2)) / 3)
                try:
                    long_percentage_of_tr_list.append((max_range_value - min_range_value) / long_slice[0].value * 100)
                except ZeroDivisionError:
                    long_percentage_of_tr_list.append(0.0)
            for short_deal, long_deal in zip(*pair_for_short):
                short_slice = single_list[short_deal.number:long_deal.number + 1]
                short_profit_list.append(short_slice[0].value - short_slice[-1].value)
                short_profit_percentage_list.append(short_profit_list[-1] / short_slice[0].value * 100)
                short_numb_of_range_values_list.append(len(short_slice))
                short_slice_value = [i.value for i in short_slice]
                max_range_value, min_range_value = max(short_slice_value), min(short_slice_value)
                short_max_up_percentage_list.append(100 - (short_slice[0].value / min_range_value * 100))
                short_max_down_percentage_list.append(100 - (short_slice[0].value / max_range_value * 100))
                average_values_range = sum(short_slice_value) / short_numb_of_range_values_list[-1]
                short_average_percentage_list.append(100 - (short_slice[0].value / average_values_range * 100))
                short_profit_percentage_given_average_list.append(
                    (short_average_percentage_list[-1] + (short_profit_percentage_list[-1] * 2)) / 3)
                try:
                    short_percentage_of_tr_list.append((max_range_value - min_range_value) / short_slice[0].value * 100)
                except ZeroDivisionError:
                    short_percentage_of_tr_list.append(0.0)
            self.long_deals_analytic = StrategiesAnalytics.ResultStrategiesAnalytics.DealDataAnalysis(
                self.long_deals_quantity, long_profit_list, long_profit_percentage_list, long_numb_of_range_values_list,
                long_max_up_percentage_list, long_max_down_percentage_list, long_profit_percentage_given_average_list,
                long_average_percentage_list, long_percentage_of_tr_list
            )
            self.short_deals_analytic = StrategiesAnalytics.ResultStrategiesAnalytics.DealDataAnalysis(
                self.short_deals_quantity, short_profit_list, short_profit_percentage_list,
                short_numb_of_range_values_list,
                short_max_up_percentage_list, short_max_down_percentage_list,
                short_profit_percentage_given_average_list,
                short_average_percentage_list, short_percentage_of_tr_list
            )
            self.recommended_trading_direction = "long" \
                if (self.long_deals_analytic.risk_index + (100 - self.long_deals_analytic.percent_of_profit_trades)) < (
                    self.short_deals_analytic.risk_index + (100 - self.short_deals_analytic.percent_of_profit_trades)) \
                else "short"

        def __str__(self):
            return str(self.__dict__), \
                   "long_deals_analytic: " + str(self.long_deals_analytic), \
                   "short_deals_analytic: " + str(self.short_deals_analytic)

    def __init__(self, indicators: Indicators):
        self.indicators = indicators

    def strategies_analytics(
            self, strategy_name: str, analytics_strategy_on_classic_candles: list[CandleArguments]
    ):
        signal_indicator, signal_indicator_list = 0, []
        close_candle_list = self.indicators.candle_info.get_only_close_candles()
        if strategy_name == "chandelier_exit_strategy":
            ce_long_and_short_list = self.indicators.return_chandelier_exit.CE_long_and_short_list
            close_candle_list = close_candle_list[len(close_candle_list) - len(ce_long_and_short_list[0]):]
            for n in range(0, len(close_candle_list)):
                prev_numb = n - 1 if n > 0 else n
                signal_indicator = StrategiesAnalytics.chandelier_exit_strategy(
                    signal_indicator, (close_candle_list[n], close_candle_list[prev_numb]),
                    (ce_long_and_short_list[0][n], ce_long_and_short_list[1][n]),
                    (ce_long_and_short_list[0][prev_numb], ce_long_and_short_list[1][prev_numb])
                )
                signal_indicator_list.append(signal_indicator)
            close_classic_candle_list = CandlesInfo(
                analytics_strategy_on_classic_candles
            ).get_only_close_candles() if analytics_strategy_on_classic_candles else None
            return StrategiesAnalytics.ResultStrategiesAnalytics(
                signal_indicator_list, close_classic_candle_list[
                                       len(close_classic_candle_list) - len(ce_long_and_short_list[0]):
                                       ] if close_classic_candle_list else close_candle_list)

    def chandelier_exit_strategy_analytics_iterator(
            self, analytics_strategy_on_classic_candles: list[CandleArguments],
            period_range: tuple[int], factor_range: tuple[float], factor_step: float, progress_bar: bool = True
    ):
        iteration_result_list, strategy_setting_list = [], []
        for period in range(*period_range):
            factor = factor_range[0]
            if progress_bar:
                Tools.print_progress_bar(period + 1, period_range[1], "StrategiesAnalytics ", length=40)
            for n in range(round(factor_range[1] / factor_step)):
                indicators = Indicators(self.indicators.candles)
                indicators.chandelier_exit(period, factor)
                strategies_analytics: StrategiesAnalytics.ResultStrategiesAnalytics = StrategiesAnalytics(
                    indicators).strategies_analytics("chandelier_exit_strategy", analytics_strategy_on_classic_candles)
                iteration_result_list.append(strategies_analytics)
                strategy_setting_list.append({"period": period, "factor": factor})
                factor = round(factor + factor_step, 2)
        return iteration_result_list, strategy_setting_list

    @staticmethod
    def chandelier_exit_strategy(
            signal_indicator, last_and_prev_close: tuple[float],
            ce_long_and_short: tuple[float], previous_ce_long_and_short: tuple[float]
    ):
        if last_and_prev_close[0] > ce_long_and_short[1] and last_and_prev_close[1] < previous_ce_long_and_short[1]:
            signal_indicator = signal_indicator + 1 if signal_indicator > -1 else 1
        elif last_and_prev_close[0] < ce_long_and_short[0] and last_and_prev_close[1] > previous_ce_long_and_short[0]:
            signal_indicator = signal_indicator - 1 if signal_indicator < 1 else -1
        else:
            signal_indicator = signal_indicator + 1 if signal_indicator > 0 else signal_indicator - 1
        return signal_indicator


class StrategyAnalyticsResultsEvaluationIterator:
    def __init__(
            self,
            iteration_result_list: list[StrategiesAnalytics.ResultStrategiesAnalytics],
            strategy_setting_list: list[dict]
    ):
        long_percent_of_profit_trades_list = []
        long_risk_index_list = []
        long_average_max_up_percentage_list = []
        long_average_profit_per_trade_percentage_list = []
        short_percent_of_profit_trades_list = []
        short_risk_index_list = []
        short_average_max_up_percentage_list = []
        short_average_profit_per_trade_percentage_list = []
        for iteration_result in iteration_result_list:
            long_deals_analytic = iteration_result.long_deals_analytic
            short_deals_analytic = iteration_result.short_deals_analytic
            long_percent_of_profit_trades_list.append(long_deals_analytic.percent_of_profit_trades)
            long_risk_index_list.append(long_deals_analytic.risk_index)
            long_average_max_up_percentage_list.append(long_deals_analytic.average_max_up_percentage)
            long_average_profit_per_trade_percentage_list.append(
                long_deals_analytic.average_profit_per_trade_percentage)
            short_percent_of_profit_trades_list.append(short_deals_analytic.percent_of_profit_trades)
            short_risk_index_list.append(short_deals_analytic.risk_index)
            short_average_max_up_percentage_list.append(short_deals_analytic.average_max_up_percentage)
            short_average_profit_per_trade_percentage_list.append(
                short_deals_analytic.average_profit_per_trade_percentage)
        self.best_long_percent_of_profit_trades = max(long_percent_of_profit_trades_list)
        self.best_long_risk_index = min(long_risk_index_list)
        self.best_long_average_max_up_percentage = max(long_average_max_up_percentage_list)
        self.best_long_average_profit_per_trade_percentage = max(long_average_profit_per_trade_percentage_list)
        self.best_short_percent_of_profit_trades = max(short_percent_of_profit_trades_list)
        self.best_short_risk_index = min(short_risk_index_list)
        self.best_short_average_max_up_percentage = min(short_average_max_up_percentage_list)
        self.best_short_average_profit_per_trade_percentage = max(short_average_profit_per_trade_percentage_list)
        for n, result_strategies_analytics in enumerate(iteration_result_list):
            long_deals_analytic = result_strategies_analytics.long_deals_analytic
            short_deals_analytic = result_strategies_analytics.short_deals_analytic
            if self.best_long_percent_of_profit_trades == long_deals_analytic.percent_of_profit_trades:
                self.best_long_percent_of_profit_trades_strategy_setting = strategy_setting_list[n]
            if self.best_long_risk_index == long_deals_analytic.risk_index:
                self.best_long_risk_index_strategy_setting = strategy_setting_list[n]
            if self.best_long_average_max_up_percentage == long_deals_analytic.average_max_up_percentage:
                self.best_long_average_max_up_percentage_strategy_setting = strategy_setting_list[n]
            if self.best_long_average_profit_per_trade_percentage == (
                    long_deals_analytic.average_profit_per_trade_percentage):
                self.best_long_average_profit_per_trade_percentage_strategy_setting = strategy_setting_list[n]
            if self.best_short_percent_of_profit_trades == short_deals_analytic.percent_of_profit_trades:
                self.best_short_percent_of_profit_trades_strategy_setting = strategy_setting_list[n]
            if self.best_short_risk_index == short_deals_analytic.risk_index:
                self.best_short_risk_index_strategy_setting = strategy_setting_list[n]
            if self.best_short_average_max_up_percentage == short_deals_analytic.average_max_up_percentage:
                self.best_short_average_max_up_percentage_strategy_setting = strategy_setting_list[n]
            if self.best_short_average_profit_per_trade_percentage == (
                    short_deals_analytic.average_profit_per_trade_percentage):
                self.best_short_average_profit_per_trade_percentage_strategy_setting = strategy_setting_list[n]

    def __str__(self):
        return f"\nИтоги оценки результатов аналитики стратегии:\n" \
               f"\nLong оценка:\n\n" \
               f"Лучший процент прибыльных сделок: {self.best_long_percent_of_profit_trades}, " \
               f"значения индикатора: {self.best_long_percent_of_profit_trades_strategy_setting}\n" \
               f"Минимальный процент индекса риска: {self.best_long_risk_index}, " \
               f"значения индикатора: {self.best_long_risk_index_strategy_setting}\n" \
               f"Максимальный средний процент роста: {self.best_long_average_max_up_percentage}, " \
               f"значения индикатора: {self.best_long_average_max_up_percentage_strategy_setting}\n" \
               f"Максимальный средний процент прибыли: {self.best_long_average_profit_per_trade_percentage}, " \
               f"значения индикатора: {self.best_long_average_profit_per_trade_percentage_strategy_setting}\n" \
               f"\nShort оценка:\n\n" \
               f"Лучший процент прибыльных сделок: {self.best_short_percent_of_profit_trades}, " \
               f"значения индикатора: {self.best_short_percent_of_profit_trades_strategy_setting}\n" \
               f"Минимальный процент индекса риска: {self.best_short_risk_index}, " \
               f"значения индикатора: {self.best_short_risk_index_strategy_setting}\n" \
               f"Максимальный средний процент роста: {self.best_short_average_max_up_percentage}, " \
               f"значения индикатора: {self.best_short_average_max_up_percentage_strategy_setting}\n" \
               f"Максимальный средний процент прибыли: {self.best_short_average_profit_per_trade_percentage}, " \
               f"значения индикатора: {self.best_short_average_profit_per_trade_percentage_strategy_setting}\n"


class PrintInfo:
    """Класс вывода информации в консоль"""

    def __init__(
            self,
            client_info: ClientInfo = None,
            positions: list = None,
            operations: list = None,
            commission: dict = None,
            instrument_info: InstrumentArguments = None,
            candles_info: CandlesInfo = None,
            trading_info: TradingDirectionArguments = None,
            settings_info: SettingsHandler = None
    ):
        self.Client_info = client_info
        self.positions = positions
        self.operations = operations
        self.commission = commission
        self.instrument_info = instrument_info
        self.candles_info = candles_info
        self.trading_info = trading_info
        self.settings_info = settings_info

    def print_info(
            self,
            latest_numb: int = 12,
            print_client_info=True,
            print_positions_info=True,
            print_operations_info=True,
            print_instrument_info_=True,
            print_commission_info_=True,
            print_candles_info_=True,
            print_refill_info=True,
            print_withdrawal_info=True,
            print_service_fee_info=True,
    ):
        if self.Client_info and print_client_info:
            print(f"\nИнформация об аккаунте '{self.Client_info.account_name}' (id={self.Client_info.account_id}):\n\n"
                  f"   ▶  Тариф: {self.Client_info.account_tariff}\n"
                  f"   ▶  Денег на счету: {self.Client_info.account_money}\n"
                  f"   ▶  Cумма обеспечения для сделок с плечом: {self.Client_info.account_liquid_portfolio}\n"
                  f"   ▶  Объем недостающих средств: {self.Client_info.account_amount_of_missing_funds}\n"
                  f"   ▶  Тип счета: {self.Client_info.account_type}\n"
                  f"   ▶  Сатус счета: {self.Client_info.account_status}\n"
                  f"   ▶  Уровень доступа к счету: {self.Client_info.account_access_level}\n"
                  f"   ▶  Сатус квалифицированного инвестора: {self.Client_info.account_qual_status}\n"
                  f"   ▶  Сатус Premium: {self.Client_info.account_prem_status}\n"
                  f"   ▶  Дата создания аккаунта: {self.Client_info.account_opened_date}")
            if print_refill_info:
                print("\nПоследние пополнения:\n")
                if len(self.Client_info.get_account_refill) > 0:
                    for refill in self.Client_info.get_account_refill[-latest_numb:]:
                        print(f"   ▶  {refill['currency']}: "
                              f"Количество = {refill['quantity']}, Дата пополнения: {refill['data']}")
                else:
                    print("   ▶  None")

            if print_withdrawal_info:
                print("\nПоследние выводы средств:\n")
                if len(self.Client_info.get_account_withdrawal_of_funds) > 0:
                    for withdrawal in self.Client_info.get_account_withdrawal_of_funds[-latest_numb:]:
                        print(f"   ▶  {withdrawal['currency']}: "
                              f"Количество = {withdrawal['quantity']}, Дата вывода: {withdrawal['data']}")
                else:
                    print("   ▶  None")

            if print_service_fee_info:
                print("\nПоследние списания комиссии за обслуживание счета:\n")
                if len(self.Client_info.get_account_service_fee) > 0:
                    for _ in self.Client_info.get_account_service_fee[-latest_numb:]:
                        print(f"   ▶  {_['quantity']} {_['currency']}: Дата списания: {_['data']}")
                else:
                    print("   ▶  None")
        if self.positions and print_positions_info:
            print("\nОткрытые позиции:\n")
            for position in self.positions:
                print(f"   ▶  {position.name} ({position.ticker}): \n"
                      f"         • Тип инстумента: {position.instrument_type}, \n"
                      f"         • Страна: {position.country_of_risk_name} ({position.country_of_risk}), \n"
                      f"         • Биржа: {position.real_exchange} ({position.exchange}), \n"
                      f"         • Сектор: {position.sector}, \n"
                      f"         • Количество: {position.quantity}, \n"
                      f"         • Цена лота на момент поркупки: "
                      f"{position.average_position_price_fifo} {position.currency},\n"
                      f"         • Текущая цена: {position.current_price} {position.currency}, \n"
                      f"         • Итог за позицию: {position.expected_yield} {position.currency}, \n"
                      f"         • Общий итог: {position.total_price_quantity} {position.currency}, \n"
                      f"         • Общий итог в процентах: {position.total_percentage} %\n")
        if self.operations and print_operations_info:
            print("\nПоследние сделки:\n")
            if len(self.operations) > 0:
                for operation in self.operations[-latest_numb:]:
                    print(f"   ▶  {operation.name} ({operation.ticker}): \n"
                          f"         • Направление сделки: {operation.trad_type}, \n"
                          f"         • Тип инстумента: {operation.instrument_type}, \n"
                          f"         • Страна: {operation.country_of_risk_name} ({operation.country_of_risk}), \n"
                          f"         • Биржа: {operation.real_exchange} ({operation.exchange}), \n"
                          f"         • Сектор: {operation.sector}, \n"
                          f"         • Количество: {operation.quantity}, \n"
                          f"         • Цена: {operation.price} {operation.currency}, \n"
                          f"         • Общая цена: {operation.payment} {operation.currency}, \n"
                          f"         • Дата сделки: {operation.date}\n")
                if self.commission and print_commission_info_:
                    print(f"\nКомиссия за {self.commission['days']} дней:\n\n"
                          f"   ▶  {self.commission['USD']} USD, "
                          f"{self.commission['RUB']} RUB, "
                          f"{self.commission['other']} other\n")
        if self.instrument_info and print_instrument_info_:
            print(f"\nИнструмент {self.instrument_info.name} ({self.instrument_info.ticker}):\n\n"
                  f"   ▶  Последняя цена: "
                  f"{InstrumentArguments().get_last_price(self.Client_info.Client, self.instrument_info.figi)}\n"
                  f"   ▶  Валюта: {self.instrument_info.currency}\n"
                  f"   ▶  Тип инструмента: {self.instrument_info.type}\n"
                  f"   ▶  Страна: {self.instrument_info.country_of_risk_name} "
                  f"({self.instrument_info.country_of_risk})\n"
                  f"   ▶  Биржа: {self.instrument_info.real_exchange} ({self.instrument_info.exchange})\n"
                  f"   ▶  Сектор: {self.instrument_info.sector}\n"
                  f"   ▶  Торговля в Short: {self.instrument_info.short_enabled_flag}\n"
                  f"   ▶  Статус: {self.instrument_info.trading_status}\n"
                  f"   ▶  Доступность торгов: {self.instrument_info.api_trade_available_flag}\n"
                  f"   ▶  Дивиденды: {self.instrument_info.div_yield_flag}\n"
                  f"   ▶  Дата первой минутной свечи: {self.instrument_info.first_1min_candle_date}\n"
                  f"   ▶  Дата первой дневной свечи: {self.instrument_info.first_1day_candle_date}")
        if self.candles_info and self.instrument_info and print_candles_info_:
            print(f"\nИнформация о свечах инструмента {self.instrument_info.name} ({self.instrument_info.ticker}):\n\n"
                  f"   ▶  Интервал свечей: {self.candles_info.candle_interval}\n"
                  f"   ▶  Цена закрытия последней свечи: "
                  f"{self.candles_info.last_close_candle} {self.instrument_info.currency}\n"
                  f"   ▶  Цены закрытия за 3 свечи: "
                  f"{self.candles_info.get_only_close_candles()[-3:]} {self.instrument_info.currency}\n"
                  f"   ▶  Цена максимума последней свечи: "
                  f"{self.candles_info.last_high_candle} {self.instrument_info.currency}\n"
                  f"   ▶  Цена минимума последней свечи: "
                  f"{self.candles_info.last_low_candle} {self.instrument_info.currency}\n"
                  f"   ▶  Разница максимума и минимума последней свечи: "
                  f"{self.candles_info.last_difference_high_low} {self.instrument_info.currency}\n"
                  f"   ▶  Цена 4ohlc последней свечи: {self.candles_info.last_4ohlc_candle} "
                  f"{self.instrument_info.currency}\n"
                  f"   ▶  Тренд закрытия последней свечи: {self.candles_info.last_close_candle_positivity_negativity}\n"
                  f"   ▶  Средний тренд закрытия за 3 свечи: "
                  f"{self.candles_info.get_ema_close_positivity_negativity(3)}\n"
                  f"   ▶  DI тренд за 3 свечи: {self.candles_info.get_di_trend(3)}\n"
                  f"   ▶  Последний объем свечи: {self.candles_info.last_volume_candle}\n"
                  f"   ▶  Средний эксп. объем за 3 свечи: {round(self.candles_info.get_volume_ema(3), 2)}\n"
                  f"   ▶  Средняя силы тренда за 3 свечи: {self.candles_info.get_adx(3)}\n"
                  f"   ▶  Средняя цена закрытия за 3 свечи: {round(self.candles_info.get_ma(3), 2)} "
                  f"{self.instrument_info.currency}\n"
                  f"   ▶  Средняя эксп. цена закрытия за 3 свечи: "
                  f"{round(self.candles_info.get_ema(3), 2)} {self.instrument_info.currency}\n"
                  f"   ▶  Время последней свечи: {self.candles_info.last_time_candle}\n")

    def print_settings_info(self):
        candle_interval_arguments = ["1 мин.", "5 мин."]
        if input(f"\n┏━━━━━━━━━━━━━━━━ • Ваши настройки • ━━━━━━━━━━━━━━━━┓\n"
                 f"   ▶  Количество дней: {self.settings_info.days}\n"
                 f"   ▶  Количество лотов к покупке: {self.settings_info.post_order_quantity} шт.\n"
                 f"   ▶  Направление сделок: {self.settings_info.trading_direction_mode}\n"
                 f"   ▶  Номер стратегии: {self.settings_info.strategy_name}\n"
                 f"   ▶  Тип свечей: {self.settings_info.str_candle_type}\n"
                 f"   ▶  Интервал свечей: {candle_interval_arguments[self.settings_info.candle_interval.value - 1]}\n"
                 f"   ▶  Длина ADX: {self.settings_info.adx_length}\n"
                 f"   ▶  Длина chandelier exit: {self.settings_info.chandelier_exit_length}\n"
                 f"   ▶  Множитель chandelier exit: {self.settings_info.chandelier_exit_factor}\n"
                 f"   ▶  Длина EMA: {self.settings_info.ema_length}\n"
                 f"   ▶  stop_loss: {self.settings_info.stop_loss}\n"
                 f"   ▶  Уровень обновления stop_loss: {self.settings_info.stop_loss_level_update}\n"
                 f"   ▶  stop_loss_quantity: {self.settings_info.stop_loss_quantity} шт.\n"
                 f"   ▶  take_profit: {self.settings_info.take_profit}\n"
                 f"   ▶  take_profit_quantity: {self.settings_info.take_profit_quantity} шт.\n"
                 f"   ▶  Выполнение заявок: {self.settings_info.real_order}\n"
                 f"   ▶  Пропуск первого сигнала: {self.settings_info.first_signal_processing}\n"
                 f"   ▶  Стиль вывода в консоль: {self.settings_info.print_style}\n"
                 f"   ▶  Запись логов: {self.settings_info.logging}\n"
                 f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
                 f"'*' вызов функции изменения настроек\n"
                 f"Проверьте введенные настройки и нажмите Enter для запуска программы:") == "*":
            SettingsHandler.change_settings(self.settings_info)

    @staticmethod
    def print_trading_info(t_inf: TradingInfo):
        if t_inf.settings.print_style == 0:
            print(f"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
                  f"   ▶  Время работы: [ {t_inf.working_hours_min} мин. ]                        "
                  f"Дата|время: [ {t_inf.start_datatime_str} ]\n   "
                  f"▶  Направление сделки: [ {t_inf.deal_direction_str} ]"
                  f"                       Режим торгов: [ {t_inf.trading_direction_mode} ]\n"
                  f" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                  f"   ▶     • Последняя цена: [ {t_inf.last_price} {t_inf.currency} ]                  "
                  f"Ticker: [ {t_inf.instrument_arguments.ticker} ]\n"
                  f"         • Верхний тригер: [ {t_inf.last_chandelier_exit_long} ]"
                  f"              Candle interval: [ {t_inf.settings.str_candle_interval} ]\n"
                  f"         • Нижний тригер:  [ {t_inf.last_chandelier_exit_short} ]\n"
                  f"         • Сила тренда:    [ {t_inf.last_adx} % ]\n"
                  f"         • EMA{t_inf.settings.ema_length}:         [ {t_inf.last_ema} ]\n"
                  f"         • Тренд закрытия последней свечи: "
                  f"[ {t_inf.indicators.candle_info.last_close_candle_positivity_negativity} ]\n"
                  f"         • Средний тренд закрытия за 10 свечей:  "
                  f"[ {t_inf.indicators.candle_info.get_ma_close_positivity_negativity(10)} ]\n"
                  f"         • Разница максимума и минимума последней свечи: "
                  f" [ {t_inf.indicators.candle_info.last_difference_high_low} ]\n"
                  f"   ▶  Цена покупки: [ {t_inf.buy_price} {t_inf.currency} ]\n"
                  f"   ▶  Процент после покупки: [ {t_inf.percentage_after_order} % ]  → "
                  f"Прибыль после покупки: [ {t_inf.amount_after_order} {t_inf.currency} ]\n"
                  f"   ▶  Время с момента последней сделки: [ {t_inf.time_since_last_trade_min} мин. ]\n"
                  f"   ▶  Take profit trigger: [ {t_inf.take_profit_trigger} % ]  → "
                  f"Stop Loss trigger: [ {t_inf.stop_loss_trigger} % ]\n"
                  f"   ▶  Количество сделок: [ {t_inf.deals_quantity} шт. ]  "
                  f"→ Номер обновления = [ {t_inf.update_number} ]\n"
                  f"   ▶  Количество Take Profit: [ {t_inf.take_profit_deals_quantity} шт. ]  → "
                  f"Количество Stop Loss: [ {t_inf.stop_loss_deals_quantity} шт. ]\n"
                  f"   ▶  Лотов в наличии: [ {t_inf.stock_quantity} шт. ]  → "
                  f"Стоимость лотов в наличии: [ {t_inf.price_lots_stock} {t_inf.currency} ]\n"
                  f"   ▶  Скорость выполнения цикла: [ {t_inf.cycle_speed} сек. ]\n"
                  f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")
        elif t_inf.settings.print_style == 1:
            print(f"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
                  f"  ▶ Время работы: {t_inf.working_hours_min} мин.        "
                  f"Дата|время: {t_inf.start_datatime_str}\n "
                  f"▶  Направление сделки: {t_inf.deal_direction_str}"
                  f"       Режим торгов: {t_inf.trading_direction_mode}\n"
                  f" ▶  • Последняя цена: {t_inf.last_price} {t_inf.currency}     "
                  f"Ticker: {t_inf.instrument_arguments.ticker}\n"
                  f"    • Верхний тригер: {t_inf.last_chandelier_exit_long}        "
                  f"Candle interval: {t_inf.settings.str_candle_interval}\n"
                  f"    • Нижний тригер:  {t_inf.last_chandelier_exit_short}\n"
                  f"    • Сила тренда:    {t_inf.last_adx} %\n"
                  f"    • EMA{t_inf.settings.ema_length}:         {t_inf.last_ema}\n"
                  f"    • Тренд закрытия последней свечи: "
                  f"{t_inf.indicators.candle_info.last_close_candle_positivity_negativity}\n"
                  f"    • Средний тренд закрытия за 10 свечей: "
                  f"{t_inf.indicators.candle_info.get_ma_close_positivity_negativity(10)}\n"
                  f"    • Разница макс. и мин. последней свечи:  "
                  f"{t_inf.indicators.candle_info.last_difference_high_low}\n"
                  f" ▶  Цена покупки: {t_inf.buy_price} {t_inf.currency}\n"
                  f" ▶  Процент после покупки: {t_inf.percentage_after_order} % → "
                  f"Прибыль: {t_inf.amount_after_order} {t_inf.currency}\n"
                  f" ▶  Время с момента последней сделки: {t_inf.time_since_last_trade_min} мин.\n"
                  f" ▶  TP trigger: {t_inf.take_profit_trigger} % → "
                  f"SL trigger: {t_inf.stop_loss_trigger} % \n"
                  f" ▶  Количество сделок: {t_inf.deals_quantity} шт. → Номер обновления = {t_inf.update_number}\n"
                  f" ▶  Take Profit: {t_inf.take_profit_deals_quantity} шт. → "
                  f"Stop Loss: {t_inf.stop_loss_deals_quantity} шт.\n"
                  f" ▶  Лотов в наличии: {t_inf.stock_quantity} шт. → "
                  f"Стоимость: {t_inf.price_lots_stock} {t_inf.currency}\n"
                  f"  ▶ Скорость выполнения цикла: {t_inf.cycle_speed} сек.\n"
                  f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")
        else:
            pass

    @staticmethod
    def print_strategy_analytics_info(res_strategy: StrategiesAnalytics.ResultStrategiesAnalytics):
        long_analytics = res_strategy.long_deals_analytic
        short_analytics = res_strategy.short_deals_analytic
        print(f"\nРезультаты аналитики стратегии:\n"
              f"Количество сделок: {res_strategy.amount_of_deals}\n"
              f"\n• Результаты аналитики сделок в long:\n\n"
              f"Количество сделок в long: {res_strategy.long_deals_quantity}\n"
              f"Процент прибыльных сделок: {long_analytics.percent_of_profit_trades}\n"
              f"Процент прибыльных сделок с учетом диапазона: "
              f"{long_analytics.percent_of_profit_trades_given_the_range}\n"
              f"Средняя прибыль за сделку: {long_analytics.average_profit_per_trade}\n"
              f"Средний процент прибыли за сделку: {long_analytics.average_profit_per_trade_percentage}\n"
              f"Средняя длительность сделки в свечах: {long_analytics.average_numb_of_values_in_range}\n"
              f"Средний процент роста после покупки: {long_analytics.average_max_up_percentage}\n"
              f"Средний процент просадки после покупки: {long_analytics.average_max_down_percentage}\n"
              f"Процент максимального роста: {long_analytics.maximum_height}\n"
              f"Процент максимальной просадки: {long_analytics.maximum_drawdown}\n"
              f"Средний процент изменчивости: {long_analytics.average_percent_variability}\n"
              f"Процент среднего истинного диапазона: {long_analytics.atr_percentage}\n"
              f"Индекс риска в процентах: {long_analytics.risk_index}\n"
              f"Нормализованный stop loss уровень: {long_analytics.normalized_stop_loss_percent_level}\n"
              f"Нормализованный take profit уровень: {long_analytics.normalized_take_profit_percent_level}\n"
              f"\n• Результаты аналитики сделок в short:\n\n"
              f"Количество сделок в short: {res_strategy.short_deals_quantity}\n"
              f"Процент прибыльных сделок: {short_analytics.percent_of_profit_trades}\n"
              f"Процент прибыльных сделок с учетом диапазона: "
              f"{short_analytics.percent_of_profit_trades_given_the_range}\n"
              f"Средняя прибыль за сделку: {short_analytics.average_profit_per_trade}\n"
              f"Средний процент прибыли за сделку: {short_analytics.average_profit_per_trade_percentage}\n"
              f"Средняя длительность сделки в свечах: {short_analytics.average_numb_of_values_in_range}\n"
              f"Средний процент роста после покупки: {short_analytics.average_max_up_percentage}\n"
              f"Средний процент просадки после покупки: {short_analytics.average_max_down_percentage}\n"
              f"Процент максимального роста: {short_analytics.maximum_height}\n"
              f"Процент максимальной просадки: {short_analytics.maximum_drawdown}\n"
              f"Средний процент изменчивости: {short_analytics.average_percent_variability}\n"
              f"Процент среднего истинного диапазона: {short_analytics.atr_percentage}\n"
              f"Индекс риска в процентах: {short_analytics.risk_index}\n"
              f"Нормализованный stop loss уровень: {short_analytics.normalized_stop_loss_percent_level}\n"
              f"Нормализованный take profit уровень: {short_analytics.normalized_take_profit_percent_level}\n"

              f"\n• Рекомендуемое направление сделок: {res_strategy.recommended_trading_direction}\n")

    @staticmethod
    def print_loading(time_sleep: float = 0.002):
        for i in range(101):
            time.sleep(time_sleep)
            print('\r', 'Загрузка данных:', int(i / 5) * '█', str(i), '%', end='') if i % 10 == 0 else None
        print('\n')

    @staticmethod
    def print_start_label():
        print("\n", r"       _________________________ _____________________", "\n",
              r"      /   _____/\__    ___/  _  \\______   \__    ___/", "\n",
              r"      \_____  \   |    | /  /_\  \|       _/ |    |   ", "\n",
              r"      /        \  |    |/    |    \    |   \ |    |   ", "\n",
              r"     /_______  /  |____|\____|__  /____|_  / |____|   ", "\n",
              r"             \/                 \/       \/           ", "\n\n")

    @staticmethod
    def print_take_profit_trigger(profit_per_trade: float):
        print(f"━━━━━━━━━ • Take Profit • ━━━━━━━━━\n"
              f"  ▶  Прибыль за сделку Take Profit: {profit_per_trade}\n")

    @staticmethod
    def print_stop_loss_trigger(profit_per_trade: float):
        print(f"━━━━━━━━━ • Stop Loss • ━━━━━━━━━\n"
              f"  ▶  Прибыль за сделку Stop Loss: {profit_per_trade}\n")

    @staticmethod
    def print_buy_trigger(buy_price: float):
        print(f"━━━━━━━━━ • BUY • ━━━━━━━━━\n"
              f"  ▶  Цена продажи: {buy_price}\n")

    @staticmethod
    def print_sell_trigger(buy_price: float):
        print(f"━━━━━━━━━ • SELL • ━━━━━━━━━\n"
              f"  ▶  Цена продажи: {buy_price}\n")

    @staticmethod
    def print_before_starting_the_main_loop(settings: SettingsHandler):
        inp = input("'*' вызов функции для изменения настроек\n"
                    "'-' выставить таймер запуска основного цикла\n"
                    "Нажмите Enter для запуска основного цикла:")
        if inp == "*":
            settings.change_settings()
            PrintInfo.print_before_starting_the_main_loop(settings)
        elif inp == "-":
            Tools.start_timer(input("\nВыставите время остановки таймера в формате ( day:hour:min ):\n"
                                    "Ввод:"))
        print()


class MarketDataHandler:
    """Класс обработки стрим запроса"""

    def __init__(self, market_data: MarketDataResponse):
        """
        self.ping = round((market_data.ping.time - dt.datetime.combine(
            dt.datetime.utcnow().date(), dt.datetime.utcnow().time(), dt.timezone.utc)
                           ).total_seconds(), 4) if market_data.ping else "None"
        """
        self.last_price = Tools.compound(market_data.last_price.price.units, market_data.last_price.price.nano) \
            if market_data.last_price else None
        self.candle = CandleHandler(market_data.candle) if market_data.candle else None

    def __str__(self):
        return str(self.__dict__)


class MainServices:
    """Класс главных сервисных инструментов"""

    def __init__(self, client_: Services):
        self.Client = client_
        self.ClientInfo = ClientInfo(self.Client)

    @staticmethod
    def get_my_settings():
        return Settings().setting_request()

    def get_current_positions(self):
        return self.ClientInfo.get_current_positions()

    def get_current_operations(self, days: int = 90):
        return self.ClientInfo.get_current_operations(days)

    def get_arguments_instrument(self, figi: str):
        return FindInstrument(figi, self.ClientInfo.instruments).instrument

    def get_historical_candles(self, figi: str, days: int, candle_interval: CandleInterval, candle_type):
        historical_candles = GetHistoricalCandles(self.Client, figi, days)
        if candle_type == "oll":
            return historical_candles.get_classic_and_ha_historical_candles(candle_interval)
        return historical_candles.get_ha_historical_candles(candle_interval) if candle_type in [2, "ha"] else \
            historical_candles.get_classic_historical_candles(candle_interval)

    def get_last_historical_candle(
            self, figi: str, candle_interval: CandleInterval, candle_type, from_minutes: int = 10, out: str = "last"
    ):
        last_historical_candles = GetHistoricalCandles(self.Client, figi, 0)
        return last_historical_candles.get_ha_last_historical_candle(candle_interval, from_minutes, out) if \
            candle_type in [2, "ha"] else \
            last_historical_candles.get_classic_last_historical_candle(candle_interval, from_minutes, out)

    def get_last_price(self, figi: str):
        return InstrumentArguments(figi, self.ClientInfo.instruments).get_last_price(self.Client)

    def find_instrument(self, out: tuple = ("figi",)):
        return FindInstrument.find_instrument(self.Client, out)

    def post_order_request(self, figi: str, quantity: int, direction: OrderDirection):
        self.Client.orders.post_order(
            order_id=str(datetime.utcnow().timestamp()),
            figi=figi,
            quantity=quantity,
            account_id=self.ClientInfo.account_id,
            direction=direction,
            order_type=OrderType.ORDER_TYPE_MARKET
        )

    def get_chandelier_exit_strategy_analytics(
            self, figi: str, days: int, candle_interval: CandleInterval, candle_type, ce_period: int, ce_factor: float
    ):
        if candle_type == 1:
            cl_candle = self.get_historical_candles(figi, days, candle_interval, 1)
            ha_candle = None
        elif candle_type == 2:
            cl_candle, ha_candle = self.get_historical_candles(figi, days, candle_interval, "oll")
        else:
            cl_candle, ha_candle = self.get_historical_candles(figi, days, candle_interval, "oll")
        indicators = Indicators(ha_candle if candle_type == 2 else cl_candle)
        indicators.chandelier_exit(ce_period, ce_factor)
        return StrategiesAnalytics(indicators).strategies_analytics("chandelier_exit_strategy", cl_candle)

    @staticmethod
    def request_iterator(figi: str, subscribe_interval: SubscriptionInterval):
        yield MarketDataRequest(
            subscribe_candles_request=SubscribeCandlesRequest(
                subscription_action=SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                instruments=[
                    CandleInstrument(figi=figi, interval=subscribe_interval)],
                waiting_close=True
            )
        )
        yield MarketDataRequest(
            subscribe_last_price_request=SubscribeLastPriceRequest(
                subscription_action=SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                instruments=[LastPriceInstrument(figi=figi)]
            )
        )
        while True:
            pass


def plotting_mode(services: MainServices):
    figi = services.find_instrument()[-1]
    candles = services.get_historical_candles(figi, 30, CandleInterval.CANDLE_INTERVAL_5_MIN, 1)
    close_candles = CandlesInfo(candles).get_only_close_candles()
    candles_numbs = [i for i in range(len(close_candles))]
    Tools.plotting(candles_numbs, close_candles)


def launch_in_strategy_analysis_mode(services: MainServices, path: str = "strategy_analysis_mode_setting.txt"):
    print("Произведен запуск в режиме анализа стратегии...\n")
    try:
        with open(path, encoding='utf-8') as filehandle:
            strategy_analysis_mode_setting = tuple(
                [rl.rstrip().split('=')[1].replace(" ", "") for rl in filehandle.readlines()])
    except FileNotFoundError:
        print("Файл с настройками не обнаружен!\n")
        raise SystemExit
    interval_argument = {"1m": CandleInterval.CANDLE_INTERVAL_1_MIN, "5m": CandleInterval.CANDLE_INTERVAL_5_MIN}
    days = int(strategy_analysis_mode_setting[0])
    candle_type = int(strategy_analysis_mode_setting[1])
    candle_interval_str = strategy_analysis_mode_setting[2]
    candle_interval = interval_argument[candle_interval_str]
    name_strategy = strategy_analysis_mode_setting[3].replace(" ", "")
    period_range = tuple([int(i) for i in strategy_analysis_mode_setting[4].split(":")])
    factor_range = tuple([float(i) for i in strategy_analysis_mode_setting[5].split(":")])
    factor_step = float(strategy_analysis_mode_setting[6])
    input(f"Ваши настройки в режиме анализа стратегии:\n\n"
          f"Количество дней = {days}\n"
          f"Тип свечей для индикатора = {candle_type}\n"
          f"Интервал свечей = {candle_interval_str}\n"
          f"Имя стратегии = {name_strategy}\n"
          f"chandelier exit диапазон периода = {period_range}\n"
          f"chandelier exit диапазон множителя = {factor_range}\n"
          f"chandelier exit шаг множителя = {factor_step}\n\n"
          f"Проверьте настройки и нажмите Enter для запуска:")
    figi = services.find_instrument()[-1]
    print("\nПолучение свечей...\n")
    if name_strategy == "chandelier_exit":
        if candle_type == 1:
            cl_candle = services.get_historical_candles(figi, days, candle_interval, 1)
            ha_candle = None
        elif candle_type == 2:
            cl_candle, ha_candle = services.get_historical_candles(figi, days, candle_interval, "oll")
        else:
            cl_candle, ha_candle = services.get_historical_candles(figi, days, candle_interval, "oll")
        indicators = Indicators(ha_candle if candle_type == 2 else cl_candle)
        analytics_iterator = StrategiesAnalytics(
            indicators).chandelier_exit_strategy_analytics_iterator(
            cl_candle if candle_type == 2 else None, period_range, factor_range, factor_step)
        print(StrategyAnalyticsResultsEvaluationIterator(*analytics_iterator))
    else:
        print("Индикатор не определен...")
        indicators = None
        cl_candle = None
    if name_strategy == "chandelier_exit":
        while True:
            inp = input("Анализ стратегии через определение настроек индикатора:\n"
                        "Определите параметры индикатора в формате ( period:factor ):")
            period, factor = inp.split(":")
            indicators.chandelier_exit(int(period), float(factor))
            strategy_analytics = StrategiesAnalytics(
                indicators).strategies_analytics("chandelier_exit_strategy", cl_candle)
            PrintInfo.print_strategy_analytics_info(strategy_analytics)
            if input("'*' завершить работу\n""Enter для продолжения:") == "*":
                raise SystemExit
    raise SystemExit


def writing_strategies(path: str = "list_strategies_name.txt"):
    with open(path, "w") as file:
        file.write(str([m for m in dir(Strategies) if not m.startswith('__')]))


def signal_processing(
        signal_indicator: int, stock_quantity: int, trading_direction_mode: str, first_signal_processing: bool
):
    if trading_direction_mode == "long":
        signal_indicator = signal_indicator - 1 if stock_quantity == 0 and signal_indicator == -1 else signal_indicator
        if first_signal_processing and signal_indicator == 1:
            signal_indicator += 1
    elif trading_direction_mode == "short":
        signal_indicator = signal_indicator + 1 if stock_quantity == 0 and signal_indicator == 1 else signal_indicator
        if first_signal_processing and signal_indicator == -1:
            signal_indicator -= 1
    return signal_indicator


def take_profit_and_stop_loss_handler(
        settings: SettingsHandler, services: MainServices,
        figi: str, amount_after_order: float,
        percentage_after_order: float, stock_quantity: int,
        take_profit_numb: int, take_profit_deals_quantity: int,
        stop_loss_numb: int, stop_loss_deals_quantity: int
):
    if percentage_after_order >= settings.take_profit[take_profit_numb]:
        take_profit_numb = take_profit_numb + 1 if take_profit_numb <= (len(settings.take_profit) - 1) else -1
        profit_per_trade = round(amount_after_order / abs(stock_quantity) * settings.take_profit_quantity, 2)
        if settings.trading_direction_mode == "long" and stock_quantity >= settings.take_profit_quantity:
            if settings.real_order:
                services.post_order_request(figi, settings.take_profit_quantity, OrderDirection.ORDER_DIRECTION_SELL)
            take_profit_deals_quantity += 1
            PrintInfo.print_take_profit_trigger(profit_per_trade)
            stock_quantity -= settings.take_profit_quantity
        elif settings.trading_direction_mode == "short" and stock_quantity <= -settings.take_profit_quantity:
            if settings.real_order:
                services.post_order_request(figi, settings.take_profit_quantity, OrderDirection.ORDER_DIRECTION_BUY)
            take_profit_deals_quantity += 1
            PrintInfo.print_take_profit_trigger(profit_per_trade)
            stock_quantity += settings.take_profit_quantity
        else:
            pass
    elif percentage_after_order < settings.stop_loss[stop_loss_numb] + \
            (0 if percentage_after_order < settings.stop_loss_level_update else -settings.stop_loss_level_update):
        stop_loss_numb = stop_loss_numb + 1 if stop_loss_numb <= (len(settings.stop_loss) - 1) else -1
        profit_per_trade = round(amount_after_order / abs(stock_quantity) * settings.stop_loss_quantity, 2)
        if settings.trading_direction_mode == "long" and stock_quantity >= settings.stop_loss_quantity:
            if settings.real_order:
                services.post_order_request(figi, settings.stop_loss_quantity, OrderDirection.ORDER_DIRECTION_SELL)
            stop_loss_deals_quantity += 1
            PrintInfo.print_stop_loss_trigger(profit_per_trade)
            stock_quantity -= settings.stop_loss_quantity
        elif settings.trading_direction_mode == "short" and stock_quantity <= -settings.stop_loss_quantity:
            if settings.real_order:
                services.post_order_request(figi, settings.stop_loss_quantity, OrderDirection.ORDER_DIRECTION_BUY)
            stop_loss_deals_quantity += 1
            PrintInfo.print_stop_loss_trigger(profit_per_trade)
            stock_quantity += settings.stop_loss_quantity
        else:
            pass
    else:
        pass
    return stock_quantity, take_profit_numb, take_profit_deals_quantity, stop_loss_numb, stop_loss_deals_quantity


def trading_direction_handler(
        last_price: float,
        signal_indicator: int,
        figi: str,
        services: MainServices,
        settings: SettingsHandler,
        trading_direction_arguments: TradingDirectionArguments
):
    if signal_indicator == 1:
        quantity = abs(trading_direction_arguments.stock_quantity) if \
            settings.trading_direction_mode == "short" else settings.post_order_quantity
        if settings.real_order and quantity != 0:
            services.post_order_request(figi, quantity, OrderDirection.ORDER_DIRECTION_BUY)
        trading_direction_arguments.buy_price = last_price
        trading_direction_arguments.stock_quantity += quantity
        trading_direction_arguments.deals_quantity += 1
        trading_direction_arguments.transaction_time = datetime.now()
        PrintInfo.print_buy_trigger(trading_direction_arguments.buy_price)
    elif signal_indicator == -1:
        quantity = abs(trading_direction_arguments.stock_quantity) if \
            settings.trading_direction_mode == "long" else settings.post_order_quantity
        if settings.real_order and quantity != 0:
            services.post_order_request(figi, quantity, OrderDirection.ORDER_DIRECTION_SELL)
        trading_direction_arguments.buy_price = last_price
        trading_direction_arguments.stock_quantity -= quantity
        trading_direction_arguments.deals_quantity += 1
        trading_direction_arguments.transaction_time = datetime.now()
        PrintInfo.print_sell_trigger(trading_direction_arguments.buy_price)
    return trading_direction_arguments


def loger(trading_info: TradingInfo, previous_stock_quantity_: int, max_lines: int = 250, del_lines: int = 25):
    if trading_info.settings.logging and abs(trading_info.stock_quantity) != abs(previous_stock_quantity_):
        with open("loger.txt", "a") as file:
            loger_write = (f"• ID сессии {trading_info.settings.session_id}| "
                           f"Log: instrument_ticker {trading_info.instrument_arguments.ticker}, "
                           f"deal_direction '{trading_info.deal_direction_str}', "
                           f"trading_mode '{trading_info.settings.trading_direction_mode}', "
                           f"stock_quantity {trading_info.stock_quantity} шт, "
                           f"price_of_lots_in_stock {trading_info.price_lots_stock} "
                           f"{trading_info.currency}, "
                           f"transaction_time {trading_info.transaction_time.time()}, "
                           f"deals_numb {trading_info.deals_quantity}") if \
                abs(trading_info.signal_indicator) in (1, -1) else \
                (f"• ID сессии {trading_info.settings.session_id}| "
                 f"Log: instrument_ticker {trading_info.instrument_arguments.ticker}, "
                 f"last_price {trading_info.last_price} {trading_info.currency}, "
                 f"stock_quantity {trading_info.stock_quantity} шт, "
                 f"percentage_after_order {trading_info.percentage_after_order}%, "
                 f"take_profit_deals_quantity {trading_info.take_profit_deals_quantity}, "
                 f"stop_loss_deals_quantity {trading_info.stop_loss_deals_quantity},"
                 f"take_profit_quantity {trading_info.settings.take_profit_quantity}, "
                 f"stop_loss_quantity {trading_info.settings.stop_loss_quantity}")
            file.write(loger_write + "\n")
        with open("loger.txt", "r") as file:
            len_file = len(file.readlines())
        if len_file >= max_lines:
            with open('loger.txt', 'r') as file:
                read_file = "".join(i for i in file.readlines()[del_lines:])
            with open('loger.txt', 'w') as file:
                file.write(read_file)


def main():
    writing_strategies()

    """ Settings """

    S = MainServices.get_my_settings()
    if type(S) == SettingsHandler:
        PrintInfo(settings_info=S).print_settings_info()

    signal_indicator = 0
    take_profit_deals_quantity, stop_loss_deals_quantity = 0, 0
    take_profit_numb, stop_loss_numb = 0, 0
    start_data_time = datetime.now()
    previous_stock_quantity = 0
    trading_direction_arguments = TradingDirectionArguments(None, 0, 0, start_data_time)

    """ Start """

    PrintInfo.print_start_label()
    threading.Thread(target=PrintInfo.print_loading, name="print_loading").start() if type(S) == SettingsHandler \
        else None

    with Client(TOKENS.Tinkoff_Token) as client:
        if S == "strategy_analysis_mode":
            launch_in_strategy_analysis_mode(MainServices(client))
        elif S == "plotting_mode":
            plotting_mode(MainServices(client))
        while len(threading.enumerate()) > 1:
            time.sleep(0.05)

        """Создание экземпляра класса MainServices и вывод информации о аккаунте"""
        ms = MainServices(client)
        PrintInfo(
            client_info=ms.ClientInfo,
            positions=ms.ClientInfo.get_account_positions,
            operations=ms.ClientInfo.get_account_operations,
            commission=ms.ClientInfo.get_account_commission,
        ).print_info()

        """Запрос, обработка и вывод информации о инструменте торговли"""
        figi = ms.find_instrument()[-1]
        instrument_arguments = ms.get_arguments_instrument(figi)
        PrintInfo(
            client_info=ms.ClientInfo,
            instrument_info=instrument_arguments,
            candles_info=CandlesInfo(GetHistoricalCandles(client, figi, 90).get_classic_1day_historical_candles())
        ).print_info(print_client_info=False)

        """Аналитика стратегии"""
        if S.strategy_analytics:
            strategy_analytics = ms.get_chandelier_exit_strategy_analytics(
                figi, S.days, S.candle_interval, S.candle_type, S.chandelier_exit_length, S.chandelier_exit_factor)
            PrintInfo.print_strategy_analytics_info(strategy_analytics)

        PrintInfo.print_before_starting_the_main_loop(S)

        """Запрос и обработка первичных данных"""
        candle_list_oll = ms.get_historical_candles(figi, S.days, S.candle_interval, S.candle_type)
        indicators = Indicators(candle_list_oll[1:])
        adx = indicators.adx(S.adx_length)
        chandelier_exit = indicators.chandelier_exit(S.chandelier_exit_length, S.chandelier_exit_factor)
        ema = indicators.ema(S.ema_length)
        last_historical_candle = ms.get_last_historical_candle(figi, S.candle_interval, S.candle_type)
        if last_historical_candle and last_historical_candle.time != candle_list_oll[-1].time:
            print(f"\nДобавлена пропущенная свеча:\n\n"
                  f"Старая последняя свеча: {str(candle_list_oll[-1])}\n"
                  f"Новая последняя свеча: {str(last_historical_candle)}\n")
            candle_list_oll.append(last_historical_candle)
        candle = candle_list_oll[-1]
        last_price = ms.get_last_price(figi)

        """Основной цикл получения и обработки данных"""
        for marketdata in client.market_data_stream.market_data_stream(
                ms.request_iterator(figi, S.subscription_candle_interval)
        ):
            """Обработка marketdata запроса"""
            start_while_time = time.time()
            marketdata = MarketDataHandler(marketdata)
            previous_candle, previous_last_price = candle, last_price
            if marketdata.candle:
                candle = marketdata.candle.ha_candle_handler(candle_list_oll[-1]) \
                    if S.candle_type in [2, "ha"] else marketdata.candle.candle
                if candle_list_oll[-1] != candle:
                    print("Candle close\n")
                    candle_list_oll.append(candle)
                indicators = Indicators(candle_list_oll)
                adx = indicators.adx_plus_one(S.adx_length, adx)
                chandelier_exit = indicators.chandelier_exit_plus_one(
                    S.chandelier_exit_length, S.chandelier_exit_factor, chandelier_exit)
                ema = indicators.ema_plus_one(S.ema_length, ema)
            if marketdata.last_price:
                last_price = marketdata.last_price
            if candle == previous_candle and last_price == previous_last_price:
                print("    Continue\n")
                continue

            """Определение и обработка сигнала"""
            signal_indicator = getattr(Strategies(signal_indicator, indicators), S.strategy_name)()
            if signal_indicator in [1, -1] or S.first_signal_processing:
                signal_indicator = signal_processing(signal_indicator, trading_direction_arguments.stock_quantity,
                                                     S.trading_direction_mode, S.first_signal_processing)
                S.first_signal_processing = False
                trading_direction_arguments = trading_direction_handler(
                    last_price, signal_indicator, figi, ms, S, TradingDirectionArguments(
                        trading_direction_arguments.buy_price, trading_direction_arguments.stock_quantity,
                        trading_direction_arguments.deals_quantity, trading_direction_arguments.transaction_time
                    )
                )

            """Обработка событий после открытия позиции"""
            if trading_direction_arguments.stock_quantity != 0:
                percentage_after_order = round((last_price - trading_direction_arguments.buy_price) *
                                               100 / trading_direction_arguments.buy_price, 2)
                if S.trading_direction_mode == "short":
                    percentage_after_order *= -1
                amount_after_order = round(last_price * (percentage_after_order / 100) *
                                           abs(trading_direction_arguments.stock_quantity), 2)
                (trading_direction_arguments.stock_quantity, take_profit_numb, take_profit_deals_quantity,
                 stop_loss_numb, stop_loss_deals_quantity) = take_profit_and_stop_loss_handler(
                    S, ms, figi, amount_after_order, percentage_after_order, trading_direction_arguments.stock_quantity,
                    take_profit_numb, take_profit_deals_quantity, stop_loss_numb, stop_loss_deals_quantity
                )
            else:
                take_profit_numb, stop_loss_numb = 0, 0
                take_profit_deals_quantity, stop_loss_deals_quantity = 0, 0
                percentage_after_order, amount_after_order = "None", "None"
                trading_direction_arguments.buy_price = "None"

            """Формирование информации по итоговым значениям торговли"""
            trading_info = TradingInfo(
                S, trading_direction_arguments, last_price, signal_indicator, instrument_arguments, indicators,
                datetime.now(), start_data_time, start_while_time, percentage_after_order, amount_after_order,
                take_profit_deals_quantity, stop_loss_deals_quantity, take_profit_numb, stop_loss_numb
            )
            """Отрисовка и запись информации по итоговым значениям торговли"""
            PrintInfo.print_trading_info(trading_info)
            loger(trading_info, previous_stock_quantity)
            previous_stock_quantity = trading_direction_arguments.stock_quantity


if __name__ == "__main__":
    main()
