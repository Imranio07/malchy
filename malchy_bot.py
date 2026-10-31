"""Malchy — Telegram bot that answers the shepherd's questions, by voice or text.
Voice is transcribed on the CPU with Meta's open Omnilingual ASR (Kyrgyz);
answers are exact facts from the latest reports and the weather forecast."""
import datetime as dt
import json
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import weather

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = str(os.environ["TELEGRAM_CHAT_ID"])
API = f"https://api.telegram.org/bot{TOKEN}"
ASR_MODEL = os.environ.get("ASR_MODEL", "omniASR_CTC_1B")
ASR_LANG = os.environ.get("ASR_LANG", "kir_Cyrl")

INTENTS = {
    "threat": ("карышкыр", "карыш", "жырткыч", "волк", "хищник"),
    "weather": ("аба", "ырай", "кар ", "кар?", "шамал", "суук", "аяз", "погод", "снег", "ветер", "мороз"),
    "missing": ("жок", "жогол", "жетиш", "пропа", "не хват", "потер"),
    "stray": ("бөлүн", "болун", "адаш", "качып", "отбил", "убеж"),
    "sick": ("оору", "кыймыл", "больн", "болеет", "не двига"),
    "count": ("канча", "сан", "сколько", "кой", "овец", "овц"),
}
KYRGYZ_HINTS = ("ү", "ө", "ң", "канча", "барбы", "кандай", "жок", "бугун", "кайсы", "карышкыр")
THREAT_KY = {"wolf": "карышкыр", "dog": "ит", "bear": "аюу", "fox": "түлкү"}
THREAT_RU = {"wolf": "волк", "dog": "собака", "bear": "медведь", "fox": "лиса"}


def api(method, **params):
    data = urllib.parse.urlencode(params).encode() if params else None
    with urllib.request.urlopen(f"{API}/{method}", data=data, timeout=60) as response:
        return json.load(response)["result"]


def send(text):
    api("sendMessage", chat_id=CHAT_ID, text=text)


def download_voice(file_id, folder):
    path = api("getFile", file_id=file_id)["file_path"]
    oga, wav = Path(folder) / "voice.oga", Path(folder) / "voice.wav"
    urllib.request.urlretrieve(f"https://api.telegram.org/file/bot{TOKEN}/{path}", oga)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(oga), "-ar", "16000", "-ac", "1", str(wav)],
                   check=True)
    return wav


def load_facts():
    history = json.loads(Path("herd_history.json").read_text()) if Path("herd_history.json").exists() else []
    history.sort(key=lambda e: e["date"])
    today = history[-1] if history else None
    prev = history[-2] if len(history) > 1 else None
    threats = {}
    for report in Path(".").glob("*_report.json"):
        threats.update(json.loads(report.read_text()).get("threats", {}))
    return today, prev, threats


def weather_risks():
    lat, lon = os.environ.get("PASTURE_LAT"), os.environ.get("PASTURE_LON")
    if not lat or not lon:
        return None
    _, rows = weather.fetch(float(lat), float(lon))
    return weather.assess(rows)


def when_ky(t):
    days = (t.date() - dt.date.today()).days
    day = {0: "бүгүн", 1: "эртең", 2: "бүрсүгүнү"}.get(days, t.strftime("%d.%m"))
    return f"{day} саат {t.hour}:00 чамасында"


def answer(question):
    q = question.lower() + " "
    ky = any(h in q for h in KYRGYZ_HINTS)
    intent = next((name for name, stems in INTENTS.items() if any(s in q for s in stems)), "help")
    today, prev, threats = load_facts()

    if intent == "help" or (today is None and intent in ("count", "missing", "stray", "sick")):
        if today is None and intent != "help":
            return "Азырынча маалымат жок — адегенде оторду санап ал." if ky else \
                "Пока нет данных — сначала посчитай стадо."
        return ("Мындай сурай аласың: канча кой бар? кайсы кой жок? аба ырайы кандай? карышкыр барбы?"
                if ky else "Можно спросить: сколько овец? кого не хватает? какая погода? был ли волк?")

    if intent == "threat":
        if not threats:
            return "Карышкыр байкалган жок." if ky else "Волк не замечен."
        names = ", ".join((THREAT_KY if ky else THREAT_RU).get(t, t) for t in threats)
        return (f"⚠️ Эскертүү: отордун жанында {names} көрүндү! Малды текшер." if ky
                else f"⚠️ Внимание: возле стада замечен {names}! Проверь стадо.")

    if intent == "weather":
        try:
            risks = weather_risks()
        except Exception:
            risks = None
        if risks is None:
            return "Аба ырайын азыр билүү мүмкүн эмес." if ky else "Сейчас не получается узнать погоду."
        if not risks:
            return "Жакынкы эки күндө аба ырайы тынч." if ky else "Ближайшие двое суток погода спокойная."
        if not ky:
            return weather.describe(risks, os.environ.get("PASTURE_NAME", "пастбище"))
        parts = []
        if "snow" in risks:
            parts.append(f"{when_ky(risks['snow']['from'])} кар жаайт, болжол менен {risks['snow']['total_cm']} см")
        if "wind" in risks:
            parts.append(f"шамал {risks['wind']['max_kmh']} км/саатка чейин күчөйт")
        if "frost" in risks:
            parts.append(f"аяз {risks['frost']['min_c']}°C чейин")
        return "⚠️ Аба ырайы: " + "; ".join(parts) + ". Малды төмөн түшүрүүнү ойлон."

    count = today["count"]
    change = count - prev["count"] if prev else None
    strays, still = today.get("stray_ids", []), today.get("motionless_ids", [])

    if intent == "missing":
        if change is not None and change < 0:
            return (f"{-change} кой жетишпейт: кечээ {prev['count']}, бүгүн {count}." if ky
                    else f"Не хватает {-change}: вчера было {prev['count']}, сегодня {count}.")
        return "Бардык кой ордунда." if ky else "Все на месте."
    if intent == "stray":
        if not strays:
            return "Отордон бөлүнгөн кой жок." if ky else "Отбившихся нет."
        ids = ", ".join(map(str, strays))
        return (f"{len(strays)} кой отордон бөлүнүп кетти: № {ids}. Жайыттын четин текшер." if ky
                else f"Отбились от стада: {len(strays)} (№ {ids}). Проверь края пастбища.")
    if intent == "sick":
        if not still:
            return "Баары кыймылдап жүрөт." if ky else "Все двигаются."
        ids = ", ".join(map(str, still))
        return (f"{len(still)} кой дээрлик кыймылдабайт (№ {ids}) — ооруп калышы мүмкүн, карап кел." if ky
                else f"Почти не двигаются: {len(still)} (№ {ids}). Возможно, больны — осмотри.")

    text = f"Бүгүн {count} кой саналды." if ky else f"Сегодня посчитано: {count}."
    if change is not None:
        if change < 0:
            text += f" Кечээ {prev['count']} болчу — {-change} кой жетишпейт." if ky else \
                f" Вчера было {prev['count']} — не хватает {-change}."
        elif change > 0:
            text += f" Кечээ {prev['count']} болчу." if ky else f" Вчера было {prev['count']}."
    return text


def main():
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    print(f"Loading {ASR_MODEL} on CPU...")
    asr = ASRInferencePipeline(model_card=ASR_MODEL, device="cpu")

    pending = api("getUpdates")
    offset = pending[-1]["update_id"] + 1 if pending else 0
    print("Malchy bot is listening. Ask by voice or text in Telegram (Ctrl+C to stop).")

    while True:
        try:
            updates = api("getUpdates", offset=offset, timeout=30)
        except Exception as error:
            print(f"[warning] {error}")
            continue
        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message", {})
            if str(message.get("chat", {}).get("id")) != CHAT_ID:
                continue
            if "voice" in message:
                with tempfile.TemporaryDirectory() as folder:
                    wav = download_voice(message["voice"]["file_id"], folder)
                    question = asr.transcribe([str(wav)], lang=[ASR_LANG], batch_size=1)[0]
                heard = f"🎙 Уктум: «{question}»\n\n"
            elif "text" in message:
                question, heard = message["text"], ""
            else:
                continue
            reply = heard + answer(question)
            print(f"Q: {question}\nA: {reply}\n")
            send(reply)


if __name__ == "__main__":
    main()
