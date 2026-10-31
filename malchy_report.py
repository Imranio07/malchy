"""Malchy — daily herd report. Safety-critical facts and advice are written by
code and sent word for word; the model only adds one warm opening sentence
with no numbers, checked by a guardrail."""
import argparse
import datetime as dt
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

import weather

HISTORY = Path("herd_history.json")
LANGUAGES = {"ru": "Russian", "ky": "Kyrgyz", "en": "English"}
ALL_CLEAR = ("всё хорошо", "все хорошо", "всё спокойно", "все спокойно", "в порядке", "нет проблем",
             "all good", "everything is fine")


def load_history():
    if HISTORY.exists():
        return json.loads(HISTORY.read_text())
    return []


def save_history(history):
    HISTORY.write_text(json.dumps(sorted(history, key=lambda e: e["date"]), indent=2, ensure_ascii=False))


THREAT_NAMES = {"wolf": "волк", "dog": "собака", "bear": "медведь", "fox": "лиса"}


def build_summary(entry, history, expected, weather_line, threats=None):
    earlier = [e for e in history if e["date"] < entry["date"]]
    prev = max(earlier, key=lambda e: e["date"]) if earlier else None
    return {
        "date": entry["date"],
        "count": entry["count"],
        "previous_count": prev["count"] if prev else None,
        "change": entry["count"] - prev["count"] if prev else None,
        "expected": expected,
        "missing_vs_expected": expected - entry["count"] if expected else None,
        "stray_ids": entry["stray_ids"],
        "motionless_ids": entry["motionless_ids"],
        "weather": weather_line,
        "threats": threats or {},
    }


def has_problems(s):
    return bool(s["threats"] or s["weather"] or s["stray_ids"] or s["motionless_ids"]
                or (s["change"] is not None and s["change"] < 0)
                or (s["missing_vs_expected"] and s["missing_vs_expected"] > 0))


def facts_message(s):
    lines = [f"Малчы — отчёт за {s['date']}"]
    for threat, info in s["threats"].items():
        name = THREAT_NAMES.get(threat, threat)
        lines.append(f"🐺 ТРЕВОГА: возле стада замечен {name} (виден на {info['frames']} кадрах). "
                     "Проверь стадо прямо сейчас!")
    if s["weather"]:
        lines.append(s["weather"])
    lines.append(f"Посчитано: {s['count']}")
    if s["change"] is not None:
        if s["change"] < 0:
            lines.append(f"Вчера было {s['previous_count']}, сегодня {s['count']} — не хватает {-s['change']}.")
        elif s["change"] > 0:
            lines.append(f"Вчера было {s['previous_count']}, сегодня на {s['change']} больше.")
        else:
            lines.append("Столько же, сколько вчера.")
    if s["missing_vs_expected"] and s["missing_vs_expected"] > 0:
        lines.append(f"До полного стада ({s['expected']}) не хватает {s['missing_vs_expected']}.")
    if s["stray_ids"]:
        ids = ", ".join(map(str, s["stray_ids"]))
        lines.append(f"Отбились от стада: {len(s['stray_ids'])} (№ {ids}). Проверь края пастбища.")
    if s["motionless_ids"]:
        ids = ", ".join(map(str, s["motionless_ids"]))
        lines.append(f"Почти не двигаются: {len(s['motionless_ids'])} (№ {ids}). Возможно, больны — осмотри.")
    if not has_problems(s):
        lines.append("Всё спокойно.")
    return "\n".join(lines)


def intro_ok(intro, s, lang):
    if re.search(r"\d", intro):
        print("[guardrail] intro contains numbers — facts must come from code only")
        return False
    if lang != "en" and re.search(r"[A-Za-z]{3,}", intro):
        print("[guardrail] intro mixes English words into the message")
        return False
    if has_problems(s) and any(p in intro.lower() for p in ALL_CLEAR):
        print("[guardrail] intro says everything is fine, but there are problems")
        return False
    if len(intro) > 220:
        print("[guardrail] intro too long")
        return False
    return True


def llm_intro(facts, lang):
    base, model = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_MODEL")
    if not base or not model:
        return None
    language = LANGUAGES.get(lang, "Russian")
    system = (
        "You are Malchy, a caring assistant for a shepherd in Kyrgyzstan. "
        f"Write ONE short, warm opening sentence in {language} for the herd report below. "
        "Mention the most urgent issue in general words. Do not use any numbers. "
        "Do not give advice and do not repeat details — the report itself follows your sentence. "
        f"Write only in {language}; never use English words. Output only the sentence."
    )
    body = json.dumps({
        "model": model,
        "temperature": 0.3,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": facts}],
    }).encode()
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'none')}"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            text = json.load(response)["choices"][0]["message"]["content"].strip()
            return text.strip('"«» \n')
    except Exception as error:
        print(f"[warning] LLM unavailable ({error}); sending facts only")
        return None


def send_telegram(text):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=30):
        return True


def main():
    parser = argparse.ArgumentParser(description="Daily herd report for the shepherd")
    parser.add_argument("--report", default="herd_report.json")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--count", type=int)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--lang", default="ru", choices=LANGUAGES)
    parser.add_argument("--lat", type=float, default=os.environ.get("PASTURE_LAT"))
    parser.add_argument("--lon", type=float, default=os.environ.get("PASTURE_LON"))
    parser.add_argument("--place", default=os.environ.get("PASTURE_NAME", "пастбище"))
    parser.add_argument("--threats-from", nargs="*", default=[])
    args = parser.parse_args()

    if args.count is not None:
        report = {"unique_animals": args.count, "stray_ids": [], "motionless_ids": []}
    else:
        report = json.loads(Path(args.report).read_text())

    weather_line = None
    if args.lat is not None and args.lon is not None and args.count is None:
        try:
            weather_line = weather.warning(float(args.lat), float(args.lon), args.place)
        except Exception as error:
            print(f"[warning] weather unavailable ({error})")

    entry = {
        "date": args.date,
        "count": report["unique_animals"],
        "stray_ids": report.get("stray_ids", []),
        "motionless_ids": report.get("motionless_ids", []),
    }
    history = [e for e in load_history() if e["date"] != args.date] + [entry]
    save_history(history)

    threats = dict(report.get("threats", {}))
    for path in args.threats_from:
        threats.update(json.loads(Path(path).read_text()).get("threats", {}))

    summary = build_summary(entry, history, args.expected, weather_line, threats)
    facts = facts_message(summary)
    intro = llm_intro(facts, args.lang)
    message = f"{intro}\n\n{facts}" if intro and intro_ok(intro, summary, args.lang) else facts
    print(message)
    if send_telegram(message):
        print("[sent to Telegram]")


if __name__ == "__main__":
    main()
