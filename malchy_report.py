"""Malchy — daily herd report for the shepherd: history by day, comparison
with the previous day, a short message (Muse model or template), Telegram."""
import argparse
import datetime as dt
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

HISTORY = Path("herd_history.json")
LANGUAGES = {"ru": "Russian", "ky": "Kyrgyz", "en": "English"}


def load_history():
    if HISTORY.exists():
        return json.loads(HISTORY.read_text())
    return []


def save_history(history):
    HISTORY.write_text(json.dumps(sorted(history, key=lambda e: e["date"]), indent=2, ensure_ascii=False))


def build_summary(entry, history, expected):
    earlier = [e for e in history if e["date"] < entry["date"]]
    prev = max(earlier, key=lambda e: e["date"]) if earlier else None
    return {
        "date": entry["date"],
        "animal": entry["animal"],
        "count": entry["count"],
        "previous_date": prev["date"] if prev else None,
        "previous_count": prev["count"] if prev else None,
        "change": entry["count"] - prev["count"] if prev else None,
        "expected": expected,
        "missing_vs_expected": expected - entry["count"] if expected else None,
        "stray_ids": entry["stray_ids"],
        "motionless_ids": entry["motionless_ids"],
    }


def template_message(s):
    lines = [f"Малчы — отчёт за {s['date']}", f"Посчитано: {s['count']}"]
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
    if not s["stray_ids"] and not s["motionless_ids"] and (s["change"] is None or s["change"] >= 0):
        lines.append("Всё спокойно.")
    return "\n".join(lines)


def llm_message(s, lang):
    base, model = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_MODEL")
    if not base or not model:
        return None
    system = (
        "You are Malchy, a helpful assistant for a shepherd in Kyrgyzstan. "
        f"Write a short daily herd report (at most 5 lines) in {LANGUAGES.get(lang, 'Russian')}. "
        "Start with the most urgent problem: missing animals first, then strays, then motionless "
        "(possibly sick) animals. Use only the numbers in the data; never invent facts. "
        "If everything is fine, say so in one line."
    )
    body = json.dumps({
        "model": model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(s, ensure_ascii=False)},
        ],
    }).encode()
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'none')}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)["choices"][0]["message"]["content"].strip()
    except Exception as error:
        print(f"[warning] LLM unavailable ({error}); using the template")
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
    args = parser.parse_args()

    if args.count is not None:
        report = {"prompt": "sheep", "unique_animals": args.count, "stray_ids": [], "motionless_ids": []}
    else:
        report = json.loads(Path(args.report).read_text())

    entry = {
        "date": args.date,
        "animal": report.get("prompt", "sheep"),
        "count": report["unique_animals"],
        "stray_ids": report.get("stray_ids", []),
        "motionless_ids": report.get("motionless_ids", []),
    }
    history = [e for e in load_history() if e["date"] != args.date] + [entry]
    save_history(history)

    summary = build_summary(entry, history, args.expected)
    message = llm_message(summary, args.lang) or template_message(summary)
    print(message)
    if send_telegram(message):
        print("[sent to Telegram]")


if __name__ == "__main__":
    main()
