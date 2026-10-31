"""Malchy — daily herd report. Facts are computed in code; the model only
rewrites them, and a guardrail rejects changed numbers or 'more' vs 'fewer'."""
import argparse
import datetime as dt
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

HISTORY = Path("herd_history.json")
LANGUAGES = {"ru": "Russian", "ky": "Kyrgyz", "en": "English"}
MORE_WORDS = ("больше", "более", "прибав", "more")


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


def allowed_numbers(s):
    nums = {s["count"], len(s["stray_ids"]), len(s["motionless_ids"]), *s["stray_ids"], *s["motionless_ids"]}
    for key in ("previous_count", "expected", "missing_vs_expected"):
        if s[key] is not None:
            nums.add(s[key])
    if s["change"] is not None:
        nums.add(abs(s["change"]))
    nums.update(int(p) for p in s["date"].split("-"))
    return nums


def passes_guardrail(text, s):
    found = {int(n) for n in re.findall(r"\d+", text)}
    extra = found - allowed_numbers(s)
    if extra:
        print(f"[guardrail] model used numbers not in the data: {sorted(extra)}")
        return False
    lowered = text.lower()
    if s["change"] is not None and s["change"] < 0 and any(w in lowered for w in MORE_WORDS):
        print("[guardrail] model said 'more' but the herd got smaller")
        return False
    if s["change"] is not None and s["change"] < 0 and -s["change"] not in found:
        print("[guardrail] model dropped the number of missing animals")
        return False
    return True


def llm_message(facts, lang):
    base, model = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_MODEL")
    if not base or not model:
        return None
    system = (
        "You are Malchy, a helpful assistant for a shepherd in Kyrgyzstan. "
        f"Rewrite the herd report below as a short, warm message (at most 5 lines) in {LANGUAGES.get(lang, 'Russian')}. "
        "Keep every number exactly as written and keep its meaning: if animals are missing, say they are missing. "
        "Do not add any new numbers or facts. Put the most urgent problem first."
    )
    body = json.dumps({
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": facts},
        ],
    }).encode()
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'none')}"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
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
    facts = template_message(summary)
    message = llm_message(facts, args.lang)
    if message is None or not passes_guardrail(message, summary):
        message = facts
    print(message)
    if send_telegram(message):
        print("[sent to Telegram]")


if __name__ == "__main__":
    main()
