"""Malchy — mountain weather warnings for the herd (Open-Meteo, free, no key)."""
import argparse
import datetime as dt
import json
import urllib.parse
import urllib.request

API = "https://api.open-meteo.com/v1/forecast"


def fetch(lat, lon, hours=48, tz="Asia/Bishkek"):
    query = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "timezone": tz, "forecast_days": 3,
        "hourly": "temperature_2m,snowfall,wind_gusts_10m",
    })
    with urllib.request.urlopen(f"{API}?{query}", timeout=30) as response:
        data = json.load(response)
    hourly = data["hourly"]
    now = dt.datetime.now()
    rows = []
    for i, stamp in enumerate(hourly["time"]):
        t = dt.datetime.fromisoformat(stamp)
        if now - dt.timedelta(hours=1) <= t <= now + dt.timedelta(hours=hours):
            rows.append({
                "time": t,
                "temp": hourly["temperature_2m"][i],
                "snow": hourly["snowfall"][i] or 0.0,
                "gust": hourly["wind_gusts_10m"][i] or 0.0,
            })
    return data.get("elevation"), [r for r in rows if r["temp"] is not None]


def assess(rows, snow_cm=1.0, gust_kmh=50.0, frost_c=-5.0):
    risks = {}
    snow_rows = [r for r in rows if r["snow"] >= 0.2]
    total = sum(r["snow"] for r in rows)
    if snow_rows and total >= snow_cm:
        risks["snow"] = {"from": snow_rows[0]["time"], "total_cm": max(1, round(total))}
    windy = [r for r in rows if r["gust"] >= gust_kmh]
    if windy:
        risks["wind"] = {"from": windy[0]["time"], "max_kmh": round(max(r["gust"] for r in windy))}
    if rows:
        coldest = min(rows, key=lambda r: r["temp"])
        if coldest["temp"] <= frost_c:
            risks["frost"] = {"at": coldest["time"], "min_c": round(coldest["temp"])}
    return risks


def when(t):
    days = (t.date() - dt.date.today()).days
    day = {0: "сегодня", 1: "завтра", 2: "послезавтра"}.get(days, t.strftime("%d.%m"))
    return f"{day} около {t.hour}:00"


def describe(risks, name):
    parts = []
    if "snow" in risks:
        parts.append(f"снег {when(risks['snow']['from'])}, всего около {risks['snow']['total_cm']} см")
    if "wind" in risks:
        parts.append(f"порывы ветра до {risks['wind']['max_kmh']} км/ч с {when(risks['wind']['from'])}")
    if "frost" in risks:
        parts.append(f"мороз до {risks['frost']['min_c']}°C {when(risks['frost']['at'])}")
    if not parts:
        return None
    return f"⚠️ Погода ({name}): " + "; ".join(parts) + ". Подумай, не спустить ли стадо ниже."


def warning(lat, lon, name, **thresholds):
    _, rows = fetch(lat, lon)
    return describe(assess(rows, **thresholds), name)


def main():
    parser = argparse.ArgumentParser(description="Mountain weather warnings for the herd")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--name", default="пастбище")
    parser.add_argument("--snow", type=float, default=1.0)
    parser.add_argument("--gust", type=float, default=50.0)
    parser.add_argument("--frost", type=float, default=-5.0)
    args = parser.parse_args()

    elevation, rows = fetch(args.lat, args.lon)
    risks = assess(rows, args.snow, args.gust, args.frost)
    print(f"{args.name}: elevation {elevation} m, {len(rows)} forecast hours")
    if rows:
        print(f"min temp {min(r['temp'] for r in rows):.0f}°C, "
              f"max gust {max(r['gust'] for r in rows):.0f} km/h, "
              f"snow {sum(r['snow'] for r in rows):.1f} cm")
    print(describe(risks, args.name) or "Погода спокойная, угроз нет.")


if __name__ == "__main__":
    main()
