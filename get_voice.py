"""Download the latest voice message sent to the Malchy bot as voice.wav."""
import json
import os
import subprocess
import urllib.request

token = os.environ["TELEGRAM_BOT_TOKEN"]
api = f"https://api.telegram.org/bot{token}"
updates = json.load(urllib.request.urlopen(f"{api}/getUpdates", timeout=30))["result"]
voices = [u["message"]["voice"] for u in updates if "voice" in u.get("message", {})]
if not voices:
    raise SystemExit("No voice messages found: send one to the bot first")
path = json.load(urllib.request.urlopen(f"{api}/getFile?file_id={voices[-1]['file_id']}", timeout=30))
urllib.request.urlretrieve(f"https://api.telegram.org/file/bot{token}/{path['result']['file_path']}", "voice.oga")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "voice.oga", "-ar", "16000", "-ac", "1", "voice.wav"],
               check=True)
print("Saved voice.wav")
