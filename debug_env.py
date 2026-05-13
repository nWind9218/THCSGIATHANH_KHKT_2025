import os
from dotenv import load_dotenv

load_dotenv()
user = os.getenv("SMTP_USERNAME")
pw = os.getenv("SMTP_PASSWORD")
host = os.getenv("SMTP_HOST")

print(f"DEBUG INFO:")
print(f"User: '{user}'")
print(f"Password length: {len(pw) if pw else 0}")
print(f"Host: '{host}'")
print(f"Password ends with space? {pw.endswith(' ') if pw else 'N/A'}")
