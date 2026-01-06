# Fix all escaped quotes in app.py
import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix all backslash-escaped quotes
content = content.replace('\\\"', '"')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed all escaped quotes in app.py")
