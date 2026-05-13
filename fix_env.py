import os

def fix_env():
    if not os.path.exists('.env'):
        print("No .env file found")
        return
        
    with open('.env', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    found = False
    for line in lines:
        if 'SMTP_PASSWORD' in line:
            parts = line.split('=', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().replace(' ', '').replace('"', '').replace("'", "")
                new_lines.append(f"{key}={value}\n")
                print(f"Fixed SMTP_PASSWORD. New value length: {len(value)}")
                found = True
        else:
            new_lines.append(line)
            
    if not found:
        print("SMTP_PASSWORD not found in file")
        
    with open('.env', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    fix_env()
