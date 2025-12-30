"""
Race Condition: là sự bất đồng bộ trong quyền truy cập
và thay đổi dữ liệu chia sẻ. Điều này thường xuyên xảy
ra khi bạn code multi-threading không đúng cách và dẫn
đến mất sự nhất quán và chính xác của các dữ liệu 
"""
import threading
counter = 0
def inc():
    global counter
    for _ in range(10):
        counter += 1    
        print(f"😘 Now counter is equal to: {counter}")

if __name__ == '__main__':
    threads = [threading.Thread(target=inc) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("✅ Done")

    """OUTPUT OF RACE CONDITION:
    😘 Now counter is equal to: 1
    😘 Now counter is equal to: 2
    😘 Now counter is equal to: 3
    😘 Now counter is equal to: 4
    😘 Now counter is equal to: 5
    😘 Now counter is equal to: 7
    😘 Now counter is equal to: 6
    😘 Now counter is equal to: 8
    😘 Now counter is equal to: 10
    😘 Now counter is equal to: 9
    😘 Now counter is equal to: 12
    😘 Now counter is equal to: 11
    😘 Now counter is equal to: 13
    😘 Now counter is equal to: 14
    😘 Now counter is equal to: 15
    😘 Now counter is equal to: 16
    😘 Now counter is equal to: 17
    😘 Now counter is equal to: 18
    😘 Now counter is equal to: 19
    😘 Now counter is equal to: 20
    ✅ Done
    """