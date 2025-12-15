import traceback
import sys
import logging

logging.basicConfig(filename='app_errors.log', level=logging.ERROR)

def func_a():
    return func_b()
def func_b():
    return 10 / 0
try:
    func_a()
except ZeroDivisionError as e:
    logging.error("Xảy ra lỗi chia cho 0", exc_info= True)
    error_string = traceback.format_exc()
    print("\n-- Traceback dưới dạng String --")
    print(error_string)

    print("\n-- In trực tiếp bằng print_exc() --")
    traceback.print_exc(file=sys.stdout)

    with open('debug_output.txt', 'w') as f:
        traceback.print_exc(file=f)