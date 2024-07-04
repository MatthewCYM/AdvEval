import google.generativeai as palm
import time


def generate_text(*args, **kwargs):
    for _ in range(20):
        try:
            rtn = palm.generate_text(*args, **kwargs)
            return rtn
        except Exception as e:
            print(e)
            time.sleep(30)
    exit(-1)
