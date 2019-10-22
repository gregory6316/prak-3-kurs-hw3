import warnings
import librosa
import numpy
import time
import os
from queue import Queue
from threading import Thread, Lock

warnings.filterwarnings('ignore')
user_path = input('Введите путь к файлу с аудио: ')
path = os.getcwd()

def result():
    if not os.path.exists(path + '/res'):
        os.mkdir(path + '/res')
    lst = []
    for i in os.walk(user_path):
        lst.append(i)
    cur_path = path + '/res'
    for address, _, _ in lst:
        if not os.path.exists(cur_path + address[len(user_path):]):
            os.mkdir(cur_path + address[len(user_path):])

def file_result(q, lock, n):
    while True:
        lock.acquire()
        try:
            f = q.get()
            if f is None:
                break
            y, sr = librosa.load(f)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            s = path + '/res' + f[len(user_path):-3] + 'npy'
        finally:
            lock.release()
        numpy.save(s, mfcc)


start_time = time.time()
result()
files = librosa.util.find_files(user_path)
files = numpy.asarray(files)
q = Queue(5)
lock = Lock()
th1 = Thread(target=file_result, args=(q,lock,1))
th2 = Thread(target=file_result, args=(q,lock,2))
th1.start()
th2.start()
for f in files:
    q.put(f)
q.put(None)
q.put(None)
th1.join()
th2.join()
print('Общее время(в секундах): ',format((time.time() - start_time)))
