import warnings
import librosa
import numpy
import time
import os
from multiprocessing import Pool

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
def result_file(f):
    y, sr = librosa.load(f)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    s = path + '/res' + f[len(user_path):-3] + 'npy'
    numpy.save(s, mfcc)

start_time = time.time()
result()
files = librosa.util.find_files(user_path)
files = numpy.asarray(files)
pool = Pool(8)
pool.map(result_file, files)
print('Общее время(в секундах): ',format((time.time() - start_time)))

