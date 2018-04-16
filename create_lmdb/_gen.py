
'''
PATH
  --VOC2007
    --labelmap_voc.prototxt
    --Annotations
      --*.xml
    --ImageSets
      --Main
        --trainval.txt
        --test.txt
    --JPEGImages
      --*.jpg
'''

import os
from sklearn.model_selection import train_test_split

def enum_file(strpath, ext=''):
  l = []
  for fn in os.listdir(strpath):
    if fn.endswith(ext):
      l.append(fn)
  return l

if __name__ == '__main__':
  with open('./VOC2007/ImageSets/Main/trainval.txt', 'w') as fv, open('./VOC2007/ImageSets/Main/test.txt', 'w') as ft:
  #, open('./VOC2007/data/trainval.txt', 'w') as fv_l, open('./VOC2007/data/test.txt', 'w') as ft_l:
    f = enum_file('./VOC2007/JPEGImages', ext='.jpg')
    f_train, f_test = train_test_split(f, test_size=0.1)

    for f in f_train:
      fv.writelines(os.path.splitext(f)[0] + '\n')
      #fv_l.writelines(f + ' ' + '0' + '\n')

    for f in f_test:
      ft.writelines(os.path.splitext(f)[0] + '\n')
      #fv_l.writelines(f + ' ' + '0' + '\n')
