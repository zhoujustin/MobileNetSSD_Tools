import numpy as np
import sys,os
caffe_root = '/home/kelale/Downloads/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

proto = 'MobileNetSSD_test.prototxt'
model = 'lpr.caffemodel'

proto = 'MobileNetSSD_deploy.prototxt'
model = 'MobileNetSSD_deploy.caffemodel'

#proto = 'MobileNetSSD_out.prototxt'
#model = 'MobileNetSSD_out.caffemodel'

def caffedump(net):
    for key in iter(net.params.keys()):
        if type(net.params[key]) is caffe._caffe.BlobVec:
            conv = net.params[key]
            for i, w in enumerate(conv):
                print (key, i, w.data)

if __name__ == '__main__':
    net = caffe.Net(proto, model, caffe.TEST)
    caffedump(net)
