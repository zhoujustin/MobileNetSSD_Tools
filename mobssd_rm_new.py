import sys,os
caffe_root = '/home/kelale/Downloads/caffe/'
sys.path.insert(0, caffe_root + 'python')

import numpy as np
import caffe

deploy_proto = 'MobileNetSSD_deploy.prototxt'
deploy_model = 'MobileNetSSD_deploy.caffemodel'

out_proto = 'MobileNetSSD_out.prototxt'
out_model = 'MobileNetSSD_out.caffemodel'

def parse_rm_proto(filein, fileout):
    with open(filein, 'r') as fi, open(fileout, 'w') as fo:
        for strline in fi.readlines():
            fo.write(strline.replace('_new', ''))
        fi.close()
        fo.close()

def rm_model(deploy_net, out_net):
    for key in iter(deploy_net.params.keys()):
        if type(deploy_net.params[key]) is caffe._caffe.BlobVec:
            conv = deploy_net.params[key]
            for i, w in enumerate(conv):
                out_net.params[key.replace('_new','')][i].data[...] = w.data
                print (key, i, w.data)
            

if __name__ == '__main__':
    parse_rm_proto(deploy_proto, out_proto)

    deploy_net = caffe.Net(deploy_proto, deploy_model, caffe.TEST)
    out_net = caffe.Net(out_proto, caffe.TEST)

    rm_model(deploy_net, out_net)
    out_net.save(out_model)
