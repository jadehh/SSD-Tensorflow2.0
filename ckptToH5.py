#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/16 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/16  下午3:32 modify by jade

import tensorflow as tf
import h5py
import os
def save_weights():
    reader = tf.train.NewCheckpointReader("/home/jade/Checkpoints/VGG/VGG.ckpt")
    h5Filename = "/home/jade/Checkpoints/VGG/VGG.h5"
    os.remove(h5Filename)
    f = h5py.File(h5Filename, 'w')
    t_g = None
    key_list = []
    for key in sorted(reader.get_variable_to_shape_map()):
        if key.endswith('weights') or key.endswith('biases'):
            keySplits = key.split(r'/')
            keyDict = ""
            for i in range(len(keySplits)):
                if i == len(keySplits) - 1:
                    a = keySplits[i]
                else:
                    a = keySplits[i] + '/'
                keyDict += a
            # keyDict = keySplits[0] + '/' + keySplits[1] + '/' + keySplits[2] + '/' + keySplits[3]
            #print(keyDict)
            f.attrs[keyDict] = "weights_names"

            key_list.append(keyDict.encode("utf-8"))
    f.attrs["layer_names"] = key_list
    for key in key_list:
        print("save",key.decode('utf8'))
        group = f[keyDict]
        print(group.attrs)

def load_attributes_from_hdf5_group(group, name):
  """Loads attributes of the specified name from the HDF5 group.

  This method deals with an inherent problem
  of HDF5 file which is not able to store
  data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

  Arguments:
      group: A pointer to a HDF5 group.
      name: A name of the attributes to load.

  Returns:
      data: Attributes data.
  """
  if name in group.attrs:
    data = [n.decode('utf8') for n in group.attrs[name]]
  else:
    data = []
    chunk_id = 0
    while '%s%d' % (name, chunk_id) in group.attrs:
      data.extend(
          [n.decode('utf8') for n in group.attrs['%s%d' % (name, chunk_id)]])
      chunk_id += 1
  return data

def readH5(h5_path):
    f = h5py.File(h5_path, 'r')
    name = "layer_names"
    if name in f.attrs:
        layer_names = []
        for n in f.attrs[name]:
            layer_names.append(n.decode('utf8'))
    for layer in layer_names:
        print(layer)
        g = f[layer]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        print(weight_names)

if __name__ == '__main__':
    save_weights()
    readH5("/home/jade/Checkpoints/VGG/VGG.h5")