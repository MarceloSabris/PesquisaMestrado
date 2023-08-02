import numpy as np
import tensorflow as tf

from util import log
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
print(tf.__version__)



def check_data_id(dataset, data_id):
    if not data_id:
        return

    wrong = []
    for id in data_id:
        if id in dataset.data:
            pass
        else:
            wrong.append(id)

    if len(wrong) > 0:
        raise RuntimeError("There are %d invalid ids, including %s" % (
            len(wrong), wrong[:5]
        ))


def create_input_ops(dataset,
                     batch_size,
                     num_threads=1,           # for creating batches
                     is_training=False,
                     data_id=None,
                     scope='inputs',
                     shuffle=True,
                     is_loadImage=False
                     ):
    '''
    Return a batched tensor for the inputs from the dataset.
    '''
    input_ops = {}
    imgs = [] 

    if data_id is None:
        data_id = dataset.ids
        log.info("input_ops [%s]: Using %d IDs from dataset", scope, len(data_id))
    else:
        log.info("input_ops [%s]: Using specified %d IDs", scope, len(data_id))

    # single operations
    with tf.device("/gpu:0"), tf.name_scope(scope):
        tf.compat.v1.disable_eager_execution()
        input_ops['id'] = tf.compat.v1.train.string_input_producer(
           tf.convert_to_tensor(data_id), shuffle=False, capacity=128
        ).dequeue(name='input_ids_dequeue')

        img, q, a,imgDecod,codImag,codImagOri = dataset.get_data(data_id[0])
  #for id in data_id:
          #   imgs.append(load_fn(id)[1])
        #imgs = np.array(imgs)
        def load_fn(id):
            # image [n, n], q: [m], a: [l]
            img, q, a, imgDecod , codImag,codImagOri  = dataset.get_data(id)
            print(id)
            #print(img.astype(np.float32))
            print(q.astype(np.float32))
            print(a.astype(np.float32))
            #print(imgDecod.astype(np.float32))
            #print( codImag.astype(np.float32))
            print(codImagOri.astype(np.float32))
            return (id, img, q.astype(np.float32), a.astype(np.float32),imgDecod,   codImag  ,codImagOri.astype(np.float32))

        input_ops['id'], input_ops['img'], input_ops['q'], input_ops['a'],input_ops['imgDecod'],input_ops['codImag'],input_ops['codImagOri']   = tf.compat.v1.py_func(
            load_fn, inp=[input_ops['id']],
            Tout=[tf.string, tf.float32, tf.float32, tf.float32, tf.float32,tf.float32,tf.float32],
            name='func'
        )

        input_ops['id'].set_shape([])
        input_ops['img'].set_shape([])
        input_ops['q'].set_shape(list(q.shape))
        input_ops['a'].set_shape(list(a.shape)) 
        input_ops['imgDecod'].set_shape([])
        input_ops['codImag'].set_shape(list(codImag.shape)) 
        input_ops['codImagOri'].set_shape(list(codImagOri.shape))
        
    # batchify
    capacity = 2 * batch_size * num_threads
    min_capacity = min(int(capacity * 0.90), 1024)

    if shuffle:
        batch_ops =  tf.compat.v1.train.shuffle_batch(
            input_ops,
            batch_size=batch_size,

            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_capacity,
        )
    else:
        batch_ops =  tf.compat.v1.train.batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
        )

         

    return input_ops, batch_ops,imgs
