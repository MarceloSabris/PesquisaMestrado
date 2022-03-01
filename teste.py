import tf_slim  as slim 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from util import log
import vqa_util as vqa

def check_data_path(path):
    if os.path.isfile(os.path.join(path, 'data.hy')) \
           and os.path.isfile(os.path.join(path, 'id.txt')):
        return True
    else:
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='Sort-of-CLEVR_default')
    config = parser.parse_args()

    path = os.path.join('./datasets', config.dataset_path)

    if check_data_path(path):
        import sort_of_clevr as dataset
    else:
        raise ValueError(path)

    
    data = dataset.return_dataset(path)
    img, q, a = data.get_data('1')
    img = vqa.visualize_iqa( img, q, a)
    
    img.savefig('questao.png')
  
    

if __name__ == '__main__':
    main()

