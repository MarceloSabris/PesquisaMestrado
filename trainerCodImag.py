
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import matplotlib.pyplot as plt
from util import log
from pprint import pprint

from input_ops import create_input_ops

import os
import time
import tensorflow as tf
import tf_slim as slim
import numpy
import json

class Trainer(object):

    @staticmethod
    
    def get_model_class(model_name):
        from model_rn_image_imageRepre_camada2_9Obj import Model
        return Model

    def __init__(self,
                 config,
                 data,
                 dataset,
                 dataset_test):
        self.config = config
        hyper_parameter_str = config.dataset_path+'_lr_'+str(config.learning_rate)
        self.train_dir = './train_dir/%s-%s-%s-%s' % (
            config.model,
            config.prefix,
            hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size
        self.data = data
        
        _, self.batch_train,imgs = create_input_ops(dataset, self.batch_size,shuffle=False,
                                               is_training=True)

        _, self.batch_test,ims = create_input_ops(dataset_test, self.batch_size,shuffle=False,                                          
                                              is_training=False)
        self.DataSetPath = os.path.join('./datasets', config.dataset_path)
        # --- create model ---
        Model = self.get_model_class(config.model)
        log.infov("Using Model class : %s", Model)
        self.model = Model(config)
        #tf.keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True)
        # --- optimizer ---
        self.global_step = tf.compat.v1.train.get_or_create_global_step(graph=None)
        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True,
                name='decaying_learning_rate'
            )
        self.QtdTest = len(dataset_test) 
        self.check_op = tf.no_op()

        self.optimizer = slim.optimize_loss(
            loss=self.model.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer='Adam',
            clip_gradients=20.0,
            name='optimizer_loss'
        )
       
        self.summary_op = tf.compat.v1.summary.merge_all()
        #import tfplot
        self.plot_summary_op =  tf.compat.v1.summary.merge_all()
      
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1000)
        
        self.summary_writer = tf.compat.v1.summary.FileWriter(self.train_dir)
    
        self.train_amount_network = config.train_amount_network
        self.pathDataSets = os.path.join('./datasets', config.dataset_path)
        self.checkpoint_secs = 600  # 10 min
        self.acuracy = [] 
        self.step = []
        self.supervisor = tf.compat.v1.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=self.checkpoint_secs,
            global_step=self.global_step,
        )

        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            # intra_op_parallelism_threads=1,
            # inter_op_parallelism_threads=1,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)
        self.session.graph._unsafe_unfinalize()

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided checkpoint path")
    
    def plot_acuracy(self):
            plt.plot(self.step,self.acuracy)
            
            #plt.ylim([0, 10])
            plt.xlabel('step')
            plt.ylabel('Cauracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(   self.train_dir + '/acuracy_datalenght.png')
         #   plt.show()

    def train(self):
        log.infov("Training Starts!")
        pprint(self.batch_train)
        #alterei aqui 
        max_steps =100000  
        output_save_step = 1000
        teste_log_Save = 2000

        for s in xrange(max_steps):
            if ( self.train_amount_network!=0 and s != 0 and s % self.train_amount_network==0):
                dataset_train = self.data.create_default_splits(os.path.join( self.pathDataSets,'Level'+str(s)),True,True)
                _, self.batch_train,imgs = create_input_ops(dataset_train, self.batch_size,
                                               is_training=True)
                
            step, accuracy, summary, loss, step_time = \
                      self.run_single_step(self.batch_train, step=s, is_train=True)
            
                      
            if s % teste_log_Save == 0:

                 # periodic inference
                accuracy_test, step_time_test = \
                    self.run_test(self.batch_test, is_train=False,QtdTest = self.QtdTest)
                self.log_step_message(step, accuracy, accuracy_test, loss, step_time,step_time_test)
                temp=[]
                temp.append('step:'+ str(step))
                temp.append('teste - time' + str(step_time_test))
                temp.append('train- time' + str(step_time))
                temp.append('accuracy:'+ str(accuracy))
                temp.append('accuracy_test:'+ str(accuracy_test))
                temp.append('loss:'+ str(loss))
                self.GravarArquivo(temp,'Logs',)




            self.summary_writer.add_summary(summary, global_step=step)         

                  

            if (s % output_save_step == 0 or s == max_steps):
                log.infov("Saved checkpoint at %d", s) 
                save_path = self.saver.save(self.session,
                                            os.path.join(self.train_dir, 'model'),
                                            global_step=step)
        self.plot_acuracy()

    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.model.accuracy, self.summary_op,
                 self.model.loss, self.check_op, self.optimizer]

        try:
            if step is not None and (step % 100 == 0):
                fetch += [self.plot_summary_op]
        except:
            pass

        fetch_values = self.session.run(
            fetch, feed_dict=self.model.get_feed_dict(batch_chunk, step=step)
        )
        [step, accuracy, summary, loss] = fetch_values[:4]

        try:
            if self.plot_summary_op in fetch:
                summary += fetch_values[-1]
        except:
            pass

        _end_time = time.time()

        return step, accuracy, summary, loss,  (_end_time - _start_time)

    def GravarArquivo ( self,data_dict,fname):
      
       print("gravar arquivo: " + fname + " qtd: " +  str(len(data_dict)))
       os.makedirs(self.train_dir, exist_ok=True)
       fname = self.train_dir + "/" + fname +".json"
       # Create file
       with open(fname, 'a') as outfile:
         json.dump(data_dict, outfile, ensure_ascii=False, indent=2) 
         outfile.write('\n')
         outfile.close() 

    def run_test(self, batch, is_train=False, repeat_times=8,QtdTest = 1):
        _start_time = time.time()
        qtdInterator = int(QtdTest/batch['a'].shape[0])
        accuracy_test = 0
        for i in range(qtdInterator):
           batch_chunk = self.session.run(batch)
           accuracy_teste_step = self.session.run(
            self.model.accuracy, feed_dict=self.model.get_feed_dict(batch_chunk, is_training=False))
           accuracy_test =   accuracy_teste_step +accuracy_test
        
        _end_time = time.time() 
        #tf.compat.v1.summary.scalar("loss/accuracy_test", accuracy_test)
        return (accuracy_test/qtdInterator),(_end_time-_start_time)

    def log_step_message(self, step, accuracy, accuracy_test, loss, step_time, step_time_training,is_train=True):
        self.acuracy.append(accuracy)
        self.step.append(step)
        if step_time == 0:
            step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Loss: {loss:.5f} " +
                "Accuracy : {accuracy:.2f} "
                "Accuracy test: {accuracy_test:.2f} " + 
                 "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) " +
                 "({sec_per_batch_test:.3f} sec/batch teste)"
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         loss=loss,
                         accuracy=accuracy*100,
                         accuracy_test=accuracy_test*100,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time , 
                         sec_per_batch_test = step_time_training
                         )
               )


def check_data_path(path):
    if os.path.isfile(os.path.join(path, 'data.hy')) \
           and os.path.isfile(os.path.join(path, 'id.txt')):
        return True
    else:
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--model', type=str, default='rn', choices=['rn', 'baseline'])
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default='Sort-of-CLEVR_teste_decode-image3')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    parser.add_argument('--train_amount_network',  type=int, default=0)
    
    config = parser.parse_args()

    path = os.path.join('./datasets', config.dataset_path)

    if check_data_path(path):
        import sort_of_clevr as dataset
    else:
        raise ValueError(path)

    config.data_info = dataset.get_data_info()
    config.conv_info = dataset.get_conv_info()
    dataset_train = dataset.create_default_splits(path,is_full =True)
    dataset_test= dataset.create_default_splits(path,is_full =True,id_filename="id_test.txt")
    trainer = Trainer(config,dataset,
                      dataset_train, dataset_test)

    log.warning("dataset: %s, learning_rate: %f",
                config.dataset_path, config.learning_rate)
    trainer.train()
   

if __name__ == '__main__':
    main()
