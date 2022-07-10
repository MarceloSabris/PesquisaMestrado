from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy.lib.function_base import append

from six.moves import xrange

from util import log

from input_ops import create_input_ops, check_data_id
from vqa_util import NUM_COLOR
from vqa_util import visualize_iqa,question2str,answer2str


import os
import time
import numpy as np
import tensorflow as tf
import tf_slim  as slim
import matplotlib.pyplot as plt
import json


class EvalManager(object):
  
    ArrayQuestoesCertas =[]
    ArrarQuestoesErradas=[]
    
    def __init__(self,dataset,checkpoint_path):
        # collection of batches (not flattened)
        self._ids = []
        self._predictions = []
        self._groundtruths = []
        self._questions = [] 
        self._images = [] 
        self.percentageData = 'model2000'
        self.ArrayQuestoesCertas =[]
        self.ArrarQuestoesErradas=[]
        self.dataset = dataset
        self._answers =[]
        self.checkpoint_path = checkpoint_path
   

    def add_batch(self, full, prediction, groundtruth):
    
        # for now, store them all (as a list of minibatch chunks)
        self._ids.append(full['id'])
        self._questions.append(full['q'])
        self._answers.append(full['a'])
        self._predictions.append(prediction)
        self._groundtruths.append(groundtruth)
    

    def report(self):

        #img, q, a = self.dataset.get_data(self._ids[0])
        #visualize_iqa( img, q, a)


        # report L2 loss
        log.info("Computing scores...")
        correct_prediction_nr = 0
        count_nr = 0
        correct_prediction_r = 0
        correctQuest =0
        errorQuest = 0
        count_r = 0
        
        
        for id,q,a, pred, gt in zip(self._ids,self._questions, self._answers ,self._predictions, self._groundtruths):
            for i in range(pred.shape[0]):
                # relational
       
                quest = question2str(q[i])
                if int(id[i]) == 39950 : 
                    print('oi')
                answer = answer2str(a[i])
                anserPred = np.zeros((len(a[i]))) 
                anserPred[np.argmax(pred[i,:])] = 1 
                anserPred1 = answer2str(anserPred)
                q_num = np.argmax(q[i][6:])
                if  q_num > 2:
                    count_r += 1
                    
                    if np.argmax(pred[i, :]) == np.argmax(gt[i, :]):
                        correct_prediction_r += 1
                        correctQuest += 1 
                        #self.ArrayQuestoesCertas.append(str(int(id[i])))
                        self.ArrayQuestoesCertas.append("qestao id numero : " +str(int(id[i])))
                        self.ArrayQuestoesCertas.append(quest)
                        self.ArrayQuestoesCertas.append(answer)
                        self.ArrayQuestoesCertas.append(anserPred1)
                        self.ArrayQuestoesCertas.append("qestao do tipo : " +str(q_num)) 
                        self.ArrayQuestoesCertas.append("tipo : Relacional")
                    else:
                        errorQuest += 1 
                        #self.ArrarQuestoesErradas.append(str(int(id[i])))
                        self.ArrarQuestoesErradas.append ("qestao id numero : " + str(int(id[i])))
                        self.ArrarQuestoesErradas.append(quest)
                        self.ArrarQuestoesErradas.append("Reposta errada:" + anserPred1)  
                        self.ArrarQuestoesErradas.append("Resposta certa:" + answer) 
                        self.ArrarQuestoesErradas.append("qestao do tipo : " +str(q_num))
                        self.ArrarQuestoesErradas.append("tipo : Relacional")

                # non-relational
                else:
                    count_nr += 1
                    if np.argmax(pred[i, :]) == np.argmax(gt[i, :]):
                        correctQuest += 1 
                        correct_prediction_nr += 1
                        #self.ArrayQuestoesCertas.append(str(int(id[i])))
                        self.ArrayQuestoesCertas.append("qestao id numero : " +str(int(id[i])))
                        self.ArrayQuestoesCertas.append(quest)
                        self.ArrayQuestoesCertas.append(answer)
                        self.ArrayQuestoesCertas.append(anserPred1)
                        self.ArrayQuestoesCertas.append("qestao do tipo : " +str(q_num)) 
                        self.ArrayQuestoesCertas.append("tipo : Nao-Relacional")
                    else:
                        errorQuest +=1
                        #self.ArrarQuestoesErradas.append ( str(int(id[i])))
                        self.ArrarQuestoesErradas.append ("qestao id numero : " + str(int(id[i])))
                        self.ArrarQuestoesErradas.append(quest)
                        self.ArrarQuestoesErradas.append("Reposta errada:" + anserPred1)  
                        self.ArrarQuestoesErradas.append("Resposta certa:" + answer) 
                        self.ArrarQuestoesErradas.append("qestao do tipo : " +str(q_num)) 
                        self.ArrarQuestoesErradas.append("tipo : Nao-Relacional")

        avg_nr = float(correct_prediction_nr)/count_nr
        log.infov("Average accuracy of non-relational questions: {}%".format(avg_nr*100))
        avg_r = float(correct_prediction_r)/count_r
        log.infov("Average accuracy of relational questions: {}%".format(avg_r*100))
        avg = float(correct_prediction_r+correct_prediction_nr)/(count_r+count_nr)
        log.infov("Average accuracy: {}%".format(avg*100))
        file_name = self.checkpoint_path.split("\\")[2]
        self.GravarArquivo(file_name,self.ArrarQuestoesErradas,"questaoErradaTrain",errorQuest )
        self.GravarArquivo(file_name,self.ArrayQuestoesCertas,"questaoCertaTrain",correctQuest)
       
       

    def GravarArquivo (self,percentageData, data_dict,fname,qtd):
      fname = fname +"_" +str(qtd) + '.json'
      print("gravar arquivo: " + fname + " qtd: " +  str(len(data_dict)))
      os.makedirs('Resultado//curriculo2//Percentage_'+ str(percentageData) , exist_ok=True)
      fname = 'Resultado//curriculo2//Percentage_'+ str(percentageData) + "/" + fname
      # Create file
      with open(fname, 'w') as outfile:
        json.dump(data_dict, outfile, ensure_ascii=False, indent=4) 
        outfile.close()

class Evaler(object):

    @staticmethod
    def get_model_class(model_name):
        from model_rn_image_imageRepre_camada2_9Obj  import Model
        return Model

    def __init__(self,
                 config,
                 dataset):
        self.config = config
        self.train_dir = config.train_dir

        log.info("self.train_dir = %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        self.dataset = dataset

        check_data_id(dataset, config.data_id)
        _, self.batch ,img  = create_input_ops(dataset, self.batch_size,
                                         data_id=config.data_id,
                                         is_training=False,
                                         shuffle=False)

        # --- create model ---
        Model = self.get_model_class(config.model)
        log.infov("Using Model class : %s", Model)
        self.model = Model(config)

        self.global_step = tf.compat.v1.train.get_or_create_global_step(graph=None)
        self.step_op = tf.no_op(name='step_no_op')

        #tf.random.set_seed(1234)

        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = tf.compat.v1.Session(config=session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.compat.v1.train.Saver(max_to_keep=100)

        self.checkpoint_path = config.checkpoint_path
        if self.checkpoint_path is None and self.train_dir:
            self.checkpoint_path = tf.train.latest_checkpoint(self.train_dir)
        if self.checkpoint_path is None:
            log.warn("No checkpoint is given. Just random initialization :-)")
            self.session.run(tf.global_variables_initializer())
        else:
            log.info("Checkpoint path : %s", self.checkpoint_path)

    def eval_run(self):
        # load checkpoint
        if self.checkpoint_path:
            self.saver.restore(self.session, self.checkpoint_path)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")

        log.info("# of examples = %d", len(self.dataset))
        length_dataset = len(self.dataset)

        max_steps = int(length_dataset / self.batch_size) +1
        log.info("max_steps = %d", max_steps)

        coord = tf.train.Coordinator()
        threads =tf.compat.v1.train.start_queue_runners(self.session,
                                               coord=coord, start=True)

        evaler = EvalManager(self.dataset,self.checkpoint_path)
        try:
            for s in xrange(max_steps):
                step, loss, step_time, batch_chunk, prediction_pred, prediction_gt = \
                    self.run_single_step(self.batch)
                self.log_step_message(s, loss, step_time)
                evaler.add_batch(batch_chunk, prediction_pred, prediction_gt)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        try:
            coord.join(threads, stop_grace_period_secs=3)
        except RuntimeError as e:
            log.warn(str(e))

        evaler.report()
        log.infov("Evaluation complete.")

    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        [step, accuracy, all_preds, all_targets, _] = self.session.run(
            [self.global_step, self.model.accuracy, self.model.all_preds, self.model.a, self.step_op],
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )
    

        _end_time = time.time()

        return step, accuracy, (_end_time - _start_time), batch_chunk, all_preds, all_targets

    def log_step_message(self, step, accuracy, step_time, is_train=False):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "batch total-accuracy (test): {test_accuracy:.2f}% " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         test_accuracy=accuracy*100,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time,
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
    parser.add_argument('--model', type=str, default='conv', choices=['rn', 'baseline'])
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--dataset_path', type=str, default='Sort-of-CLEVR_default')
    parser.add_argument('--data_id', nargs='*', default=None)
    config = parser.parse_args()

    path = os.path.join('./datasets', config.dataset_path)

    if check_data_path(path):
        import sort_of_clevr as dataset
    else:
        raise ValueError(path)

    config.data_info = dataset.get_data_info()
    config.conv_info = dataset.get_conv_info()
    dataset_train = dataset.create_default_splits(path,is_full=True, id_filename="tipoCV2.txt")
    
   
    evaler = Evaler(config, dataset_train)


    log.warning("dataset: %s", config.dataset_path)

    evaler.eval_run()

if __name__ == '__main__':
    main()
