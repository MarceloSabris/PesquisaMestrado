from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
#import plaidml.keras
import os
#plaidml.keras.install_backend()

import numpy
from six.moves import xrange

import matplotlib.pyplot as plt
from util import log

from pprint import pprint

from input_ops import create_input_ops
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from vqa_util import NUM_COLOR
from vqa_util import visualize_iqa,question2str,answer2str

import time
import tensorflow as tf
import tf_slim as slim
import numpy
import json
import logging
from datetime import datetime
import random
import DataBase as Postgree

tf.get_logger().setLevel(logging.ERROR)
tf.debugging.set_log_device_placement(True) 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#tf.enable_eager_execution() 
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
        #self.train_dir = './train_dir/%s-%s-%s-%s' % (
        #    config.model,
        #    config.prefix,
        #    hyper_parameter_str,
        #    time.strftime("%Y%m%d-%H%M%S")
        #)
        self.train_dir = './train_dir/%s' % (config.train_dir)

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)
        
        # --- input ops ---
        self.batch_size = config.batch_size
        self.data = data
        
        _, self.batch_train,imgs = create_input_ops(dataset[0], self.batch_size,shuffle=False,
                                               is_training=True,is_loadImage= self.config.is_loadImage)

        _, self.batch_test,ims = create_input_ops(dataset_test, self.batch_size,shuffle=False,                                          
                                              is_training=False,is_loadImage= self.config.is_loadImage)
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
        self.dataset = dataset[0]
        self.datasets = dataset
        self.dataset_test = dataset_test
        self.check_op = tf.no_op()
        self.trainPosition = 0 
        self._ids = []
        self._predictions = []
        self._groundtruths = []
        self._questions = [] 
        self._answers=[]
        self._images = [] 
        self._predictionsTrain = []
        self._groundtruthsTrain = []
        self._questionsTrain = [] 
        self._answersTrain=[]
        self._imagesTrain = [] 
        
        self.ArrayQuestoesCertas = [] 
        self.ArrarQuestoesErradas =[]
        self.check_pathSaveTrain= self.config.check_pathSaveTrain
        self.ArrayTotalQuestoesCertas =[0,0,0,0,0]
        self.ArrayTotalQuestoesErradas =[0,0,0,0,0]
       

        self.testAcuracy = 0 
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
      
        self.saver = tf.compat.v1.train.Saver(max_to_keep=10000)
        
        self.summary_writer = tf.compat.v1.summary.FileWriter(self.train_dir)
    
        self.train_amount_network = config.train_amount_network
        self.pathDataSets = os.path.join('./datasets', config.dataset_path)
        self.checkpoint_secs = 60000  # 10 min
        self.acuracy = [] 
        self.step = []
        self.supervisor = tf.compat.v1.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=3000,
            save_model_secs=self.checkpoint_secs,
            global_step=self.global_step,
        )
        tf.test.is_gpu_available()

        #gpu_config = tf.GPUOptions()
        #gpu_config.visible_device_list = "1"

        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
             intra_op_parallelism_threads=8,
             inter_op_parallelism_threads=8,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True),
            device_count={'GPU': 2},
        )
        #self.tf_config.gpu_options.allow_growth=True
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)
        self.session.graph._unsafe_unfinalize()

        self.ckpt_path = config.checkpoint
      
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided checkpoint path")
      
    def add_batch (self, id,questions,ans, prediction, groundtruth):
        # for now, store them all (as a list of minibatch chunks)
        self._ids.append(id)
        self._questions.append(questions)
        self._answers.append(ans)
        self._predictions.append(prediction)
        self._groundtruths.append(groundtruth)

    
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
      
        #alterei aqui 
        max_steps_datasets =100000
        max_step_dataset = 50000
        tempogravarlog  =0                            
        #output_save_step = 4000
        teste_log_Save = 1500
        stepTimeTotalExecution = 0  
        _start_time_total = time.time()
        step_time_test_Total = 0
        totalTempoGravarArquivoLog = 0
        TotalTempoGravaRede =0 
        _tempoPorRodada = time.time()
        #if ( self.config.train_type!='full'):
            #      if (self.config.train_type!='full_after'):
            #          if ( self.config.train_amount_network ==  s):
            #            dataset_train = self.data.create_default_splits(os.path.join(self.config.pathDataSets,self.config.train_nameDataset),True,True)
            #            _, self.batch_train,imgs = create_input_ops(dataset_train, self.config.batch_size,
            #                                   is_training=True)
            #if ( self.config.train_amount_network ==  'multiply'): 
            #            nameposition=0
            #            for steps in self.config.train_amount_network.split(','):
            #                if s== int(steps):
            #                   dataset_train = self.data.create_default_splits(os.path.join( self.config.pathDataSets,self.config.train_nameDataset.split(',')[nameposition]),True,True)
            #                   _, self.batch_train,imgs = create_input_ops(dataset_train, self.config.batch_size,
            #                                   is_training=True)
        stepControl = 0   
        accuracy_test= 0.0
       
        while stepControl < max_steps_datasets  : 
          datasetcont = -1
          for cont in self.config.orderDataset.split(',') :
            accuracy = 0.00
            #accuracy_total = 0.001
            datasetcont = datasetcont+1
            self.dataset = self.datasets[int(cont)]
            dataset = self.datasets[int(cont)]
            _, batch_train,imgs = create_input_ops(self.datasets[int(cont)], self.config.batch_size,
                                               is_training=True,is_loadImage=self.config.is_loadImage)
            self.trainPosition =0
            s=1
            stepdataset = 0
            log.info("mudou o database")
            while stepdataset < int(self.config.StepChangeGroup.split(',')[int(cont)] )  :
              for s in xrange(dataset.maxGrups -1):
                if stepdataset > int(self.config.StepChangeGroup.split(',')[int(cont)] ) : 
                    break
                stepdataset =stepdataset+1
                #log.infov(s)
                
                step, accuracy, summary, loss, step_time = \
                    self.run_single_step(batch_train, step=stepControl,teste_log_Save= teste_log_Save, is_train=True)
                stepTimeTotalExecution = step_time + stepTimeTotalExecution
                #accuracy_total = accuracy +accuracy_total
                self.trainPosition = self.trainPosition + 1      
                stepControl = stepControl+1    
                if stepControl % teste_log_Save == 0 :

                 # periodic inference
                    accuracy_test, step_time_test,questaotipo0,questaotipo1,questaotipo2,questaotipo3,questaotipo4 = \
                        self.run_test('Teste' ,step,self.dataset_test)
                    step_time_test_Total = step_time_test + step_time_test_Total
                    self.log_step_message(step, accuracy, accuracy_test, loss, step_time,step_time_test)
                    tempogravarlog = time.time()
                    temp=[]
                    temp.append('step:'+ str(step))
                    temp.append('teste - time:' + str(step_time_test))
                    temp.append('train- time:' + str(step_time))
                    temp.append('accuracy:'+ str(accuracy))
                    temp.append('accuracy_test:'+ str(accuracy_test))
                    temp.append('loss:'+ str(loss))
                    temp.append('dataset:'+str(datasetcont))
                    temp.append('questaotipo0:'+str(questaotipo0))
                    temp.append('questaotipo1:'+str(questaotipo1))
                    temp.append('questaotipo2:'+str(questaotipo2))
                    temp.append('questaotipo3:'+str(questaotipo3))
                    temp.append('questaotipo4:'+str(questaotipo4))            
                    self.GravarArquivo1(temp,'Logs')
                    Postgree.add_new_row(step,self.config.train_dir ,accuracy,accuracy_test,questaotipo0,questaotipo1, questaotipo2,questaotipo3,questaotipo4,self.config.GrupDataset.split('|')[int(cont)] )
                    totalTempoGravarArquivoLog = totalTempoGravarArquivoLog + (time.time() -  tempogravarlog)


                    self.summary_writer.add_summary(summary, global_step=stepControl)         

                #log.infov( 'Tempo total rodada' + str((time.time() - _tempoPorRodada)))
                    now = datetime.now()
                #self.run_test('Treino' ,step,self.dataset)
                    current_time = now.strftime("%H:%M:%S")

                    inicioTempoGravaRede = time.time()
                    log.infov( "%s Saved checkpoint at %d",stepControl,  s) 
                    save_path = self.saver.save(self.session,
                                            os.path.join(self.train_dir, 'model'),
                                            global_step=step)
                    TotalTempoGravaRede = TotalTempoGravaRede + (time.time() -inicioTempoGravaRede )
                if stepdataset % 5000 == 0: 
                 
                 self.data.updateIdsDataSet( os.path.join('./datasets', self.config.dataset_path  ), self.datasets[int(cont)], grupoDatasets= self.config.GrupDataset.split('|')[int(cont)])
              
                 dataset = self.datasets[int(cont)]
                 _, self.batch_train,imgs = create_input_ops(dataset, self.config.batch_size,
                                               is_training=True,is_loadImage=self.config.is_loadImage)
        self.plot_acuracy()
        _end_time_total = time.time()
        log.info('Tempo total de validacao'+ str(step_time_test_Total))
        log.info( 'Tempo total treinamento' + str((stepTimeTotalExecution)))
        log.info( 'Tempo total para gravar log' + str((totalTempoGravarArquivoLog)))
        log.info( 'Tempo total para gravar rede' + str((TotalTempoGravaRede)))
        log.info( 'Tempo total' + str((_end_time_total - _start_time_total)))
        
       

    def run_single_step(self, batch,teste_log_Save, step=None, is_train=True):
        _start_time = time.time()
        qtd = 100
        #batch_chunk = self.session.run(batch)
        treino=[]
        dataset = 0

        
        
        if (self.trainPosition > self.dataset.maxGrups -1):
            self.trainPosition = 0
        treino = self.dataset.batch[self.trainPosition]
        fetch = [self.global_step, self.model.accuracy, self.summary_op,
                         self.model.loss, self.check_op, self.optimizer,  self.model.all_preds, self.model.a]
      
          #treino= tf.convert_to_tensor(treino)
        try:
               if step is not None and (step % 100 == 0):
                 fetch += [self.plot_summary_op]
        except:
               pass
       
        fetch_values = self.session.run(
                  fetch, feed_dict=self.model.get_feed_dict2([treino[1],treino[2],treino[3],treino[4],treino[5],treino[6],fetch], step=step)
                )
        posicao = step
        [step, accuracy, summary, loss] = fetch_values[:4]
       
        #if posicao % teste_log_Save == 0 :
        #   self.add_batch( treino[0],treino[2],treino[3] ,fetch_values[6], fetch_values[7])
        #   self.report(step,'treinamento')
           
        try:
                if self.plot_summary_op in fetch:
                    summary += fetch_values[-1]
        except:
                pass

        _end_time = time.time()
        return step, accuracy, summary, loss,  (_end_time - _start_time)

    def GravarArquivo1 ( self,data_dict,fname):
      
       print("gravar arquivo: " + fname + " qtd: " +  str(len(data_dict)))
       os.makedirs(self.train_dir, exist_ok=True)
       fname = self.train_dir + "/" + fname +".json"
       # Create file
       with open(fname, 'a') as outfile:
         json.dump(data_dict, outfile, ensure_ascii=False, indent=2) 
         outfile.write('\n')
         outfile.close() 

    def run_test(self,tipo,step ,dataset,is_train=False):
        _start_time = time.time()
        treino=[]
        accuracy_test = 0
        i =0
        while (i < dataset.maxGrups -1):
            treino = dataset.batch[i]
            [accuracy_teste_step, all_preds, all_targets]  = self.session.run(
                [self.model.accuracy, self.model.all_preds, self.model.a], feed_dict=self.model.get_feed_dict2([treino[1],treino[2],treino[3],treino[4],treino[5],treino[6]], is_training=False))
            accuracy_test =   accuracy_teste_step +accuracy_test
            self.add_batch( treino[0],treino[2],treino[3] ,all_preds, all_targets)
            i=i+1
        accuracy_test,avg_nr,avg_r = self.report(step,tipo)
        _end_time = time.time() 
        tf.compat.v1.summary.scalar("loss/accuracy_test", (accuracy_test))
        self._ids=[]
        self._questions=[]
        self._answers=[]
        self._predictions=[]
        self._groundtruths=[]
        self.ArrarQuestoesErradas = []
        self.ArrayQuestoesCertas =[]
        questaotipo0 = self.ArrayTotalQuestoesCertas[0]/ ( self.ArrayTotalQuestoesErradas[0] +  self.ArrayTotalQuestoesCertas[0] )
        questaotipo1 = self.ArrayTotalQuestoesCertas[1]/ (self.ArrayTotalQuestoesErradas[1] +  self.ArrayTotalQuestoesCertas[1])
        questaotipo2 = self.ArrayTotalQuestoesCertas[2]/ (self.ArrayTotalQuestoesErradas[2] + self.ArrayTotalQuestoesCertas[2])
        questaotipo3 = self.ArrayTotalQuestoesCertas[3]/ (self.ArrayTotalQuestoesErradas[3] +  self.ArrayTotalQuestoesCertas[3])
        questaotipo4 = self.ArrayTotalQuestoesCertas[4]/ (self.ArrayTotalQuestoesErradas[4] +  self.ArrayTotalQuestoesCertas[4])
        totalQuestoes = self.ArrayTotalQuestoesErradas[0] +  self.ArrayTotalQuestoesCertas[0] + self.ArrayTotalQuestoesErradas[1] +  self.ArrayTotalQuestoesCertas[1] + self.ArrayTotalQuestoesErradas[2] + self.ArrayTotalQuestoesCertas[2] + self.ArrayTotalQuestoesErradas[3] +  self.ArrayTotalQuestoesCertas[3] + self.ArrayTotalQuestoesErradas[4] +  self.ArrayTotalQuestoesCertas[4] 
        totalQuestoesCertas = self.ArrayTotalQuestoesCertas[0] +   self.ArrayTotalQuestoesCertas[1] + self.ArrayTotalQuestoesCertas[2] +   self.ArrayTotalQuestoesCertas[3] +   self.ArrayTotalQuestoesCertas[4]
        
        self.ArrayTotalQuestoesCertas =[0,0,0,0,0] 
        self.ArrayTotalQuestoesErradas =[0,0,0,0,0]
        return (totalQuestoesCertas/totalQuestoes),(_end_time-_start_time),questaotipo0,questaotipo1,questaotipo2,questaotipo3,questaotipo4


    def report(self,step,tipo):

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
                #if int(id[i]) == 39950 : 
                #    print('oi')
                answer = answer2str(a[i])
                anserPred = np.zeros((len(a[i]))) 
                anserPred[np.argmax(pred[i,:])] = 1 
                anserPred1 = answer2str(anserPred)
                q_num = np.argmax(q[i][6:])
                if  q_num >= 2:
                    count_r += 1
                    
                    if np.argmax(pred[i, :]) == np.argmax(gt[i, :]):
                        self.ArrayTotalQuestoesCertas[q_num] +=1
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
                        self.ArrayTotalQuestoesErradas[q_num] += 1
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
                        self.ArrayTotalQuestoesCertas[q_num] +=1
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
                        self.ArrayTotalQuestoesErradas[q_num] += 1
                        #self.ArrarQuestoesErradas.append ( str(int(id[i])))
                        self.ArrarQuestoesErradas.append ("qestao id numero : " + str(int(id[i])))
                        self.ArrarQuestoesErradas.append(quest)
                        self.ArrarQuestoesErradas.append("Reposta errada:" + anserPred1)  
                        self.ArrarQuestoesErradas.append("Resposta certa:" + answer) 
                        self.ArrarQuestoesErradas.append("qestao do tipo : " +str(q_num)) 
                        self.ArrarQuestoesErradas.append("tipo : Nao-Relacional")

        avg_nr = float(correct_prediction_nr)/count_nr
        log.info("Average accuracy of non-relational questions: {}%".format(avg_nr*100))
        avg_r = float(correct_prediction_r)/count_r
        log.info("Average accuracy of relational questions: {}%".format(avg_r*100))
        avg = float(correct_prediction_r+correct_prediction_nr)/(count_r+count_nr)
        log.info("Average accuracy: {}%".format(avg*100))
        file_folder = self.check_pathSaveTrain
        return avg,avg_nr,avg_r
        #self.GravarArquivo(errorQuest,self.ArrarQuestoesErradas,"questaoErrada" +tipo ,file_folder,step )
        #self.GravarArquivo(correctQuest,self.ArrayQuestoesCertas,"questaoCerta"+ tipo,file_folder,step)
      
       

    def GravarArquivo (self,qtd, data_dict,fname, file_folder,step):
      fname = fname +"_" +str(qtd) + '.json'
      print("gravar arquivo: " + fname + " qtd: " +  str(len(data_dict)))
      os.makedirs(file_folder+'//'+'processamento'+str(step), exist_ok=True)
      fname = file_folder+'processamento'+str(step) + "/" + fname
      # Create file
      with open(fname, 'w') as outfile:
        json.dump(data_dict, outfile, ensure_ascii=False, indent=4) 
        outfile.close()
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
def check_data_path(path,id):
    if os.path.isfile(os.path.join(path, 'data.hy')) \
           and os.path.isfile(os.path.join(path,id)):
        return True
    else:
        return False

def runDynamicTrainner (config): 
    GrupDataset = config.GrupDataset
    path = os.path.join('./datasets', config.dataset_path  )
    import sort_of_clevr as DataSetClevr
    
    config.data_info = DataSetClevr.get_data_info()
    config.conv_info = DataSetClevr.get_conv_info()
   
    tipo =0
    dataset_train = []
    if len(GrupDataset )> 0: 
         for Grup in config.GrupDataset.split('|'):
            dataset_train.append(DataSetClevr.create_default_splits_perc(path,is_full =True,grupoDatasets=Grup,is_loadImage=config.is_loadImage))

    
    dataset_test= DataSetClevr.create_default_splits(path,is_full =True,id_filename="id_test.txt",is_loadImage=config.is_loadImage)

    

    trainner = None 
    trainer = Trainer(config,DataSetClevr,
                      dataset_train, dataset_test)

    log.warning("dataset: %s, learning_rate: %f",
                config.dataset_path, config.learning_rate)
    trainer.train()
      

def main():
    

    import tensorflow as tf
    tf.test.is_gpu_available()

    #gpu_config = tf.GPUOptions()
    #gpu_config.visible_device_list = "0"

    #session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_config))
    import argparse
    #import os
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--model', type=str, default='rn', choices=['rn', 'baseline'])
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default='Sort-of-CLEVR_teste_decode-image3')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    parser.add_argument('--train_amount_network',  type=int, default=0)
    parser.add_argument('--train_type',  type=str , default='full')
    parser.add_argument('--train_nameDataset',  type=str , default='id')
    
    parser.add_argument('--GrupDataset',  type=str , default='0,1,2|3,4')
    parser.add_argument('--OrdemDados',  type=str , default='E,E')
    parser.add_argument('--check_pathSaveTrain', type=str , default='id')
    parser.add_argument('--train_dir', type=str , default='padrao')
    parser.add_argument('--orderDataset', type=str , default='1,0')
    parser.add_argument('--StepChangeGroup', type=str , default='50000,50000')
    parser.add_argument('--is_loadImage', type=str , default=False)
    config = parser.parse_args()
    print('****************************************************')
    print(config.is_loadImage)
    if type(config.is_loadImage) == type(True): 
       config.is_loadImage = config.is_loadImage
    else:
       if config.is_loadImage == 'True' : 
            config.is_loadImage = True 
       else:
            config.is_loadImage = False
    path = os.path.join('./datasets', config.dataset_path  )
    #porcentual = config.train_nameInitalDataset.split(',')
    GrupDataset = config.GrupDataset.split('|')

    import sort_of_clevr as DataSetClevr
    
    config.data_info = DataSetClevr.get_data_info()
    config.conv_info = DataSetClevr.get_conv_info()
   
    tipo =0
    dataset_train = []
    if len(GrupDataset )> 0: 
         for Grup in config.GrupDataset.split('|'):
            dataset_train.append(DataSetClevr.create_default_splits_perc(path,is_full =True,grupoDatasets=Grup,is_loadImage=config.is_loadImage))

    
    
    dataset_test= DataSetClevr.create_default_splits(path,is_full =True,id_filename="id_test.txt",is_loadImage=config.is_loadImage)

    


    trainer = Trainer(config,DataSetClevr,
                      dataset_train, dataset_test)

    log.warning("dataset: %s, learning_rate: %f",
                config.dataset_path, config.learning_rate)
    trainer.train()



if __name__ == '__main__':
    main()
