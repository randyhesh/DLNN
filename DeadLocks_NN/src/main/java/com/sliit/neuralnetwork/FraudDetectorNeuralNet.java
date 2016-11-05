package com.sliit.neuralnetwork;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.metadata.*;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.vectorizer.Vectorizer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.datavec.RecordReaderFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.stats.StatsUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.web.context.ServletContextAware;

import javax.servlet.ServletContext;
import java.io.*;
import java.util.*;

/**
 * Deep Mulilayer Neural Network to detect frauds.
 */
public class FraudDetectorNeuralNet {

    private static final Log log = LogFactory.getLog(FraudDetectorNeuralNet.class);
    private int inputs = 4;
    private int outputNum = 2;
    private int batchSize = 64;
    private int nEpochs = 10;
    private int iterations = 1;
    private int seed = 123;
    private int nCores = 3;
    private ServletContext servletContext;
    private String uploadDirectory;
    private DataSetIterator minTrainSet;
    private MultiLayerNetwork model = null;

    public FraudDetectorNeuralNet(){



    }

    public void buildModel() throws NeuralException {

        log.info("load model....");
        try {
            log.info("build model...");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .iterations(iterations)
                    .regularization(true).l2(1e-5)
                    .updater(Updater.ADADELTA)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .learningRate(0.005)
                    .momentum(0.08)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(inputs).nOut(outputNum)
                            .activation("tanh")
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(1, new DenseLayer.Builder().nIn(outputNum).nOut(2)
                            .activation("tanh")
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .weightInit(WeightInit.XAVIER)
                            .activation("softmax")
                            .nIn(2).nOut(outputNum).build())
                    .backprop(true).pretrain(false)
                    .build();

            model = new MultiLayerNetwork(conf);
            model.init();
            //model.setUpdater(Updater.ADAGRAD);
        } catch (Exception e) {
            log.error("Error ocuured while building neural netowrk :"+e.getMessage());
            throw new NeuralException(e.getLocalizedMessage(),e);
        }
    }

    public void buildDynamicModel() throws NeuralException{

        try {
            log.info("loadDynamic Model...");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .iterations(iterations)
                    .regularization(true).l2(1e-5)
                    .updater(Updater.ADADELTA)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .learningRate(0.005)
                    .momentum(0.08)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(inputs).nOut(8)
                            .activation("tanh")
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(1, new DenseLayer.Builder().nIn(8).nOut(5)
                            .activation("tanh")
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(2, new DenseLayer.Builder().nIn(5).nOut(2)
                            .activation("tanh")
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .weightInit(WeightInit.XAVIER)
                            .activation("softmax")
                            .nIn(2).nOut(outputNum).build())
                    .backprop(true).pretrain(false)
                    .build();
            model = new MultiLayerNetwork(conf);
            model.init();
        }
        catch(Exception e){

            log.error("Error ocuured while building neural netowrk :"+e.getMessage());
            throw new NeuralException(e.getLocalizedMessage(),e);
        }
    }

    public boolean generateModel(Map<String,String> params){

        boolean status = false;
        try {

            inputs = Integer.parseInt(params.get("inputs"));
            outputNum = Integer.parseInt(params.get("outputs"));
            String name = params.get("model");
            if(inputs < 10) {

                buildModel();
            }
            else{

                buildDynamicModel();
            }
            loadSaveNN(name, true);
            status = true;
        }catch (NeuralException e){

            log.error("Error occurred:"+e.getLocalizedMessage());
        }
        return status;
    }
    public String trainModel(String neural_model,JavaRDD<DataSet> rddDataSet,SparkDataSet sparkDataSet) throws NeuralException {

        try {
            //SparkDataSet sparkDataSet = SparkDataSet.getInstance("fraud_data","local[*]",0,2);
            //JavaRDD<DataSet> rddDataSet = sparkDataSet.generateTrainingDataset("hdfs://localhost:9000/user/asantha/fraud_data/datasamplefraud.csv");
            log.info("Training model...");
            loadSaveNN(neural_model,false);
            if(model== null){

                buildModel();
            }
            model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1/5)));
            //setup the spark training
            ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1)
                    .workerPrefetchNumBatches(2) //Asynchronusly prefetch upto 2 batches
                    .saveUpdater(true)
                    .repartionData(Repartition.Always)
                    .repartitionStrategy(RepartitionStrategy.Balanced)
                    .averagingFrequency(5)
                    .batchSizePerWorker(8) //number of examples that each worket gets,per fit operation
                    .build();
            SparkDl4jMultiLayer network = new SparkDl4jMultiLayer(sparkDataSet.getSc(),model,tm);
            network.setCollectTrainingStats(true);
            int nEpochs = 100;
            for(int i=0;i<nEpochs;i++){

                model = network.fit(rddDataSet);
            }
            SparkTrainingStats stats = network.getSparkTrainingStats();
            StatsUtils.exportStatsAsHtml(stats,"SparkTrainingStatus.html",sparkDataSet.getSc());
            Evaluation evaluation = network.evaluate(rddDataSet);
            String statMsg = evaluation.stats();
            log.info(statMsg);
            loadSaveNN(neural_model,true);
            return statMsg;

        } catch (Exception e) {

            log.error("Error ocuured while building neural netowrk :"+e.getMessage());
            throw new NeuralException(e.getLocalizedMessage(),e);
        }
    }

    public String detectFraud(List<String[]> inputs,String neural_model,String path) throws NeuralException, IOException {

        /*if(model == null){

            loadSaveNN(neural_model,false);
        }
        NormalizeDataset normalizer = new NormalizeDataset();
        File uploadDirecotry = new File(path);
        File outputTest = normalizer.readFromList(inputs,uploadDirecotry);
        org.datavec.api.records.reader.RecordReader recordReader = new org.datavec.api.records.reader.impl.csv.CSVRecordReader(0,",");
        try {
            recordReader.initialize(new org.datavec.api.split.FileSplit(outputTest));
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        DataSetIterator iterator = new org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator(recordReader,1,0,2);
        INDArray outputArr = null;
        while(iterator.hasNext()){

            DataSet ds = iterator.next();
            System.out.println(ds);
            outputArr = model.output(ds.getFeatureMatrix());
            System.out.println(outputArr);
        }
        String msg = "";
        assert outputArr != null;
        double result = Double.parseDouble(Nd4j.argMax(outputArr,1).toString());
        if(result==0){

            msg = "normal";
        }else{

            msg = "fraud";
        }
        System.out.println(msg);*/
        return "";
    }

    private void loadSaveNN(String name,boolean save){


        File directory = new File(uploadDirectory);
        File[] allNN = directory.listFiles();
        boolean status = false;
        try {

            if(model == null && save){

                buildModel();
            }
            if(allNN != null && allNN.length > 0) {
                for (File NN : allNN) {

                    String fnme = FilenameUtils.removeExtension(NN.getName());
                    if (NN.getName().equals(fnme)) {

                        status = true;
                        if (save) {

                            ModelSerializer.writeModel(model,NN,true);

                        } else {

                            model = ModelSerializer.restoreMultiLayerNetwork(NN);
                        }
                        break;
                    }
                }
            }
            if(!status && save){

                //File tempFIle = File.createTempFile(name,".zip",directory);
                File tempFile = new File(directory.getAbsolutePath()+"/"+name+".zip");
                if(!tempFile.exists()){

                    tempFile.createNewFile();
                }
                ModelSerializer.writeModel(model,tempFile,true);
            }
        }catch(IOException e) {

            log.error("Error occurred :"+e.getLocalizedMessage());
        } catch (NeuralException e) {

            log.error("Error occurred :"+e.getLocalizedMessage());
        }
    }

    public static void main(String[] args) {

        FraudDetectorNeuralNet model = new FraudDetectorNeuralNet();
        try {
            //model.buildModel();
            //model.trainModel();
            String[] data = {"normal","1","3232235777","1456888991","1456889891"};
            String[] data2 = {"fraud","2","3232235677","1456888891","1456889691"};
            List<String[]> dataList = new ArrayList<String[]>();
            dataList.add(data);
            //dataList.add(data2);
            model.detectFraud(dataList,"temp3","");
        } catch (NeuralException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void setUploadDirectory(String uploadDirectory) {

        this.uploadDirectory = uploadDirectory;

    }
}
