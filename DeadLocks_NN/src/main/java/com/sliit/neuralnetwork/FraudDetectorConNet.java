package com.sliit.neuralnetwork;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaRDD;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;

/**
 * Convolutional Neural Network.
 */
public class FraudDetectorConNet {

    private static final Log log = LogFactory.getLog(FraudDetectorConNet.class);
    private int nChannels = 1;
    private int outputNum = 10;
    private int batchSize = 64;
    private int nEpochs = 10;
    private int iterations = 1;
    private int seed = 123;
    private DataSetIterator minTrainSet;
    private MultiLayerNetwork model = null;

    public FraudDetectorConNet(){


    }

    public void buildModel() throws NeuralException{

        log.info("load Model...");
        try{

            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .regularization(true).l2(0.0005)
                    .learningRate(0.01)//.biasLearningRate(0.02)
                    //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                    .weightInit(WeightInit.XAVIER)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Updater.NESTEROVS).momentum(0.9)
                    .list()
                    .layer(0, new ConvolutionLayer.Builder(5, 5)
                            .nIn(nChannels)
                            .stride(1, 1)
                            .nOut(20)
                            .activation("identity")
                            .build())
                    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2,2)
                            .stride(2,2)
                            .build())
                    .layer(2, new ConvolutionLayer.Builder(5, 5)
                            .stride(1, 1)
                            .nOut(50)
                            .activation("identity")
                            .build())
                    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2,2)
                            .stride(2,2)
                            .build())
                    .layer(4, new DenseLayer.Builder().activation("relu")
                            .nOut(500).build())
                    .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nOut(outputNum)
                            .activation("softmax")
                            .build())
                    .backprop(true).pretrain(false);
            new ConvolutionLayerSetup(builder,28,28,1);

            MultiLayerConfiguration conf = builder.build();
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
        }catch(Exception e){
            log.error("Error ocuured while building neural netowrk :"+e.getMessage());
            throw new NeuralException(e.getLocalizedMessage(),e);
        }
    }

    public void trainModel() throws NeuralException{

        try {
            RecordReader recordReader = new CSVRecordReader(0,",");
            SparkDataSet sparkDataSet = SparkDataSet.getInstance("fraud_data","local[*]",38,3);
            JavaRDD<DataSet> rddDataSet = sparkDataSet.generateTrainingDataset("hdfs://localhost:9000/user/asantha/fraud_data/fraud_dataset.csv");
            log.info("Train model...");
            if(model== null){

                buildModel();
            }
            model.setListeners(new ScoreIterationListener(1));
            ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1)
                    .workerPrefetchNumBatches(2) //Asynchronusly prefetch upto 2 batches
                    .saveUpdater(true)
                    .averagingFrequency(8) //average frequency per iteration this should be high for increase the efficiency
                    .batchSizePerWorker(8) //number of examples that each worket gets,per fit operation
                    .build();
            SparkDl4jMultiLayer network = new SparkDl4jMultiLayer(sparkDataSet.getSc(),model,tm);
            network.fit(rddDataSet);
        } catch (Exception e) {

            log.error("Error ocuured while building neural netowrk :"+e.getMessage());
            throw new NeuralException(e.getLocalizedMessage(),e);
        }
    }

    public void detectFraud(DataSet dataSet) throws NeuralException{

        if(model == null){

            buildModel();
        }
        log.info("output :"+model.output(dataSet.getLabels()));
    }

    public static void main(String[] args) {

        FraudDetectorConNet model = new FraudDetectorConNet();
        try {

            model.buildModel();
            model.trainModel();
        } catch (NeuralException e) {

            log.error("Error Occurred:"+e.getMessage());
        }
    }
}
