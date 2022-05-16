import * as tf from '@tensorflow/tfjs-node-gpu';
import logger from './logger.js';
import pkg from './dataLoader.js';

const batchSize = 24;
const trainEpochs = 25;
const loader = pkg.loader;
const metrics = pkg.metricsHandler;
const IMAGE_H = pkg.IMAGE_H;
const IMAGE_W = pkg.IMAGE_W;

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_H, IMAGE_W, 1],
        kernelSize: [5, 5],
        filters: 128,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [3, 3] }));
    model.add(tf.layers.conv2d({ kernelSize: [5, 5], filters: 64, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: [3, 3] }));
    model.add(tf.layers.conv2d({ kernelSize: [3, 3], filters: 32, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.conv2d({ kernelSize: [3, 3], filters: 16, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 2048, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));
    const optimizer = tf.train.adam(0.0001, 0.9, 0.999);
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

const trainData = loader.getData("train");
const valData = loader.getData("val");

let model = createModel();
model.summary(90, [0.32, 0.61, 0.89, 1], (message) =>
    logger.log(message, false)
);


await model.fit(trainData.images, trainData.labels, {
    batchSize,
    validationData: [valData.images, valData.labels],
    epochs: trainEpochs,
    callbacks: {
        onEpochEnd: async (batch, logs) => {
            logger.log(logs);
            metrics.addEpochLogs(logs);
        }
    }
});

trainData.images.dispose();
trainData.labels.dispose();
valData.images.dispose();
valData.labels.dispose();

const testData = loader.getData("test", false);
const evalOutput = model.evaluate(testData.images, testData.labels);
logger.log(
    `Evaluation result:` +
    `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
    `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

const predictions = model.predict(testData.images);
predictions.print();

metrics.regResult(predictions, loader.dataLabels);
logger.log(metrics.confMatrix);
metrics.exportToJSON();

const modelSavePath = './model'

await model.save(`file://${modelSavePath}`);
logger.log(`Saved model to path: ${modelSavePath}`);