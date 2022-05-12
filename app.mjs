import * as tf from '@tensorflow/tfjs-node-gpu';
import logger from './logger.js'
import pkg from './dataLoader.js';

const batchSize = 24;
const trainEpochs = 25;
const validationSplit = 0.15;
const loader = pkg.loader;
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
    model.add(tf.layers.conv2d({ kernelSize: [3, 3], filters: 30, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.conv2d({ kernelSize: [3, 3], filters: 30, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 2048, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));

    model.compile({
        optimizer: 'adam',
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
        }
    }
});

trainData.images.dispose();
trainData.labels.dispose();
valData.images.dispose();
valData.labels.dispose();


const testData = loader.getData("test");
const evalOutput = model.evaluate(testData.images, testData.labels);
logger.log(
    `Evaluation result:` +
    `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
    `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

const predictions = model.predict(testData.images);
console.log(predictions.shape)
//const out = tf.math.confusionMatrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], predictions, 4);
const ls =  Array.from(testData.labels.argMax(1).dataSync());
const output = Array.from(predictions.argMax(1).dataSync());
//console.log(output);
predictions.print();
output.forEach((item, index) => console.log(`${index} : ${item}`));

const modelSavePath = './model'

await model.save(`file://${modelSavePath}`);
logger.log(`Saved model to path: ${modelSavePath}`);
