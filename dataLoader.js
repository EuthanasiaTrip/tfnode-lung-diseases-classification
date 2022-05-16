const fs = require('fs');
const tf = require('@tensorflow/tfjs-node-gpu');
const logger = require('./logger.js');
const ConfusionMatrix = require('ml-confusion-matrix');

const IMAGE_H = 200;
const IMAGE_W = 200;

class DataHandler {
    constructor() {
        this.dataLabels = [];
    }

    getData(dataType, shuffle = true) {
        const dataFolder = `data/${dataType}`;
        let data = [];
        let classes = [];
        this.dataLabels = [];

        fs.readdirSync(dataFolder).forEach(file => {
            classes.push(file);
        });

        classes.forEach((label, index) => {
            const folder = dataFolder + '/' + label;
            logger.log(`Resizing and preparing array of 3D tensors. Class: ${label}`);
            fs.readdirSync(folder).forEach(img => {
                let imageTensor = tf.node.decodeJpeg(fs.readFileSync(folder + '/' + img), 1);
                data.push(
                    tf.image.resizeBilinear(imageTensor, [IMAGE_H, IMAGE_W])
                );
                imageTensor.dispose();
                this.dataLabels.push(index);
            });
        });
        if (shuffle) {
            tf.util.shuffleCombo(data, this.dataLabels);
        }
        logger.log("Stacking 3D tensors into 4D tensor...");
        const images = tf.stack(data);
        const labels = tf.oneHot(tf.tensor1d(this.dataLabels, 'int32'), classes.length).toFloat();

        data.forEach(element =>
            element.dispose()
        );
        return { images, labels }
    }
}

class MetricsHandler {
    constructor() {
        this.valAcc = [];
        this.valLoss = [];
        this.loss = [];
        this.acc = [];
        this.exportFolder = './results';
        this._setFolder();

        this.exportFile = this.exportFolder + '/Results ' + new Date().toISOString()
            .replace(/T/, ' ')
            .replace(/\..+/, '')
            .replace(':', '-')
            .replace(':', '-') + ".json";
    }

    _setFolder(){
        if(!fs.existsSync(this.exportFolder)){
            fs.mkdirSync(this.exportFolder);
        }
    }

    addEpochLogs(logs) {
        this.valAcc.push(logs.val_acc);
        this.valLoss.push(logs.val_loss);
        this.loss.push(logs.loss);
        this.acc.push(logs.acc);
    }

    regResult(predictions, labels) {
        this.predictions = predictions;
        const output = Array.from(predictions.argMax(1).dataSync());
        const CM = ConfusionMatrix.fromLabels(labels, output);
        this.accuracy = CM.getAccuracy();
        this.confMatrix = CM.getMatrix();
        this.falseCount = CM.getFalseCount();
        this.trueCount = CM.getTrueCount();
    }

    exportToJSON() {
        const metrics = {
            val_acc: this.valAcc,
            val_loss: this.valLoss,
            loss: this.loss,
            acc: this.acc,
            confMatrix: this.confMatrix,
            accuracy: this.accuracy,
            falseCount: this.falseCount,
            trueCount: this.trueCount,
        }
        fs.writeFileSync(this.exportFile, JSON.stringify(metrics));
    }
}

module.exports = { IMAGE_H, IMAGE_W, loader: new DataHandler(), metricsHandler: new MetricsHandler() };