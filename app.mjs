import * as tf from '@tensorflow/tfjs-node-gpu';
import logger from './logger.js'
import ConfusionMatrix from 'ml-confusion-matrix'

const trueLabels =      [0, 1, 0, 1, 1, 0, 0];
const predictedLabels = [1, 1, 0, 1, 0, 0, 0];


const CM = ConfusionMatrix.fromLabels(trueLabels, predictedLabels);
console.log(CM.getMatrix());

