import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

// Grab a reference to the MNIST input values (pixel data).
const INPUTS = TRAINING_DATA.inputs;

// Grab a reference to the MNIST output values.
const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the two arrays in the same way so inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Input feature Array is 2 dimensional.
const INPUTS_TENSOR = tf.tensor2d(INPUTS);

// Output feature Array is 1 dimensional.
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

// Create an d define model architecture.
const model = tf.sequential();

model.add(tf.layers.dense({ inputShape: [784], units: 30, activation: "relu" }));
model.add(tf.layers.dense({ units: 15, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

model.summary();

train();

async function train() {
	// Comile the model with the defined optimizer and specify our loss function to use.
	model.compile({
		optimizer: "adam",
		loss: "categoricalCrossentropy",
		metrics: ["accuracy"],
	});

	let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
		shuffle: true, // Ensure data is shuffled before using each epoch
		validationSplit: 0.2, // Keep 20% of the data for validation testing
		batchSize: 512, // Update weights after every 512 examples.
		epochs: 50, // Go over the data 50 times.
		callbacks: { onEpochEnd: logProgress },
	});

  console.log('Accuracy: ', results.history.acc);
  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();
  evaluate();   // Once trained we can evaluate 
}

const PREDICTION_ELEMENT = document.getElementById("prediction");

function evaluate() {
  const OFFSET = Math.floor((Math.random() * INPUTS.length)); // Select random int from example inputs size

  let answer = tf.tidy(function() {
    let newInput = tf.tensor1d(INPUTS[OFFSET]);

    let output = model.predict(newInput.expandDims());
    output.print();
    return output.squeeze().argMax();
  });

  answer.array().then(function(index) {
    PREDICTION_ELEMENT.innerText = index;
    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');
    answer.dispose();
    drawImage(INPUTS[OFFSET]);
  })
}

const CANVAS = document.getElementById("canvas");
const CTX = CANVAS.getContext('2d');

function drawImage(digit) {
  CTX.willReadFrequently = true; // For performance recommended by the browser
  var imageData = CTX.getImageData(0, 0, 28, 28);

  for (let i = 0; i < digit.length; i++) {
    imageData.data[i * 4] = digit[i] * 255;  // Red Channel.
    imageData.data[i * 4 + 1] = digit[i] * 255; // Green Channel
    imageData.data[i * 4 + 2] = digit[i] * 255; // Blue Channel
    imageData.data[i * 4 + 3] = 255;  // Alpha Channel
  }

  // Render the updated array of data to the canvas itself
  CTX.putImageData(imageData, 0, 0);

  // Peform a new classification after a certain interval.
  setTimeout(evaluate, 2000);
}
function logProgress() {
  console.log("Data for epoch " + epoch + ": " + logs.loss);
}