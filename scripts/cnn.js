import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js";

const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

function normalize(tensor, min, max) {
	const result = tf.tidy(function () {
		const MIN_VALUES = tf.scalar(min);
		const MAX_VALUES = tf.scalar(max);

		const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
		const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
		const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

		return NORMALIZED_VALUES;
	});

	return result;
}

// Input feature Array is 2 dimensional.
const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

// Map output index to label
const LOOKUP = [
	"T-shirt",
	"Trouser",
	"Pullover",
	"Dress",
	"Coat",
	"Sandal",
	"Shirt",
	"Sneaker",
	"Bag",
	"Ankle boot",
];

var interval = 2000;
const RANGER = document.getElementById("ranger");
const DOM_SPEED = document.getElementById("domSpeed");
const PREDICTION_ELEMENT = document.getElementById("prediction");

// When user drags slider update interval
RANGER.addEventListener("input", function (e) {
	interval = this.value;
	DOM_SPEED.innerText = "Change speed of classifications! Currently: " + interval + "ms";
});

// Create and define model architecture.
const model = tf.sequential();

// Add the first convolutional layer
model.add(
	tf.layers.conv2d({
		inputShape: [28, 28, 1],
		filters: 16,
		kernelSize: 3, // Square Filter of 3 by 3. could also specify rectangle ef [2, 3].
		stride: 1,
		padding: "same",
		activation: "relu",
	})
);

// Add the first maxPooling layer. Transform 28x28px input into 14x14px
model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

// Add the second convolutional layer
model.add(
	tf.layers.conv2d({
		filters: 32,
		kernelSize: 3,
		strides: 1,
		padding: "same",
		activation: "relu",
	})
);

// Add the second maxPooling layer. 7x7 outputs
model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

// Flatten the outputs from maxPooling to convert to an array
model.add(tf.layers.flatten());

model.add(tf.layers.dense({units: 128, activation: 'relu'}));

model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

model.summary();

train();

async function train() {
  model.compile({
    optimizer: 'adam', // Adam changes the learning rate over time which is usefull
    loss: 'categoricalCrossentropy', // As this is a classification problem, dont use MSE
    metrics: ['accuracy']
  });

  const RESHAPED_INPUTS = INPUTS_TENSOR.reshape([INPUTS.length, 28,28, 1]);
  let results = await model.fit(RESHAPED_INPUTS, OUTPUTS_TENSOR, {
    shuffle: true,
    validationSplit: 0.15,
    epochs: 30,  // Go over the data 30 times
    batchSize: 256,
    callbacks: {onEpochEnd: logProgress}
  });

  RESHAPED_INPUTS.dispose();
  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();
  evaluate();
}

function evaluate() {
  // Select a random index from all the example images we have in the training data arrays.
  const OFFSET = Math.floor((Math.random() * INPUTS.length));
  
  // Clean up created tensors automatically.
  let answer = tf.tidy(function() {
    let newInput = normalize(tf.tensor1d(INPUTS[OFFSET]), 0, 255);
    
    let output = model.predict(newInput.reshape([1, 28, 28, 1]));
    output.print();
    
    return output.squeeze().argMax();    
  });
  
  answer.array().then(function(index) {
    PREDICTION_ELEMENT.innerText = LOOKUP[index];
    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');
    answer.dispose();
    drawImage(INPUTS[OFFSET]);
  });
}

const CANVAS = document.getElementById('canvas');
const CTX = CANVAS.getContext('2d');

function drawImage(item) {
  CTX.willReadFrequently = true;
  var imageData = CTX.getImageData(0, 0, 28, 28);

 for (let i = 0; i < item.length; i++) {
  imageData.data[i * 4] = item[i] * 255; // Red
  imageData.data[i * 4 + 1] = item[i] * 255; // Green
  imageData.data[i * 4 + 2] = item[i] * 255; // Blue
  imageData.data[i * 4 + 3] = 255; // Alpha
 }

 // Render the updated array of data to the canvas itself
 CTX.putImageData(imageData, 0, 0);

 // Perform a new classification depending the interval chosen.
 setTimeout(evaluate, interval);
}

function logProgress(epoch, logs) {
  console.log("Data for epoch " + epoch + ": " + logs);
}