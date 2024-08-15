// Generate input numbers from 1 to 20 inclusive
const INPUTS = [];
for (let i = 1; i <= 20; i++) {
	INPUTS.push(i);
}

// Generate outputs that simply each input multiplied by itself,
// to generate some non linear data.
const OUTPUTS = [];
for (let i = 0; i < INPUTS.length; i++) {
	OUTPUTS.push(INPUTS[i] * INPUTS[i]);
}

// Input feature Array of Arrays need 2D tensor to store.
const INPUTS_TENSOR = tf.tensor1d(INPUTS);

// Ouput can stay 1 dimensional.
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

// Function to take a Tensor and normalize values
// with respect to each column of values contained in that Tensor.
function normalize(tensor, min, max) {
	const result = tf.tidy(function () {
		// Find the minimum value contained in the Tensor.
		const MIN_VALUES = min || tf.min(tensor, 0);

		// Find the maximum value contained in the Tensor.
		const MAX_VALUES = max || tf.max(tensor, 0);

		// Now subtract the MIN_VALUE from every value in the Tensor
		// And store the results in a new Tensor.
		const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

		// Calculate the range size of possible values.
		const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

		// Calculate the adjusted values divided by the range size as a new Tensor.
		const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

		return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
	});
	return result;
}

// Normalize all input feature arrays and then
// dispose of the original none normalized Tensors.
const FEATURE_RESULTS = normalize(INPUTS_TENSOR);

console.log("Normalize Values:");
FEATURE_RESULTS.NORMALIZED_VALUES.print();

console.log("Min values:");
FEATURE_RESULTS.MIN_VALUES.print();

console.log("Max values:");
FEATURE_RESULTS.MAX_VALUES.print();

INPUTS_TENSOR.dispose();

// Create and define model architecture
const model = tf.sequential();

// const LEARNING_RATE = 0.01; // Choose learning rate that's suitable for the data we are using.
let optimizer;
let learning_rate;
let epochVal;

function caseOf7(valueToPredict) {
	learning_rate = 0.0001;
	optimizer = tf.train.sgd(learning_rate);
	epochVal = 200;

	// Add the first layer with 3 neurons (units) processing 1 inpyt feature value,
	// and an activation function.
	model.add(tf.layers.dense({ inputShape: [1], units: 25, activation: "relu" })); // update to 25 units

	// Add a hidden layer with a desired number of neurons and an activation function.
	// Each neuron in this layer takes inputs from all neurons in the first layer to
	// produce input for the single neuron in the output layer below.
	model.add(tf.layers.dense({ units: 5, activation: "relu" })); // update units to 5

	// Add another layer with a single neuron that takes input from all the neuron in the second hidden layer.
	// This is called the output layer that will produce the final prediction result.
	model.add(tf.layers.dense({ units: 1 }));
	model.summary();

	train(valueToPredict, optimizer, epochVal);
}

// console.log("Result for Prediction Value of 7:");
// await caseOf7(7);

function caseOf14(valueToPredict) {
	learning_rate = 0.0001;
	optimizer = tf.train.sgd(learning_rate);
	epochVal = 200;

	// Add the first layer with 3 neurons (units) processing 1 inpyt feature value,
	// and an activation function.
	model.add(tf.layers.dense({ inputShape: [1], units: 25, activation: "relu" })); // update to 25 units

	// Add a hidden layer with a desired number of neurons and an activation function.
	// Each neuron in this layer takes inputs from all neurons in the first layer to
	// produce input for the single neuron in the output layer below.
	model.add(tf.layers.dense({ units: 5, activation: "relu" })); // update units to 5

	// Add another layer with a single neuron that takes input from all the neuron in the second hidden layer.
	// This is called the output layer that will produce the final prediction result.
	model.add(tf.layers.dense({ units: 1 }));
	model.summary();

	train(valueToPredict, optimizer, epochVal);
}

console.log("Result for Prediction Value of 14:");
await caseOf14(14);

async function train(value, optimizerVal, epochVal) {
	// Compile the model with the defined learning rate and specify a loss function to use.
	model.compile({
		optimizer: optimizerVal,
		loss: "meanSquaredError",
	});

	// Finally do the training itself
	let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
		callbacks: { onEpochEnd: logProgress },
		shuffle: true, // Ensure data is shuffled in case it was in an order.
		batchSize: 2, // As we have a lot of training data, batch size is set to 64.
		epochs: epochVal, // Go over the data 10 times!
	});

	OUTPUTS_TENSOR.dispose();
	FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

	console.log(
		"Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1])
	);

	evaluate(value); // Once trained evaluate the model/
}

function evaluate(value) {
	// Predict answer for a single piece of data
	tf.tidy(function () {
		let newInput = normalize(
			tf.tensor1d([value]),
			FEATURE_RESULTS.MIN_VALUES,
			FEATURE_RESULTS.MAX_VALUES
		);

		let output = model.predict(newInput.NORMALIZED_VALUES);
		console.log("Prediction Result:");
		output.print();
	});

	// Clean up memory use
	FEATURE_RESULTS.MIN_VALUES.dispose();
	FEATURE_RESULTS.MAX_VALUES.dispose();

	model.dispose();

	console.log(tf.memory().numTensors);
}

async function saveModelOffline() {
	await model.save("localstorage://tfjs/linear-regression-model");
	console.log("Model saved successfully", model);
}

saveModelOffline();

function logProgress(epoch, logs) {
	console.log("Data for epoch " + epoch, Math.sqrt(logs.loss));
	// if (epoch == 70) {
	// 	optimizer.setLearningRate(learning_rate / 2);
	// }
}
