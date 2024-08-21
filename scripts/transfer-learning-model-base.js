
const STATUS = document.getElementById("status");
const VIDEO = document.getElementById("webcam");
const ENABLE_CAM_BUTTON = document.getElementById("enableCam");
const RESET_BUTTON = document.getElementById("reset");
const TRAIN_BUTTON = document.getElementById("train");
const MOBILE_NET_INPUT_HEIGHT = 224;
const MOBILE_NET_INPUT_WIDTH = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];

ENABLE_CAM_BUTTON.addEventListener("click", enableCam);
TRAIN_BUTTON.addEventListener("click", trainAndPredict);
RESET_BUTTON.addEventListener("click", reset);

// Add more buttons in HTML to allow classification of more classes of data
let dataCollectorButtons = document.querySelectorAll("button.dataCollector");
for (let i = 0; i < dataCollectorButtons.length; i++) {
	dataCollectorButtons[i].addEventListener("mousedown", gatherDataForClass);
	dataCollectorButtons[i].addEventListener("mouseup", gatherDataForClass);
	// For mobile
	dataCollectorButtons[i].addEventListener("touchend", gatherDataForClass);

	// Populate the human readable names for classes.
	CLASS_NAMES.push(dataCollectorButtons[i].getAttribute("data-name"));
}

let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;
let mobileNetBase = undefined;

function customPrint(line) {
	let p = document.createElement("p");
	p.innerText = line;
	document.body.appendChild(p);
}

/**
 * Loads the MobileNet model and warms it up so ready for use.
 */
async function loadMobileNetFeatureModel() {
	const URL =
		"https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json";
	mobilenet = await tf.loadLayersModel(URL);
	STATUS.innerText = "MonileNet v2 loaded successfully!";
	mobilenet.summary(null, null, customPrint);

	const layer = mobilenet.getLayer("global_average_pooling2d_1");
	mobileNetBase = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
	mobileNetBase.summary();

	// Warm up the model by passing zeros through it once.
	tf.tidy(function () {
		let answer = mobileNetBase.predict(
			tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
		);
		console.log("answer.shape ", answer.shape);
	});
}

loadMobileNetFeatureModel();

let model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [1280], units: 64, activation: "relu" }));
model.add(tf.layers.dense({ units: CLASS_NAMES.length, activation: "softmax" }));

model.summary();

// Compile the model with the defined optimizer and specify a loss funtion to use.
model.compile({
	// Adam changes the learning rate over time which is useful
	optimizer: "adam",
	// Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
	// Else, categoricalCrossentropy is used if more than 2 classes.
	loss: CLASS_NAMES.length === 2 ? "binaryCrossentropy" : "categoricalCrossentropy",
	// As this is a  classification problem you can record accuracy in the logs
	metrics: ["accuracy"],
});

/**
 * Check if getUserMedia is supported for webcam access.
 */
function hasGetUserMedia() {
	return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

/**
 * Enable the webcam with video constraints applied.
 */
function enableCam() {
	if (hasGetUserMedia) {
		// getUserMedia parameters
		const constraints = {
			video: true,
			width: 640,
			height: 480,
		};

		// Activate the webcam stream
		navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
			VIDEO.srcObject = stream;
			VIDEO.addEventListener("loadeddata", function () {
				videoPlaying = true;
				ENABLE_CAM_BUTTON.classList.add("removed");
			});
		});
	} else {
		console.warn("getUserMedia() is not supported by the browser");
	}
}

/**
 * Hanlde Data Gather for button mouseup/mousedown.
 */
function gatherDataForClass() {
	let classNumber = parseInt(this.getAttribute("data-1hot"));
	gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
	dataGatherLoop();
}

function calculateFeaturesOnCurrentFrame() {
	return tf.tidy(function () {
		// Grab pixels from current VIDEO frame.
		let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
		// Resize video frame tensor to be 224x224 pixels which is needed by MobileNet for input.
		let resizedTensorFrame = tf.image.resizeBilinear(
			videoFrameAsTensor,
			[MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
			true
		);

		let normalizesTensorFrame = resizedTensorFrame.div(255);

		return mobileNetBase.predict(normalizesTensorFrame.expandDims()).squeeze();
	});
}

/**
 * When a button used to gather data is pressed, record feature vectors along with class type to arrays.
 */
function dataGatherLoop() {
	// Only gather data if webcam is on and a relevant button is pressed.
	if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
		// Ensure tensor are cleaned up
		let imageFeatures = calculateFeaturesOnCurrentFrame();

		trainingDataInputs.push(imageFeatures);
		trainingDataOutputs.push(gatherDataState);

		// Initialize array index element if currently undefined.
		if (examplesCount[gatherDataState] === undefined) {
			examplesCount[gatherDataState] = 0;
		}

		// Increment counts of examples fr user interface to show
		examplesCount[gatherDataState]++;

		STATUS.innerText = "";
		for (let n = 0; n < CLASS_NAMES.length; n++) {
			STATUS.innerText += CLASS_NAMES[n] + " data count: " + examplesCount[n] + ". ";
		}

		window.requestAnimationFrame(dataGatherLoop);
	}
}

/**
 * Once data collected actually perform the transfer learning.
 */
async function trainAndPredict() {
	predict = false;
	console.log("training Input", trainingDataInputs);
	console.log("training Output ", trainingDataOutputs);
	tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

	let outputsAsTensor = tf.tensor1d(trainingDataInputs, "int32");
	let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
	let inputsAsTensor = tf.stack(trainingDataInputs);

	let results = await model.fit(inputsAsTensor, oneHotOutputs, {
		shuffle: true,
		batchSize: 5,
		epochs: 5,
		callbacks: { onEpochEnds: logProgress },
	});

	outputsAsTensor.dispose();
	oneHotOutputs.dispose();
	inputsAsTensor.dispose();

	predict = true;

	// Make combined model for download

	let combinedModel = tf.sequential();
	combinedModel.add(mobileNetBase);
	combinedModel.add(model);

	combinedModel.compile({
		optimizer: "adam",
		loss: CLASS_NAMES === 2 ? "binaryCrossentropy" : "categoricalCrossentropy",
	});

	combinedModel.summary();
	await combinedModel.save("download://my-model");
	predictLoop();
}

/**
 * Log training progress.
 */
function logProgress(epoch, logs) {
	console.log("Data for epoch " + epoch, logs);
}

/**
 * Make live predictions from webcam once trained.
 */
function predictLoop() {
	if (predict) {
		tf.tidy(function () {
			let imageFeatures = calculateFeaturesOnCurrentFrame();
			let prediction = model.predict(imageFeatures.expandDims()).squeeze();
			console.log("Prediction Result: ", prediction);
			let highestIndex = prediction.argMax().arraySynx();
			let predictionArray = prediction.arraySynx();
			STATUS.innerText =
				"Prediction: " +
				CLASS_NAMES[highestIndex] +
				" with " +
				Math.floor(predictionArray[highestIndex] * 100) +
				"% confidence.";
		});

		window.requestAnimationFrame(prediction);
	}
}

/**
 * Purge data and start over. Note this coes not dispose of the loaded
 * MobileNet model and MLP head tensors as they will be needed to train
 * a new model
 */
function reset() {
	predict = false;
	examplesCount.splice(0);
	for (let i = 0; i < trainingDataInputs.length; i++) {
		trainingDataInputs[i].dispose();
	}
	trainingDataInputs.splice(0);
	trainingDataOutputs.splice(0);
	STATUS.innerText = "No data collected";

	console.log("Tensor in memory: " + tf.memory().numTensor);
}
