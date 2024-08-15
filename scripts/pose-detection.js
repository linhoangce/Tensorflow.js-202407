let EXAMPLE_IMG = document.getElementById("exampleImg");

function drawPoints(ctx, x, y, radius, color) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI, true);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = color;
  ctx.stroke();
}

async function loadModel() {
	const detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING };
	const detector = await poseDetection.createDetector(
		poseDetection.SupportedModels.MoveNet,
		detectorConfig
	);

	let exampleInputTensor = document.getElementById('exampleImg');
  let imageTensor = tf.browser.fromPixels(exampleInputTensor);
  console.log(imageTensor.shape);

  let cropStartPoint = [15, 170, 0];
  let cropSize = [345, 345, 3];
  let croppedTensor = tf.slice(imageTensor,  cropStartPoint, cropSize);

  let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).cast('int32');
  console.log(resizedTensor.shape);

  console.log(tf.expandDims(resizedTensor));

	const poses = await detector.estimatePoses(imageTensor);
  console.log(poses[0].keypoints);
  console.log(poses[0].keypoints.length);

  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');

	poses[0].keypoints.forEach(keypoint => {
    const {x, y, score, name} = keypoint;
    if( score > 0.3) {
      drawPoints(ctx, x, y, 1, 'red');
    }
  });

}

loadModel();

