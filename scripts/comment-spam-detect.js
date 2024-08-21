const POST_COMMENT_BTN = document.getElementById("post");
const COMMENT_TEXT = document.getElementById("comment");
const COMMENTS_LIST = document.getElementById("commentsList");
// CSS styling class to indicate comment is being pricessed when posting to provide visual feedback to users.
const PROCESSING_CLASS = "processing";

var currentUserName = "Anonymous";

/**
 * Function to handle the processing of submitted comments.
 */
function handleCommentPost() {
	// Only continue if you are not already processing the comment.
	if (!POST_COMMENT_BTN.classList.contains(PROCESSING_CLASS)) {
		// Set style to show processing in case it takes a long time
		POST_COMMENT_BTN.classList.add(PROCESSING_CLASS);
		COMMENT_TEXT.classList.add(PROCESSING_CLASS);

		// Grab the comment text from DOM
		let currentComment = COMMENT_TEXT.innerText;

		// Convert sentence to lower case which ML Model expects
		// Strip all characters that are not alphanumeric or spaces
		// then split on spaces to create a word array
		let lowercaseSentenceArray = currentComment
			.toLocaleLowerCase()
			.replace(/[^\w\s]/g, " ")
			.split(" ");
		console.log("lower case sentence: ", lowercaseSentenceArray);

		let li = document.createElement("li");

		loadAndPredict(tokenize(lowercaseSentenceArray), li).then(function () {
			POST_COMMENT_BTN.classList.remove(PROCESSING_CLASS);
			COMMENT_TEXT.classList.remove(PROCESSING_CLASS);

			let p = document.createElement("p");
			p.innerText = COMMENT_TEXT.innerText;
			let spanName = document.createElement("span");
			spanName.setAttribute("class", "username");
			spanName.innerText = currentUserName;

			let spanDate = document.createElement("span");
			spanDate.setAttribute("class", "timestamp");
			let curDate = new Date();
			spanDate.innerText = curDate.toLocaleString();

			li.appendChild(spanName);
			li.appendChild(spanDate);
			li.appendChild(p);
			COMMENTS_LIST.prepend(li);

			// Reset comment text
			COMMENT_TEXT.innerText = "";
		});
	}
}

POST_COMMENT_BTN.addEventListener("click", handleCommentPost);

const MODEL_JSON_URL =
	"/public/model.json";
const SPAM_THRESHOLD = 0.75;
var model = undefined;

async function loadAndPredict(inputTensor, domComment) {
	// Load the model.json and binart files you hosted. Note this is an
	// asynchronous operation
	if (model === undefined) {
		model = await tf.loadLayersModel(MODEL_JSON_URL);
	}

	// Once model has been loaded we can call model.predict and pass to an input
	// in the form of a Tensor. Then store the result
	var results = await model.predict(inputTensor);

	// Print the result to the console for us to inspect.
	results.print();

	// let dataArray = results.dataSync();
	// if (dataArray[1] > SPAM_THRESHOLD) {
	// 	domComment.classList.add("spam");
	// }

	results.data().then((dataArray)=>{
    if (dataArray[1] > SPAM_THRESHOLD) {
      domComment.classList.add('spam');
    } else {
      // Emit socket.io comment event for server to handle containing
      // all the comment data you would need to render the comment on
      // a remote client's front end.
      socket.emit('comment', {
        username: currentUserName,
        timestamp: domComment.querySelectorAll('span')[1].innerText,
        comment: domComment.querySelectorAll('p')[0].innerText
      });
    }
  })
}

import * as DICTIONARY from "/public/dictionary.js";

// The number of input elements the ML Model is expecting.
const ENCODING_LENGTH = 20;

/**
 * Function that takes an array of words, converts words to tokens,
 * and then returns a Tensor representation of the tokenization that
 * can be used as input to the machine learning model.
 * @param wordArray
 * @returns
 */
function tokenize(wordArray) {
	// Start the array with the START token
	let returnArray = [DICTIONARY.START];

	// Loop through the words in the sentence needing to be encoded.
	// If word is found in dictionary, add that number else add the UNKNOWN token
	for (var i = 0; i < wordArray.length; i++) {
		let encoding = DICTIONARY.LOOKUP[wordArray[i]];
		returnArray.push(encoding === undefined ? DICTIONARY.UNKNOWN : encoding);
	}

	while (returnArray.length < ENCODING_LENGTH) {
		returnArray.push(DICTIONARY.PAD);
	}

	console.log(returnArray);

	return tf.tensor([returnArray]);
}

// Connect to Socket.io on the Node.js backend.
var socket = io.connect();


function handleRemoteComments(data) {
  // Render a new comment to DOM from a remote client.
  let li = document.createElement('li');
  let p = document.createElement('p');
  p.innerText = data.comment;

  let spanName = document.createElement('span');
  spanName.setAttribute('class', 'username');
  spanName.innerText = data.username;

  let spanDate = document.createElement('span');
  spanDate.setAttribute('class', 'timestamp');
  spanDate.innerText = data.timestamp;

  li.appendChild(spanName);
  li.appendChild(spanDate);
  li.appendChild(p);
  
  COMMENTS_LIST.prepend(li);
}


// Add event listener to receive remote comments that passed
// spam check.
socket.on('remoteComment', handleRemoteComments);