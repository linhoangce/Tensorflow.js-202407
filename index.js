const http = require('http');
const express = require("express");
const app = express();
const server = http.createServer(app);

// Require socket.io and then make it use the http server above.
// This allows us to expose correct socket.io library JS for use
// in the client side JS.

const { Server } = require('socket.io');
const io = new Server(server);

app.use(express.json());
const fs = require("fs");
const port = 8000;

app.use("/scripts", express.static("./scripts"));
app.use("/styles", express.static("./styles"));
app.use("/html", express.static("./html"));
app.use("/public", express.static("./public"));


app.get("/", function (req, res) {
	let doc = fs.readFileSync("./html/object-detection.html", "utf8");

	res.send(doc);
});

app.get("/property", function (req, res) {
	let doc = fs.readFileSync("./html/property-price-prediction.html", "utf8");

	res.send(doc);
});

app.get("/pose", function (req, res) {
	let doc = fs.readFileSync("./html/pose-detection.html", "utf8");

	res.send(doc);
});

app.get("/linear", function (req, res) {
	let doc = fs.readFileSync("./html/linear-regression.html", "utf8");

	res.send(doc);
});

app.get("/non-linear", function (req, res) {
	let doc = fs.readFileSync("./html/non-linear-regression.html", "utf8");

	res.send(doc);
});

app.get("/multi-layer", function (req, res) {
	let doc = fs.readFileSync("./html/multi-layer-classification.html", "utf8");

	res.send(doc);
});

app.get("/cnn", (req, res) => {
	let doc = fs.readFileSync("./html/cnn.html", "utf-8");
	res.send(doc);
});

app.get("/transfer-learning", (req, res) => {
	let doc = fs.readFileSync("./html/transfer-learning.html", "utf-8");
	res.send(doc);
});

app.get("/transfer-learning-model-base", (req, res) => {
	let doc = fs.readFileSync("./html/transfer-learning-model-base.html", "utf-8");
	res.send(doc);
});

app.get("/comment-filter", (req, res) => {
	let doc = fs.readFileSync("./html/comment-spam-detect.html", "utf-8");
	res.send(doc);
});

app.get("/vocab-converter", (req, res) => {
	let doc = fs.readFileSync("./html/vocab-converter.html", "utf-8");
	res.send(doc);
});

// Handle socket.io client connect event.
io.on('connect', socket => {
  console.log('Client connected');

  // If you wanted you could emit existing comments from some DB
  // to client to render upon connect.
  // socket.emit('storedComments', commentObjectArray);  
 
  // Listen for "comment" event from a connected client.
  socket.on('comment', (data) => {
    // Relay this comment data to all other connected clients
    // upon receiving.
    socket.broadcast.emit('remoteComment', data);
  });
});

server.listen(port, () => {
	console.log(`Listening on port ${port}.`);
});
