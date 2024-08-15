const express = require("express");
const app = express();

app.use(express.json());
const fs = require("fs");
const port = 8000;

app.use("/scripts", express.static("./scripts"));
app.use("/styles", express.static("./styles"));
app.use("html", express.static("./html"));

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

app.listen(port, () => {
	console.log(`Listening on port ${port}.`);
});
