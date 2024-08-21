var filesInput = document.getElementById("file");

filesInput.addEventListener("change", function (event) {
	var files = event.target.files;
	var file = files[0];
	var reader = new FileReader();

	reader.addEventListener("load", function (event) {
		var textFile = event.target;
		var lines = textFile.result.split("\n");

		// remove blank trailing line if present
		if (lines[lines.length - 1] === "") {
			lines = lines.splice(0, lines.length - 1);
		}

		var outputFile = "";
		var lookup = "";

		for (let n = 0; n < lines.length; n++) {
			let wordIdPair = lines[n].split(" ");
			// Handle special token
			if (wordIdPair[0].includes("<")) {
				outputFile +=
					"export const " + wordIdPair[0].replace(/[<>]/g, "") + " = " + wordIdPair[1] + ";\n";
			} else {
				// Regular word
				lookup +=
					'"' + wordIdPair[0] + '": ' + wordIdPair[1] + (n === lines.length - 1 ? "\n" : ",\n");
			}
		}

    outputFile += 'export const LOOKUP = {\n' + lookup + '}\n';

    download('dictionary.js', outputFile);
	});

  reader.readAsText(file);
});

function download(filename, text) {
  var blob = new Blob([text], {type: "text/plain"});
  const href = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.download = filename;
  a.href = href;
  document.body.appendChild(a);
  a.click();
}
