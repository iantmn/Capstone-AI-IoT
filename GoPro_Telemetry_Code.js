

const gpmfExtract = require('gpmf-extract');
const goproTelemetry = require(`gopro-telemetry`);
const fs = require('fs');

const file = fs.readFileSync('C:/Users/gijsl/Downloads/GH010015.mp4');


gpmfExtract(file)
  .then(extracted => {
    goproTelemetry(extracted, {}, telemetry => {
      fs.writeFileSync('output_path_Skipp.json', JSON.stringify(telemetry));
      console.log('Telemetry saved as JSON');
    });
  })
  .catch(error => console.error(error));

