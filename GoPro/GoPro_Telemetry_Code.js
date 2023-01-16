

const gpmfExtract = require('gpmf-extract');
const goproTelemetry = require(`gopro-telemetry`);
const fs = require('fs');

const file = fs.readFileSync('D:/DCIM/100GOPRO/GL020034_LRV.MP4');



gpmfExtract(file)
  .then(extracted => {
    goproTelemetry(extracted, {}, telemetry => {
      fs.writeFileSync('Timo_fietsen_GH020034.json', JSON.stringify(telemetry));
      console.log('Telemetry saved as JSON');
    });
  })
  .catch(error => console.error(error));

