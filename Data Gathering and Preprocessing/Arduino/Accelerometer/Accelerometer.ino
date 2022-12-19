// Change integers/pins if module is connected elsewhere
#define VDD     19 // 59 for Arduino mega
#define GROUND  18 // 58 for Arduino mega

const int xpin = A3;
const int ypin = A2;
const int zpin = A1;

void setup() {
  // Change baudrate if it takes too much time.
  // Don't forget to change it also in the python code!
  Serial.begin(9600);

  // Set the input pins A4 and A5 to output and set them to gnd and vdd
  pinMode(GROUND, OUTPUT);
  pinMode(VDD, OUTPUT);
  digitalWrite(GROUND, LOW);
  digitalWrite(VDD, HIGH);
}

void loop() {
  // Read the pins and print the results to the serial connection
  Serial.print(analogRead(xpin)); // Value between 0 and 1023
  Serial.print(',');
  Serial.print(analogRead(ypin));
  Serial.print(',');
  Serial.println(analogRead(zpin));
  delay(100);
}
