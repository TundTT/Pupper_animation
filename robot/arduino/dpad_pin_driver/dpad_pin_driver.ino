// Drives one digital output pin HIGH/LOW per D-pad direction, controlled over serial
// by the ros2_ws/src/dpad_serial_bridge ROS 2 node.
//
// Protocol: newline-terminated 2-character lines, "<U|R|D|L><0|1>", e.g. "U1\n" = up
// pressed (pin HIGH), "D0\n" = down released (pin LOW). Malformed lines are ignored.
// Baud rate must match the node's `baud_rate` param.

const int PIN_UP = 3;
const int PIN_RIGHT = 4;
const int PIN_DOWN = 5;
const int PIN_LEFT = 6;

void setup() {
  Serial.begin(115200);

  pinMode(PIN_UP, OUTPUT);
  pinMode(PIN_RIGHT, OUTPUT);
  pinMode(PIN_DOWN, OUTPUT);
  pinMode(PIN_LEFT, OUTPUT);

  digitalWrite(PIN_UP, LOW);
  digitalWrite(PIN_RIGHT, LOW);
  digitalWrite(PIN_DOWN, LOW);
  digitalWrite(PIN_LEFT, LOW);
}

int pinForDirection(char direction) {
  switch (direction) {
    case 'U': return PIN_UP;
    case 'R': return PIN_RIGHT;
    case 'D': return PIN_DOWN;
    case 'L': return PIN_LEFT;
    default: return -1;
  }
}

void loop() {
  static String line;

  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      if (line.length() == 2) {
        int pin = pinForDirection(line[0]);
        char state = line[1];
        if (pin != -1 && (state == '0' || state == '1')) {
          digitalWrite(pin, state == '1' ? HIGH : LOW);
          // DEBUG: bench-test echo. Remove once verified against real hardware,
          // since the ROS node never reads this and it's just Serial Monitor noise.
          Serial.print("OK pin ");
          Serial.print(pin);
          Serial.println(state == '1' ? " -> HIGH" : " -> LOW");
        } else {
          Serial.print("IGNORED malformed line: \"");
          Serial.print(line);
          Serial.println("\"");
        }
      } else {
        Serial.print("IGNORED wrong-length line (");
        Serial.print(line.length());
        Serial.print(" chars): \"");
        Serial.print(line);
        Serial.println("\"");
      }
      line = "";
    } else if (c != '\r') {
      line += c;
      // Guard against a malformed/oversized line never seeing '\n'.
      if (line.length() > 8) {
        line = "";
      }
    }
  }
}
