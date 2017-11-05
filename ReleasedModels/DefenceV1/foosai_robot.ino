//////////////////////////////////////////////////////////////
////////// FoosAI - Defensemen V1 model
//////////////////////////////////////////////////////////////
// See https://github.com/glmcdona/FoosAI for more information.
//
// This is the arduino code to run the foosbot robot. There are three main components:
//   1. Webcam hooked up to a PC. The webcam has a view of the foosball table.
//   2. Computer running deep-learning AI model that is trained on human foosball player defenders.
//   3. The computer sends the predicted rod position changes to this arduino code via serial USB. This script controls the stepper motors to take those actions.
//////////////////////////////////////////////////////////////

#include <AccelStepper.h> // http://www.airspayce.com/mikem/arduino/AccelStepper/

// Maximum accelerations for the spinning and rotation. It stalls if you increase this too much.
float spin_max_accel = 6000.0;
float belt_max_accel = 6000.0;


int max_position = 300; // 200 is one rotation

// Switch to turn robot on/off
int TGLPIN = 3;

// Stepper motor that controls the rotation of the rod
int dirPin = 8;
int stepperPin = 7;
AccelStepper stepperSpin (1, stepperPin, dirPin);

// Stepper motor that controls the translation of the rod
int dirPinBelt = 10;
int stepperPinBelt = 9;
AccelStepper stepperBelt (1, stepperPinBelt, dirPinBelt);

void setup()
{
	// On/off switch
	pinMode(TGLPIN, INPUT);

	Serial.begin(115200);

	// Stepper motors
	stepperSpin.setMaxSpeed(8000); // 200 * 10 = 2000 for 10 rotations a second
	stepperSpin.setAcceleration(spin_max_accel); // Sets the acceleration/deceleration rate. The desired acceleration in steps per second per second
	stepperBelt.setMaxSpeed(8000);
	stepperBelt.setAcceleration(belt_max_accel);
}

int count = 0;
int tglOn = 0;
bool first_run = true;

void loop()
{
	tglOn = digitalRead(TGLPIN);
	
	if( tglOn )
	{
		if( first_run )
		{
			// Reset our positions
			stepperSpin.setCurrentPosition(0);
			stepperBelt.setCurrentPosition(0);
			
			// Run position test
			stepperBelt.runToNewPosition(max_position/2);
			stepperSpin.runToNewPosition(200);
			delay(100);
			stepperBelt.runToNewPosition(max_position);
			stepperSpin.runToNewPosition(400);
			delay(100);
			stepperBelt.runToNewPosition(max_position/2);
			stepperSpin.runToNewPosition(600);
			delay(100);
			stepperBelt.runToNewPosition(0);
			stepperSpin.runToNewPosition(0);
			delay(100);
			first_run = false;
		}
		else
		{
			if (Serial.available() >= 4*3)
			{
				// Update the current requested position change
				float dpos[] = {0.0, 0.0, 0.0}; // Goalie, Defensemen, Attacking 3-bar
				if ( Serial.readBytes((char*)dpos, 4*3) == 4*3 )
				{
					// Update our desired rod position based on current position
					int desired_belt_position = stepperBelt.currentPosition() + ((int) dpos[1]*max_position)
					
					// Limit the rod position
					if( desired_belt_position < 0 )
						desired_belt_position = 0;
					if( desired_belt_position > max_position )
						desired_belt_position = max_position;
					
					stepperBelt.moveTo(desired_belt_position);
				}
			}
			stepperBelt.run();
		}
	}
	else
	{
		first_run = true;
	}
	count++;
}



