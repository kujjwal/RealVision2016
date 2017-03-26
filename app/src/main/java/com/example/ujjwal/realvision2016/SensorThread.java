package com.example.ujjwal.realvision2016;

/**
 * Created by Ujjwal on 1/8/17.
 */

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.widget.TextView;

public class SensorThread extends Thread implements SensorEventListener{


    public SensorManager mSensorManager;
    public Sensor gravSensor;
    public Context context;
    TextView gravSensorView;

    public SensorThread(Context c){

        mSensorManager = MainActivity.mSensorManager;

        gravSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
        context = c;

        gravSensorView = (TextView) ((AppCompatActivity)c).findViewById(R.id.gravView);
        gravSensorView.setText("Connected");
    }

    //SENSORS
    @Override
    public final void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Do something here if sensor accuracy changes.
    }

    @Override
    public final void onSensorChanged(SensorEvent event) {
        // The light sensor returns a single value.

//        Log.e(MainActivity.TAG, "WORKING stuff");
        // Many sensors return 3 values, one for each axis.
        float x = event.values[0];
        float y = event.values[1];
        float z = event.values[2];
        // Do something with this sensor value.

        gravSensorView.setText("x: " + x + ", y: " + y + ", z: " + z);

//        Log.e(TAG,"x: " + x + ", y: " + y + ", z: " + z);


    }


    public void onActivityResume(){
        mSensorManager.registerListener(this, gravSensor, SensorManager.SENSOR_DELAY_NORMAL);
    }

    public void onActivityPause(){
        mSensorManager.unregisterListener(this);
    }
}
