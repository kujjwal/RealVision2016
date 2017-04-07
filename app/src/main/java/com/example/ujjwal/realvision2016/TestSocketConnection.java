package com.example.ujjwal.realvision2016;

import android.util.Log;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;

/**
 * Created by Ujjwal on 4/6/17.
 */

public class TestSocketConnection {

    private final String TAG = "TestSocketConnection";

    public void testConnection() throws IOException {
        ServerSocket ss = new ServerSocket(8001);
        ss.setReuseAddress(true);

        Socket s = ss.accept();
        DataInputStream dis = new DataInputStream(s.getInputStream());
        boolean isAimed = dis.readBoolean();
        String centerString = dis.readUTF();
        String distString = dis.readUTF();
        double[] center = new double[] { Double.parseDouble(centerString.split(",")[0].trim()),
                Double.parseDouble(centerString.split(",")[1].trim())};
        double dist = Double.parseDouble(distString.split(",")[1].trim());
        Log.d(TAG, "IsAimed: " + isAimed);
        Log.d(TAG, "Center: " + center[0] + ", " + center[1]);
        Log.d(TAG, "Dist: " + dist);
    }
}
