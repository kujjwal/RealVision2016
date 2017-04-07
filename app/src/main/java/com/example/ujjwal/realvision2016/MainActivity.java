package com.example.ujjwal.realvision2016;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.hardware.Camera;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.RelativeLayout;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.text.DecimalFormat;
import java.util.Calendar;
import java.util.Enumeration;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    //from robot
    protected static final String mTAG = "VisionServer";
    protected static final int DROID_SIDE_SERVER_PORT = 5000;
    protected String mLocalServerIp;
    private Handler mHandler = new Handler();
    private ServerSocket mLocalServerSocket;

    public Mat screenMat;

    //TODO: Change the IP and Port
    String RIO_SIDE_SERVER_IP = "10.6.49.2";
    int RIO_SIDE_SERVER_PORT = 5805;

    int MAX_X = 288, MAX_Y = 352; //cam resolution
    int green = Color.parseColor("#43fa00"), red = Color.parseColor("#fa0000");

    Handler handler = new Handler();

    static String TAG = "Vision649";
    static String fileName = "vision649.txt";

    //TargetFinder targetFinder;
    FindingRects findingRects;

    RelativeLayout colorScreen;
    RelativeLayout checkBox;
    boolean isGreen = false;
    //TODO: Change based on lighting conditions at tourney
    boolean dark = false;

    Thread clientThread;
    private PortraitCameraView mOpenCvCameraView;
    private Camera.Parameters mCameraParameters;
    public Camera mCamera;
    public static SensorManager mSensorManager;

    SensorThread sensorThread;

    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    mOpenCvCameraView.enableView();
                    Log.d(this.getClass().getSimpleName(), "CAMERA SUCCESS");
                } break;
                default:
                {
                    super.onManagerConnected(status);

                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        ////*******//////

        // Check if we have write permission
        int permission = ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE);

        RequestUserPermission requestUserPermission = new RequestUserPermission(this);
        requestUserPermission.verifyStoragePermissions();

        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                    this,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }

        ////*******//////


        setContentView(R.layout.activity_main);

        //Check if opencv is working
        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        //Log.e("Camera", "" + Camera.getNumberOfCameras());
        mCamera = Camera.open();
//        mCamera.setDisplayOrientation(90);


        mLocalServerIp = getLocalIpAddress();
        Log.d(mTAG, "Server IP " + mLocalServerIp);

        Thread fst = new Thread(new ServerThread());
        fst.start();

        colorScreen = (RelativeLayout) findViewById(R.id.colorBack);

        mOpenCvCameraView = (PortraitCameraView) findViewById(R.id.HelloOpenCvView_java);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setMaxFrameSize(600, 600);

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);

        //targetFinder = new TargetFinder();
        findingRects = new FindingRects();

        sensorThread = new SensorThread(this);
        sensorThread.start();

        screenMat = new Mat();
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

        mOpenCvCameraView.releaseCamera();

        sensorThread.onActivityPause();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this,
                mLoaderCallback);

        //targetFinder = new TargetFinder();
        findingRects = new FindingRects();

        sensorThread.onActivityResume();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

        sensorThread.interrupt();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }



    ///CAMERA

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    Map<String, Object> results;

    @Override
    public Mat onCameraFrame(PortraitCameraView.CvCameraViewFrame inputFrame) {
        //TODO this is where we will add image processing

        Camera.Parameters parameters = mOpenCvCameraView.mCamera.getParameters();
        Log.d(TAG, "1: " + parameters.getExposureCompensation());
        parameters.setExposureCompensation(20);
        Log.d(TAG, "2: " + parameters.getExposureCompensation());
        Mat original = new Mat();

        double t1 = System.currentTimeMillis();
        inputFrame.rgba().copyTo(original);
        if (!original.empty()){
            ProcessImage imageProcessor = new ProcessImage(original);
            imageProcessor.start();
        }

        Log.d("->timelog", "Pulling data t: " + (System.currentTimeMillis() - t1));

        if (!screenMat.empty()){
            return screenMat;
        }
        else{
            return inputFrame.rgba();
        }
    }


    private void writeToFile(String fName, String data) {


        try {
            //verifyStoragePermissions(this);

            File root = new File(Environment.getExternalStorageDirectory(), "Vision");
            File filepath = new File(root, fName);  // file path to save
            FileWriter fileOut = new FileWriter(filepath);

            fileOut.write("");
            fileOut.append(data + "\n");

            Calendar cal = Calendar.getInstance();
            cal.add(Calendar.DATE, 1);

            fileOut.append("" + cal.getTime());
            fileOut.close();
        }
        catch (IOException e) {
            Log.e("Exception", "File write failed: " + e.toString());
        }
    }

    public synchronized void updateMat(Mat m){
        screenMat = m;
    }

    public class ProcessImage extends Thread{
        Mat m0;

        public ProcessImage(Mat mat){
            m0 = mat;
        }

        @Override
        public void run(){

            Mat m = new Mat();
            Mat original = new Mat();
            m0.copyTo(m);
            m.copyTo(original);

            //initial
            double _time = Calendar.getInstance().getTimeInMillis();

            //results = targetFinder.shapeDetectTarget(m);

            /*if(dark) {
                results = targetFinder.darkAlg(m);
            } else {
                results = targetFinder.lightAlg(m);
            }*/
            //results = targetFinder.lightAlg(m);
            //results = findingRects.findingRects(m);
            File root = new File(Environment.getExternalStorageDirectory(), "Vision");
            File filepath = new File(root, "Screenshot_20170319-005626~3.png");  // file path to save
            Mat templ = Imgcodecs.imread(filepath.getPath());
            int method = Imgproc.TM_CCOEFF_NORMED;
            //results = findingRects.templateMatching(m, )
            //results = findingRects.templateMatching(m, templ, method);
            //results = targetFinder.shapeDetectTarget10(m);
            results = findingRects.ujjwaliscool(m);
            Log.d("->timelog", "SHAPE DETECT TARGET FULL t: " + (Calendar.getInstance().getTimeInMillis() - _time));
            _time = Calendar.getInstance().getTimeInMillis();

            Center c = (Center)results.get("center");
            //m = (Mat)results.get("m");
            //Mat thresh = (Mat)results.get("thresh");
            //Mat subImage = (Mat)results.get("subImage");
            //Mat blobMat = (Mat)results.get("blobMat");
            //Mat combined = (Mat)results.get("combined");
            //Mat hsv = (Mat)results.get("hsv");
            //        Mat ycrcb = (Mat)results.get("ycrcb");
            Rect roi = (Rect)results.get("roi");
            double dist = (Double) results.get("dist");
            Mat hsv = (Mat) results.get("hsv");

            Log.d("->timelog", "Pulling data t: " + (Calendar.getInstance().getTimeInMillis() - _time));
            _time = Calendar.getInstance().getTimeInMillis();


            DecimalFormat df = new DecimalFormat("#.##");

            String formattedCenter = "C (" + df.format(c.x) + ", " + df.format(c.y) + ")";

            //Log.d(TAG, formattedCenter);

            if (!(c.equals(TargetFinder.NO_CENTER) || c.equals(new Center(Double.NaN, Double.NaN)))) {
                Imgproc.putText(original, "FOUND: " + formattedCenter, new Point(10, 20), Core.FONT_HERSHEY_PLAIN, 0.5, new Scalar(255, 0, 0));
                Imgproc.line(original, new Point(c.x + 10, c.y), new Point(c.x - 10, c.y), new Scalar(255, 0, 0), 2);
                Imgproc.line(original, new Point(c.x, c.y + 10), new Point(c.x, c.y - 10), new Scalar(255, 0, 0), 2);
                Imgproc.rectangle(original, roi.tl(), roi.br(), new Scalar(0, 255, 0));
                // update the background
                //TODO Tweak vals based on what works
                if (c.y > MAX_Y / 2.0 - 10 && c.y < MAX_Y / 2.0 + 10) {
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            colorScreen.setBackgroundColor(green);
                            isGreen = true;
                        }
                    });
                } else {
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            colorScreen.setBackgroundColor(red);
                            isGreen = false;
                        }
                    });
                }
                /*handler.post(new Runnable() {
                    @Override
                    public void run() {
                        colorScreen.setBackgroundColor(green);
                        isGreen = true;
                    }
                });*/
            }
            else {
                Log.d(TAG, "No target detected");
                Imgproc.putText(original, "No Target Detected", new Point(10, 20), Core.FONT_HERSHEY_PLAIN, 0.5, new Scalar(0, 0, 255));
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        colorScreen.setBackgroundColor(red);
                        isGreen = false;
                    }
                });
            }

            Log.d("->timelog", "draw crap on image t: " + (Calendar.getInstance().getTimeInMillis() - _time));
            _time = Calendar.getInstance().getTimeInMillis();

            //draw boundaries
            Imgproc.line(original, new Point(0, TargetFinder.MIN_Y_COORD), new Point(TargetFinder.RES_X, TargetFinder.MIN_Y_COORD), new Scalar(20, 20, 20), 2);
            Imgproc.line(original, new Point(0, TargetFinder.MAX_Y_COORD), new Point(TargetFinder.RES_X, TargetFinder.MAX_Y_COORD), new Scalar(20, 20, 20), 2);

            //TODO
            updateMat(original);

            //writeToFile(fileName, "" + df.format(c.x) + " " + df.format(c.y));//+ " <> Time: " + df.format(System.currentTimeMillis()));

            Log.d("->timelog", "draw 2 lines and write to file t: " + (Calendar.getInstance().getTimeInMillis() - _time));
            _time = Calendar.getInstance().getTimeInMillis();


            //Log.d("T>>>>>", "" + (System.currentTimeMillis() - t1));

            //m.release();
            //thresh.release();
            //subImage.release();
            //combined.release();
            //blobMat.release();
            //hsv.release();

            try {
                Socket s = new Socket(InetAddress.getLocalHost(), RIO_SIDE_SERVER_PORT); //set up on Robot
                Log.d(TAG, "Connecting socket: " + s.getInetAddress().getHostAddress());
                DataOutputStream dos = new DataOutputStream(s.getOutputStream());

                if (c != new Center(-1, -1) && dist != 0.0) {
                    //dos.writeUTF(isGreen);
                    dos.writeBoolean(isGreen);
                    dos.writeUTF("" + c.x + ", " + c.y); //EG sent like:  120.2, 222.3
                    dos.writeUTF("Distance, " + dist); //EG sent like: Distance, 100.0
                    dos.flush();
                    dos.close();
                }

                DataInputStream dis = new DataInputStream(s.getInputStream());
                System.out.println(dis.toString());

                checkBox = (RelativeLayout) findViewById(R.id.dataIn);

                if(dis.readBoolean()) {
                    checkBox.setBackgroundColor(green);
                } else {
                    checkBox.setBackgroundColor(red);
                }
                s.close();
            }
            catch (UnknownHostException e){
                Log.e(TAG, "Unknown Host Error: " + e.getMessage(), e);
            }
            catch (IOException e){
                Log.e(TAG, "Here we go, IO Error: " + e.getMessage(), e);
            }
        }
    }


    private void switchToVision() {
        Log.d(mTAG, "Bringing Application to Front");

        Intent intent = new Intent(this, MainActivity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK); // You need this if starting
// the activity from a service
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP ); // You need this if starting

        intent.setAction(Intent.ACTION_MAIN);
        intent.addCategory(Intent.CATEGORY_LAUNCHER);
        startActivity(intent);
        mCamera.release();
        finish();
    }


    private void startIpWebcam() {
        Log.d(mTAG, "Bringing IpWebcam to Front");
        Intent launcher = new Intent().setAction(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_HOME);
        Intent ipwebcam =
                new Intent()
                        .setClassName("com.pas.webcam", "com.pas.webcam.Rolling")
                        .putExtra("cheats", new String[] {
                                "set(Video,320,240)",
                                "reset(Photo)",
                                "set(Awake,true)",
                                "reset(Port)",                 // Use default port 8080
                        })
                        .putExtra("hidebtn1", true)                // Hide help button
                        .putExtra("caption2", "Run in background") // Change caption on "Actions..."
                        .putExtra("intent2", launcher)             // And give button another purpose
                        .putExtra("returnto", new Intent().setClassName(MainActivity.this, MainActivity.class.getName())); // Set activity to return to
        startActivity(ipwebcam);


    }


    // Gets the ip address of phone's network
    private String getLocalIpAddress() {
        try {
            for (Enumeration<NetworkInterface> en = NetworkInterface.getNetworkInterfaces(); en.hasMoreElements();) {
                NetworkInterface intf = en.nextElement();
                for (Enumeration<InetAddress> enumIpAddr = intf.getInetAddresses(); enumIpAddr.hasMoreElements();) {
                    InetAddress inetAddress = enumIpAddr.nextElement();
                    if (!inetAddress.isLoopbackAddress()) {
                        return inetAddress.getHostAddress().toString();
                    }
                }
            }
        } catch (SocketException ex) {
            Log.e("ServerActivity", ex.toString());
        }
        return null;
    }

    public class ServerThread implements Runnable {

        public void run() {
            try {
                if (mLocalServerIp != null) {
                    Log.d(mTAG, "Listening on IP: " + mLocalServerIp);

                    mLocalServerSocket = new ServerSocket(DROID_SIDE_SERVER_PORT);
                    mLocalServerSocket.setReuseAddress(true);
                    while (true) {
                        // LISTEN FOR INCOMING CLIENTS
                        Socket client = mLocalServerSocket.accept();
                        Log.d(mTAG, "Connected.");

                        try {
                            BufferedReader in = new BufferedReader(
                                    new InputStreamReader(client.getInputStream()));
                            String command = null;
                            while ((command = in.readLine()) != null) {
                                Log.d("ServerActivity", command);
                                mHandler.post(new ProcessCommandRunnable(command));
                            }
                        } catch (Exception e) {
                            Log.d(mTAG, "Oops. Connection interrupted. Please reconnect your phones.");
                            e.printStackTrace();
                        }
                    }
                } else {
                    Log.d(mTAG, "Couldn't detect internet connection.");
                }

            } catch (final Exception e) {
                Log.d(mTAG, "Error" + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    private void processCommand(String command) {
        if (command.equals("webcam")) {
            startIpWebcam();
        } else if (command.equals("vision")) {
            switchToVision();
        } else {
            Log.e(mTAG, "Unknown command" + command);
        }
    }


    private class ProcessCommandRunnable implements Runnable {
        private final String mCommand;

        ProcessCommandRunnable(final String command) {
            mCommand = command;
        }

        public void run() {
            System.out.println(mCommand);
            if (mCommand.equals("webcam")) {
                startIpWebcam();
            } else if (mCommand.equals("vision")) {
                switchToVision();
            } else {
                Log.e(mTAG, "Unknown command" + mCommand);
            }
        }
    }

    static {
        System.loadLibrary("opencv_java3");
    }
}


/*public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }
    static{ System.loadLibrary("opencv_java3"); }
}*/
