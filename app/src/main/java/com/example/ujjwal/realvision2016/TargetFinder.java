package com.example.ujjwal.realvision2016;

/**
 * Created by Ujjwal on 1/8/17.
 */

import android.os.Environment;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.LineSegmentDetector;
import org.opencv.imgproc.Moments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static android.R.attr.y;
import static android.os.Build.VERSION_CODES.N;
import static java.lang.Math.PI;
import static java.lang.Math.acos;
import static java.lang.Math.sqrt;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.imgproc.Imgproc.contourArea;

public class TargetFinder {
    public static double CAM_UP_ANGLE = 5; //TODO
    public static double CAM_DOWN_ANGLE = 110;

    public static double RES_X = 640;
    public static double RES_Y = 640;

    public static double WIDTH_TARGET = 15.0; //in
    public static double STANDARD_VIEW_ANGLE = 0.454885;//0.9424778; //radians, for an Axis Camera 206 /////...54 degrees

    public static double MAX_Y_COORD = 800; ///RES_Y / 1.15; //293 //TODO find the actual angle of camera and the corresponding max y coord
    public static double MIN_Y_COORD = 700; //RES_Y / 4.3; //70

    public static double X_TARGET = 160;
    public static double K_PIX = 1.0 / 400;

    public static Center NO_CENTER = new Center(-1, -1);

    public static final double RATIO1 = 3.0;
    public static final double RATIO2 = 6.0;
    public static final double RATIO_ERROR_RATE = 0.1;
    public static final double ERROR_RATE_COORDS = 10;

    public static double POS_1_CAM_X = 160; //pixels
    public static double POS_2_CAM_X = 160; //pixels
    public static double POS_3_CAM_X = 160; //pixels
    public static double POS_4_CAM_X = 160; //pixels
    public static double POS_5_CAM_X = 160; //pixels

    public Mat hierarchy;
    public List<MatOfPoint> contours;
    public Mat finalImage;
    public Mat dilate, erode;

    Rect roi;

    //holds the hsv constants (enables faster editing through adb shell)
    static String HSVFileName = "hsv.txt";
    static String TAG = "TargetFinder";

    //Outputs
    private ArrayList<Line> findLinesOutput = new ArrayList<Line>();
    private ArrayList<Line> filterLinesOutput = new ArrayList<Line>();
    private Mat rgbThresholdOutput = new Mat();
    private Mat blurOutput = new Mat();
    private Mat cvDilateOutput = new Mat();
    private Mat hslThresholdOutput = new Mat();
    private ArrayList<MatOfPoint> findContoursOutput = new ArrayList<MatOfPoint>();
    private ArrayList<MatOfPoint> filterContoursOutput = new ArrayList<MatOfPoint>();
    private ArrayList<MatOfPoint> convexHullsOutput = new ArrayList<MatOfPoint>();

    /*static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }*/

    public TargetFinder() {
        dilate = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE, new Size(3, 3));
        erode = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE, new Size(3, 3));
    }

    //old method

    /**
     * Algorithm:
     * <p>
     * calls performThresh (see below)
     * find contours
     * find the largest object that is in the frame & is greater than the max size and in the upper half of the frame
     *
     * @return Center found (NO_CENTER if none found)
     */

    public Center findOneRetroTarget(Mat image) {

        Center center = new Center(-1, -1); //default
        Rect r1 = new Rect();


        ///
        finalImage = performThresh(image);
        /************/
        //CONTOURS AND OBJECT DETECTION
        contours = new ArrayList<>();
        hierarchy = new Mat();

        Moments mu;

        // find contours
        Imgproc.findContours(finalImage, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        boolean noValid = true;
        // if any contour exist...
        if (hierarchy.size().height > 0 && hierarchy.size().width > 0) {
            int largest = 0;

            // for each remaining contour, find the biggest
            for (int i = 0; i < contours.size(); i++) {
                double area = contourArea(contours.get(i));
                mu = Imgproc.moments(contours.get(i));
                double y_coord = mu.get_m01() / mu.get_m00();
                //greater than min size AND in the upper part of photo AND greater than the last biggest
                if (area > 20.0 && y_coord < MAX_Y_COORD && area >= contourArea(contours.get(largest))) {
                    noValid = false;
                    largest = i;
                    //NetworkTable tab = NetworkTable.getTable("Obj " + i);

                    //Center:    mu.m10()/mu.m00() , mu.m01()/mu.m00()

                }
            }

            roi = Imgproc.boundingRect(contours.get(largest));
            mu = Imgproc.moments(contours.get(largest));

            //ASSUME LARGEST is the target, now calc dist

            //old
            double dist = calcDistAxis206(roi.width, WIDTH_TARGET, 320, STANDARD_VIEW_ANGLE);

            center = new Center(mu.get_m10() / mu.get_m00(), mu.get_m01() / mu.get_m00());

        }

        if (noValid) {
            return NO_CENTER;
        } else {
            return center;
        }
    }

    //also old

    /**
     * Algorithm:
     * <p>
     * Gaussian Blur
     * convert to HSV
     * HSV threshold
     * dilate
     * erode
     * dilate
     * Blur again
     * binary threshold to get rid of small elements
     * dilate
     *
     * @return threshed Mat
     */

    public Mat performThresh(Mat image) {
        Mat imageHSV, erode, dilate;
        imageHSV = new Mat();


        //BLUR
        Imgproc.GaussianBlur(image, image, new Size(11, 11), 0);

        //read HSV from text file
        imageHSV = viewHSVFromFile(image);


        //DILATE > ERODE > DILATE
        dilate = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE, new Size(3, 3));
        Imgproc.dilate(imageHSV, imageHSV, dilate);//dilate
        erode = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE, new Size(3, 3));
        Imgproc.erode(imageHSV, imageHSV, erode);
        Imgproc.dilate(imageHSV, imageHSV, dilate);

        //BLUR PART 2
        Imgproc.GaussianBlur(imageHSV, imageHSV, new Size(15, 15), 0);

        //THRESHING
        Imgproc.threshold(imageHSV, imageHSV, 73, 255, Imgproc.THRESH_BINARY);

        //DILATE ONCE MORE
        Imgproc.dilate(imageHSV, imageHSV, dilate);


        erode.release();
        dilate.release();
        System.gc();

        return imageHSV;
    }


    /**
     * @return converted binary Mat based on text file's constants after the HSV thresh
     */

    ///CURRENT BEST: "(33,7,250)->(180,120,255)"
    public Mat viewHSVFromFile(Mat m) {
        Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2HSV);

        //Core.inRange(imageHSV, new Scalar(78, 124, 213), new Scalar(104, 255, 255), imageHSV);

        Scalar[] vals = readElements(HSVFileName);

        if (vals == null) {
            vals = new Scalar[2];
            //            vals[0] = new Scalar(60, 41, 218);    OLLLDDDDD
            //            vals[1] = new Scalar(94, 255, 255);   OLDDDER
            vals[0] = new Scalar(33, 4, 230);
            vals[1] = new Scalar(85, 45, 255);
            Log.d(TAG, "File read failed ---- defaulting");
        }

        Core.inRange(m, vals[0], vals[1], m);

        return m;
    }

    //reading from text file
    public Scalar[] readElements(String filename) {
        try {
            File root = new File(Environment.getExternalStorageDirectory(), "Vision");
            File filepath = new File(root, filename);  // file path to save
            BufferedReader in = new BufferedReader(new FileReader(filepath));

            String str1 = in.readLine();
            String str2 = in.readLine();
            String[] line1 = str1.split(",");
            String[] line2 = str2.split(",");


            Scalar[] s = new Scalar[]{new Scalar(Integer.parseInt(line1[0]), Integer.parseInt(line1[1]), Integer.parseInt(line1[2]))
                    , new Scalar(Integer.parseInt(line2[0]), Integer.parseInt(line2[1]), Integer.parseInt(line2[2]))};
            Log.d(TAG, "MIN: (" + s[0].val[0] + ", " + s[0].val[1] + ", " + s[0].val[2]
                    + ") MAX: (" + s[1].val[0] + ", " + s[1].val[1] + ", " + s[1].val[2] + ")");
            return s;
        } catch (Exception e) {
            Log.e("Exception", "File READ failed: " + e.toString());
            return null;
        }
    }

    //for testing purposes
    public Mat testingAlgorithm(Mat m) {

        Imgproc.GaussianBlur(m, m, new org.opencv.core.Size(11, 11), 0);

        Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2GRAY);

        Imgproc.threshold(m, m, 245, 255, Imgproc.THRESH_TOZERO);


        return m;

    }

    /**
     * Hough lines alg
     */
    public Map<String, Object> shapeDetectTarget3(Mat m) {
        long _time = Calendar.getInstance().getTimeInMillis();

        Center center = NO_CENTER;
        Mat hsv = new Mat();
        Mat ycrcb = new Mat();

        Imgproc.GaussianBlur(m, m, new org.opencv.core.Size(5, 5), 0);

        Log.d("->timelog", "GAUSSIAN t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();

        HSVThread hsvSide = new HSVThread(m);
        hsvSide.start();
        GrayThread graySide = new GrayThread(m);
        graySide.start();

        try {
            graySide.join();
            hsvSide.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        Log.d("->timelog", "Both threads t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();


        //find overlap
        Mat combined = new Mat();
        try {
            Core.bitwise_and(hsvSide.getFinalMat(), graySide.getFinalMat(), m); //think &&
        } catch (Exception e) {
            return new HashMap<>(); //dun got broked, don' do this
        }

        //Imgproc.dilate(combined, m, dilate); //dilate to be safe

        Mat thresh = new Mat(), subImage = new Mat();
        m.copyTo(thresh); //copy filtered image

        Log.d("->timelog", "Bit AND & dilate t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();

        /*Mat lines = new Mat();
        int threshold = 80;
        int minLineSize = 40;
        int lineGap = 20;

        Imgproc.HoughLinesP(thresh, lines, 1, Math.PI / 2, threshold,
                minLineSize, lineGap);

        ArrayList<Point> corners = new ArrayList<Point>();
        for (int i = 0; i < lines.cols(); i++) {
            for (int j = i + 1; j < lines.cols(); j++) {
                Point pt = computeIntersect(lines.get(0, i), lines.get(0, j));
                if (pt.x >= 0 && pt.y >= 0)
                    corners.add(pt);
            }
        }

        if (corners.size() < 4) {
            //return "";
        }

        Point centerNew = new Point(0, 0);
        // Get mass center
        for (int i = 0; i < corners.size(); i++) {
            centerNew.x += corners.get(i).x;
            centerNew.y += corners.get(i).y;
        }
        centerNew.x = (centerNew.x / corners.size());
        centerNew.y = (centerNew.y / corners.size());

        ArrayList<Point> top = new ArrayList<Point>();
        ArrayList<Point> bottom = new ArrayList<Point>();

        for (int i = 0; i < corners.size(); i++) {
            if (centerNew.y > corners.get(i).y) {

                top.add(corners.get(i));

            } else {
                bottom.add(corners.get(i));

            }

        }
        if (top.size() > 0 && bottom.size() > 0) {
            double largest_top_x = top.get(0).x;
            double smallest_top_x = top.get(0).x;
            double largest_top_y = top.get(0).y;
            double smallest_top_y = top.get(0).y;

            int min_top, max_top, min_bottom, max_bottom;

            double largest_bottom_x = bottom.get(0).x;
            double smallest_bottom_x = bottom.get(0).x;
            double largest_bottom_y = bottom.get(0).y;
            double smallest_bottom_y = bottom.get(0).y;

            for (int i = 0; i < top.size(); i++) {
                if (top.get(i).x > largest_top_x) {
                    largest_top_x = top.get(i).x;
                    min_top = i;
                } else if (top.get(i).x < smallest_top_x) {
                    smallest_top_x = top.get(i).x;
                    max_top = i;
                }

                if (top.get(i).y > largest_top_y) {
                    largest_top_y = top.get(i).y;
                    min_top = i;
                } else if (top.get(i).y < smallest_top_y) {
                    smallest_top_y = top.get(i).y;
                    max_top = i;
                }

            }

            for (int i = 0; i < bottom.size(); i++) {
                if (bottom.get(i).x > largest_bottom_x) {
                    largest_bottom_x = bottom.get(i).x;
                    min_top = i;
                } else if (bottom.get(i).x < smallest_bottom_x) {
                    smallest_bottom_x = bottom.get(i).x;
                    max_top = i;
                }

                if (bottom.get(i).y > largest_bottom_y) {
                    largest_bottom_y = bottom.get(i).y;
                    min_top = i;
                } else if (bottom.get(i).y < smallest_bottom_y) {
                    smallest_bottom_y = bottom.get(i).y;
                    max_top = i;
                }

            }
            center = new Center(centerNew.x, centerNew.y);
            //Core.circle(untouched, centerNew, 20, new Scalar(255, 0, 0), 5); // p1
        }*/
        //TODO Change the values here in order to create the real hashmap(and get this to work)
        return new HashMap<String, Object>();
    }

    public Map<String, Object> shapeDetectTarget10(Mat m) {
        Mat test = m;
        Mat dst = new Mat();
        Imgproc.GaussianBlur(test, dst, new Size(1, 1), 23, 11);
        Imgproc.dilate(dst, test, new Mat(), new Point(-1, -1), 28, Core.BORDER_DEFAULT, new Scalar(-1));
        double[] hslThresholdHue = {63.12949640287769, 104.27505995165401};
        double[] hslThresholdSaturation = {98.44632768361582, 255.0};
        double[] hslThresholdLuminance = {43.57014388489208, 241.43617021276594};
        hslThreshold(test, hslThresholdHue, hslThresholdSaturation, hslThresholdLuminance, dst);
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(dst, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        // Step Filter_Contours0:
        ArrayList<MatOfPoint> filterContoursContours = findContoursOutput;
        double filterContoursMinArea = 100;
        double filterContoursMinPerimeter = 0;
        double filterContoursMinWidth = 0;
        double filterContoursMaxWidth = 1000;
        double filterContoursMinHeight = 0;
        double filterContoursMaxHeight = 1000;
        double[] filterContoursSolidity = {0, 100};
        double filterContoursMaxVertices = 1000000;
        double filterContoursMinVertices = 0;
        double filterContoursMinRatio = 0;
        double filterContoursMaxRatio = 1000;
        filterContours(filterContoursContours, filterContoursMinArea, filterContoursMinPerimeter, filterContoursMinWidth,
                filterContoursMaxWidth, filterContoursMinHeight, filterContoursMaxHeight, filterContoursSolidity,
                filterContoursMaxVertices, filterContoursMinVertices, filterContoursMinRatio, filterContoursMaxRatio, filterContoursOutput);

        List<MatOfPoint> outputContours = filterContoursOutput;
        final MatOfInt hull = new MatOfInt();
        outputContours.clear();
        for (int i = 0; i < contours.size(); i++) {
            final MatOfPoint contour = contours.get(i);
            final MatOfPoint mopHull = new MatOfPoint();
            Imgproc.convexHull(contour, hull);
            mopHull.create((int) hull.size().height, 1, CvType.CV_32SC2);
            for (int j = 0; j < hull.size().height; j++) {
                int index = (int) hull.get(j, 0)[0];
                double[] point = new double[]{contour.get(index, 0)[0], contour.get(index, 0)[1]};
                mopHull.put(j, 0, point);
            }
            outputContours.add(mopHull);
        }

        Log.d("# of contours", "" + outputContours.size());
        if (outputContours.size() == 1) {
            MatOfPoint2f matOfPoint2f = new MatOfPoint2f();
            Imgproc.approxPolyDP(new MatOfPoint2f(outputContours.get(0).toArray()), matOfPoint2f, 8.0, true);
            Rect roi = Imgproc.boundingRect(new MatOfPoint(matOfPoint2f.toArray()));
            Point br = roi.br();
            Point tl = roi.tl();
            Center mid = new Center((br.x + tl.x) / 2, (tl.y + br.y) / 2);
            Map<String, Object> foundElements = new HashMap<>();
            foundElements.put("center", mid);
            foundElements.put("roi", roi);
            double dist = calcDistAxis206(roi.width, WIDTH_TARGET, 320, STANDARD_VIEW_ANGLE);
            foundElements.put("dist", dist);
            return foundElements;
        } else {
            Map<String, Object> fE = new HashMap<>();
            fE.put("center", NO_CENTER);
            fE.put("roi", new Rect());
            fE.put("dist", 0.0);
            return fE;
        }
    }

    //Also return the distance
    public Map<String, Object> lightAlg(Mat m) {
        // Step CV_dilate0:
        Mat cvDilateSrc = m;
        Mat cvDilateKernel = new Mat();
        Point cvDilateAnchor = new Point(-1, -1);
        double cvDilateIterations = 9.0;
        int cvDilateBordertype = Core.BORDER_CONSTANT;
        Scalar cvDilateBordervalue = new Scalar(-1);
        cvDilate(cvDilateSrc, cvDilateKernel, cvDilateAnchor, cvDilateIterations, cvDilateBordertype, cvDilateBordervalue, cvDilateOutput);

        // Step HSL_Threshold0:
        Mat hslThresholdInput = cvDilateOutput;
        double[] hslThresholdHue = {0.0, 180.0};
        double[] hslThresholdSaturation = {206.38489208633095, 255.0};
        double[] hslThresholdLuminance = {181.16007194244605, 237.68251273344652};
        hslThreshold(hslThresholdInput, hslThresholdHue, hslThresholdSaturation, hslThresholdLuminance, hslThresholdOutput);

        // Step Find_Contours0:
        Mat findContoursInput = hslThresholdOutput;
        boolean findContoursExternalOnly = false;
        findContours(findContoursInput, findContoursExternalOnly, findContoursOutput);

        // Step Filter_Contours0:
        ArrayList<MatOfPoint> filterContoursContours = findContoursOutput;
        double filterContoursMinArea = 1500.0;
        double filterContoursMinPerimeter = 200.0;
        double filterContoursMinWidth = 100;
        double filterContoursMaxWidth = 500;
        double filterContoursMinHeight = 100;
        double filterContoursMaxHeight = 500;
        double[] filterContoursSolidity = {0, 100};
        double filterContoursMaxVertices = 1000000;
        double filterContoursMinVertices = 0;
        double filterContoursMinRatio = 0;
        double filterContoursMaxRatio = 1000;
        filterContours(filterContoursContours, filterContoursMinArea, filterContoursMinPerimeter, filterContoursMinWidth, filterContoursMaxWidth, filterContoursMinHeight, filterContoursMaxHeight, filterContoursSolidity, filterContoursMaxVertices, filterContoursMinVertices, filterContoursMinRatio, filterContoursMaxRatio, filterContoursOutput);

        /* New Values that might work
        *double filterContoursMinArea = 2000.0;
        double filterContoursMinPerimeter = 250.0;
        double filterContoursMinWidth = 50;
        double filterContoursMaxWidth = 1000;
        double filterContoursMinHeight = 50;
        double filterContoursMaxHeight = 1000;
        double[] filterContoursSolidity = {0, 100};
        double filterContoursMaxVertices = 1000000;
        double filterContoursMinVertices = 0;
        double filterContoursMinRatio = 0;
        double filterContoursMaxRatio = 1000;*/


        ArrayList<MatOfPoint> convexHullsContours = filterContoursOutput;
        convexHulls(convexHullsContours, convexHullsOutput);

        ArrayList<MatOfPoint> conts = convexHullsOutput;
        Log.d(TAG, "Contours: " + conts.size());
        if (conts.size() == 1 /*|| conts.size() == 2*/) {
            Rect roi2 = Imgproc.boundingRect(conts.get(0));
            Center c = new Center((roi2.br().x + roi2.tl().x) / 2, (roi2.br().y + roi2.tl().y) / 2);
            Map<String, Object> fE = new HashMap<>();
            double dist = calcDistAxis206(roi2.width, WIDTH_TARGET, 320, STANDARD_VIEW_ANGLE);
            fE.put("center", c);
            fE.put("roi", roi2);
            fE.put("dist", dist);
            return fE;
        } else if (conts.size() == 0) {
            Map<String, Object> fE = new HashMap<>();
            fE.put("center", NO_CENTER);
            fE.put("roi", new Rect());
            fE.put("dist", 0.0);
            return fE;
        } else {
            Map<Integer, Double> vals = new HashMap<>();
            double finalArea = Double.MAX_VALUE;
            int finalAreaNum = -1;
            for (int i = 0; i < conts.size(); i++) {
                MatOfPoint contour = conts.get(i);
                Rect rect = Imgproc.boundingRect(contour);
                double area = Imgproc.contourArea(contour);
                double rectArea = rect.area();
                //vals.put(i, Math.abs(rectArea - area));
                if (Math.abs(rectArea - area) < finalArea) {
                    finalArea = Math.abs(rectArea - area);
                    finalAreaNum = i;
                }
            }

            Rect roi2 = Imgproc.boundingRect(conts.get(finalAreaNum));
            Center c = new Center((roi2.br().x + roi2.tl().x) / 2, (roi2.br().y + roi2.tl().y) / 2);
            Map<String, Object> fE = new HashMap<>();
            double dist = calcDistAxis206(roi2.width, WIDTH_TARGET, 320, STANDARD_VIEW_ANGLE);
            fE.put("center", c);
            fE.put("roi", roi2);
            fE.put("dist", dist);
            return fE;
        }
    }

    /**
     * This is the primary method that runs the entire pipeline and updates the outputs.
     */
    //Dark alg
    //Also return distance
    public Map<String, Object> darkAlg(Mat source0) {
        // Step CV_dilate0:
        Mat cvDilateSrc = source0;
        Mat cvDilateKernel = new Mat();
        Point cvDilateAnchor = new Point(-1, -1);
        double cvDilateIterations = 25.0;
        int cvDilateBordertype = Core.BORDER_CONSTANT;
        Scalar cvDilateBordervalue = new Scalar(-1);
        cvDilate(cvDilateSrc, cvDilateKernel, cvDilateAnchor, cvDilateIterations, cvDilateBordertype, cvDilateBordervalue, cvDilateOutput);

        // Step HSL_Threshold0:
        Mat hslThresholdInput = cvDilateOutput;
        double[] hslThresholdHue = {0.0, 151.27659574468086};
        double[] hslThresholdSaturation = {127.25988700564972, 255.0};
        double[] hslThresholdLuminance = {144.0677966101695, 255.0};
        hslThreshold(hslThresholdInput, hslThresholdHue, hslThresholdSaturation, hslThresholdLuminance, hslThresholdOutput);

        /*Mat findContoursInput = hslThresholdOutput;
        HSVThread2 hsvSide = new HSVThread2(findContoursInput);
        hsvSide.start();
        GrayThread2 graySide = new GrayThread2(findContoursInput);
        graySide.start();

        try {
            graySide.join();
            hsvSide.join();
        } catch (InterruptedException e){
            e.printStackTrace();
        }

        //find overlap
        Mat combined = new Mat();
        try {
            Core.bitwise_and(hsvSide.getFinalMat(), graySide.getFinalMat(), findContoursInput); //think &&
        }
        catch(Exception e){
            return new HashMap<>(); //dun got broked, don' do this
        }*/


        // Step Find_Contours0:
        Mat findContoursInput = hslThresholdOutput;
        //Imgproc.cvtColor(findContoursInput, findContoursInput, Imgproc.COLOR_HSV2BGR);
        boolean findContoursExternalOnly = false;
        findContours(findContoursInput, findContoursExternalOnly, findContoursOutput);

        // Step Filter_Contours0:
        ArrayList<MatOfPoint> filterContoursContours = findContoursOutput;

        Log.d(TAG, "Contours b4: " + filterContoursContours.size());

        double filterContoursMinArea = 700;
        double filterContoursMinPerimeter = 250;
        double filterContoursMinWidth = 50;
        double filterContoursMaxWidth = 1000;
        double filterContoursMinHeight = 50;
        double filterContoursMaxHeight = 1000;
        double[] filterContoursSolidity = {0, 100};
        double filterContoursMaxVertices = 1000000;
        double filterContoursMinVertices = 0;
        double filterContoursMinRatio = 0;
        double filterContoursMaxRatio = 1000;
        filterContours(filterContoursContours, filterContoursMinArea, filterContoursMinPerimeter, filterContoursMinWidth, filterContoursMaxWidth, filterContoursMinHeight, filterContoursMaxHeight, filterContoursSolidity, filterContoursMaxVertices, filterContoursMinVertices, filterContoursMinRatio, filterContoursMaxRatio, filterContoursOutput);

        // Step Convex_Hulls0:
        ArrayList<MatOfPoint> convexHullsContours = filterContoursOutput;
        convexHulls(convexHullsContours, convexHullsOutput);

        /*// Step HSL_Threshold0:
        Mat hslThresholdInput = source0;
        double[] hslThresholdHue = {84.0, 180.0};
        double[] hslThresholdSaturation = {13.758992805755396, 79.660441426146};
        double[] hslThresholdLuminance = {173.87894768930616, 255.0};
        hslThreshold(hslThresholdInput, hslThresholdHue, hslThresholdSaturation, hslThresholdLuminance, hslThresholdOutput);

        // Step Find_Contours0:
        Mat findContoursInput = hslThresholdOutput;
        boolean findContoursExternalOnly = false;
        findContours(findContoursInput, findContoursExternalOnly, findContoursOutput);

        // Step Filter_Contours0:
        ArrayList<MatOfPoint> filterContoursContours = findContoursOutput;
        double filterContoursMinArea = 600.0;
        double filterContoursMinPerimeter = 400.0;
        double filterContoursMinWidth = 50.0;
        double filterContoursMaxWidth = 200.0;
        double filterContoursMinHeight = 100.0;
        double filterContoursMaxHeight = 1000;
        double[] filterContoursSolidity = {0, 100};
        double filterContoursMaxVertices = 1000000;
        double filterContoursMinVertices = 0;
        double filterContoursMinRatio = 0;
        double filterContoursMaxRatio = 1000;
        filterContours(filterContoursContours, filterContoursMinArea, filterContoursMinPerimeter, filterContoursMinWidth, filterContoursMaxWidth, filterContoursMinHeight, filterContoursMaxHeight, filterContoursSolidity, filterContoursMaxVertices, filterContoursMinVertices, filterContoursMinRatio, filterContoursMaxRatio, filterContoursOutput);

        // Step Convex_Hulls0:
        ArrayList<MatOfPoint> convexHullsContours = filterContoursOutput;
        convexHulls(convexHullsContours, convexHullsOutput);*/

        ArrayList<MatOfPoint> conts = convexHullsOutput;
        Log.d(TAG, "Contours: " + conts.size());
        if (conts.size() == 1 /*|| conts.size() == 2*/) {
            Rect roi2 = Imgproc.boundingRect(conts.get(0));
            Center c = new Center((roi2.br().x + roi2.tl().x) / 2, (roi2.br().y + roi2.tl().y) / 2);
            Map<String, Object> fE = new HashMap<>();
            double dist = calcDistAxis206(roi2.width, WIDTH_TARGET, 320, STANDARD_VIEW_ANGLE);
            fE.put("center", c);
            fE.put("roi", roi2);
            fE.put("dist", dist);
            return fE;
        } else if (conts.size() == 0) {
            Map<String, Object> fE = new HashMap<>();
            fE.put("center", NO_CENTER);
            fE.put("roi", new Rect());
            fE.put("dist", 0.0);
            return fE;
        } else {
            Map<Integer, Double> vals = new HashMap<>();
            double finalArea = Double.MAX_VALUE;
            int finalAreaNum = -1;
            for (int i = 0; i < conts.size(); i++) {
                MatOfPoint contour = conts.get(i);
                Rect rect = Imgproc.boundingRect(contour);
                double area = Imgproc.contourArea(contour);
                double rectArea = rect.area();
                //vals.put(i, Math.abs(rectArea - area));
                if (Math.abs(rectArea - area) < finalArea) {
                    finalArea = Math.abs(rectArea - area);
                    finalAreaNum = i;
                }
            }

            Rect roi2 = Imgproc.boundingRect(conts.get(finalAreaNum));
            Center c = new Center((roi2.br().x + roi2.tl().x) / 2, (roi2.br().y + roi2.tl().y) / 2);
            Map<String, Object> fE = new HashMap<>();
            double dist = calcDistAxis206(roi2.width, WIDTH_TARGET, 320, STANDARD_VIEW_ANGLE);
            fE.put("center", c);
            fE.put("roi", roi2);
            fE.put("dist", dist);
            return fE;
        }
    }

    /**
     * This method is a generated getter for the output of a RGB_Threshold.
     * @return Mat output from RGB_Threshold.
     */
    public Mat rgbThresholdOutput() {
        return rgbThresholdOutput;
    }

    /**
     * Segment an image based on color ranges.
     * @param input The image on which to perform the RGB threshold.
     * @param red The min and max red.
     * @param green The min and max green.
     * @param blue The min and max blue.
     * @param out The image in which to store the output.
     */
    private void rgbThreshold(Mat input, double[] red, double[] green, double[] blue,
                              Mat out) {
        Imgproc.cvtColor(input, out, Imgproc.COLOR_BGR2RGB);
        Core.inRange(out, new Scalar(red[0], green[0], blue[0]),
                new Scalar(red[1], green[1], blue[1]), out);
    }

    /**
     * This method is a generated getter for the output of a CV_dilate.
     *
     * @return Mat output from CV_dilate.
     */
    public Mat cvDilateOutput() {
        return cvDilateOutput;
    }

    /**
     * This method is a generated getter for the output of a HSL_Threshold.
     *
     * @return Mat output from HSL_Threshold.
     */
    public Mat hslThresholdOutput() {
        return hslThresholdOutput;
    }

    /**
     * This method is a generated getter for the output of a Find_Contours.
     *
     * @return ArrayList<MatOfPoint> output from Find_Contours.
     */
    public ArrayList<MatOfPoint> findContoursOutput() {
        return findContoursOutput;
    }

    /**
     * This method is a generated getter for the output of a Filter_Contours.
     *
     * @return ArrayList<MatOfPoint> output from Filter_Contours.
     */
    public ArrayList<MatOfPoint> filterContoursOutput() {
        return filterContoursOutput;
    }

    /**
     * This method is a generated getter for the output of a Convex_Hulls.
     *
     * @return ArrayList<MatOfPoint> output from Convex_Hulls.
     */
    public ArrayList<MatOfPoint> convexHullsOutput() {
        return convexHullsOutput;
    }

    /**
     * An indication of which type of filter to use for a blur.
     * Choices are BOX, GAUSSIAN, MEDIAN, and BILATERAL
     */
    enum BlurType {
        BOX("Box Blur"), GAUSSIAN("Gaussian Blur"), MEDIAN("Median Filter"),
        BILATERAL("Bilateral Filter");

        private final String label;

        BlurType(String label) {
            this.label = label;
        }

        public static BlurType get(String type) {
            if (BILATERAL.label.equals(type)) {
                return BILATERAL;
            } else if (GAUSSIAN.label.equals(type)) {
                return GAUSSIAN;
            } else if (MEDIAN.label.equals(type)) {
                return MEDIAN;
            } else {
                return BOX;
            }
        }

        @Override
        public String toString() {
            return this.label;
        }
    }

    /**
     * Softens an image using one of several filters.
     *
     * @param input        The image on which to perform the blur.
     * @param type         The blurType to perform.
     * @param doubleRadius The radius for the blur.
     * @param output       The image in which to store the output.
     */
    private void blur(Mat input, BlurType type, double doubleRadius,
                      Mat output) {
        int radius = (int) (doubleRadius + 0.5);
        int kernelSize;
        switch (type) {
            case BOX:
                kernelSize = 2 * radius + 1;
                Imgproc.blur(input, output, new Size(kernelSize, kernelSize));
                break;
            case GAUSSIAN:
                kernelSize = 6 * radius + 1;
                Imgproc.GaussianBlur(input, output, new Size(kernelSize, kernelSize), radius);
                break;
            case MEDIAN:
                kernelSize = 2 * radius + 1;
                Imgproc.medianBlur(input, output, kernelSize);
                break;
            case BILATERAL:
                Imgproc.bilateralFilter(input, output, -1, radius, radius);
                break;
        }
    }

    /**
     * Expands area of higher value in an image.
     *
     * @param src         the Image to dilate.
     * @param kernel      the kernel for dilation.
     * @param anchor      the center of the kernel.
     * @param iterations  the number of times to perform the dilation.
     * @param borderType  pixel extrapolation method.
     * @param borderValue value to be used for a constant border.
     * @param dst         Output Image.
     */
    private void cvDilate(Mat src, Mat kernel, Point anchor, double iterations,
                          int borderType, Scalar borderValue, Mat dst) {
        if (kernel == null) {
            kernel = new Mat();
        }
        if (anchor == null) {
            anchor = new Point(-1, -1);
        }
        if (borderValue == null) {
            borderValue = new Scalar(-1);
        }
        Imgproc.dilate(src, dst, kernel, anchor, (int) iterations, borderType, borderValue);
    }

    /**
     * Segment an image based on hue, saturation, and luminance ranges.
     *
     * @param input The image on which to perform the HSL threshold.
     * @param hue   The min and max hue
     * @param sat   The min and max saturation
     * @param lum   The min and max luminance
     * @param out   The image in which to store the output.
     */
    private void hslThreshold(Mat input, double[] hue, double[] sat, double[] lum,
                              Mat out) {
        Imgproc.cvtColor(input, out, Imgproc.COLOR_BGR2HLS);
        Core.inRange(out, new Scalar(hue[0], lum[0], sat[0]),
                new Scalar(hue[1], lum[1], sat[1]), out);
    }

    /**
     * Sets the values of pixels in a binary image to their distance to the nearest black pixel.
     *
     * @param input    The image on which to perform the Distance Transform.
     * @param contours The image in which to store the output.
     */
    private void findContours(Mat input, boolean externalOnly,
                              List<MatOfPoint> contours) {
        Mat hierarchy = new Mat();
        contours.clear();
        int mode;
        if (externalOnly) {
            mode = Imgproc.RETR_EXTERNAL;
        } else {
            if (input.type() == CvType.CV_8UC1) {
                mode = Imgproc.RETR_LIST;
            } else if (input.type() == CvType.CV_32SC1) {
                mode = Imgproc.RETR_FLOODFILL;
            } else {
                mode = Imgproc.RETR_LIST;
            }
        }
        int method = Imgproc.CHAIN_APPROX_SIMPLE;
        try {
            Imgproc.findContours(input, contours, hierarchy, mode, method);
        } catch (CvException c) {
            contours = new ArrayList<>();
        }
    }


    /**
     * Filters out contours that do not meet certain criteria.
     *
     * @param inputContours  is the input list of contours
     * @param output         is the the output list of contours
     * @param minArea        is the minimum area of a contour that will be kept
     * @param minPerimeter   is the minimum perimeter of a contour that will be kept
     * @param minWidth       minimum width of a contour
     * @param maxWidth       maximum width
     * @param minHeight      minimum height
     * @param maxHeight      maximimum height
     * @param solidity       the minimum and maximum solidity of a contour
     * @param minVertexCount minimum vertex Count of the contours
     * @param maxVertexCount maximum vertex Count
     * @param minRatio       minimum ratio of width to height
     * @param maxRatio       maximum ratio of width to height
     */
    private void filterContours(List<MatOfPoint> inputContours, double minArea,
                                double minPerimeter, double minWidth, double maxWidth, double minHeight, double
                                        maxHeight, double[] solidity, double maxVertexCount, double minVertexCount, double
                                        minRatio, double maxRatio, List<MatOfPoint> output) {
        final MatOfInt hull = new MatOfInt();
        output.clear();
        //operation
        for (int i = 0; i < inputContours.size(); i++) {
            final MatOfPoint contour = inputContours.get(i);
            final Rect bb = Imgproc.boundingRect(contour);
            if (bb.width < minWidth || bb.width > maxWidth) continue;
            if (bb.height < minHeight || bb.height > maxHeight) continue;
            final double area = Imgproc.contourArea(contour);
            if (area < minArea) continue;
            if (Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true) < minPerimeter)
                continue;
            Imgproc.convexHull(contour, hull);
            MatOfPoint mopHull = new MatOfPoint();
            mopHull.create((int) hull.size().height, 1, CvType.CV_32SC2);
            for (int j = 0; j < hull.size().height; j++) {
                int index = (int) hull.get(j, 0)[0];
                double[] point = new double[]{contour.get(index, 0)[0], contour.get(index, 0)[1]};
                mopHull.put(j, 0, point);
            }
            final double solid = 100 * area / Imgproc.contourArea(mopHull);
            if (solid < solidity[0] || solid > solidity[1]) continue;
            if (contour.rows() < minVertexCount || contour.rows() > maxVertexCount) continue;
            final double ratio = bb.width / (double) bb.height;
            if (ratio < minRatio || ratio > maxRatio) continue;
            output.add(contour);
        }
    }

    /**
     * Compute the convex hulls of contours.
     *
     * @param inputContours  The contours on which to perform the operation.
     * @param outputContours The contours where the output will be stored.
     */
    private void convexHulls(List<MatOfPoint> inputContours,
                             ArrayList<MatOfPoint> outputContours) {
        final MatOfInt hull = new MatOfInt();
        outputContours.clear();
        for (int i = 0; i < inputContours.size(); i++) {
            final MatOfPoint contour = inputContours.get(i);
            final MatOfPoint mopHull = new MatOfPoint();
            Imgproc.convexHull(contour, hull);
            mopHull.create((int) hull.size().height, 1, CvType.CV_32SC2);
            for (int j = 0; j < hull.size().height; j++) {
                int index = (int) hull.get(j, 0)[0];
                double[] point = new double[]{contour.get(index, 0)[0], contour.get(index, 0)[1]};
                mopHull.put(j, 0, point);
            }
            outputContours.add(mopHull);
        }
    }


    /**
     * This is the primary method that runs the entire pipeline and updates the outputs.
     *//*
    public Map<String, Object> shapeDetectTarget10(Mat source0) {
        // Step HSL_Threshold0:
        Mat hslThresholdInput = source0;
        double[] hslThresholdHue = {77.6978417266187, 92.45733788395904};
        double[] hslThresholdSaturation = {171.98741007194243, 255.0};
        double[] hslThresholdLuminance = {43.57014388489208, 255.0};
        hslThreshold(hslThresholdInput, hslThresholdHue, hslThresholdSaturation, hslThresholdLuminance, hslThresholdOutput);

        // Step Find_Contours0:
        Mat findContoursInput = hslThresholdOutput;
        boolean findContoursExternalOnly = false;
        findContours(findContoursInput, findContoursExternalOnly, findContoursOutput);

        // Step Filter_Contours0:
        Log.d("Before filtering cont", "" + findContoursOutput.size());
        ArrayList<MatOfPoint> filterContoursContours = findContoursOutput;
        double filterContoursMinArea = 85.0;
        double filterContoursMinPerimeter = 0.0;
        double filterContoursMinWidth = 0.0;
        double filterContoursMaxWidth = 1000.0;
        double filterContoursMinHeight = 0.0;
        double filterContoursMaxHeight = 1000.0;
        double[] filterContoursSolidity = {0, 100};
        double filterContoursMaxVertices = 1000000.0;
        double filterContoursMinVertices = 0.0;
        double filterContoursMinRatio = 0.0;
        double filterContoursMaxRatio = 1000.0;
        filterContours(filterContoursContours, filterContoursMinArea, filterContoursMinPerimeter, filterContoursMinWidth, filterContoursMaxWidth, filterContoursMinHeight, filterContoursMaxHeight, filterContoursSolidity, filterContoursMaxVertices, filterContoursMinVertices, filterContoursMinRatio, filterContoursMaxRatio, filterContoursOutput);

        // Step Convex_Hulls0:
        ArrayList<MatOfPoint> convexHullsContours = filterContoursOutput;
        *//*Get the low contour, then get the max left and max right points, then get the top contour, and get the min left and min right
        * points, then create rect from those. Use moments alg to do the rest*//*
        //convexHulls(convexHullsContours, convexHullsOutput);

        // Step Find Extreme points
        *//*ArrayList<Double> leftsX = new ArrayList<>();
        double leftPointX = 1080.0;
        double rightPointX = 5.0;
        double leftPointY = 0;
        double rightPointY = 0;
        ArrayList<Double> rightsX = new ArrayList<>();
        ArrayList<Double> leftsY = new ArrayList<>();
        ArrayList<Double> rightsY = new ArrayList<>();
        for(MatOfPoint p : convexHullsContours*//**//*We currently are not doing the convex hulls alg.*//**//*) {
            List<Point> allPoints = p.toList();
            for(Point p2 : allPoints) {
                if(p2.x < leftPointX) {
                    leftPointX = p2.x;
                    leftPointY = p2.y;
                }
            }
            leftsX.add(leftPointX);
            leftsY.add(leftPointY);
            leftPointX = 1080.0;
        }
        for(MatOfPoint p : convexHullsContours*//**//*We currently are not doing the convex hulls alg.*//**//*) {
            List<Point> allPoints = p.toList();
            for(Point p2 : allPoints) {
                if(p2.x > rightPointX) {
                    rightPointX = p2.x;
                    rightPointY = p2.y;
                }
            }
            rightsX.add(rightPointX);
            rightsY.add(rightPointY);
            rightPointX = 5.0;
        }
        if(rightsX.size() == 2 && leftsX.size() == 2) {
            double xLeftBottom = (leftsX.get(0) + leftsX.get(1))/2;
            double xRightBottom = (rightsX.get(0) + rightsX.get(1))/2;
            double yLeft = (leftsY.get(0) + leftsY.get(1))/2;
            double yRight = (rightsY.get(0) + rightsY.get(1))/2;
            double yEnd = (yLeft + yRight)/2;
            double xEnd = (xLeftBottom + xRightBottom)/2;
            Center c = new Center(xEnd, yEnd);
            Rect roi = new Rect(new Point(Math.min(leftsX.get(0), leftsX.get(1)), Math.min(leftsY.get(0), leftsY.get(1))),
                    new Point(Math.max(rightsX.get(0), rightsX.get(1)), Math.max(rightsY.get(0), rightsY.get(1))));
            Map<String, Object> foundElements = new HashMap<>();
            foundElements.put("center", c);
            foundElements.put("roi", roi);
            return foundElements;
        } else {
            Map<String, Object> foundElements = new HashMap<>();
            foundElements.put("center", NO_CENTER);
            foundElements.put("roi", new Rect());
            return foundElements;
        }*//*
        Log.d("TAG", "Come here 537");
        Log.d("TAG", "" + convexHullsContours.size());
        if(convexHullsContours.size() == 2) {
            Log.d("Contours", "Found 2 contours");
            MatOfPoint contour1 = convexHullsContours.get(0);
            MatOfPoint contour2 = convexHullsContours.get(1);
            List<Point> c1PointList = contour1.toList();
            List<Point> c2PointList = contour2.toList();
            Rect roi2;
            if(c1PointList.get(0).y > c2PointList.get(0).y) {
                //Contour 2 is the bottom one
                Point bottomLeft = new Point(Double.MAX_VALUE, 0);
                Point bottomRight = new Point(0, Double.MAX_VALUE);
                Point topLeft = new Point(Double.MAX_VALUE, 0);
                Point topRight = new Point(0, Double.MAX_VALUE);
                for(Point p : c2PointList) {
                    if(p.x < bottomLeft.x) {
                        bottomLeft = p;
                    }
                }
                for(Point p : c2PointList) {
                    if(p.x > bottomRight.x) {
                        bottomRight = p;
                    }
                }
                for(Point p : c1PointList) {
                    if(p.x < topLeft.x) {
                        topLeft = p;
                    }
                }
                for(Point p : c1PointList) {
                    if(p.x > topRight.x) {
                        topRight = p;
                    }
                }
                roi2 = new Rect(bottomLeft, topRight);
            } else {
                //Contour 1 is the bottom one
                Point bottomLeft = new Point(Double.MAX_VALUE, 0);
                Point bottomRight = new Point(0, Double.MAX_VALUE);
                Point topLeft = new Point(Double.MAX_VALUE, 0);
                Point topRight = new Point(0, Double.MAX_VALUE);
                for(Point p : c1PointList) {
                    if(p.x < bottomLeft.x) {
                        bottomLeft = p;
                    }
                }
                for(Point p : c1PointList) {
                    if(p.x > bottomRight.x) {
                        bottomRight = p;
                    }
                }
                for(Point p : c2PointList) {
                    if(p.x < topLeft.x) {
                        topLeft = p;
                    }
                }
                for(Point p : c2PointList) {
                    if(p.x > topRight.x) {
                        topRight = p;
                    }
                }
                roi2 = new Rect(bottomLeft, topRight);
            }

            List<Moments> mu = new ArrayList<Moments>(convexHullsContours.size());
            Map<Double, Double> coords = new LinkedHashMap<>();
            for (int i = 0; i < convexHullsContours.size(); i++) {
                mu.add(i, Imgproc.moments(convexHullsContours.get(i), false));
                Moments p = mu.get(i);
                double x = p.get_m10() / p.get_m00();
                double y = p.get_m01() / p.get_m00();
                coords.put(x, y);
                //Core.circle(rgbaImage, new Point(x, y), 4, new Scalar(255,49,0,255));
            }
            Center mid = centerOfPoints(coords);
            Map<String, Object> foundElements = new HashMap<>();
            foundElements.put("center", mid);
            foundElements.put("roi", roi2);
            return foundElements;
        } else {
            Map<String, Object> foundElements = new HashMap<>();
            foundElements.put("center", NO_CENTER);
            foundElements.put("roi", new Rect());
            return foundElements;
        }
    }

    public Center centerOfPoints(Map<Double, Double> coords) {
        List<Center> centers = new ArrayList<>();
        Center mid;
        for(Double d : coords.keySet()) {
            centers.add(new Center(d, coords.get(d)));
        }
        Center c1 = centers.get(0);
        Center c2 = centers.get(1);
        mid = new Center((c1.x + c2.x)/2, (c1.y + c2.y)/2);
        return mid;
    }

    private static double min(double[] array) {
        double min = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] < min) {
                min = array[i];
            }
        }
        return min;
    }

    *//**
     * This method is a generated getter for the output of a HSL_Threshold.
     * @return Mat output from HSL_Threshold.
     *//*
    public Mat hslThresholdOutput() {
        return hslThresholdOutput;
    }

    *//**
     * This method is a generated getter for the output of a Find_Contours.
     * @return ArrayList<MatOfPoint> output from Find_Contours.
     *//*
    public ArrayList<MatOfPoint> findContoursOutput() {
        return findContoursOutput;
    }

    *//**
     * This method is a generated getter for the output of a Filter_Contours.
     * @return ArrayList<MatOfPoint> output from Filter_Contours.
     *//*
    public ArrayList<MatOfPoint> filterContoursOutput() {
        return filterContoursOutput;
    }

    *//**
     * This method is a generated getter for the output of a Convex_Hulls.
     * @return ArrayList<MatOfPoint> output from Convex_Hulls.
     *//*
    public ArrayList<MatOfPoint> convexHullsOutput() {
        return convexHullsOutput;
    }


    *//**
     * Segment an image based on hue, saturation, and luminance ranges.
     *
     * @param input The image on which to perform the HSL threshold.
     * @param hue The min and max hue
     * @param sat The min and max saturation
     * @param lum The min and max luminance
     * @param out The image in which to store the output.
     *//*
    private void hslThreshold(Mat input, double[] hue, double[] sat, double[] lum,
                              Mat out) {
        Imgproc.cvtColor(input, out, Imgproc.COLOR_BGR2HLS);
        Core.inRange(out, new Scalar(hue[0], lum[0], sat[0]),
                new Scalar(hue[1], lum[1], sat[1]), out);
    }

    *//**
     * Sets the values of pixels in a binary image to their distance to the nearest black pixel.
     * @param input The image on which to perform the Distance Transform.
     *//*
    private void findContours(Mat input, boolean externalOnly,
                              List<MatOfPoint> contours) {
        Mat hierarchy = new Mat();
        contours.clear();
        int mode;
        if (externalOnly) {
            mode = Imgproc.RETR_EXTERNAL;
        }
        else {
            mode = Imgproc.RETR_LIST;
        }
        int method = Imgproc.CHAIN_APPROX_SIMPLE;
        Imgproc.findContours(input, contours, hierarchy, mode, method);
    }


    *//**
     * Filters out contours that do not meet certain criteria.
     * @param inputContours is the input list of contours
     * @param output is the the output list of contours
     * @param minArea is the minimum area of a contour that will be kept
     * @param minPerimeter is the minimum perimeter of a contour that will be kept
     * @param minWidth minimum width of a contour
     * @param maxWidth maximum width
     * @param minHeight minimum height
     * @param maxHeight maximimum height
     * @param solidity the minimum and maximum solidity of a contour
     * @param minVertexCount minimum vertex Count of the contours
     * @param maxVertexCount maximum vertex Count
     * @param minRatio minimum ratio of width to height
     * @param maxRatio maximum ratio of width to height
     *//*
    private void filterContours(List<MatOfPoint> inputContours, double minArea,
                                double minPerimeter, double minWidth, double maxWidth, double minHeight, double
                                        maxHeight, double[] solidity, double maxVertexCount, double minVertexCount, double
                                        minRatio, double maxRatio, List<MatOfPoint> output) {
        final MatOfInt hull = new MatOfInt();
        output.clear();
        //operation
        for (int i = 0; i < inputContours.size(); i++) {
            final MatOfPoint contour = inputContours.get(i);
            final Rect bb = Imgproc.boundingRect(contour);
            if (bb.width < minWidth || bb.width > maxWidth) continue;
            if (bb.height < minHeight || bb.height > maxHeight) continue;
            final double area = Imgproc.contourArea(contour);
            if (area < minArea) continue;
            if (Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true) < minPerimeter) continue;
            Imgproc.convexHull(contour, hull);
            MatOfPoint mopHull = new MatOfPoint();
            mopHull.create((int) hull.size().height, 1, CvType.CV_32SC2);
            for (int j = 0; j < hull.size().height; j++) {
                int index = (int)hull.get(j, 0)[0];
                double[] point = new double[] { contour.get(index, 0)[0], contour.get(index, 0)[1]};
                mopHull.put(j, 0, point);
            }
            final double solid = 100 * area / Imgproc.contourArea(mopHull);
            if (solid < solidity[0] || solid > solidity[1]) continue;
            if (contour.rows() < minVertexCount || contour.rows() > maxVertexCount)	continue;
            final double ratio = bb.width / (double)bb.height;
            if (ratio < minRatio || ratio > maxRatio) continue;
            output.add(contour);
        }
    }

    */

    /**
     * Compute the convex hulls of contours:
     *//*
    private void convexHulls(List<MatOfPoint> inputContours,
                             ArrayList<MatOfPoint> outputContours) {
        final MatOfInt hull = new MatOfInt();
        outputContours.clear();
        for (int i = 0; i < inputContours.size(); i++) {
            final MatOfPoint contour = inputContours.get(i);
            final MatOfPoint mopHull = new MatOfPoint();
            Imgproc.convexHull(contour, hull);
            mopHull.create((int) hull.size().height, 1, CvType.CV_32SC2);
            for (int j = 0; j < hull.size().height; j++) {
                int index = (int) hull.get(j, 0)[0];
                double[] point = new double[] {contour.get(index, 0)[0], contour.get(index, 0)[1]};
                mopHull.put(j, 0, point);
            }
            outputContours.add(mopHull);
        }
    }*/
    public Map<String, Object> shapeDetectTarget5(Mat m) {
        long _time = Calendar.getInstance().getTimeInMillis();

        Center center = NO_CENTER;
        Mat hsv = new Mat();
        Mat ycrcb = new Mat();

        Imgproc.GaussianBlur(m, m, new org.opencv.core.Size(5, 5), 0);

        //Imgproc.medianBlur(m, m, 11);

        Log.d("->timelog", "GAUSSIAN t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        List<MatOfInt> hull = new ArrayList<MatOfInt>();


        //Imgproc.cvtColor(m, ycrcb, Imgproc.COLOR_RGB2YCrCb);
//        Imgproc.cvtColor(m, ycrcb, Imgproc.COLOR_RGB2Luv);
//        Scalar[] scalars = readElements("Lab.txt");
//        Log.d(TAG, scalars.toString());
//        Core.inRange(ycrcb, scalars[0], scalars[1],ycrcb);

        //Core.inRange(hsv, new Scalar(25, 2, 180), new Scalar(120, 255, 255), hsv); //relatively loose
//        //TROLL NOT ACTUALLY HSV HAHA
//        Imgproc.cvtColor(m, hsv, Imgproc.COLOR_RGB2YCrCb);
//        Scalar[] scalars = readElements("ycrcb.txt");
//        Core.inRange(hsv, scalars[0], scalars[1],hsv);

        HSVThread hsvSide = new HSVThread(m);
        hsvSide.start();
        GrayThread graySide = new GrayThread(m);
        graySide.start();

        try {
            graySide.join();
            hsvSide.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        Log.d("->timelog", "Both threads t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();


        //find overlap
        Mat combined = new Mat();
        try {
            Core.bitwise_and(hsvSide.getFinalMat(), graySide.getFinalMat(), m); //think &&
        } catch (Exception e) {
            return new HashMap<>(); //dun got broked, don' do this
        }

        //Imgproc.dilate(combined, m, dilate); //dilate to be safe

        Mat thresh = new Mat(), subImage = new Mat();
        m.copyTo(thresh); //copy filtered image

        Log.d("->timelog", "Bit AND & dilate t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();

        Mat hierarchy = new Mat();
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Log.d("->timelog", "Get contours: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();

        List<Double> differences = new ArrayList<Double>();
        List<Rect> boundingRects = new ArrayList<Rect>();

        for (MatOfPoint contour : contours) {
            Rect bounding = new Rect();
            bounding = Imgproc.boundingRect(contour);
            double contourArea = Imgproc.contourArea(contour);
            double rectArea = bounding.area();
            differences.add(Math.abs(contourArea - rectArea));
            boundingRects.add(bounding);
        }

        Log.d("->timelog", "Found differences in rectangle sizes " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();

        double min = Double.MAX_VALUE;
        int index = -1;
        for (int i = 0; i < differences.size(); i++) {
            if (differences.get(i) < min) {
                min = differences.get(i);
                index = i;
            }
        }

        Log.d("->timelog", "Found smallest difference" + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();

        Map<String, Object> foundElements = new HashMap<>();
        Rect finalRect = boundingRects.get(index);
        foundElements.put("center", new Center((finalRect.tl().x + finalRect.br().x) / 2, (finalRect.tl().x + finalRect.br().x) / 2));
        foundElements.put("roi", finalRect);
        return foundElements;
    }

    /**
     * This is the primary method that runs the entire pipeline and updates the outputs.
     *//*
    public Map<String, Object> shapeDetectTarget(Mat source0) {
        // Step CV_erode0:
        Mat cvErodeSrc = source0;
        Mat cvErodeKernel = new Mat();
        Point cvErodeAnchor = new Point(-1, -1);
        double cvErodeIterations = 8.0;
        int cvErodeBordertype = Core.BORDER_REFLECT;
        Scalar cvErodeBordervalue = new Scalar(-1);
        cvErode(cvErodeSrc, cvErodeKernel, cvErodeAnchor, cvErodeIterations, cvErodeBordertype, cvErodeBordervalue, cvErodeOutput);

        // Step RGB_Threshold0:
        Mat rgbThresholdInput = cvErodeOutput;
        double[] rgbThresholdRed = {177.57133276429704, 255.0};
        double[] rgbThresholdGreen = {0.0, 255.0};
        double[] rgbThresholdBlue = {0.0, 255.0};
        rgbThreshold(rgbThresholdInput, rgbThresholdRed, rgbThresholdGreen, rgbThresholdBlue, rgbThresholdOutput);

        // Step Find_Lines0:
        *//*Mat findLinesInput = rgbThresholdOutput;
        findLines(findLinesInput, findLinesOutput);*//*

        // Step Find_Contours0:
        Mat findContoursInput = rgbThresholdOutput;
        boolean findContoursExternalOnly = true;
        findContours(findContoursInput, findContoursExternalOnly, findContoursOutput);

        // Step Filter_Lines0:
        *//*ArrayList<Line> filterLinesLines = findLinesOutput;
        double filterLinesMinLength = 12.0;
        double[] filterLinesAngle = {0.0, 360.0};
        filterLines(filterLinesLines, filterLinesMinLength, filterLinesAngle, filterLinesOutput);*//*

        // Step Filter_Contours0:
        ArrayList<MatOfPoint> filterContoursContours = findContoursOutput;
        double filterContoursMinArea = 50.0;
        double filterContoursMinPerimeter = 75.0;
        double filterContoursMinWidth = 300.0;
        double filterContoursMaxWidth = 20000.0;
        double filterContoursMinHeight = 0;
        double filterContoursMaxHeight = 20000.0;
        double[] filterContoursSolidity = {0.0, 100.0};
        double filterContoursMaxVertices = 1000000;
        double filterContoursMinVertices = 0;
        double filterContoursMinRatio = 0;
        double filterContoursMaxRatio = 1000;
        filterContours(filterContoursContours, filterContoursMinArea, filterContoursMinPerimeter, filterContoursMinWidth, filterContoursMaxWidth, filterContoursMinHeight, filterContoursMaxHeight, filterContoursSolidity, filterContoursMaxVertices, filterContoursMinVertices, filterContoursMinRatio, filterContoursMaxRatio, filterContoursOutput);

        // Step Convex_Hulls0:
        ArrayList<MatOfPoint> convexHullsContours = filterContoursOutput;
        convexHulls(convexHullsContours, convexHullsOutput);

        // Step Get Vertices0:
        ArrayList<Point> newConvexHull = new ArrayList<>();
        for(MatOfPoint point : convexHullsOutput) {
            for(Point p : point.toList()) {
                if(p.y > MIN_Y_COORD && p.y < MAX_Y_COORD) {
                    newConvexHull.add(p);
                }
            }
        }
        ArrayList<Point> inLine1 = new ArrayList<>();
        ArrayList<Point> inLine2 = new ArrayList<>();
        if(newConvexHull.size() == 4) {
            int inline = -1;
            for(int i = 1; i < 4; i++) {
                Point curr = newConvexHull.get(0);
                if((curr.x - ERROR_RATE_COORDS < newConvexHull.get(i).x) && (curr.x + ERROR_RATE_COORDS > newConvexHull.get(i).x)){
                    inLine1.add(curr);
                    inLine1.add(newConvexHull.get(i));
                    inline = i;
                }
            }
            if(inline == 1) {
                inLine2.add(newConvexHull.get(2));
                inLine2.add(newConvexHull.get(3));
            } else if(inline == 2) {
                inLine2.add(newConvexHull.get(1));
                inLine2.add(newConvexHull.get(3));
            } else {
                inLine2.add(newConvexHull.get(1));
                inLine2.add(newConvexHull.get(2));
            }

            Center c = new Center(((inLine1.get(0).x + inLine2.get(0).x)/2), ((inLine1.get(0).y + inLine1.get(1).y)/2));
            Map<String, Object> foundElements = new HashMap<>();
            foundElements.put("center", c);
            MatOfPoint mop = new MatOfPoint(newConvexHull.get(0), newConvexHull.get(1), newConvexHull.get(2), newConvexHull.get(3));
            Rect roi = Imgproc.boundingRect(mop);
            foundElements.put("roi", roi);
            return foundElements;
        } else {
            Map<String, Object> foundElements = new HashMap<>();
            foundElements.put("center", NO_CENTER);
            foundElements.put("roi", new Rect());
            return foundElements;
        }
    }



    /**
     * Algorithm:
     *
     *  Gaussian Blur
     *  cvt to gray
     *  threshold (Tozero)
     *      //TODO add erode and dilates and other thresh fcts here
     *  Use canny edge detector to generate mask of edges
     *  Find contours based on new thresh img
     *  Calculate if the contour has 4 edges, and if it is a rectangle
     *  Put all rectangles into an array
     *  Find rectangle that is directly in line/underneath the other to get target
     *  Calculate the center and other features
     *
     * @return
     *
     *  Dictionary of objects
     *  -> "center" Center of found object
     *  -> "roi" Bounding rect of selected object
     */
    public Map<String, Object> shapeDetectTarget4(Mat m) {
        try {
            long _time = Calendar.getInstance().getTimeInMillis();

            Center center = NO_CENTER;
            Mat hsv = new Mat();
            Mat ycrcb = new Mat();

            Imgproc.GaussianBlur(m, m, new org.opencv.core.Size(5, 5), 0);

            Log.d("->timelog", "GAUSSIAN t: " + (Calendar.getInstance().getTimeInMillis() - _time));
            _time = Calendar.getInstance().getTimeInMillis();

            HSVThread hsvSide = new HSVThread(m);
            hsvSide.start();
            GrayThread graySide = new GrayThread(m);
            graySide.start();

            try {
                graySide.join();
                hsvSide.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            Log.d("->timelog", "Both threads t: " + (Calendar.getInstance().getTimeInMillis() - _time));
            _time = Calendar.getInstance().getTimeInMillis();


            //find overlap
            Mat combined = new Mat();
            try {
                Core.bitwise_and(hsvSide.getFinalMat(), graySide.getFinalMat(), m); //think &&
            } catch (Exception e) {
                return new HashMap<>(); //dun got broked, don' do this
            }

            //Imgproc.dilate(combined, m, dilate); //dilate to be safe

            Mat thresh = new Mat(), subImage = new Mat();
            m.copyTo(thresh); //copy filtered image

            Log.d("->timelog", "Bit AND & dilate t: " + (Calendar.getInstance().getTimeInMillis() - _time));
            _time = Calendar.getInstance().getTimeInMillis();

            Mat edges = new Mat();
            //TODO: Change thresholds based on what works
            Imgproc.Canny(thresh, edges, 100, 300);
            Log.d("->timelog", "Using canny edge detection t: " + (Calendar.getInstance().getTimeInMillis() - _time));
            _time = Calendar.getInstance().getTimeInMillis();

            //Imgproc.findContours(m, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            Mat newHierarchy = new Mat();
            List<MatOfPoint> contours2 = new ArrayList<MatOfPoint>();
            Imgproc.findContours(edges, contours2, newHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);


            Log.d("->timelog", "find contours with edges t: " + (Calendar.getInstance().getTimeInMillis() - _time));
            _time = Calendar.getInstance().getTimeInMillis();

            List<MatOfPoint2f> allRectangles = new ArrayList<>();
            List<Rect> enclosingRects = new ArrayList<>();

            MatOfPoint2f matOfPoint2f = new MatOfPoint2f();
            MatOfPoint2f approxCurve = new MatOfPoint2f();

            boolean noValid = true;
            Moments mu;
            MatOfPoint2f approx = new MatOfPoint2f();
            // if any contour exist...
            Log.d(TAG, "Contour count: " + contours2.size());
            if (contours2.size() > 0) {
                int largest = 0;

                // for each remaining contour, find the biggest
                for (int h = 0; h < contours2.size(); h++) {
                    MatOfPoint contour = contours2.get(h);
                    Rect rect = Imgproc.boundingRect(contour);
                    double contourArea = contourArea(contour);
                    matOfPoint2f.fromList(contour.toList());
                    Imgproc.approxPolyDP(matOfPoint2f, approxCurve, Imgproc.arcLength(matOfPoint2f, true) * 0.02, true);
                    long total = approxCurve.total();

                    if (total == 4) {
                        List<Double> cos = new ArrayList<>();
                        Point[] points = approxCurve.toArray();
                        for (int j = 2; j < total + 1; j++) {
                            //TODO Might need to change, since another angle could be the vertex*//*
                            cos.add(angleBetween(points[(int) (j % total)], points[j - 2], points[j - 1]));
                        }
                        Collections.sort(cos);
                        Double minCos = cos.get(0);
                        Double maxCos = cos.get(cos.size() - 1);
                        boolean isRect = minCos >= -0.1 && maxCos <= 0.3;
                        if (isRect) {
                            double ratio = Math.abs(1 - (double) rect.width / rect.height);
                            //drawText(rect.tl(), ratio <= 0.02 ? "SQU" : "RECT");
                            boolean isSquare = ratio <= 0.02;
                            if (!isSquare && ((ratio == RATIO1 - RATIO_ERROR_RATE || ratio == RATIO1 + RATIO_ERROR_RATE) || (ratio == RATIO2 - RATIO_ERROR_RATE || ratio == RATIO2 + RATIO_ERROR_RATE))) {
                                //Do stuff bc object is a rectangle
                                //Adding to the list of Rectangles
                                allRectangles.add(approxCurve);
                                enclosingRects.add(rect);
                            } else {
                                contours2.remove(contour);
                            }
                        }
                    }
                }

                Log.d("->timelog", "filtering blobs t: " + (Calendar.getInstance().getTimeInMillis() - _time));
                _time = Calendar.getInstance().getTimeInMillis();

                roi = Imgproc.boundingRect(contours2.get(largest));
                mu = Imgproc.moments(contours2.get(largest));
                //center = new Center(mu.get_m10() / mu.get_m00(), mu.get_m01() / mu.get_m00());
//            Log.d(TAG, "m00: " + mu.get_m00());

            }

        /*for (int idx = 0; idx >= 0; idx = (int) newHierarchy.get(0, idx)[0]) {
            MatOfPoint contour = contours2.get(idx);
            Rect rect = Imgproc.boundingRect(contour);
            double contourArea = Imgproc.contourArea(contour);
            matOfPoint2f.fromList(contour.toList());
            Imgproc.approxPolyDP(matOfPoint2f, approxCurve, Imgproc.arcLength(matOfPoint2f, true) * 0.02, true);
            long total = approxCurve.total();
            *//*if (total == 3) { // is triangle
                // do things for triangle
            }
            if (total >= 4 && total <= 6) {
                List<Double> cos = new ArrayList<>();
                Point[] points = approxCurve.toArray();
                for (int j = 2; j < total + 1; j++) {
                    cos.add(angleBetween(points[(int) (j % total)], points[j - 2], points[j - 1]));
                }
                Collections.sort(cos);
                Double minCos = cos.get(0);
                Double maxCos = cos.get(cos.size() - 1);
                boolean isRect = total == 4 && minCos >= -0.1 && maxCos <= 0.3;
                boolean isPolygon = (total == 5 && minCos >= -0.34 && maxCos <= -0.27) || (total == 6 && minCos >= -0.55 && maxCos <= -0.45);
                if (isRect) {
                    double ratio = Math.abs(1 - (double) rect.width / rect.height);
                    //drawText(rect.tl(), ratio <= 0.02 ? "SQU" : "RECT");
                    boolean isSquare = ratio <= 0.02;
                    if(!isSquare) {
                        //Do stuff bc object is a rectangle
                        //Adding to the list of Rectangles
                    }
                }
                if (isPolygon) {
                   //drawText(rect.tl(), "Polygon");
                }
            }*//*
            if(total == 4) {
                List<Double> cos = new ArrayList<>();
                Point[] points = approxCurve.toArray();
                for (int j = 2; j < total + 1; j++) {
                    *//*TODO Might need to change, since another angle could be the vertex*//*
                    cos.add(angleBetween(points[(int) (j % total)], points[j - 2], points[j - 1]));
                }
                Collections.sort(cos);
                Double minCos = cos.get(0);
                Double maxCos = cos.get(cos.size() - 1);
                boolean isRect = minCos >= -0.1 && maxCos <= 0.3;
                if (isRect) {
                    double ratio = Math.abs(1 - (double) rect.width / rect.height);
                    //drawText(rect.tl(), ratio <= 0.02 ? "SQU" : "RECT");
                    boolean isSquare = ratio <= 0.02;
                    if(!isSquare && ((ratio == RATIO1 - RATIO_ERROR_RATE || ratio == RATIO1 + RATIO_ERROR_RATE) || (ratio == RATIO2 - RATIO_ERROR_RATE || ratio == RATIO2 + RATIO_ERROR_RATE))) {
                        //Do stuff bc object is a rectangle
                        //Adding to the list of Rectangles
                        allRectangles.add(approxCurve);
                        enclosingRects.add(rect);
                    } else {
                        contours2.remove(contour);
                    }
                }
            }
        }*/
            int size = enclosingRects.size();
            int largest = 0;
            int h = 0;
            Rect previous = null;
            boolean firstTime = true;
            for (MatOfPoint contour : contours2) {
                MatOfPoint cont = contour;
                double area = contourArea(cont);
                mu = Imgproc.moments(cont);
                //number of corners
                approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(new MatOfPoint2f(cont.toArray()), approx, 8, true);

                Point circ_center = new Point();
                float[] radius = new float[1];

                double y = mu.get_m01() / mu.get_m00();

                Imgproc.minEnclosingCircle(new MatOfPoint2f(cont.toArray()), circ_center, radius);
                boolean circle = Math.abs(area - PI * radius[0] * radius[0]) < 5;
                //greater than min size AND in the upper part of photo AND greater than the last biggest
                //TODO Change the area if it does not work.
                if ((area > 30.0 && area >= contourArea(contours2.get(largest)) && approx.rows() == 4 && y < MAX_Y_COORD && y > MIN_Y_COORD) && firstTime) { //&& !circle)) {
                    noValid = false;
                    largest = h;
                    previous = Imgproc.boundingRect(cont);
                    enclosingRects.add(previous);
                    firstTime = false;
                } else if (area > 30.0 && area >= contourArea(contours2.get(largest)) && approx.rows() == 4 && y < MAX_Y_COORD && y > MIN_Y_COORD) {
                    noValid = false;
                    largest = h;
                    enclosingRects.remove(previous);
                    enclosingRects.add(Imgproc.boundingRect(cont));
                }
                h++;
            }
            if (size > 0) {
                int largestRectIndex = 0;
                for (Rect t : enclosingRects) {
                    double area = t.area();
                    //greater than min size AND in the upper part of photo AND greater than the last biggest
                    //TODO Change the area if it does not work.
                    if (area > 30.0 && area >= t.area() && approx.rows() == 4 && y < MAX_Y_COORD && y > MIN_Y_COORD) { //&& !circle)) {
                        noValid = false;
                        largestRectIndex = h;
                    }
                }
                Rect biggest = enclosingRects.get(largestRectIndex);
                Point tl = biggest.tl();
                Point br = biggest.br();
                double midX = (br.x - tl.x) / 2;
                double midY = (tl.y - br.y) / 2;
                Center c = new Center(midX, midY);
                Map<String, Object> foundElements = new HashMap<>();
                foundElements.put("center", c);
                foundElements.put("roi", biggest);
                return foundElements;
            } else {
                Map<String, Object> foundElements = new HashMap<>();
                foundElements.put("center", NO_CENTER);
                foundElements.put("roi", new Rect());
                return foundElements;
            }
            /*int size = enclosingRects.size();
            if(size > 0) {
                if(size == 2) {
                    List<Center> centers = new ArrayList<>();
                    Log.d(TAG, "We're good, bc we detect either 2 rectangles");
                    // if any contour exist...
                    //Log.d(TAG, "Blob count: " + blobContours.size());
                    List<Moments> mu1 = new ArrayList<Moments>(enclosingRects.size());
                    for (int i = 0; i < enclosingRects.size(); i++) {
                        mu1.add(i, Imgproc.moments(contours2.get(i), false));
                        Moments p = mu1.get(i);
                        int x = (int) (p.get_m10() / p.get_m00());
                        int y = (int) (p.get_m01() / p.get_m00());
                        //Core.circle(rgbaImage, new Point(x, y), 4, new Scalar(255,49,0,255));
                        Center c = new Center((double) x, (double) y);
                        centers.add(c);
                    }
                    //Get centers to see which is to top rectangle
                    Center c1 = centers.get(0);
                    Center c2 = centers.get(1);
                    if (c1.y > c2.y) {
                        Map<String, Object> foundElements = new HashMap<>();
                        foundElements.put("center", c1);
                        foundElements.put("roi", enclosingRects.get(0));
                        return foundElements;
                    } else {
                        Map<String, Object> foundElements = new HashMap<>();
                        foundElements.put("center", c2);
                        foundElements.put("roi", enclosingRects.get(1));
                        return foundElements;
                    }
                } else if(size > 2){
                    //Log.d(TAG, "" + contours2.size());
                    //Log.e(TAG, "Something went wrong idk");
                    int largest = 0;
                    int h = 0;
                    Rect previous = null;
                    boolean firstTime = true;
                    for(MatOfPoint contour : contours2) {
                        MatOfPoint cont = contour;
                        double area = Imgproc.contourArea(cont);
                        mu = Imgproc.moments(cont);
                        //number of corners
                        approx = new MatOfPoint2f();
                        Imgproc.approxPolyDP(new MatOfPoint2f(cont.toArray()), approx,8, true);

                        Point circ_center = new Point();
                        float[] radius = new float[1];

                        double y = mu.get_m01() / mu.get_m00();

                        Imgproc.minEnclosingCircle(new MatOfPoint2f(cont.toArray()), circ_center, radius);
                        boolean circle = Math.abs(area - PI * radius[0] * radius[0]) < 5;
                        //greater than min size AND in the upper part of photo AND greater than the last biggest
                        //TODO Change the area if it does not work.
                        if ((area > 30.0 && area >= Imgproc.contourArea(contours2.get(largest))&& approx.rows() == 4 && y < MAX_Y_COORD && y > MIN_Y_COORD) && firstTime){ //&& !circle)) {
                            noValid = false;
                            largest = h;
                            previous = Imgproc.boundingRect(cont);
                            enclosingRects.add(previous);
                            firstTime = false;
                        } else if(area > 30.0 && area >= Imgproc.contourArea(contours2.get(largest))&& approx.rows() == 4 && y < MAX_Y_COORD && y > MIN_Y_COORD) {
                            noValid = false;
                            largest = h;
                            enclosingRects.remove(previous);
                            enclosingRects.add(Imgproc.boundingRect(cont));
                        }
                        h++;
                    }

                    Log.d("->timelog", "filtering blobs t: " + (Calendar.getInstance().getTimeInMillis() - _time));
                    _time = Calendar.getInstance().getTimeInMillis();

                    roi = Imgproc.boundingRect(contours2.get(largest));
                    mu = Imgproc.moments(contours2.get(largest));
                    center = new Center(mu.get_m10() / mu.get_m00(), mu.get_m01() / mu.get_m00());

                    Map<String, Object> foundElements = new HashMap<>();
                    foundElements.put("center", center);
                    if(!noValid) {
                        foundElements.put("roi", enclosingRects.get(0));
                    } else {
                        foundElements.put("roi", new Rect());
                    }
                    return foundElements;
                } else if(size == 1) {
                    Log.d(TAG, "Let's pray that this rectangle is the one.");
                    List<Center> centers = new ArrayList<>();
                    // if any contour exist...
                    //Log.d(TAG, "Blob count: " + blobContours.size());
                    List<Moments> mu2 = new ArrayList<Moments>(contours2.size());
                    for (int i = 0; i < contours2.size(); i++) {
                        mu2.add(i, Imgproc.moments(contours2.get(i), false));
                        Moments p = mu2.get(i);
                        int x = (int) (p.get_m10() / p.get_m00());
                        int y = (int) (p.get_m01() / p.get_m00());
                        //Core.circle(rgbaImage, new Point(x, y), 4, new Scalar(255,49,0,255));
                        Center c = new Center((double) x, (double) y);
                        centers.add(c);
                    }
                    Map<String, Object> foundElements = new HashMap<>();
                    foundElements.put("center", centers.get(0));
                    foundElements.put("roi", Imgproc.boundingRect(contours2.get(0)));
                    return foundElements;
                } else {
                    Log.d(TAG, "We don't detect any rectangles");
                    Map<String, Object> foundElements = new HashMap<>();
                    foundElements.put("center", NO_CENTER);
                    foundElements.put("roi", new Rect());
                    return foundElements;
                }
            } else {
                Log.d(TAG, "No contours");
                Map<String, Object> foundElements = new HashMap<>();
                foundElements.put("center", NO_CENTER);
                foundElements.put("roi", new Rect());
                return foundElements;
            }*/

            /*if (!(contours2.size() > 2)) {
                Log.d(TAG, "We're f***ing screwed bc we detect more than 2 rectangles.");
                Map<String, Object> newMap = new HashMap<String, Object>();
                newMap.put("center", NO_CENTER);
                newMap.put("roi", new Rect());
                return newMap;
            } else if (contours2.size() == 2) {
                List<Center> centers = new ArrayList<>();
                Log.d(TAG, "We're good, bc we detect either 2 rectangles");
                // if any contour exist...
                //Log.d(TAG, "Blob count: " + blobContours.size());
                List<Moments> mu1 = new ArrayList<Moments>(contours2.size());
                for (int i = 0; i < contours2.size(); i++) {
                    mu1.add(i, Imgproc.moments(contours2.get(i), false));
                    Moments p = mu1.get(i);
                    int x = (int) (p.get_m10() / p.get_m00());
                    int y = (int) (p.get_m01() / p.get_m00());
                    //Core.circle(rgbaImage, new Point(x, y), 4, new Scalar(255,49,0,255));
                    Center c = new Center((double) x, (double) y);
                    centers.add(c);
                }
                //Get centers to see which is to top rectangle
                Center c1 = centers.get(0);
                Center c2 = centers.get(1);
                if (c1.y > c2.y) {
                    Map<String, Object> foundElements = new HashMap<>();
                    foundElements.put("center", c1);
                    foundElements.put("roi", enclosingRects.get(0));
                    return foundElements;
                } else {
                    Map<String, Object> foundElements = new HashMap<>();
                    foundElements.put("center", c2);
                    foundElements.put("roi", enclosingRects.get(1));
                    return foundElements;
                }
            } else if (contours2.size() == 1) {
                Log.d(TAG, "Let's pray that this rectangle is the one.");
                List<Center> centers = new ArrayList<>();
                // if any contour exist...
                //Log.d(TAG, "Blob count: " + blobContours.size());
                List<Moments> mu2 = new ArrayList<Moments>(contours2.size());
                for (int i = 0; i < contours2.size(); i++) {
                    mu2.add(i, Imgproc.moments(contours2.get(i), false));
                    Moments p = mu2.get(i);
                    int x = (int) (p.get_m10() / p.get_m00());
                    int y = (int) (p.get_m01() / p.get_m00());
                    //Core.circle(rgbaImage, new Point(x, y), 4, new Scalar(255,49,0,255));
                    Center c = new Center((double) x, (double) y);
                    centers.add(c);
                }
                Map<String, Object> foundElements = new HashMap<>();
                foundElements.put("center", centers.get(0));
                foundElements.put("roi", enclosingRects.get(0));
                return foundElements;
            } else {
                //Log.d(TAG, "" + contours2.size());
                //Log.e(TAG, "Something went wrong idk");
                int largest = 0;
                int h = 0;
                for(MatOfPoint contour : contours2) {
                    MatOfPoint cont = contour;
                    double area = Imgproc.contourArea(cont);
                    mu = Imgproc.moments(cont);
                    //number of corners
                    approx = new MatOfPoint2f();
                    Imgproc.approxPolyDP(new MatOfPoint2f(cont.toArray()), approx,8, true);

                    Point circ_center = new Point();
                    float[] radius = new float[1];

                    double y = mu.get_m01() / mu.get_m00();

                    Imgproc.minEnclosingCircle(new MatOfPoint2f(cont.toArray()), circ_center, radius);
                    boolean circle = Math.abs(area - PI * radius[0] * radius[0]) < 5;
                    //greater than min size AND in the upper part of photo AND greater than the last biggest
                    if (area > 50.0 && area >= Imgproc.contourArea(contours2.get(largest))&& approx.rows() == 4 && y < MAX_Y_COORD && y > MIN_Y_COORD){ //&& !circle)) {
                        noValid = false;
                        largest = h;
                    }
                    h++;
                }

                Log.d("->timelog", "filtering blobs t: " + (Calendar.getInstance().getTimeInMillis() - _time));
                _time = Calendar.getInstance().getTimeInMillis();

                roi = Imgproc.boundingRect(contours2.get(largest));
                mu = Imgproc.moments(contours2.get(largest));
                center = new Center(mu.get_m10() / mu.get_m00(), mu.get_m01() / mu.get_m00());

                Map<String, Object> foundElements = new HashMap<>();
                foundElements.put("center", center);
                foundElements.put("roi", enclosingRects.get(0));
                return foundElements;
            }*/
        } catch (Exception e) {
            Log.d(TAG, e.getMessage(), e);
            Map<String, Object> foundElements = new HashMap<>();
            foundElements.put("center", NO_CENTER);
            foundElements.put("roi", new Rect());
            return foundElements;
        }
    }

    //Finds lines that make a rect
    public Map<String, Object> findLinedRects(Mat m) {
        // Step Find_Lines0:
        Mat findLinesInput = m;
        findLines(findLinesInput, findLinesOutput);

        // Step Filter_Lines0:
        ArrayList<Line> filterLinesLines = findLinesOutput;
        double filterLinesMinLength = 25.0;
        double[] filterLinesAngle = {0.0, 100.2376910016978};
        filterLines(filterLinesLines, filterLinesMinLength, filterLinesAngle, filterLinesOutput);
        return new HashMap<String, Object>();
    }

    /**
     * This method is a generated getter for the output of a Find_Lines.
     * @return ArrayList<Line> output from Find_Lines.
     */
    public ArrayList<Line> findLinesOutput() {
        return findLinesOutput;
    }

    /**
     * This method is a generated getter for the output of a Filter_Lines.
     * @return ArrayList<Line> output from Filter_Lines.
     */
    public ArrayList<Line> filterLinesOutput() {
        return filterLinesOutput;
    }


    public static class Line {
        public final double x1, y1, x2, y2;
        public Line(double x1, double y1, double x2, double y2) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
        }
        public double lengthSquared() {
            return Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2);
        }
        public double length() {
            return Math.sqrt(lengthSquared());
        }
        public double angle() {
            return Math.toDegrees(Math.atan2(y2 - y1, x2 - x1));
        }
    }
    /**
     * Finds all line segments in an image.
     * @param input The image on which to perform the find lines.
     * @param lineList The output where the lines are stored.
     */
    private void findLines(Mat input, ArrayList<Line> lineList) {
        final LineSegmentDetector lsd = Imgproc.createLineSegmentDetector();
        final Mat lines = new Mat();
        lineList.clear();
        if (input.channels() == 1) {
            lsd.detect(input, lines);
        } else {
            final Mat tmp = new Mat();
            Imgproc.cvtColor(input, tmp, Imgproc.COLOR_BGR2GRAY);
            lsd.detect(tmp, lines);
        }
        if (!lines.empty()) {
            for (int i = 0; i < lines.rows(); i++) {
                lineList.add(new Line(lines.get(i, 0)[0], lines.get(i, 0)[1],
                        lines.get(i, 0)[2], lines.get(i, 0)[3]));
            }
        }
    }

    /**
     * Filters out lines that do not meet certain criteria.
     * @param inputs The lines that will be filtered.
     * @param minLength The minimum length of a line to be kept.
     * @param angle The minimum and maximum angle of a line to be kept.
     * @param outputs The output lines after the filter.
     */
    private void filterLines(List<Line> inputs,double minLength,double[] angle,
                             List<Line> outputs) {
        /*outputs = inputs.stream()
                .filter(line -> line.lengthSquared() >= Math.pow(minLength,2))
                .filter(line -> (line.angle() >= angle[0] && line.angle() <= angle[1])
                        || (line.angle() + 180.0 >= angle[0] && line.angle() + 180.0 <= angle[1]))
                .collect(Collectors.toList());*/
        List<Line> out1 = new ArrayList<>();
        for(Line l : inputs) {
            if(l.lengthSquared() >= Math.pow(minLength, 2)) {
                out1.add(l);
            }
        }
        /*for(Line l : out1) {
            if((l.angle() >= angle[0] && l.angle() <= angle[1]) || (l.angle() + 180.0 >= angle[0] && l.angle() + 180.0 <= angle[1])) {
                outputs.add(l);
            }
        }*/
        List<Line> horiz = new ArrayList<>();
        List<Line> vert = new ArrayList<>();
        for(Line line : out1) {
            if(line.angle() >= angle[0] && line.angle() <= angle[0] + 12) {
                horiz.add(line);
            } if(line.angle() <= angle[1] && line.angle() >= angle[1] - 12) {
                vert.add(line);
            }
        }
        //We have separated into 2 categories.
        for (int i = 0; i < horiz.size(); ++i) {
            for (int j = i + 1; j < horiz.size(); ++j) {
                // Use arrayList.get(i) and arrayList.get(j).
                Line horiz1 = horiz.get(i);
                Line horiz2 = horiz.get(j);
                for(int k = 0; k < vert.size(); ++k) {
                    for(int l = k + 1; l < vert.size(); ++l) {
                        Line vert1 = vert.get(k);
                        Line vert2 = vert.get(l);
                        double angle1 = Math.abs(vert1.angle() - horiz1.angle());
                        double angle2 = Math.abs(horiz2.angle() - vert2.angle());
                        if(Math.abs(Math.cos(angle1)) < 0.21 && Math.abs(Math.cos(angle2)) < 0.21) {
                            //These four points make a rect

                        }
                    }
                }
            }
        }
    }


    //Finds rects based on opencv examples
    public Map<String, Object> findRects(Mat m) {
        List<List<Point>> rects = new ArrayList<>();
        Mat pyr = new Mat();
        Mat timg = new Mat();
        Mat gray = new Mat();
        Mat gray0 = new Mat(m.size(), CV_8U);
        Imgproc.pyrDown(m, pyr, new Size(m.cols() / 2, m.rows() / 2));
        Imgproc.pyrUp(pyr, timg, m.size());
        List<MatOfPoint> contours = new ArrayList<>();
        for (int c = 0; c < 3; c++) {
            int[] ch = new int[]{c, 0};
            ArrayList<Mat> timgList = new ArrayList<>();
            timgList.add(timg);
            ArrayList<Mat> gray0List = new ArrayList<>();
            gray0List.add(gray0);
            Core.mixChannels(timgList, gray0List, new MatOfInt(ch));
            for (int l = 0; l < 11 /*TODO: Change?*/; l++) {
                if (l == 0) {
                    Imgproc.Canny(gray0, gray, 245, 255, 5, false);
                    Imgproc.dilate(gray, gray, new Mat(), new Point(-1, -1), 1);
                } else {
                    //gray = gray0 >= (l+1)*255/N;
                    //I think this is what it means??
                    //Core.compare(gray0, new Mat(gray0.size(), gray0.type(), new Scalar((l + 1) * 255 / N)), gray, Core.CMP_GE);
                    Core.compare(gray0, new Scalar((l+1)*255/N), gray, Core.CMP_GE);
                }
                Imgproc.findContours(gray, contours, new Mat(), Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
                Log.d(TAG, "Number of contours: " + contours.size());
                if(contours.size() != 0) {
                    Log.d(TAG, "Channel num: " + contours.get(0).channels());
                }
                MatOfPoint2f approx = new MatOfPoint2f();
                // test each contour
                for (int i = 0; i < contours.size(); i++) {
                    // approximate contour with accuracy proportional
                    // to the contour perimeter
                    MatOfPoint now = contours.get(i);
                    Imgproc.approxPolyDP(new MatOfPoint2f(now.toArray()), approx, Imgproc.arcLength(new MatOfPoint2f(now), true) * 0.02, true);
                    // square contours should have 4 vertices after approximation
                    // relatively large area (to filter out noisy contours)
                    // and be convex.
                    // Note: absolute value of an area is used because
                    // area may be positive or negative - in accordance with the
                    // contour orientation
                    if (approx.toList().size() == 4 && Math.abs(contourArea(approx)) > 1000 && Imgproc.isContourConvex(new MatOfPoint(approx)))
                    {
                        double maxCosine = 0;

                        for (int j = 2; j < 5; j++) {
                            // find the maximum cosine of the angle between joint edges
                            double cosine = Math.abs(angle(approx.toList().get(j % 4), approx.toList().get(j - 2), approx.toList().get(j - 1)));
                            maxCosine = Math.max(maxCosine, cosine);
                        }

                        // if cosines of all angles are small
                        // (all angles are ~90 degree) then write quandrange
                        // vertices to resultant sequence
                        if (maxCosine < 0.3) {
                            rects.add(approx.toList());
                        }
                    }
                }
            }
        }
        if (rects.size() != 0) {
            List<Rect> rects1 = new ArrayList<>();
            for (List<Point> lp : rects) {
                Point v1 = lp.get(0);
                Point v2 = lp.get(1);
                Point v3 = lp.get(2);
                Point v4 = lp.get(3);
                Point bl;
                Point tr;
                int finalNum = -1;
                int finalNum2 = -1;
                int finalNum3 = -1;
                int finalNum4 = -1;
                double[] xcoords = new double[]{v1.x, v2.x, v3.x, v4.x};
                Arrays.sort(xcoords);
                for (int i = 0; i < 4; i++) {
                    Point p = lp.get(i);
                    if (p.x == xcoords[0]) {
                        finalNum = i;
                        break;
                    }
                }
                for (int i = 0; i < 4; i++) {
                    Point p = lp.get(i);
                    if (p.x == xcoords[1]) {
                        finalNum2 = i;
                        break;
                    }
                }
                if (lp.get(finalNum).y < lp.get(finalNum2).y) {
                    //finalNum is the bottom left
                    bl = lp.get(finalNum);
                } else {
                    //finalNum2 is the bottom left
                    bl = lp.get(finalNum2);
                }
                for (int i = 0; i < 4; i++) {
                    Point p = lp.get(i);
                    if (p.x == xcoords[2]) {
                        finalNum3 = i;
                        break;
                    }
                }
                for (int i = 0; i < 4; i++) {
                    Point p = lp.get(i);
                    if (p.x == xcoords[3]) {
                        finalNum4 = i;
                        break;
                    }
                }
                if (lp.get(finalNum3).y < lp.get(finalNum4).y) {
                    //finalNum3 is the top right
                    tr = lp.get(finalNum3);
                } else {
                    //finalNum4 is the top right
                    tr = lp.get(finalNum4);
                }
                Rect r = new Rect(bl, tr);
                rects1.add(r);
            }
            Rect roi2 = null;
            double largestArea = Double.MIN_VALUE;
            for (Rect r : rects1) {
                if (r.area() > largestArea) {
                    roi2 = r;
                }
            }
            Center c = new Center((roi2.br().x + roi2.tl().x) / 2, (roi2.br().y + roi2.tl().y) / 2);
            Map<String, Object> fE = new HashMap<>();
            double dist = calcDistAxis206(roi2.width, WIDTH_TARGET, 320, STANDARD_VIEW_ANGLE);
            fE.put("center", c);
            fE.put("roi", roi2);
            fE.put("dist", dist);
            return fE;
        } else {
            Map<String, Object> fE = new HashMap<>();
            fE.put("center", NO_CENTER);
            fE.put("roi", new Rect());
            fE.put("dist", 0.0);
            return fE;
        }
    }

    double angle(Point pt1, Point pt2, Point pt0) {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
    }

    // Old 2015-16

    /**
     * Algorithm:
     * <p>
     * Gaussian Blur
     * cvt to gray
     * threshold (Tozero)
     * //TODO add erode and dilates and other thresh fcts here
     * find contours
     * find, draw, and fill convex hulls
     * subtract the threshed image from the one with filled convex hulls to get the new filled portions (blobs)
     * ->this allows us to roughly find a U shape
     * find contours of the blobs
     * approximate the polygon (should be a rectangle)
     * find the largest object & is greater than the max size & has 4 corners (and is not circular)
     * Calculate its center and other features
     *
     * @return Dictionary of objects
     * -> "center" Center of found object
     * -> "roi" Bounding rect of selected object
     * -> "m" after convex hull fill
     * -> "hsv"
     * -> "thresh" the mat after initial thresholding
     * -> "subImage" subtraced image
     * -> "blobMat" after find contours of blobs
     */

    public Map<String, Object> shapeDetectTarget2(Mat m) {
        long _time = Calendar.getInstance().getTimeInMillis();

        Center center = NO_CENTER;
        Mat hsv = new Mat();
        Mat ycrcb = new Mat();

        Imgproc.GaussianBlur(m, m, new org.opencv.core.Size(5, 5), 0);

        //Imgproc.medianBlur(m, m, 11);

        Log.d("->timelog", "GAUSSIAN t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        List<MatOfInt> hull = new ArrayList<MatOfInt>();


        //Imgproc.cvtColor(m, ycrcb, Imgproc.COLOR_RGB2YCrCb);
//        Imgproc.cvtColor(m, ycrcb, Imgproc.COLOR_RGB2Luv);
//        Scalar[] scalars = readElements("Lab.txt");
//        Log.d(TAG, scalars.toString());
//        Core.inRange(ycrcb, scalars[0], scalars[1],ycrcb);

        //Core.inRange(hsv, new Scalar(25, 2, 180), new Scalar(120, 255, 255), hsv); //relatively loose
//        //TROLL NOT ACTUALLY HSV HAHA
//        Imgproc.cvtColor(m, hsv, Imgproc.COLOR_RGB2YCrCb);
//        Scalar[] scalars = readElements("ycrcb.txt");
//        Core.inRange(hsv, scalars[0], scalars[1],hsv);

        HSVThread hsvSide = new HSVThread(m);
        hsvSide.start();
        GrayThread graySide = new GrayThread(m);
        graySide.start();

        try {
            graySide.join();
            hsvSide.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        Log.d("->timelog", "Both threads t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();


        //find overlap
        Mat combined = new Mat();
        try {
            Core.bitwise_and(hsvSide.getFinalMat(), graySide.getFinalMat(), m); //think &&
        } catch (Exception e) {
            return new HashMap<>(); //dun got broked, don' do this
        }

        //Imgproc.dilate(combined, m, dilate); //dilate to be safe

        Mat thresh = new Mat(), subImage = new Mat();
        m.copyTo(thresh); //copy filtered image

        Log.d("->timelog", "Bit AND & dilate t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();

        Imgproc.findContours(m, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        Log.d("->timelog", "find contours t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();

        MatOfInt hullInt = new MatOfInt();
        MatOfPoint hullPointMat;
        List<Point> hullPointList = new ArrayList<>();
        List<MatOfPoint> hullPoints = new ArrayList<>();

        //draw convex Hulls
        for (int k = 0; k < contours.size(); k++) {
            MatOfPoint cont = contours.get(k);

            Imgproc.convexHull(cont, hullInt);

            hullPointList.clear();
            for (int j = 0; j < hullInt.toList().size(); j++) {
                hullPointList.add(cont.toList().get(hullInt.toList().get(j)));
            }

            hullPointMat = new MatOfPoint();
            hullPointMat.fromList(hullPointList);
            hullPoints.add(hullPointMat);
        }
        m = new Mat();
        thresh.copyTo(m); //bring the old one back and draw new stuff on it

        Log.d("->timelog", "draw Convex hulls t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();
//
//        //at this point thresh and m are the same (no contours just the threshold)
//        //draw the full contours on both
//
//        Imgproc.drawContours(thresh, contours, -1, new Scalar(255, 255, 255), 1);
//        Imgproc.fillPoly(thresh, contours, new Scalar(255, 255, 255));
////
        //Imgproc.drawContours(m, contours, -1, new Scalar(255, 255, 255), 1);
//        Imgproc.fillPoly(m, contours, new Scalar(255, 255, 255));
////
//        Imgproc.drawContours(m, hullPoints, -1, new Scalar(255, 255, 255), 1);
//        Imgproc.fillPoly(m,hullPoints, new Scalar(255,255,255));


        MatOfPoint2f approx;
        Log.d(TAG, "H SIZE: " + hullPoints.size());
        for (int i = 0; i < hullPoints.size(); i++) {

            Imgproc.fillConvexPoly(m, hullPoints.get(i), new Scalar(255, 255, 255));
        }


        Log.d("->timelog", "fillConvexPoly t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();


        //now we are left with whatever was filled in
        Core.subtract(m, thresh, subImage);

        Log.d("->timelog", "subtract images t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();

        Imgproc.erode(subImage, subImage, erode);
        //Imgproc.erode(subImage, subImage, erode);
        Imgproc.threshold(subImage, subImage, 250, 255, Imgproc.THRESH_BINARY);

        Log.d("->timelog", "2nd binary thresh t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();

        Mat blobMat = new Mat();
        subImage.copyTo(blobMat);
        List<MatOfPoint> blobContours = new ArrayList<>();
        Imgproc.findContours(blobMat, blobContours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        Log.d("->timelog", "2nd find contours t: " + (Calendar.getInstance().getTimeInMillis() - _time));
        _time = Calendar.getInstance().getTimeInMillis();


        boolean noValid = true;
        Moments mu;
        approx = new MatOfPoint2f();
        // if any contour exist...
        Log.d(TAG, "Blob count: " + blobContours.size());
        if (blobContours.size() > 0) {
            int largest = 0;

            // for each remaining contour, find the biggest
            for (int h = 0; h < blobContours.size(); h++) {
                MatOfPoint cont = blobContours.get(h);
                double area = contourArea(cont);
                mu = Imgproc.moments(cont);
                //number of corners
                approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(new MatOfPoint2f(cont.toArray()), approx, 8, true);

                Point circ_center = new Point();
                float[] radius = new float[1];

                double y = mu.get_m01() / mu.get_m00();

                Imgproc.minEnclosingCircle(new MatOfPoint2f(cont.toArray()), circ_center, radius);
                boolean circle = Math.abs(area - PI * radius[0] * radius[0]) < 5;
                //greater than min size AND in the upper part of photo AND greater than the last biggest
                if (area > 50.0 && area >= contourArea(blobContours.get(largest)) && approx.rows() == 4 && y < MAX_Y_COORD && y > MIN_Y_COORD) { //&& !circle)) {
                    noValid = false;
                    largest = h;

                }


            }

            Log.d("->timelog", "filtering blobs t: " + (Calendar.getInstance().getTimeInMillis() - _time));
            _time = Calendar.getInstance().getTimeInMillis();

            roi = Imgproc.boundingRect(blobContours.get(largest));
            mu = Imgproc.moments(blobContours.get(largest));
            center = new Center(mu.get_m10() / mu.get_m00(), mu.get_m01() / mu.get_m00());
//            Log.d(TAG, "m00: " + mu.get_m00());

        }

        //the array we return
        //          1:
        Map<String, Object> foundElements = new HashMap<>();
//        List<Object> foundElements = new ArrayList<>();


        if (noValid) {
            foundElements.put("center", NO_CENTER);

        } else {
            foundElements.put("center", center);
        }

        foundElements.put("roi", roi);
        foundElements.put("m", m);
        foundElements.put("hsv", hsv);
        foundElements.put("thresh", thresh);
        foundElements.put("combined", combined);
        foundElements.put("subImage", subImage);
        foundElements.put("blobMat", blobMat);
//        foundElements.put("ycrcb", ycrcb);


        return foundElements;
    }


    //old
    public double calcDistAxis206(double obj_pix, double obj_in, double view_pix, double max_cam_angle) {
        return view_pix * obj_in / (2 * Math.tan(max_cam_angle) * obj_pix);
    }

    public double angleBetween(Point v1, Point v2, Point v3) {
        Point v4 = new Point();
        v4.x = v1.x - v3.x;
        v4.y = v1.y - v3.y;
        Point v5 = new Point();
        v5.x = v2.x - v3.x;
        v5.y = v2.y - v3.y;

        double len1 = sqrt(v4.x * v4.x + v4.y * v4.y);
        double len2 = sqrt(v5.x * v5.x + v5.y * v5.y);

        double dot = v4.x * v5.x + v4.y * v5.y;

        double a = dot / (len1 * len2);

        if (a >= 1.0)
            return 0.0;
        else if (a <= -1.0)
            return PI;
        else
            return acos(a); // 0..PI
    }

    public class HSVThread extends Thread {

        Mat m;
        boolean ready;

        public HSVThread(Mat mat) {
            m = new Mat();
            mat.copyTo(m);
            ready = false; //kinda redundant but meh
        }

        @Override
        public void run() {
            if (m != null) {
                //hsv side
                Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2HSV);
                Scalar[] scalars = readElements(HSVFileName);
                Core.inRange(m, scalars[0], scalars[1], m); //"20,3,215 - > 75,250,250"
                //used to be 15,2,210 -> 100,255,255


                Imgproc.dilate(m, m, dilate);//dilate
                Imgproc.erode(m, m, erode);
                //Imgproc.dilate(m, m, dilate);
                Imgproc.dilate(m, m, dilate);
                ready = true;
            }
        }

        public Mat getFinalMat() {
            if (ready) {
                return m;
            }
            return new Mat();
        }
    }

    public class GrayThread extends Thread {

        Mat m;
        boolean ready;

        public GrayThread(Mat mat) {
            m = new Mat();
            mat.copyTo(m);
            ready = false; //kinda redundant but meh
        }

        @Override
        public void run() {
            if (m != null) {
                //brightness side
                Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2GRAY);
                Imgproc.threshold(m, m, 180, 250, Imgproc.THRESH_TOZERO);
//                Imgproc.threshold(m,m,200, 255, Imgproc.THRESH_BINARY);

                Imgproc.dilate(m, m, dilate);//dilate
                Imgproc.erode(m, m, erode);
                //Imgproc.dilate(m, m, dilate);
                Imgproc.dilate(m, m, dilate);
                ready = true;
            }
        }

        public Mat getFinalMat() {
            if (ready) {
                return m;
            }
            return new Mat();
        }
    }


    public class HSVThread2 extends Thread {

        Mat m;
        boolean ready;

        public HSVThread2(Mat mat) {
            m = new Mat();
            mat.copyTo(m);
            ready = false; //kinda redundant but meh
        }

        @Override
        public void run() {
            if (m != null) {
                //hsv side
                Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2HSV);
                //Scalar[] scalars = readElements(HSVFileName);
                //Core.inRange(m, scalars[0], scalars[1], m); //"20,3,215 - > 75,250,250"
                //used to be 15,2,210 -> 100,255,255

                ready = true;
            }
        }

        public Mat getFinalMat() {
            if (ready) {
                return m;
            }
            return new Mat();
        }
    }

    public class GrayThread2 extends Thread {

        Mat m;
        boolean ready;

        public GrayThread2(Mat mat) {
            m = new Mat();
            mat.copyTo(m);
            ready = false; //kinda redundant but meh
        }

        @Override
        public void run() {
            if (m != null) {
                //brightness side
                Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2GRAY);
                ready = true;
            }
        }

        public Mat getFinalMat() {
            if (ready) {
                return m;
            }
            return new Mat();
        }
    }


//        Imgproc.cornerHarris(m, cornerMat,2,3,0.04);
//        Core.normalize(cornerMat, cornerMat, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());
//        Core.convertScaleAbs(cornerMat, cornerMat);

//        for (int x = 0; x < cornerMat.rows(); x++){
//            for (int y = 0; y < cornerMat.cols(); y++) {
//                if (cornerMat.get(x,y)[0] > 200 ){
//                    Imgproc.circle(m, new Point(x,y), 5, new Scalar(255,0,0));
//                }
//            }
//        }
}