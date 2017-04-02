package com.example.ujjwal.realvision2016;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.example.ujjwal.realvision2016.TargetFinder.MAX_Y_COORD;
import static com.example.ujjwal.realvision2016.TargetFinder.MIN_Y_COORD;
import static com.example.ujjwal.realvision2016.TargetFinder.NO_CENTER;
import static com.example.ujjwal.realvision2016.TargetFinder.STANDARD_VIEW_ANGLE;
import static com.example.ujjwal.realvision2016.TargetFinder.WIDTH_TARGET;
import static java.lang.Math.PI;
import static org.opencv.imgproc.Imgproc.approxPolyDP;
import static org.opencv.imgproc.Imgproc.contourArea;

/**
 * Created by Ujjwal on 3/19/17.
 */

public class FindingRects {

    private Rect roi;
    private Center center = new Center(-1, -1);
    private String TAG = "FindingRects";
    private Mat erode, dilate;
    public final double FOCAL_LENGTH = 4.0;

    //Outputs
    private Mat blurOutput = new Mat();
    private Mat rgbThresholdOutput = new Mat();
    private Mat hslThresholdOutput = new Mat();
    private ArrayList<MatOfPoint> findContoursOutput = new ArrayList<MatOfPoint>();
    private Mat cvErodeOutput = new Mat();
    private Mat cvDilateOutput = new Mat();
    private Mat cvCvtcolor0Output = new Mat();
    private Mat rgbThreshold0Output = new Mat();
    private ArrayList<MatOfPoint> findContours0Output = new ArrayList<MatOfPoint>();
    private Mat maskOutput = new Mat();
    private Mat cvCvtcolor1Output = new Mat();
    private Mat rgbThreshold1Output = new Mat();
    private ArrayList<MatOfPoint> findContours1Output = new ArrayList<MatOfPoint>();
    private ArrayList<MatOfPoint> filterContoursOutput = new ArrayList<MatOfPoint>();
    private ArrayList<MatOfPoint> convexHullsOutput = new ArrayList<MatOfPoint>();


    public FindingRects() {
        dilate = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE, new Size(3, 3));
        erode = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE, new Size(3, 3));
    }

    public void newMethod2(Mat m) {
        Mat initial = m;
        Mat opening = new Mat();
        Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(m,m,0, 255, Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);
        //Will this work as well?
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3));
        Imgproc.morphologyEx(m, opening, Imgproc.MORPH_OPEN, kernel);
        Mat bg = new Mat();
        Imgproc.dilate(opening, bg, kernel, new Point(-1, -1), 3);
        Mat dist = new Mat();
        Imgproc.distanceTransform(bg, dist, Imgproc.DIST_L2, 5);
        Mat fg = new Mat();
        Core.MinMaxLocResult mmr = Core.minMaxLoc(dist);
        Imgproc.threshold(dist, fg, 0.7*mmr.maxVal, 255, 0);
        Mat unknown = new Mat();
        Core.subtract(bg, fg, unknown);
        Mat markers = new Mat();
        Imgproc.connectedComponents(fg, markers);
        //TODO is this the correct scalar value? (255)
        Core.add(markers, new Scalar(0, 0, 1), markers);
        for(int i = 0; i < markers.rows(); i++) {
            for(int j = 0; j < markers.cols(); j++) {
                double[] data = unknown.get(i, j);
                if(data[0] == 255) {
                    markers.put(i, j, 0);
                }
            }
        }
        Imgproc.watershed(initial, markers);
        for(int i = 0; i < markers.rows(); i++) {
            for(int j = 0; j < markers.cols(); j++) {
                double[] data = markers.get(i, j);
                if(data[0] == -1) {
                    double[] newData = new double[] {255, 0, 0};
                    initial.put(i, j, newData);
                }
            }
        }
        //Initial is the final picture with the data
    }

    /**
     * 1678 Algorithm(Lucas)
     * 0.5th, gaussian blur radius 3
     * First, use just hsv thresholding to find the image(Fine tune)
     * Second, find contours
     * 2.5th, Use erode + dilate?
     * Third, filter contours(less than 0.5% of img->remove, more than 10% of img->remove, and some others from grip)
     * Fourth, choose topmost contour–Based on x-coords, not y-coords(Angle of phone must be low->Get ret. tape in center of img)*/
    public Map<String, Object> citrusMethod(Mat m) {
        Mat initial = m;

        //Gaussian Blur
        int radius = 3;
        Imgproc.GaussianBlur(initial, initial, new Size(6*radius + 1, 6*radius + 1), radius);

        //HSV Thresholding
        //TODO Test these values further
        //double[] hue = {0.0, 94.43123938879455};

        Mat hsv = thresh(initial);

        //Finding contours
        List<MatOfPoint> conts = new ArrayList<>();
        Imgproc.findContours(hsv, conts, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        //Erode + Dilate? Nothing rn
        /**/

        //Filter conts
        Map<String, Object> fe = new HashMap<>();
        fe.put("hsv", hsv);

        Log.d(TAG, "Num of conts: " + conts.size());
        if(conts.size() > 0) {
            Log.d(TAG, "Num of conts: " + conts.size());
            List<MatOfPoint> newConts = new ArrayList<>();
            Size imgSize = hsv.size();
            double imgArea = imgSize.area();
            Log.d(TAG, "Width: " + imgSize.width);
            Log.d(TAG, "Height: " + imgSize.height);
            Log.d(TAG, "Image Area: " + imgArea);
            for(int i = 0; i < conts.size(); i++) {
                MatOfPoint contcpy = conts.get(i);
                double contArea = contourArea(contcpy);
                double percent = contArea / imgArea;
                Log.d(TAG, "Percent: " + percent);
                Log.d(TAG, "Width: " + contcpy.width());
                Log.d(TAG, "Height: " + contcpy.height());
                Log.d(TAG, "Contour Area: " + contArea);
                //TODO for this condition, add more of the stuff from grip and change some stuff.
                if(!((percent <= 0.001 || percent >= 0.03 /*TODO Find which percentage works best(Used to be 0.5, 10->Now is 0.1%,3%)*/) /*||  (contcpy.width() > 150 || contcpy.height() > 175 || contcpy.width() < 25 || contcpy.height() < 30)*/)) {
                    newConts.add(contcpy);
                }
            }

            Log.d(TAG, "Num of newConts: " + newConts.size());
            if(newConts.size() > 0) {
                //Choosing topmost(bottom)
                Log.d(TAG, "Number of newConts: " + newConts.size());
                Moments mu;
                MatOfPoint corresponding = new MatOfPoint();
                double maxX = Double.MAX_VALUE;
                for(int i = 0; i < newConts.size(); i++) {
                    MatOfPoint cont = newConts.get(i);
                    mu = Imgproc.moments(cont);
                    double x = mu.get_m10() / mu.get_m00();
                    if(x < maxX) {
                        maxX = x;
                        corresponding = cont;
                    }
                } // Corresponding must be the largest contour
                Rect bound = Imgproc.boundingRect(corresponding);
                Moments finalMom = Imgproc.moments(corresponding);
                Center center = new Center(finalMom.get_m10() / finalMom.get_m00(), finalMom.get_m01() / finalMom.get_m00());
                fe.put("center", center);
                double dist = calcDistAxis206(bound.width, WIDTH_TARGET, 320, STANDARD_VIEW_ANGLE);
                fe.put("dist", dist);
                fe.put("roi", bound);
            } else {
                fe.put("center", NO_CENTER);
                fe.put("dist", 0.0);
                fe.put("roi", new Rect());
            }
        } else {
            fe.put("center", NO_CENTER);
            fe.put("dist", 0.0);
            fe.put("roi", new Rect());
        }
        return fe;
    }

    public Mat thresh(Mat initial) {
        double[] hue = {0.0, 180.0};
        double[] sat = {0.0, 255.0};
        double[] val = {230.0, 250.0};
        Mat hsv = new Mat();
        Imgproc.cvtColor(initial, hsv, Imgproc.COLOR_BGR2HSV);
        Core.inRange(hsv, new Scalar(hue[0], sat[0], val[0]), new Scalar(hue[1], sat[1], val[1]), hsv);
        return hsv;
    }

    /**
    * First: Gaussian Blur radius of 3
     * 2nd: erode(4) + dilate(11)
     * 3rd: RGB threshold
     * 4th: HSL threshold(Separate Threads?)
     * 5th: Mask rgb and hsl threshed imgs
     * 6th: Find contours + Filter conts
     * 7th: Get convex hulls, and use as approximate new contours(Also approxpolydp + fillconvexpoly)
     * Take bounding box + subtract, bc difference would be small(Also aspect ratio of 4/15 and 2/15)
     * Crop img?
     * After finding biggest rect, get center of mass using moments
     *
     * @return Dictionary of objects
     * -> "center" Center of found object
     * -> "roi" Bounding rect of selected object
     * -> "m" after convex hull fill
     * -> "hsv"
     * -> "thresh" the mat after initial thresholding
     * -> "subImage" subtraced image
     * -> "blobMat" after find contours of blobs*/
    public Map<String, Object> newMethod3(Mat m) {
        Mat initial = m;
        double blurRadius = 2.7027027027027026;
        int radius = (int) (blurRadius + 0.5);
        Imgproc.GaussianBlur(initial, initial, new Size(6*radius + 1, 6*radius + 1), radius);
        Imgproc.erode(initial, initial, new Mat(), new Point(-1, -1), 4, Core.BORDER_CONSTANT, new Scalar(-1));
        Imgproc.dilate(initial, initial, new Mat(), new Point(-1, -1), 11, Core.BORDER_CONSTANT, new Scalar(-1));
        /*HSVThread hsvSide = new HSVThread(initial);
        hsvSide.start();
        RGBThread rgbSide = new RGBThread(initial);
        rgbSide.start();

        try {
            rgbSide.join();
            hsvSide.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        //Essentially same as masking
        Mat bitwise = new Mat();
        try {
            bitwise = hsvSide.getFinalMat();
            //Core.bitwise_or(rgbSide.getFinalMat(), hsvSide.getFinalMat(), bitwise); //think &&
        } catch (Exception e) {
            Log.d(TAG, "Broken, rip");
            return new HashMap<>(); //dun got broked, don' do this
        }*/
        Mat hls = new Mat();
        Imgproc.cvtColor(initial, hls, Imgproc.COLOR_BGR2HLS);
        double[] hue = {0.0, 94.48494812631102};
        double[] luminance = {240.7823741007194, 255.0};
        double[] saturation = {163.73885965854183, 255.0};
        Scalar[] scalars = new Scalar[] {new Scalar(hue[0], luminance[0] ,saturation[0]), new Scalar(hue[1], luminance[1], saturation[1])};
        Core.inRange(hls, scalars[0], scalars[1], hls);
        Log.d(TAG, "Done HSVThread");

        Mat rgb = new Mat();
        Imgproc.cvtColor(initial, rgb, Imgproc.COLOR_BGR2RGB);
        double[] red = {0.0, 252.73936170212767};
        double[] green = {252.5988700564972, 255.0};
        double[] blue = {206.49717514124293, 255.0};
        Scalar[] scalars2 = new Scalar[] {new Scalar(red[0], green[0], blue[0]), new Scalar(red[1], green[1], blue[1])};
        Core.inRange(rgb, scalars2[0], scalars2[1], rgb);
        Log.d(TAG, "Done RGBThread");

        Mat bitwise = new Mat();
        try {
            Core.bitwise_and(hls, rgb, bitwise);
        } catch (Exception e) {
            Log.d(TAG, "Broken, rip");
            return new HashMap<>();
        }

        List<MatOfPoint> conts = new ArrayList<>();
        List<MatOfPoint> newConts = new ArrayList<>();
        //Probs not using this but idk
        Mat hierarchy = new Mat();
        //TODO Should it be retr_external or retr_list?
        Imgproc.findContours(bitwise, conts, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Log.d(TAG, "Conts b4 filtering: " + conts.size());
        //TODO fillConvexPoly here?
        //Maybe this is not necessary since we are using RETR_EXTERNAL, but idk
        for(MatOfPoint mop : conts) {
            Imgproc.fillConvexPoly(bitwise, mop, new Scalar(255, 255, 255));
        }
        Imgproc.findContours(bitwise, newConts, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<MatOfPoint> filtConts = new ArrayList<>();
        double filterContoursMinArea = 500.0;
        double filterContoursMinPerimeter = 220.0;
        double filterContoursMinWidth = 25.0;
        double filterContoursMaxWidth = 150.0;
        double filterContoursMinHeight = 35.0;
        double filterContoursMaxHeight = 175.0;
        double[] filterContoursSolidity = {0.0, 100};
        double filterContoursMaxVertices = 200.0;
        double filterContoursMinVertices = 4.0;
        double filterContoursMinRatio = 0.25;
        double filterContoursMaxRatio = 0.8;
        filterContours(newConts, filterContoursMinArea, filterContoursMinPerimeter, filterContoursMinWidth, filterContoursMaxWidth, filterContoursMinHeight, filterContoursMaxHeight, filterContoursSolidity, filterContoursMaxVertices, filterContoursMinVertices, filterContoursMinRatio, filterContoursMaxRatio, filtConts);
        ArrayList<MatOfPoint> convConts = new ArrayList<>();
        convexHulls(filtConts, convConts);

        Map<String, Object> foundElements = new HashMap<>();
        MatOfPoint largest = new MatOfPoint();
        Moments mu;
        double finalArea = Double.MAX_VALUE;
        Log.d(TAG, "Number of contours: " + convConts.size());
        if(convConts.size() > 0) {
            Log.d(TAG, "Condition registered");
            for(int i = 0; i < convConts.size(); i++) {
                MatOfPoint mop = convConts.get(i);
                MatOfPoint2f mop2f = new MatOfPoint2f();
                //TODO Test new epsilon values of approxPolyDP
                Imgproc.approxPolyDP(new MatOfPoint2f(mop.toArray()), mop2f, 5.0, true);
                mu = Imgproc.moments(new MatOfPoint(mop2f.toArray()));
                double x = mu.get_m10() / mu.get_m00();
                Rect bound = Imgproc.boundingRect(new MatOfPoint(mop2f.toArray()));
                double area = Math.abs(bound.area() - contourArea(new MatOfPoint(mop2f.toArray())));
                if(area < finalArea && mop2f.rows() == 4/*TODO Check this(MatOfPoint2f.rows()==4)*/ && 0 <= x && 144 >= x) {
                    finalArea = area;
                    //TODO Make sure that var area will always be less than the initial value of finalArea(Double.MAX_VALUE)
                    largest = new MatOfPoint(mop2f.toArray());
                    Log.d(TAG, "Got new largest");
                }
            }
            Rect boundingRect = Imgproc.boundingRect(largest);
            mu = Imgproc.moments(largest);
            Center center = new Center(mu.get_m10() / mu.get_m00(), mu.get_m01() / mu.get_m00());
            foundElements.put("center", center);
            double dist = calcDistAxis206(boundingRect.width, WIDTH_TARGET, 320, STANDARD_VIEW_ANGLE);
            foundElements.put("dist", dist);
            foundElements.put("roi", boundingRect);
            Log.d(TAG, "Executed successfully with contour");
        } else {
            foundElements.put("center", NO_CENTER);
            foundElements.put("dist", 0.0);
            foundElements.put("roi", new Rect());
            Log.d(TAG, "No contours");
        }
        return foundElements;
    }

    private void hslThreshold(Mat input, double[] hue, double[] sat, double[] lum,
                              Mat out) {
        Imgproc.cvtColor(input, out, Imgproc.COLOR_BGR2HLS);
        Core.inRange(out, new Scalar(hue[0], lum[0], sat[0]),
                new Scalar(hue[1], lum[1], sat[1]), out);
    }

    /**
     * An indication of which type of filter to use for a blur.
     * Choices are BOX, GAUSSIAN, MEDIAN, and BILATERAL
     */
    enum BlurType{
        BOX("Box Blur"), GAUSSIAN("Gaussian Blur"), MEDIAN("Median Filter"),
        BILATERAL("Bilateral Filter");

        private final String label;

        BlurType(String label) {
            this.label = label;
        }

        public static BlurType get(String type) {
            if (BILATERAL.label.equals(type)) {
                return BILATERAL;
            }
            else if (GAUSSIAN.label.equals(type)) {
                return GAUSSIAN;
            }
            else if (MEDIAN.label.equals(type)) {
                return MEDIAN;
            }
            else {
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
     * @param input The image on which to perform the blur.
     * @param type The blurType to perform.
     * @param doubleRadius The radius for the blur.
     * @param output The image in which to store the output.
     */
    private void blur(Mat input, BlurType type, double doubleRadius,
                      Mat output) {
        int radius = (int)(doubleRadius + 0.5);
        int kernelSize;
        switch(type){
            case BOX:
                kernelSize = 2 * radius + 1;
                Imgproc.blur(input, output, new Size(kernelSize, kernelSize));
                break;
            case GAUSSIAN:
                kernelSize = 6 * radius + 1;
                Imgproc.GaussianBlur(input,output, new Size(kernelSize, kernelSize), radius);
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

    public double findDist(double pixelWidth, double focalLength, double realWidth) {
        return (realWidth * focalLength)/pixelWidth;
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
                Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2HLS);
                double[] hslThresholdHue = {0.0, 94.48494812631102};
                double[] hslThresholdSaturation = {163.73885965854183, 255.0};
                double[] hslThresholdLuminance = {240.7823741007194, 255.0};
                Scalar[] scalars = new Scalar[] {new Scalar(hslThresholdHue[0], hslThresholdLuminance[0] ,hslThresholdSaturation[0]), new Scalar(hslThresholdHue[1], hslThresholdLuminance[1], hslThresholdSaturation[1])};
                Core.inRange(m, scalars[0], scalars[1], m); //"20,3,215 - > 75,250,250"
                //used to be 15,2,210 -> 100,255,255

                ready = true;
                Log.d(TAG, "Done HSVThread");
            }
        }

        public Mat getFinalMat() {
            if (ready) {
                return m;
            }
            return new Mat();
        }
    }

    public class RGBThread extends Thread {

        Mat m;
        boolean ready;

        public RGBThread(Mat mat) {
            m = new Mat();
            mat.copyTo(m);
            ready = false; //kinda redundant but meh
        }

        @Override
        public void run() {
            if (m != null) {
                //brightness side
                Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2RGB);
                double[] rgbThresholdRed = {0.0, 252.73936170212767};
                double[] rgbThresholdGreen = {252.5988700564972, 255.0};
                double[] rgbThresholdBlue = {206.49717514124293, 255.0};
                Scalar[] scalars = new Scalar[] {new Scalar(rgbThresholdRed[0], rgbThresholdGreen[0], rgbThresholdBlue[0]), new Scalar(rgbThresholdRed[1], rgbThresholdGreen[1], rgbThresholdBlue[1])};
                Core.inRange(m, scalars[0], scalars[1], m); //"20,3,215 - > 75,250,250"
                //              Imgproc.threshold(m,m,200, 255, Imgproc.THRESH_BINARY);

                ready = true;
                Log.d(TAG, "Done RGBThread");
            }
        }

        public Mat getFinalMat() {
            if (ready) {
                return m;
            }
            return new Mat();
        }
    }

    public void stackOverflowMethod(Mat m) {
        Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2RGB);
        Mat initial = m;
        Mat threshed = new Mat();
        List<Mat> channels = new ArrayList<>();
        Core.split(initial, channels);
        Core.inRange(channels.get(1), new Scalar(250), new Scalar(270), threshed);
        Mat structEl = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE, new Size());
        Imgproc.morphologyEx(threshed, threshed, Imgproc.MORPH_OPEN, structEl);
        Imgproc.erode(threshed, threshed, erode);
        Imgproc.dilate(threshed, threshed, dilate);
        FeatureDetector blobDetector = FeatureDetector.create(FeatureDetector.SURF);
        MatOfKeyPoint realBlobs = new MatOfKeyPoint();
        blobDetector.detect(threshed, realBlobs);
        List<KeyPoint> blobs = realBlobs.toList();
        Log.d(TAG, "Number of blobs: " + blobs.size());
        if(blobs.size() > 0) {
            for(KeyPoint kp : blobs) {
                //kp.size
            }
        }
    }

    public Map<String, Object> newMethod(Mat m) {
        Map<String, Object> foundElements = new HashMap<>();
        Imgproc.GaussianBlur(m, m, new Size(31,31), 5);
        Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2HLS);
        Core.inRange(m, new Scalar(71.0, 105.0, 140.0), new Scalar(101.0, 255.0, 236.0), m);

        Point tl = new Point(m.cols()/2, m.rows());
        Point br = new Point(m.cols(), 0);
        Rect rect = new Rect(tl, br);

        Mat crop = new Mat(m, rect);

        List<MatOfPoint> conts = new ArrayList<>();
        List<Point> finalPoints = new ArrayList<>();
        Imgproc.findContours(crop, conts, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        // Step Filter_Contours0:
        /*double filterContoursMinArea = 1000.0;
        double filterContoursMinPerimeter = 200.0;
        double filterContoursMinWidth = 25.0;
        double filterContoursMaxWidth = 125.0;
        double filterContoursMinHeight = 35.0;
        double filterContoursMaxHeight = 150.0;
        double[] filterContoursSolidity = {93.75, 100};
        double filterContoursMaxVertices = 1000000.0;
        double filterContoursMinVertices = 4.0;
        double filterContoursMinRatio = 0.3;
        double filterContoursMaxRatio = 1.0;
        filterContours(conts, filterContoursMinArea, filterContoursMinPerimeter, filterContoursMinWidth, filterContoursMaxWidth, filterContoursMinHeight, filterContoursMaxHeight, filterContoursSolidity, filterContoursMaxVertices, filterContoursMinVertices, filterContoursMinRatio, filterContoursMaxRatio, conts);
        */Log.d(TAG, "Num of conts: " + conts.size());
        if(conts.size() > 0) {
            for(MatOfPoint mop : conts) {
                List<Point> points = mop.toList();
                finalPoints.addAll(points);
            }
            MatOfPoint finalCont = new MatOfPoint(finalPoints.toArray(new Point[finalPoints.size()]));
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.convexHull(finalCont, new MatOfInt());
            approxPolyDP(new MatOfPoint2f(finalCont.toArray()), approx, 8, true);
            finalCont = new MatOfPoint(approx.toArray());
            Rect boundingRect = Imgproc.boundingRect(finalCont);
            foundElements.put("center", new Center((boundingRect.br().x + boundingRect.tl().x)/2, (boundingRect.br().y + boundingRect.tl().y)/2));
            double dist = calcDistAxis206(boundingRect.width, WIDTH_TARGET, 320, STANDARD_VIEW_ANGLE);
            foundElements.put("dist", dist);
            foundElements.put("roi", boundingRect);
        } else {
            foundElements.put("center", NO_CENTER);
            //double dist = calcDistAxis206(roi.width, WIDTH_TARGET, 320, STANDARD_VIEW_ANGLE);
            foundElements.put("dist", 0.0);
            foundElements.put("roi", new Rect());
        }
        return foundElements;
    }



    public Map<String, Object> templateMatching(Mat img, Mat templ, int match_method) {
        // / Create the result matrix
        int result_cols = img.cols() - templ.cols() + 1;
        int result_rows = img.rows() - templ.rows() + 1;
        Mat result = new Mat(result_rows, result_cols, CvType.CV_32F);

        // / Do the Matching and Normalize
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGRA2BGR);
        img.convertTo(img, CvType.CV_32F);
        templ.convertTo(templ, CvType.CV_32F);
        Imgproc.matchTemplate(img, templ, result, match_method);
        Core.normalize(result, result, 0, 1, Core.NORM_MINMAX, -1, new Mat());

        // / Localizing the best match with minMaxLoc
        Core.MinMaxLocResult mmr = Core.minMaxLoc(result);

        Point matchLoc;
        if (match_method == Imgproc.TM_SQDIFF || match_method == Imgproc.TM_SQDIFF_NORMED) {
            matchLoc = mmr.minLoc;
        } else {
            matchLoc = mmr.maxLoc;
        }

        // / Show me what you got
        //Rect roi = new Rect(img, matchLoc, new Point(matchLoc.x + templ.cols(), matchLoc.y + templ.rows()), new Scalar(0, 255, 0));
        roi = new Rect((int) matchLoc.x, (int) matchLoc.y, templ.cols(), templ.rows());

        Map<String, Object> foundElements = new HashMap<>();
        if (roi.equals(null)) {
            foundElements.put("center", NO_CENTER);
            foundElements.put("dist", 0.0);
            foundElements.put("roi", new Rect());
        } else {
            double dist = calcDistAxis206(roi.width, WIDTH_TARGET, 320, STANDARD_VIEW_ANGLE);
            foundElements.put("dist", dist);
            foundElements.put("center", center);
            foundElements.put("roi", roi);
        }
        return foundElements;
    }

    public Map<String, Object> findingRects(Mat m) {
        // Step CV_erode0:
        Mat cvErodeSrc = m;
        Mat cvErodeKernel = new Mat();
        Point cvErodeAnchor = new Point(-1, -1);
        double cvErodeIterations = 15.0;
        int cvErodeBordertype = Core.BORDER_CONSTANT;
        Scalar cvErodeBordervalue = new Scalar(-1);
        cvErode(cvErodeSrc, cvErodeKernel, cvErodeAnchor, cvErodeIterations, cvErodeBordertype, cvErodeBordervalue, cvErodeOutput);

        // Step CV_dilate0:
        Mat cvDilateSrc = cvErodeOutput;
        Mat cvDilateKernel = new Mat();
        Point cvDilateAnchor = new Point(-1, -1);
        double cvDilateIterations = 15.0;
        int cvDilateBordertype = Core.BORDER_CONSTANT;
        Scalar cvDilateBordervalue = new Scalar(-1);
        cvDilate(cvDilateSrc, cvDilateKernel, cvDilateAnchor, cvDilateIterations, cvDilateBordertype, cvDilateBordervalue, cvDilateOutput);

        // Step CV_cvtColor0:
        Mat cvCvtcolor0Src = cvDilateOutput;
        int cvCvtcolor0Code = Imgproc.COLOR_BGR2HSV;
        cvCvtcolor(cvCvtcolor0Src, cvCvtcolor0Code, cvCvtcolor0Output);


        // Step RGB_Threshold0:
        Mat rgbThreshold0Input = cvCvtcolor0Output;
        double[] rgbThreshold0Red = {202.17514124293785, 255.0};
        double[] rgbThreshold0Green = {0.0, 255.0};
        double[] rgbThreshold0Blue = {0.0, 255.0};
        rgbThreshold(rgbThreshold0Input, rgbThreshold0Red, rgbThreshold0Green, rgbThreshold0Blue, rgbThreshold0Output);

        // Step Find_Contours0:
        Mat findContours0Input = rgbThreshold0Output;
        boolean findContours0ExternalOnly = false;
        findContours(findContours0Input, findContours0ExternalOnly, findContours0Output);

        // Step Mask0:
        Mat maskInput = cvDilateOutput;
        Mat maskMask = rgbThreshold0Output;
        mask(maskInput, maskMask, maskOutput);
        //Mat m2 = maskOutput;
        //m2.convertTo(m2, CvType.CV_8S);

        // Step CV_cvtColor1:
        Mat cvCvtcolor1Src = maskOutput;
        int cvCvtcolor1Code = Imgproc.COLOR_BGR2RGB;
        cvCvtcolor(cvCvtcolor1Src, cvCvtcolor1Code, cvCvtcolor1Output);

        // Step RGB_Threshold1:
        Mat rgbThreshold1Input = cvCvtcolor1Output;
        double[] rgbThreshold1Red = {0.0, 255.0};
        double[] rgbThreshold1Green = {153.6723163841808, 255.0};
        double[] rgbThreshold1Blue = {0.0, 255.0};
        rgbThreshold(rgbThreshold1Input, rgbThreshold1Red, rgbThreshold1Green, rgbThreshold1Blue, rgbThreshold1Output);

        // Crop img
        Mat initial = rgbThreshold1Output;
        Mat crop = new Mat(initial, new Rect(initial.cols()*2/3, 0, initial.cols(), initial.rows() / 3));

        // Step Find_Contours1:
        Mat findContours1Input = crop;
        boolean findContours1ExternalOnly = false;
        findContours(findContours1Input, findContours1ExternalOnly, findContours1Output);

        // Step Filter_Contours0:
        ArrayList<MatOfPoint> filterContoursContours = findContours1Output;
        double filterContoursMinArea = 1000.0;
        double filterContoursMinPerimeter = 250.0;
        double filterContoursMinWidth = 25.0;
        double filterContoursMaxWidth = 125.0;
        double filterContoursMinHeight = 35.0;
        double filterContoursMaxHeight = 150.0;
        double[] filterContoursSolidity = {0, 100};
        double filterContoursMaxVertices = 1000000.0;
        double filterContoursMinVertices = 4.0;
        double filterContoursMinRatio = 0.4;
        double filterContoursMaxRatio = 1.0;
        filterContours(filterContoursContours, filterContoursMinArea, filterContoursMinPerimeter, filterContoursMinWidth, filterContoursMaxWidth, filterContoursMinHeight, filterContoursMaxHeight, filterContoursSolidity, filterContoursMaxVertices, filterContoursMinVertices, filterContoursMinRatio, filterContoursMaxRatio, filterContoursOutput);

        // Step Convex_Hulls0:
        ArrayList<MatOfPoint> convexHullsContours = filterContoursOutput;
        convexHulls(convexHullsContours, convexHullsOutput);

        List<MatOfPoint> conts = convexHullsOutput;

        for(int i = 0; i < conts.size(); i++) {

            //Imgproc.fillCon
        }

        boolean noValid = true;
        Moments mu;
        MatOfPoint2f approx = new MatOfPoint2f();
        // if any contour exist...
        //Log.d(TAG, "Blob count: " + blobContours.size());
        if (conts.size() > 0) {
            int largest = 0;

            // for each remaining contour, find the biggest
            for (int h = 0; h < conts.size(); h++) {
                MatOfPoint cont = conts.get(h);
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
                if (area > 50.0 && area >= contourArea(conts.get(largest)) && approx.rows() == 4 && y < MAX_Y_COORD && y > MIN_Y_COORD) { //&& !circle)) {
                    noValid = false;
                    largest = h;

                }


            }

            //Log.d("->timelog", "filtering blobs t: " + (Calendar.getInstance().getTimeInMillis() - _time));
            //_time = Calendar.getInstance().getTimeInMillis();

            roi = Imgproc.boundingRect(conts.get(largest));
            mu = Imgproc.moments(conts.get(largest));
            center = new Center(mu.get_m10() / mu.get_m00(), mu.get_m01() / mu.get_m00());
        }
        Map<String, Object> foundElements = new HashMap<>();

        if (noValid) {
            foundElements.put("center", NO_CENTER);
            foundElements.put("dist", 0.0);
        } else {
            double dist = calcDistAxis206(roi.width, WIDTH_TARGET, 320, STANDARD_VIEW_ANGLE);
            foundElements.put("dist", dist);
            foundElements.put("center", center);
        }

        foundElements.put("roi", roi);
        return foundElements;
    }

    public double calcDistAxis206(double obj_pix, double obj_in, double view_pix, double max_cam_angle) {
        return view_pix * obj_in / (2 * Math.tan(max_cam_angle) * obj_pix);
    }

    /**
     * This method is a generated getter for the output of a CV_erode.
     * @return Mat output from CV_erode.
     */
    public Mat cvErodeOutput() {
        return cvErodeOutput;
    }

    /**
     * This method is a generated getter for the output of a CV_dilate.
     * @return Mat output from CV_dilate.
     */
    public Mat cvDilateOutput() {
        return cvDilateOutput;
    }

    /**
     * This method is a generated getter for the output of a CV_cvtColor.
     * @return Mat output from CV_cvtColor.
     */
    public Mat cvCvtcolor0Output() {
        return cvCvtcolor0Output;
    }

    /**
     * This method is a generated getter for the output of a RGB_Threshold.
     * @return Mat output from RGB_Threshold.
     */
    public Mat rgbThreshold0Output() {
        return rgbThreshold0Output;
    }

    /**
     * This method is a generated getter for the output of a Find_Contours.
     * @return ArrayList<MatOfPoint> output from Find_Contours.
     */
    public ArrayList<MatOfPoint> findContours0Output() {
        return findContours0Output;
    }

    /**
     * This method is a generated getter for the output of a Mask.
     * @return Mat output from Mask.
     */
    public Mat maskOutput() {
        return maskOutput;
    }

    /**
     * This method is a generated getter for the output of a CV_cvtColor.
     * @return Mat output from CV_cvtColor.
     */
    public Mat cvCvtcolor1Output() {
        return cvCvtcolor1Output;
    }

    /**
     * This method is a generated getter for the output of a RGB_Threshold.
     * @return Mat output from RGB_Threshold.
     */
    public Mat rgbThreshold1Output() {
        return rgbThreshold1Output;
    }

    /**
     * This method is a generated getter for the output of a Find_Contours.
     * @return ArrayList<MatOfPoint> output from Find_Contours.
     */
    public ArrayList<MatOfPoint> findContours1Output() {
        return findContours1Output;
    }

    /**
     * This method is a generated getter for the output of a Filter_Contours.
     * @return ArrayList<MatOfPoint> output from Filter_Contours.
     */
    public ArrayList<MatOfPoint> filterContoursOutput() {
        return filterContoursOutput;
    }

    /**
     * This method is a generated getter for the output of a Convex_Hulls.
     * @return ArrayList<MatOfPoint> output from Convex_Hulls.
     */
    public ArrayList<MatOfPoint> convexHullsOutput() {
        return convexHullsOutput;
    }


    /**
     * Expands area of lower value in an image.
     * @param src the Image to erode.
     * @param kernel the kernel for erosion.
     * @param anchor the center of the kernel.
     * @param iterations the number of times to perform the erosion.
     * @param borderType pixel extrapolation method.
     * @param borderValue value to be used for a constant border.
     * @param dst Output Image.
     */
    private void cvErode(Mat src, Mat kernel, Point anchor, double iterations,
                         int borderType, Scalar borderValue, Mat dst) {
        if (kernel == null) {
            kernel = new Mat();
        }
        if (anchor == null) {
            anchor = new Point(-1,-1);
        }
        if (borderValue == null) {
            borderValue = new Scalar(-1);
        }
        Imgproc.erode(src, dst, kernel, anchor, (int)iterations, borderType, borderValue);
    }

    /**
     * Expands area of higher value in an image.
     * @param src the Image to dilate.
     * @param kernel the kernel for dilation.
     * @param anchor the center of the kernel.
     * @param iterations the number of times to perform the dilation.
     * @param borderType pixel extrapolation method.
     * @param borderValue value to be used for a constant border.
     * @param dst Output Image.
     */
    private void cvDilate(Mat src, Mat kernel, Point anchor, double iterations,
                          int borderType, Scalar borderValue, Mat dst) {
        if (kernel == null) {
            kernel = new Mat();
        }
        if (anchor == null) {
            anchor = new Point(-1,-1);
        }
        if (borderValue == null){
            borderValue = new Scalar(-1);
        }
        Imgproc.dilate(src, dst, kernel, anchor, (int)iterations, borderType, borderValue);
    }

    /**
     * Filter out an area of an image using a binary mask.
     * @param input The image on which the mask filters.
     * @param mask The binary image that is used to filter.
     * @param output The image in which to store the output.
     */
    private void mask(Mat input, Mat mask, Mat output) {
        mask.convertTo(mask, CvType.CV_8U);
        Core.bitwise_xor(output, output, output);
        input.copyTo(output, mask);
    }

    /**
     * Converts an image from one color space to another.
     * @param src Image to convert.
     * @param code conversion code.
     * @param dst converted Image.
     */
    private void cvCvtcolor(Mat src, int code, Mat dst) {
        Imgproc.cvtColor(src, dst, code);
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
     * Sets the values of pixels in a binary image to their distance to the nearest black pixel.
     * @param input The image on which to perform the Distance Transform.
     * //@param type The Transform.
     * //@param maskSize the size of the mask.
     * //@param output The image in which to store the output.
     */
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


    /**
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

    /**
     * Compute the convex hulls of contours.
     * @param inputContours The contours on which to perform the operation.
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
                double[] point = new double[] {contour.get(index, 0)[0], contour.get(index, 0)[1]};
                mopHull.put(j, 0, point);
            }
            outputContours.add(mopHull);
        }
    }

}
